"""
Document Studio API: transform a chat / reference documents / a brief into a templated,
multi-section document and export it as PDF or PPTX.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel

from ...core.config import settings
from ...core.db import get_conn
from ...docgen import store, images, custom_templates
from ...docgen.templates import list_templates, get_template
from ...docgen.pipeline import generate_document
from ...docgen.render_pdf import render_pdf
from ...docgen.render_pptx import render_pptx

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/docgen", tags=["docgen"])

# Limit concurrent (LLM-heavy) generations process-wide.
_GEN_SEM = asyncio.Semaphore(max(1, int(os.getenv("DOCGEN_CONCURRENCY", "2"))))
_SOURCE_MAX_CHARS = int(os.getenv("DOCGEN_SOURCE_MAX_CHARS", "30000"))


class GenerateIn(BaseModel):
    template_id: str
    mode: str = "brief"            # chat | kb | brief
    chat_id: Optional[str] = None
    doc_ids: Optional[List[str]] = None
    brief: Optional[str] = None
    title: Optional[str] = None
    org: Optional[str] = None
    custom_template: Optional[str] = None
    with_images: bool = False


@router.get("/templates")
def get_templates():
    # Built-in templates first, then any uploaded custom templates.
    return {"templates": list_templates() + store.list_custom_templates()}


@router.get("/image-status")
def image_status():
    return images.image_backend_status()


_TEMPLATE_DIR = os.path.join(settings.DATA_DIR, "docgen_templates")
_MAX_TEMPLATE_BYTES = int(os.getenv("DOCGEN_MAX_TEMPLATE_BYTES", str(25 * 1024 * 1024)))


@router.post("/templates/upload")
async def upload_template(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > _MAX_TEMPLATE_BYTES:
        raise HTTPException(status_code=413, detail="Template file too large")
    try:
        blueprint = custom_templates.parse_template_file(file.filename or "template", data)
    except Exception as e:
        logger.error("DocGen template parse failed: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Could not parse template: {e}")
    # Persist the raw file so PPTX templates can be rendered INTO (theme-preserving) on export.
    source_path = None
    try:
        os.makedirs(_TEMPLATE_DIR, exist_ok=True)
        ext = os.path.splitext(file.filename or "")[1].lower() or ".bin"
        source_path = os.path.join(_TEMPLATE_DIR, f"{abs(hash((file.filename, len(data))))}{ext}")
        with open(source_path, "wb") as f:
            f.write(data)
    except Exception:
        source_path = None
    tid = store.save_custom_template(blueprint, source_path=source_path)
    bp = store.get_custom_template(tid) or {}
    return {"template": {
        "id": tid, "name": bp.get("name", "Custom Template"), "icon": "File",
        "persona": bp.get("persona", "General Assistant"),
        "description": bp.get("description", "Uploaded custom template."),
        "sections": [s.get("title", "") for s in bp.get("section_blueprint", [])],
        "default_doc_type": bp.get("default_doc_type", "Document"),
        "theme": bp.get("theme", "midnight"), "supports_images": False, "custom": True,
        "source_format": bp.get("source_format", ""),
    }}


def _build_source_text(inp: GenerateIn) -> str:
    """Assemble the grounding text from the chosen source (bounded)."""
    cap = _SOURCE_MAX_CHARS
    if inp.mode == "chat" and inp.chat_id:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE chat_id=? ORDER BY created_at ASC",
                (inp.chat_id,),
            ).fetchall()
        text = "\n".join(f"{(r[0] or 'user').upper()}: {r[1]}" for r in rows if r[1])
        return text[:cap]
    if inp.mode == "kb" and inp.doc_ids:
        ids = [d for d in inp.doc_ids if d][:50]
        if not ids:
            return ""
        ph = ",".join("?" for _ in ids)
        with get_conn() as conn:
            rows = conn.execute(
                f"SELECT d.filename, c.text FROM chunks c JOIN documents d ON c.doc_id = d.id "
                f"WHERE c.doc_id IN ({ph}) ORDER BY c.doc_id, c.chunk_index",
                ids,
            ).fetchall()
        parts: List[str] = []
        total = 0
        for _fn, txt in rows:
            if not txt:
                continue
            parts.append(txt)
            total += len(txt)
            if total >= cap:
                break
        return "\n\n".join(parts)[:cap]
    return ""  # brief mode: no source grounding


async def _run_generation(job_id: str, inp: GenerateIn, tmpl: dict) -> None:
    try:
        source_text = await asyncio.get_running_loop().run_in_executor(None, _build_source_text, inp)

        async def progress(stage: str, pct: int):
            if stage in store.ACTIVE_STATUSES or stage in ("planning", "writing", "assembling"):
                store.update_status(job_id, stage, stage=stage)

        # For uploaded custom templates, pass the full blueprint as a template override.
        template_override = tmpl if tmpl.get("custom") else None

        async with _GEN_SEM:
            store.update_status(job_id, "planning", stage="planning")
            doc = await generate_document(
                template_id=tmpl["id"],
                brief=(inp.brief or "").strip(),
                source_text=source_text,
                title=(inp.title or "").strip(),
                org=(inp.org or tmpl.get("default_org") or "EchoMind"),
                custom_template=(inp.custom_template or "").strip(),
                template=template_override,
                with_images=bool(inp.with_images),
                progress=progress,
            )
        # Fill template defaults the pipeline may have left blank.
        doc = doc or {}
        if not doc.get("org"):
            doc["org"] = (inp.org or tmpl.get("default_org") or "EchoMind")
        if not doc.get("doc_type"):
            doc["doc_type"] = tmpl.get("default_doc_type") or "Document"
        doc["template_id"] = tmpl["id"]
        doc["persona"] = tmpl.get("persona", "")
        if (inp.title or "").strip():
            doc["title"] = inp.title.strip()
        store.update_status(job_id, "done", stage="done", doc=doc)
        logger.info("DocGen: job %s done (%d blocks)", job_id, len(doc.get("blocks", [])))
    except Exception as e:
        logger.error("DocGen: job %s failed: %s", job_id, e, exc_info=True)
        store.update_status(job_id, "error", error=str(e))


@router.post("/generate")
async def generate(inp: GenerateIn, background_tasks: BackgroundTasks):
    if inp.mode not in ("chat", "kb", "brief"):
        raise HTTPException(status_code=400, detail="mode must be chat, kb, or brief")
    if inp.mode == "chat" and not inp.chat_id:
        raise HTTPException(status_code=400, detail="chat_id required for mode=chat")
    if inp.mode == "kb" and not inp.doc_ids:
        raise HTTPException(status_code=400, detail="doc_ids required for mode=kb")
    if inp.mode == "brief" and not (inp.brief or "").strip():
        raise HTTPException(status_code=400, detail="brief required for mode=brief")

    # Resolve a custom (uploaded) template by id, else a built-in.
    tmpl = None
    if (inp.template_id or "").startswith("ctpl_"):
        tmpl = store.get_custom_template(inp.template_id)
        if not tmpl:
            raise HTTPException(status_code=404, detail="Custom template not found")
    if tmpl is None:
        tmpl = get_template(inp.template_id)
    title = (inp.title or "").strip() or tmpl.get("name", "Document")
    job_id = store.create_job(title=title, template_id=tmpl["id"], persona=tmpl.get("persona", ""), mode=inp.mode)
    background_tasks.add_task(_run_generation, job_id, inp, tmpl)
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs")
def list_generation_jobs(limit: int = 50):
    return {"jobs": store.list_jobs(limit=limit)}


@router.get("/jobs/{job_id}")
def get_generation_job(job_id: str):
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Document job not found")
    return job


@router.delete("/jobs/{job_id}")
def delete_generation_job(job_id: str):
    if not store.delete_job(job_id):
        raise HTTPException(status_code=404, detail="Document job not found")
    return {"ok": True, "deleted": job_id}


def _safe_filename(title: str, ext: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", (title or "document")).strip("_") or "document"
    return f"{base[:60]}.{ext}"


@router.get("/jobs/{job_id}/export")
def export_generation_job(job_id: str, format: str = "pdf"):
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Document job not found")
    if job.get("status") != "done" or not job.get("document"):
        raise HTTPException(status_code=400, detail="Document not ready to export")
    fmt = (format or "pdf").lower().strip()
    doc = job["document"]
    try:
        if fmt == "pptx":
            data = None
            # If generated from an uploaded PPTX template, render INTO it to preserve its theme/master.
            tid = job.get("template_id") or ""
            if tid.startswith("ctpl_"):
                bp = store.get_custom_template(tid)
                src = (bp or {}).get("_source_path")
                if bp and (bp.get("source_format") == "pptx") and src and os.path.exists(src):
                    try:
                        with open(src, "rb") as f:
                            data = custom_templates.render_into_pptx_template(f.read(), doc)
                    except Exception as e:
                        logger.warning("DocGen: render-into-template failed (%s); using default deck", e)
                        data = None
            if data is None:
                data = render_pptx(doc)
            media = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            fn = _safe_filename(job.get("title", "document"), "pptx")
        else:
            data = render_pdf(doc)
            media = "application/pdf"
            fn = _safe_filename(job.get("title", "document"), "pdf")
    except ImportError as e:
        raise HTTPException(status_code=501, detail=f"Export library not available: {e}")
    except Exception as e:
        logger.error("DocGen export failed for %s: %s", job_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    return Response(content=data, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="{fn}"',
                             # Always serve a freshly-rendered file; never let the browser
                             # hand back a previously-downloaded (stale) export for this URL.
                             "Cache-Control": "no-store, must-revalidate"})
