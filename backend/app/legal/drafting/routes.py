"""Firm templates and matter drafts (S3.1, DOC-01; decision D-045).

Templates (firm library):
- `GET  /api/legal/templates`: approved versions for everyone; drafts too for those who manage templates.
- `POST /api/legal/templates`, `POST …/{tid}/versions`: write a template or a new version (`templates.manage`).
- `POST …/{tid}/versions/{vid}/styles`: attach the firm's Word file to a draft version.
- `POST …/{tid}/versions/{vid}/approve`: a managing partner other than the author, for that hash (`templates.approve`).
- `POST …/{tid}/versions/{vid}/retire`, `POST /api/legal/templates/presets`.

Drafts (matter):
- `POST /api/legal/matters/{id}/drafts`: draft from an approved template (`ai.draft`).
- `GET  …/drafts`: the latest version of each draft the person can see.
- `POST …/drafts/{aid}/edit`: a lawyer fills or corrects fields, or rewrites a written passage (a new version).
- `GET  …/drafts/{aid}/preview.pdf|docx`: the draft rendered, marked as a draft until approved (no-store).
Review and filing go through the S2.7 routes (`/artifacts/{aid}/review`, `/file`).
"""
from __future__ import annotations

import copy
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .. import artifacts, policy
from ..audit.record import audited, note
from ..matters.routes import principal_of, require, require_law_edition
from . import engine, render, templates

router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
DOC_KIND = {"engagement_letter": "correspondence", "demand_letter": "correspondence", "nda": "contract", "memo": "memo",
            "contract_review": "memo", "board_minutes": "note", "custom": "other"}


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Not Found")


class FieldIn(BaseModel):
    key: str = Field(max_length=41)
    label: Optional[str] = Field(default=None, max_length=80)
    kind: str = Field(default="text", max_length=20)
    source: str = Field(default="documents", max_length=20)
    question: Optional[str] = Field(default=None, max_length=300)
    from_: Optional[str] = Field(default=None, alias="from", max_length=20)


class TemplateIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    kind: Literal["engagement_letter", "nda", "demand_letter", "memo", "contract_review", "board_minutes", "custom"] = "custom"
    body: str = Field(min_length=1, max_length=60_000)
    fields: list[FieldIn] = Field(default_factory=list, max_length=60)


class VersionIn(BaseModel):
    body: str = Field(min_length=1, max_length=60_000)
    fields: list[FieldIn] = Field(default_factory=list, max_length=60)


class ApproveIn(BaseModel):
    content_sha256: str = Field(min_length=64, max_length=64)


def _fields(items: list[FieldIn]) -> list[dict]:
    return [{k: v for k, v in {"key": f.key, "label": f.label, "kind": f.kind, "source": f.source, "question": f.question,
                               "from": f.from_}.items() if v is not None} for f in items]


def _manager(p: dict) -> bool:
    return policy.authorize(p, "templates.manage").allowed or policy.authorize(p, "templates.approve").allowed


# ── Templates ────────────────────────────────────────────────────────────────────
@router.get("/templates")
def list_templates(request: Request):
    p = principal_of(request)
    return {"templates": templates.listing(p["org_id"], drafts=_manager(p)), "can": {"manage": policy.authorize(p, "templates.manage").allowed,
                                                                                   "approve": policy.authorize(p, "templates.approve").allowed}}


@router.post("/templates")
@audited("template.create", object_type="template")
def create_template(inp: TemplateIn, request: Request):
    p = principal_of(request)
    require(p, "templates.manage")
    try:
        t = templates.create(p["org_id"], inp.name, inp.kind, inp.body, _fields(inp.fields), p["sub"])
    except templates.Invalid as e:
        raise HTTPException(status_code=422, detail=str(e))
    note(object_id=t["id"], details={"kind": inp.kind, "fields": len(inp.fields)})
    return {"template": t}


@router.post("/templates/presets")
@audited("template.presets", object_type="template")
def install_presets(request: Request):
    p = principal_of(request)
    require(p, "templates.manage")
    n = templates.install_presets(p["org_id"], p["sub"])
    note(details={"installed": n})
    return {"installed": n, "templates": templates.listing(p["org_id"], drafts=True)}


@router.post("/templates/{template_id}/versions")
@audited("template.version", object_type="template", object_param="template_id")
def new_version(template_id: str, inp: VersionIn, request: Request):
    p = principal_of(request)
    require(p, "templates.manage")
    try:
        t = templates.add_version(p["org_id"], template_id, inp.body, _fields(inp.fields), p["sub"])
    except templates.Invalid as e:
        raise HTTPException(status_code=422, detail=str(e))
    except LookupError:
        raise _not_found()
    note(details={"version": t["versions"][0]["version"]})
    return {"template": t}


@router.post("/templates/{template_id}/versions/{version_id}/styles")
@audited("template.styles", object_type="template", object_param="template_id")
async def attach_styles(template_id: str, version_id: str, request: Request, file: UploadFile = File(...)):
    p = principal_of(request)
    await run_in_threadpool(require, p, "templates.manage")
    data = await file.read(5_000_001)
    if len(data) > 5_000_000:
        raise HTTPException(status_code=413, detail="The Word file is larger than 5 MB.")
    try:
        t = await run_in_threadpool(templates.set_styles, p["org_id"], template_id, version_id, file.filename or "styles.docx", data, p["sub"])
    except templates.Invalid as e:
        raise HTTPException(status_code=422, detail=str(e))
    except templates.Conflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    except LookupError:
        raise _not_found()
    note(details={"bytes": len(data)})
    return {"template": t}


@router.post("/templates/{template_id}/versions/{version_id}/approve")
@audited("template.approve", object_type="template", object_param="template_id")
def approve_version(template_id: str, version_id: str, inp: ApproveIn, request: Request):
    p = principal_of(request)
    require(p, "templates.approve")
    try:
        t = templates.approve(p["org_id"], template_id, version_id, p["sub"], inp.content_sha256)
    except templates.Conflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    except LookupError:
        raise _not_found()
    note(details={"version": next(v["version"] for v in t["versions"] if v["id"] == version_id)})
    return {"template": t}


@router.post("/templates/{template_id}/versions/{version_id}/retire")
@audited("template.retire", object_type="template", object_param="template_id")
def retire_version(template_id: str, version_id: str, request: Request):
    p = principal_of(request)
    require(p, "templates.approve")
    try:
        return {"template": templates.retire(p["org_id"], template_id, version_id)}
    except LookupError:
        raise _not_found()


# ── Drafts ───────────────────────────────────────────────────────────────────────
class DraftIn(BaseModel):
    template_id: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=200)
    documents: Optional[list[str]] = Field(default=None, max_length=50)
    note: Optional[str] = Field(default=None, max_length=1000)


class FieldEdit(BaseModel):
    key: str = Field(max_length=41)
    value: Optional[str] = Field(default=None, max_length=2000)


class BlockEdit(BaseModel):
    index: int = Field(ge=0, le=500)
    text: str = Field(max_length=10_000)


class DraftEdit(BaseModel):
    content_sha256: str = Field(min_length=64, max_length=64)
    fields: list[FieldEdit] = Field(default_factory=list, max_length=60)
    blocks: list[BlockEdit] = Field(default_factory=list, max_length=60)
    reason: Optional[str] = Field(default=None, max_length=500)


def _username(uid: str) -> str:
    from ...core.db import get_conn
    with get_conn() as conn:
        r = conn.execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone()
    return r[0] if r else "someone"


def _draft(p: dict, matter_id: str, artifact_id: str) -> dict:
    a = artifacts.get(matter_id, artifact_id)
    if not a or a["kind"] != "legal_draft" or not artifacts.visible_to(a, p["sub"]):
        raise _not_found()
    return a


@router.post("/matters/{matter_id}/drafts")
@audited("draft.create", object_type="matter", object_param="matter_id")
async def create_draft(matter_id: str, inp: DraftIn, request: Request):
    import uuid
    p = principal_of(request)
    await run_in_threadpool(require, p, "ai.draft", matter_id)
    found = await run_in_threadpool(templates.approved, p["org_id"], inp.template_id)
    if not found:
        raise HTTPException(status_code=409, detail="This template has no approved version yet; a managing partner approves it first.")
    template, version = found
    content, inputs = await engine.draft(p, matter_id, template, version, inp.title.strip(), inp.documents or None, (inp.note or "").strip() or None)
    a = await run_in_threadpool(artifacts.create, matter_id, "legal_draft", "draft", str(uuid.uuid4()), content, inputs, p["sub"], "model")
    note(details={"artifact": a["id"], "template_version": version["version"], **content["summary"], "gaps": content["gaps"]})
    return {"draft": a}


@router.get("/matters/{matter_id}/drafts")
def list_drafts(matter_id: str, request: Request):
    from ...core.db import get_conn
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    with get_conn() as conn:
        rows = conn.execute("SELECT a.id FROM legal_ai_artifacts a WHERE a.matter_id=? AND a.kind='legal_draft' AND a.version = "
                            "(SELECT MAX(b.version) FROM legal_ai_artifacts b WHERE b.subject_id=a.subject_id AND b.kind='legal_draft') "
                            "ORDER BY a.created_at DESC", (matter_id,)).fetchall()
    out = []
    for (aid,) in rows:
        a = artifacts.get(matter_id, aid)
        if a and artifacts.visible_to(a, p["sub"]):
            out.append({"id": a["id"], "draft_id": a["subject_id"], "title": a["content"]["title"], "version": a["version"], "status": a["status"],
                        "gaps": a["content"]["gaps"], "template": a["content"]["template"], "created_by_name": a["created_by_name"],
                        "created_at": a["created_at"], "document_id": a.get("document_id"),
                        "sent_back": bool(a["decisions"] and a["decisions"][-1]["decision"] == "changes_requested")})
    return {"drafts": out}


@router.post("/matters/{matter_id}/drafts/{artifact_id}/edit")
@audited("draft.edit", object_type="ai_artifact", object_param="artifact_id")
def edit_draft(matter_id: str, artifact_id: str, inp: DraftEdit, request: Request):
    """A lawyer's changes make a new version: a field filled or corrected (status "entered", by them),
    or a written passage rewritten (its sources kept, marked as edited). The firm's template wording
    is not edited here: that is a new template version."""
    p = principal_of(request)
    require(p, "ai.draft", matter_id)
    cur = _draft(p, matter_id, artifact_id)
    latest = artifacts.latest(matter_id, "draft", cur["subject_id"], "legal_draft")
    if cur["content_sha256"] != inp.content_sha256 or latest["id"] != cur["id"]:
        raise HTTPException(status_code=409, detail="This draft changed since you opened it. Reload and try again.")
    content = copy.deepcopy(cur["content"])
    editor = _username(p["sub"])
    by_key = {f["key"]: f for f in content["fields"]}
    for e in inp.fields:
        f = by_key.get(e.key)
        if not f:
            raise HTTPException(status_code=422, detail=f"There is no field {e.key}.")
        value = " ".join((e.value or "").split()) or None
        if value != f.get("value"):
            f.update(value=value, status="entered" if value else "needs_input", edited_by=editor, sources=[] if value else f["sources"])
    for e in inp.blocks:
        if e.index >= len(content["blocks"]) or content["blocks"][e.index].get("origin") not in ("written", "edited"):
            raise HTTPException(status_code=422, detail="Only a written passage can be changed here; the template's wording is the firm's.")
        b = content["blocks"][e.index]
        b.update(text=e.text.strip() or b["text"], origin="edited", edited_by=editor, unsupported=[])
    content = engine.assemble(content)
    out = artifacts.create(matter_id, "legal_draft", "draft", cur["subject_id"], content,
                           {**cur["inputs"], "edited_from": cur["version"], "reason": inp.reason}, p["sub"], "person")
    note(details={"version": out["version"], "fields": len(inp.fields), "passages": len(inp.blocks), "gaps": content["gaps"]})
    return {"draft": out}


def _rendered(p: dict, matter_id: str, artifact_id: str, fmt: str) -> tuple[bytes, str]:
    a = _draft(p, matter_id, artifact_id)
    approval = artifacts.approval_of(a)
    if fmt == "pdf":
        return render.to_pdf(a["content"], approval, a["version"]), "application/pdf"
    found = templates.get(p["org_id"], a["content"]["template"]["id"])
    version = next((v for v in (found or {}).get("versions", []) if v["id"] == a["content"]["template"]["version_id"]), None)
    styles = templates.styles(p["org_id"], version) if version else None
    return render.to_docx(a["content"], approval, styles, a["version"]), DOCX


@router.get("/matters/{matter_id}/drafts/{artifact_id}/preview.{fmt}")
@audited("draft.preview", object_type="ai_artifact", object_param="artifact_id")
def preview(matter_id: str, artifact_id: str, fmt: Literal["pdf", "docx"], request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    data, mime = _rendered(p, matter_id, artifact_id, fmt)
    note(details={"format": fmt, "bytes": len(data)})
    return Response(data, media_type=mime, headers={"Cache-Control": "no-store",
                                                     "Content-Disposition": f'attachment; filename="draft-{artifact_id[:8]}.{fmt}"'})


def filed(p: dict, a: dict, approval: dict) -> tuple[bytes, str, str, str]:
    """What filing an approved draft makes: a Word document in the firm's styles (used by the S2.7 file route)."""
    found = templates.get(p["org_id"], a["content"]["template"]["id"])
    version = next((v for v in (found or {}).get("versions", []) if v["id"] == a["content"]["template"]["version_id"]), None)
    styles = templates.styles(p["org_id"], version) if version else None
    name = "".join(ch if ch.isalnum() or ch in " -_" else "-" for ch in a["content"]["title"]).strip()[:80] or "draft"
    return (render.to_docx(a["content"], approval, styles), f"{name}.docx", DOCX,
            DOC_KIND.get(a["content"]["template"]["kind"], "other"))
