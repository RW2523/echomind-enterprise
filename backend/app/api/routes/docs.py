from __future__ import annotations
import json
import os
import shutil
import httpx
from fastapi import APIRouter, UploadFile, File, HTTPException

from ...core.config import settings
from ...core.db import get_conn
from ...rag.parse import parse_any
from ...rag.index import index
from ...rag_platform_client import (
    is_configured as rag_platform_configured,
    upload_doc as rag_upload_doc,
    list_docs as rag_list_docs,
    delete_doc as rag_delete_doc,
    get_usage as rag_get_usage,
    get_data_preview as rag_get_data_preview,
    delete_all_docs as rag_delete_all_docs,
)

router = APIRouter(prefix="/docs", tags=["docs"])


def _vector_db_usage_bytes() -> int:
    """Total size of vector DB files: FAISS index, meta JSON, sparse meta, SQLite DB."""
    total = 0
    for path in (
        settings.FAISS_PATH,
        settings.META_PATH,
        settings.SPARSE_META_PATH,
        settings.FAISS_TRANSCRIPT_PATH,
        settings.META_TRANSCRIPT_PATH,
        settings.SPARSE_TRANSCRIPT_META_PATH,
        settings.DB_PATH,
    ):
        if path and os.path.exists(path):
            try:
                total += os.path.getsize(path)
            except OSError:
                pass
    return total


def _rag_unavailable_usage():
    """Fallback when RAG platform is unreachable (e.g. still starting)."""
    return {"usage_bytes": 0, "capacity_bytes": None}


@router.get("/usage")
async def storage_usage():
    """Return vector DB storage usage and disk capacity (for sidebar usage bar)."""
    if rag_platform_configured():
        try:
            return await rag_get_usage()
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return _rag_unavailable_usage()
    usage_bytes = _vector_db_usage_bytes()
    capacity_bytes = None
    try:
        disk = shutil.disk_usage(settings.DATA_DIR)
        capacity_bytes = disk.total
    except OSError:
        pass
    return {"usage_bytes": usage_bytes, "capacity_bytes": capacity_bytes}


@router.get("/list")
async def list_docs():
    """List uploaded documents only (exclude transcript entries; those appear in Transcripts panel)."""
    if rag_platform_configured():
        try:
            return await rag_list_docs()
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return {"documents": []}
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, filename, filetype, created_at FROM documents WHERE filename NOT LIKE 'transcript_%' ORDER BY created_at DESC"
        ).fetchall()
    return {"documents": [{"id": r[0], "filename": r[1], "filetype": r[2], "created_at": r[3]} for r in rows]}


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    if rag_platform_configured():
        data = await file.read()
        content_type = file.content_type or "application/octet-stream"
        return await rag_upload_doc(data, file.filename or "document", content_type)
    data = await file.read()
    filetype, text = parse_any(file.filename, data)
    res = await index.add_document(file.filename, filetype, text, {"filename": file.filename, "filetype": filetype})
    return {"ok": True, **res}


@router.delete("/{doc_id}")
async def delete_doc(doc_id: str):
    if rag_platform_configured():
        try:
            return await rag_delete_doc(doc_id)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise HTTPException(status_code=404, detail="Document not found") from e
            raise HTTPException(status_code=502, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    await index.delete_document(doc_id)
    return {"ok": True, "deleted": doc_id}


def _data_preview_transcripts_from_backend():
    """Build transcripts list from backend DB for data-preview."""
    with get_conn() as conn:
        try:
            rows = conn.execute(
                "SELECT id, title, tags_json, echotag, created_at, length(raw_text), length(polished_text) FROM transcripts ORDER BY created_at DESC"
            ).fetchall()
        except Exception:
            rows = []
    out = []
    for r in rows:
        tid, title, tags_json, echotag, created_at = r[0], r[1], r[2], r[3], r[4]
        raw_len = r[5] if len(r) > 5 else 0
        polished_len = r[6] if len(r) > 6 else 0
        tags = []
        if tags_json:
            try:
                tags = json.loads(tags_json) if isinstance(tags_json, str) else (tags_json or [])
            except Exception:
                pass
        out.append({
            "id": tid,
            "title": title or tid,
            "tags": tags,
            "echotag": echotag or "",
            "created_at": created_at or "",
            "raw_length": raw_len or 0,
            "polished_length": polished_len or 0,
        })
    return out


@router.get("/data-preview")
async def data_preview():
    """Full data preview: documents, chunks, transcripts (for Usage popover)."""
    if rag_platform_configured():
        try:
            preview = await rag_get_data_preview()
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return {
                "documents": [],
                "chunks": [],
                "transcripts": _data_preview_transcripts_from_backend(),
            }
        transcripts_out = _data_preview_transcripts_from_backend()
        return {"documents": preview.get("documents", []), "chunks": preview.get("chunks", []), "transcripts": transcripts_out}

    with get_conn() as conn:
        docs = conn.execute(
            "SELECT id, filename, filetype, created_at, meta_json FROM documents ORDER BY created_at DESC"
        ).fetchall()
        chunks = conn.execute(
            "SELECT id, doc_id, chunk_index, substr(text, 1, 200) as text_preview FROM chunks ORDER BY doc_id, chunk_index"
        ).fetchall()
        transcripts = conn.execute(
            "SELECT id, title, tags_json, echotag, created_at, length(raw_text) as raw_len, length(polished_text) as polished_len FROM transcripts ORDER BY created_at DESC"
        ).fetchall()
    documents = [{"id": r[0], "filename": r[1], "filetype": r[2], "created_at": r[3], "meta_json": r[4]} for r in docs]
    chunks_out = [{"id": r[0], "doc_id": r[1], "chunk_index": r[2], "text_preview": (r[3] or "") + ("..." if (r[3] and len(r[3]) >= 200) else "")} for r in chunks]
    transcripts_out = []
    for r in transcripts:
        tid, title, tags_json, echotag, created_at, raw_len, polished_len = r
        tags = []
        if tags_json:
            try:
                tags = json.loads(tags_json) if isinstance(tags_json, str) else (tags_json or [])
            except Exception:
                pass
        transcripts_out.append({
            "id": tid,
            "title": title or tid,
            "tags": tags,
            "echotag": echotag or "",
            "created_at": created_at or "",
            "raw_length": raw_len or 0,
            "polished_length": polished_len or 0,
        })
    return {"documents": documents, "chunks": chunks_out, "transcripts": transcripts_out}


@router.post("/delete-all")
async def delete_all_data():
    """Delete all data: documents, chunks, transcripts, chats, messages. Uses bulk clear for speed."""
    if rag_platform_configured():
        try:
            await rag_delete_all_docs()
        except (httpx.ConnectError, httpx.ConnectTimeout):
            pass  # Clear backend only; RAG platform will have stale data until it is up
    with get_conn() as conn:
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM documents")
        conn.execute("DELETE FROM transcripts")
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM chats")
        conn.commit()
    if not rag_platform_configured():
        index.clear_all()
    return {"ok": True, "message": "All data deleted."}
