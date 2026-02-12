from __future__ import annotations
import json
import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException

from ...core.config import settings
from ...core.db import get_conn
from ...rag.parse import parse_any
from ...rag.index import index

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


@router.get("/usage")
def storage_usage():
    """Return vector DB storage usage and disk capacity (for sidebar usage bar)."""
    usage_bytes = _vector_db_usage_bytes()
    capacity_bytes = None
    try:
        disk = shutil.disk_usage(settings.DATA_DIR)
        capacity_bytes = disk.total
    except OSError:
        pass
    return {"usage_bytes": usage_bytes, "capacity_bytes": capacity_bytes}


@router.get("/list")
def list_docs():
    """List uploaded documents only (exclude transcript entries; those appear in Transcripts panel)."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, filename, filetype, created_at FROM documents WHERE filename NOT LIKE 'transcript_%' ORDER BY created_at DESC"
        ).fetchall()
    return {"documents": [{"id": r[0], "filename": r[1], "filetype": r[2], "created_at": r[3]} for r in rows]}


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    filetype, text = parse_any(file.filename, data)
    res = await index.add_document(file.filename, filetype, text, {"filename": file.filename, "filetype": filetype})
    return {"ok": True, **res}


@router.delete("/{doc_id}")
async def delete_doc(doc_id: str):
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    await index.delete_document(doc_id)
    return {"ok": True, "deleted": doc_id}


@router.get("/data-preview")
def data_preview():
    """Full data preview: documents, chunks, transcripts (for Usage popover)."""
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
    """Delete all data: documents (and index), chunks, transcripts, chats, messages."""
    with get_conn() as conn:
        doc_ids = [r[0] for r in conn.execute("SELECT id FROM documents").fetchall()]
    for doc_id in doc_ids:
        try:
            await index.delete_document(doc_id)
        except Exception:
            pass
    with get_conn() as conn:
        conn.execute("DELETE FROM transcripts")
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM chats")
        conn.commit()
    return {"ok": True, "message": "All data deleted."}
