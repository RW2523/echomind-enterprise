from __future__ import annotations
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
    for path in (settings.FAISS_PATH, settings.META_PATH, settings.SPARSE_META_PATH, settings.DB_PATH):
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
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, filename, filetype, created_at FROM documents ORDER BY created_at DESC"
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
