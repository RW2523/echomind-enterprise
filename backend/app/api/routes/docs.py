from __future__ import annotations
import json
import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException

from ...core.config import settings
from ...core.db import get_conn
from ...rag.parse import parse_any
from ...rag.chunking import chunk_document
from ...rag.index import index
from ...qdrant.client import is_qdrant_enabled

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
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    filename = (file.filename or "").strip() or "document"
    if is_qdrant_enabled():
        try:
            from ...ingestion.pipeline_docs import run_pipeline_docs
            res = await run_pipeline_docs(filename, data, doc_title=filename)
            if res.get("error"):
                raise HTTPException(status_code=400, detail=res["error"])
            return {"ok": True, "doc_id": res["doc_id"], "chunks": res.get("chunks_count", 0)}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Indexing failed: {e!s}")
    try:
        filetype, text = parse_any(filename, data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e!s}")
    if not (text or "").strip():
        raise HTTPException(status_code=400, detail="No text extracted from file")
    try:
        res = await index.add_document(filename, filetype, text, {"filename": filename, "filetype": filetype})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e!s}")
    try:
        from ...catalog.dao import insert_document
        insert_document(
            res.get("doc_id", ""),
            title=filename,
            file_type=filetype,
            num_chunks=res.get("chunks", 0),
        )
    except Exception:
        pass
    return {"ok": True, **res}


@router.post("/chunk-preview")
async def chunk_preview(file: UploadFile = File(...)):
    """
    Preview how a document would be chunked without adding it to the index.
    Returns extracted text length, detected doc type, and list of chunks (with is_parent: true = section header, false = embed chunk).
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    filename = (file.filename or "").strip() or "document"
    try:
        filetype, text = parse_any(filename, data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e!s}")
    if not (text or "").strip():
        raise HTTPException(status_code=400, detail="No text extracted from file")
    try:
        chunks = chunk_document(text, "preview")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunking failed: {e!s}")
    doc_type = chunks[0].doc_type.value if chunks else "user"
    embed_count = sum(1 for c in chunks if not c.is_parent)
    out = []
    for c in chunks:
        out.append({
            "chunk_index": c.chunk_index,
            "text": c.text,
            "doc_type": c.doc_type.value,
            "is_parent": c.is_parent,
            "section": c.section,
            "char_count": len(c.text),
        })
    return {
        "filename": filename,
        "filetype": filetype,
        "extracted_length": len(text),
        "doc_type": doc_type,
        "total_chunks": len(chunks),
        "embed_count": embed_count,
        "chunks": out,
    }


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
    """Delete all data: documents, chunks, transcripts, chats, messages. Uses bulk clear for speed."""
    with get_conn() as conn:
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM documents")
        conn.execute("DELETE FROM documents_catalog")
        conn.execute("DELETE FROM transcripts")
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM chats")
        conn.commit()
    index.clear_all()
    return {"ok": True, "message": "All data deleted."}
