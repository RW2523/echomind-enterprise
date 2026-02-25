"""Doc upload, list, delete, usage, data-preview API."""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException

from app.ingestion.pipeline_docs import run_document_pipeline
from app.catalog.dao import get_catalog, list_catalogs, delete_catalog, delete_all_catalogs
from app.qdrant.delete import delete_document_points, clear_documents_collection, clear_transcripts_collection

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/docs", tags=["docs"])


def _doc_to_list_item(row: dict) -> dict:
    """Map catalog row to frontend shape: id, filename, filetype, created_at."""
    uploaded = row.get("uploaded_at")
    created_at = ""
    if uploaded is not None:
        try:
            created_at = datetime.fromtimestamp(int(uploaded), tz=timezone.utc).isoformat()
        except Exception:
            created_at = str(uploaded)
    return {
        "id": row.get("doc_id", ""),
        "filename": row.get("title", ""),
        "filetype": row.get("file_type", ""),
        "created_at": created_at,
    }


@router.post("/upload")
async def upload_doc(
    file: UploadFile = File(...),
) -> dict:
    """Upload PDF/DOCX/PPTX/TXT/MD/CSV/XLSX. Returns ok, doc_id, chunks."""
    allowed = (
        "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "text/plain", "text/markdown", "text/csv",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    if file.content_type and file.content_type not in allowed:
        name = (file.filename or "").lower()
        if not any(name.endswith(ext) for ext in (".pdf", ".docx", ".pptx", ".txt", ".md", ".csv", ".xlsx")):
            raise HTTPException(400, "Unsupported file type")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    filename = file.filename or "document"
    doc_id = run_document_pipeline(filename, data, source_path=filename)
    meta = get_catalog(doc_id)
    chunks = (meta.get("num_chunks") or 0) if meta else 0
    return {"ok": True, "doc_id": doc_id, "chunks": chunks}


@router.get("/list")
async def list_docs() -> dict:
    """List documents from catalog (shape: id, filename, filetype, created_at)."""
    items = list_catalogs()
    return {"documents": [_doc_to_list_item(r) for r in items]}


@router.get("/usage")
async def docs_usage() -> dict:
    """Storage usage for sidebar. RAG platform does not track bytes; return 0 / null."""
    return {"usage_bytes": 0, "capacity_bytes": None}


@router.get("/data-preview")
async def data_preview() -> dict:
    """Documents + empty chunks/transcripts for Usage popover (chunks/transcripts in Qdrant only)."""
    items = list_catalogs()
    documents = []
    for r in items:
        uploaded = r.get("uploaded_at")
        created_at = ""
        if uploaded is not None:
            try:
                created_at = datetime.fromtimestamp(int(uploaded), tz=timezone.utc).isoformat()
            except Exception:
                created_at = str(uploaded)
        documents.append({
            "id": r.get("doc_id", ""),
            "filename": r.get("title", ""),
            "filetype": r.get("file_type", ""),
            "created_at": created_at,
            "meta_json": None,
        })
    return {"documents": documents, "chunks": [], "transcripts": []}


@router.delete("/{doc_id}")
async def delete_doc(doc_id: str) -> dict:
    """Delete document from catalog and Qdrant."""
    if not get_catalog(doc_id):
        raise HTTPException(404, "Document not found")
    delete_document_points(doc_id)
    delete_catalog(doc_id)
    return {"ok": True, "deleted": doc_id}


@router.post("/delete-all")
async def delete_all_docs() -> dict:
    """Clear all documents and transcripts in RAG platform (catalog + Qdrant)."""
    delete_all_catalogs()
    clear_documents_collection()
    clear_transcripts_collection()
    return {"ok": True, "message": "All RAG data deleted."}


@router.get("/{doc_id}")
async def get_doc(doc_id: str) -> dict:
    """Get one document metadata from catalog."""
    row = get_catalog(doc_id)
    if not row:
        raise HTTPException(404, "Document not found")
    return row
