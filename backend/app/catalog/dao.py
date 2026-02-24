"""
Catalog DAO: insert_document, get_document_by_id, list_documents, query_documents_for_clarification.
"""
from __future__ import annotations
import json
from typing import List, Dict, Any, Optional
from .db import get_conn


def insert_document(
    doc_id: str,
    title: str,
    doc_type: str = "document",
    file_type: str = "",
    tags: Optional[List[str]] = None,
    num_pages: Optional[int] = None,
    num_chunks: Optional[int] = None,
    source_path: Optional[str] = None,
    summary_short: Optional[str] = None,
    summary_chapters: Optional[List[Dict]] = None,
    uploaded_at: Optional[str] = None,
) -> None:
    """Insert or replace a document in documents_catalog."""
    from ..utils.ids import now_iso
    uploaded_at = uploaded_at or now_iso()
    tags_json = json.dumps(tags or [])
    summary_chapters_json = json.dumps(summary_chapters) if summary_chapters is not None else None
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO documents_catalog
               (doc_id, title, doc_type, file_type, uploaded_at, tags_json, num_pages, num_chunks, source_path, summary_short, summary_chapters_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, title, doc_type, file_type, uploaded_at, tags_json, num_pages, num_chunks, source_path, summary_short, summary_chapters_json),
        )
        conn.commit()


def get_document_by_id(doc_id: str) -> Optional[Dict[str, Any]]:
    """Return one document from catalog by doc_id or None."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT doc_id, title, doc_type, file_type, uploaded_at, tags_json, num_pages, num_chunks, source_path, summary_short, summary_chapters_json FROM documents_catalog WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
    if not row:
        return None
    tags = []
    try:
        if row[5]:
            tags = json.loads(row[5])
    except Exception:
        pass
    chapters = None
    try:
        if row[10]:
            chapters = json.loads(row[10])
    except Exception:
        pass
    return {
        "doc_id": row[0],
        "title": row[1],
        "doc_type": row[2],
        "file_type": row[3],
        "uploaded_at": row[4],
        "tags": tags,
        "num_pages": row[6],
        "num_chunks": row[7],
        "source_path": row[8],
        "summary_short": row[9],
        "summary_chapters": chapters,
    }


def list_documents(limit: int = 100) -> List[Dict[str, Any]]:
    """List documents from catalog, most recent first."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT doc_id, title, doc_type, file_type, uploaded_at, tags_json, num_pages, num_chunks FROM documents_catalog ORDER BY uploaded_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    out = []
    for row in rows:
        tags = []
        try:
            if row[5]:
                tags = json.loads(row[5])
        except Exception:
            pass
        out.append({
            "doc_id": row[0],
            "title": row[1],
            "doc_type": row[2],
            "file_type": row[3],
            "uploaded_at": row[4],
            "tags": tags,
            "num_pages": row[6],
            "num_chunks": row[7],
        })
    return out


def query_documents_for_clarification(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Simple text search on title/tags for clarification (e.g. 'which document?'). Returns matching docs."""
    with get_conn() as conn:
        q = f"%{query.strip()}%"
        rows = conn.execute(
            "SELECT doc_id, title, doc_type, file_type, uploaded_at FROM documents_catalog WHERE title LIKE ? OR tags_json LIKE ? ORDER BY uploaded_at DESC LIMIT ?",
            (q, q, limit),
        ).fetchall()
    return [{"doc_id": r[0], "title": r[1], "doc_type": r[2], "file_type": r[3], "uploaded_at": r[4]} for r in rows]
