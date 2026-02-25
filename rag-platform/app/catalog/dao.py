"""
DAO for documents_catalog: insert, get, list, update.
"""
from __future__ import annotations
import json
from typing import Any, List, Optional

from app.catalog.db import get_conn, init_db


def insert_catalog(
    doc_id: str,
    title: str,
    doc_type: Optional[str] = None,
    file_type: Optional[str] = None,
    uploaded_at: Optional[int] = None,
    tags: Optional[List[str]] = None,
    num_pages: Optional[int] = None,
    num_chunks: Optional[int] = None,
    source_path: Optional[str] = None,
    summary_short: Optional[str] = None,
    summary_chapters: Optional[dict] = None,
) -> None:
    init_db()
    import time
    ts = uploaded_at if uploaded_at is not None else int(time.time())
    tags_json = json.dumps(tags) if tags is not None else None
    summary_chapters_json = json.dumps(summary_chapters) if summary_chapters is not None else None
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO documents_catalog
            (doc_id, title, doc_type, file_type, uploaded_at, tags, num_pages, num_chunks, source_path, summary_short, summary_chapters)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                title,
                doc_type or "",
                file_type or "",
                ts,
                tags_json,
                num_pages,
                num_chunks,
                source_path,
                summary_short,
                summary_chapters_json,
            ),
        )
        conn.commit()


def get_catalog(doc_id: str) -> Optional[dict]:
    init_db()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM documents_catalog WHERE doc_id = ?", (doc_id,))
        row = cur.fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def list_catalogs(limit: int = 100) -> List[dict]:
    init_db()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM documents_catalog ORDER BY uploaded_at DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
    return [_row_to_dict(r) for r in rows]


def delete_catalog(doc_id: str) -> bool:
    """Delete one catalog row. Returns True if a row was deleted."""
    init_db()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM documents_catalog WHERE doc_id = ?", (doc_id,))
        conn.commit()
        return cur.rowcount > 0


def delete_all_catalogs() -> int:
    """Delete all catalog rows. Returns number of rows deleted."""
    init_db()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM documents_catalog")
        conn.commit()
        return cur.rowcount


def _row_to_dict(row) -> dict:
    d = dict(row) if hasattr(row, "keys") else {}
    if "tags" in d and isinstance(d["tags"], str):
        try:
            d["tags"] = json.loads(d["tags"])
        except Exception:
            pass
    if "summary_chapters" in d and isinstance(d["summary_chapters"], str):
        try:
            d["summary_chapters"] = json.loads(d["summary_chapters"])
        except Exception:
            pass
    return d
