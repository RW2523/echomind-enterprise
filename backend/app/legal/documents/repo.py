"""Documents, versions, findings and jobs in the database (S1.6).

Every read takes the matter id and filters by it: a document id is never resolved on its own, so
an id copied from one matter cannot be opened through another.
"""
from __future__ import annotations

import uuid
from typing import Optional

from ...core.db import get_conn
from ..policy import now_iso

KINDS = ("contract", "correspondence", "pleading", "evidence", "memo", "note", "other")
STATUSES = ("draft", "final", "executed", "received")

_V_COLS = ("id", "document_id", "matter_id", "number", "filename", "mime", "size_bytes", "sha256", "pages",
           "wrapped_key", "uploaded_by", "uploaded_at", "index_status", "index_error", "chunk_count", "indexed_at")
_D_COLS = ("id", "org_id", "matter_id", "title", "kind", "status", "current_version_id", "created_by", "created_at",
           "updated_at", "version")


class Conflict(Exception):
    pass


def new_id() -> str:
    return str(uuid.uuid4())


def _username(conn, user_id: Optional[str]) -> Optional[str]:
    if not user_id:
        return None
    row = conn.execute("SELECT username FROM users WHERE id=?", (user_id,)).fetchone()
    return row[0] if row else None


# ── Writing ──────────────────────────────────────────────────────────────────────
def create_document(org_id: str, matter_id: str, title: str, kind: str, status: str, by: str) -> str:
    did = new_id()
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_documents (id, org_id, matter_id, title, kind, status, created_by, created_at) "
                     "VALUES (?,?,?,?,?,?,?,?)", (did, org_id, matter_id, title, kind, status, by, now_iso()))
        conn.commit()
    return did


def add_version(version_id: str, document_id: str, matter_id: str, filename: str, mime: str, size: int, sha256: str,
                pages: Optional[int], wrapped_key: str, by: str) -> dict:
    """Add the next version and make it current, and queue its indexing job — in one transaction."""
    now = now_iso()
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            number = conn.execute("SELECT COALESCE(MAX(number), 0) + 1 FROM legal_document_versions WHERE document_id=?",
                                  (document_id,)).fetchone()[0]
            conn.execute("INSERT INTO legal_document_versions (id, document_id, matter_id, number, filename, mime, size_bytes, "
                         "sha256, pages, wrapped_key, uploaded_by, uploaded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (version_id, document_id, matter_id, number, filename, mime, size, sha256, pages, wrapped_key, by, now))
            conn.execute("UPDATE legal_documents SET current_version_id=?, updated_at=?, version=version+1 WHERE id=?",
                         (version_id, now, document_id))
            org = conn.execute("SELECT org_id FROM legal_documents WHERE id=?", (document_id,)).fetchone()[0]
            conn.execute("INSERT INTO legal_jobs (id, kind, org_id, matter_id, object_id, requested_by, created_at) "
                         "VALUES (?,?,?,?,?,?,?)", (new_id(), "document.index", org, matter_id, version_id, by, now))
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    return {"number": number}


def update_document(matter_id: str, document_id: str, changes: dict, expected_version: int) -> None:
    allowed = {k: v for k, v in changes.items() if k in ("title", "kind", "status") and v is not None}
    if not allowed:
        return
    sets = ", ".join(f"{k}=?" for k in allowed)
    with get_conn() as conn:
        cur = conn.execute(f"UPDATE legal_documents SET {sets}, updated_at=?, version=version+1 "
                           "WHERE id=? AND matter_id=? AND version=?",
                           (*allowed.values(), now_iso(), document_id, matter_id, expected_version))
        conn.commit()
    if cur.rowcount != 1:
        raise Conflict("This document was changed by someone else. Reload and try again.")


def requeue(matter_id: str, version_id: str, by: str, org_id: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE legal_document_versions SET index_status='queued', index_error=NULL WHERE id=? AND matter_id=?",
                     (version_id, matter_id))
        conn.execute("INSERT INTO legal_jobs (id, kind, org_id, matter_id, object_id, requested_by, created_at) "
                     "VALUES (?,?,?,?,?,?,?)", (new_id(), "document.index", org_id, matter_id, version_id, by, now_iso()))
        conn.commit()


# ── Reading (always within one matter) ───────────────────────────────────────────
def _version_out(conn, row) -> dict:
    v = dict(zip(_V_COLS, row))
    return {"id": v["id"], "number": v["number"], "filename": v["filename"], "mime": v["mime"], "size_bytes": v["size_bytes"],
            "sha256": v["sha256"], "pages": v["pages"], "uploaded_by": _username(conn, v["uploaded_by"]),
            "uploaded_at": v["uploaded_at"], "index_status": v["index_status"], "index_error": v["index_error"],
            "chunk_count": v["chunk_count"], "indexed_at": v["indexed_at"]}


def _findings(conn, version_id: Optional[str]) -> dict:
    if not version_id:
        return {}
    return {k: n for k, n in conn.execute("SELECT kind, COUNT(*) FROM legal_findings WHERE version_id=? GROUP BY kind",
                                          (version_id,))}


def _document_out(conn, row) -> dict:
    d = dict(zip(_D_COLS, row))
    cur = conn.execute(f"SELECT {', '.join(_V_COLS)} FROM legal_document_versions WHERE id=?",
                       (d["current_version_id"],)).fetchone() if d["current_version_id"] else None
    return {"id": d["id"], "title": d["title"], "kind": d["kind"], "status": d["status"], "version": d["version"],
            "created_by": _username(conn, d["created_by"]), "created_at": d["created_at"], "updated_at": d["updated_at"],
            "current_version": _version_out(conn, cur) if cur else None,
            "findings": _findings(conn, d["current_version_id"])}


def list_documents(matter_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(f"SELECT {', '.join(_D_COLS)} FROM legal_documents WHERE matter_id=? "
                            "ORDER BY COALESCE(updated_at, created_at) DESC", (matter_id,)).fetchall()
        return [_document_out(conn, r) for r in rows]


def get_document(matter_id: str, document_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(f"SELECT {', '.join(_D_COLS)} FROM legal_documents WHERE id=? AND matter_id=?",
                           (document_id, matter_id)).fetchone()
        return _document_out(conn, row) if row else None


def list_versions(matter_id: str, document_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(f"SELECT {', '.join(_V_COLS)} FROM legal_document_versions WHERE document_id=? AND matter_id=? "
                            "ORDER BY number DESC", (document_id, matter_id)).fetchall()
        return [_version_out(conn, r) for r in rows]


def version_row(matter_id: str, document_id: str, version_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(f"SELECT {', '.join(_V_COLS)} FROM legal_document_versions WHERE id=? AND document_id=? AND matter_id=?",
                           (version_id, document_id, matter_id)).fetchone()
    return dict(zip(_V_COLS, row)) if row else None


def current_sha(matter_id: str, document_id: str) -> Optional[str]:
    with get_conn() as conn:
        row = conn.execute("SELECT v.sha256 FROM legal_documents d JOIN legal_document_versions v ON v.id = d.current_version_id "
                           "WHERE d.id=? AND d.matter_id=?", (document_id, matter_id)).fetchone()
    return row[0] if row else None


def same_content_in_matter(matter_id: str, sha256: str, exclude_document: Optional[str] = None) -> Optional[str]:
    """Title of another document in this matter whose current version has identical content."""
    with get_conn() as conn:
        row = conn.execute("SELECT d.title FROM legal_documents d JOIN legal_document_versions v ON v.id = d.current_version_id "
                           "WHERE d.matter_id=? AND v.sha256=? AND d.id IS NOT ?", (matter_id, sha256, exclude_document)).fetchone()
    return row[0] if row else None
