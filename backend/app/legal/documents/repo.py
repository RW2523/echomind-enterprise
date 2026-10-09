"""Documents, versions, findings and jobs in the database (S1.6).

Every read takes the matter id and filters by it: a document id is never resolved on its own, so
an id copied from one matter cannot be opened through another.

Reads made for a person also pass their id (S1.11): a document synced from a connector is
visible only if its source names them, and only while its connector is enabled and has synced
successfully within its freshness window (fail closed).
"""
from __future__ import annotations

import json
import uuid
from typing import Optional

from ...core.db import get_conn
from ..policy import now_iso

KINDS = ("contract", "correspondence", "pleading", "evidence", "memo", "note", "other")
STATUSES = ("draft", "final", "executed", "received")

_V_COLS = ("id", "document_id", "matter_id", "number", "filename", "mime", "size_bytes", "sha256", "pages",
           "wrapped_key", "uploaded_by", "uploaded_at", "index_status", "index_error", "chunk_count", "indexed_at",
           "progress_json", "status")
_D_COLS = ("id", "org_id", "matter_id", "title", "kind", "status", "current_version_id", "created_by", "created_at",
           "updated_at", "version", "connector_id", "remote_id", "source_state")

# Fragment over `legal_documents d` with ONE positional parameter: the person's user id.
VISIBLE_TO = ("d.deleted_at IS NULL "
              "AND (d.acl_restricted = 0 OR EXISTS (SELECT 1 FROM legal_document_acl a WHERE a.document_id = d.id AND a.user_id = ?)) "
              "AND (d.connector_id IS NULL OR EXISTS (SELECT 1 FROM legal_connectors k WHERE k.id = d.connector_id "
              "AND k.status = 'active' AND k.last_ok_at IS NOT NULL "
              "AND (julianday('now') - julianday(k.last_ok_at)) * 1440 <= k.stale_after_minutes))")


def _where(uid: Optional[str]) -> tuple[str, tuple]:
    """Visibility for a person (uid), or for the system (None: only "not deleted")."""
    return (VISIBLE_TO, (uid,)) if uid else ("d.deleted_at IS NULL", ())


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
                pages: Optional[int], wrapped_key: str, by: str, status: Optional[str] = None) -> dict:
    """Add the next version and make it current, and queue its indexing job — in one transaction.
    The version keeps its own status (default: the document's); the document's status follows it."""
    now = now_iso()
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            number = conn.execute("SELECT COALESCE(MAX(number), 0) + 1 FROM legal_document_versions WHERE document_id=?",
                                  (document_id,)).fetchone()[0]
            status = status or conn.execute("SELECT status FROM legal_documents WHERE id=?", (document_id,)).fetchone()[0]
            conn.execute("INSERT INTO legal_document_versions (id, document_id, matter_id, number, filename, mime, size_bytes, "
                         "sha256, pages, wrapped_key, uploaded_by, uploaded_at, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (version_id, document_id, matter_id, number, filename, mime, size, sha256, pages, wrapped_key, by, now, status))
            conn.execute("UPDATE legal_documents SET current_version_id=?, status=?, updated_at=?, version=version+1 WHERE id=?",
                         (version_id, status, now, document_id))
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
        if cur.rowcount == 1 and "status" in allowed:            # the current version carries the status (S2.1c)
            conn.execute("UPDATE legal_document_versions SET status=? WHERE id=(SELECT current_version_id FROM legal_documents "
                         "WHERE id=?)", (allowed["status"], document_id))
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
            "chunk_count": v["chunk_count"], "indexed_at": v["indexed_at"],
            "progress": json.loads(v["progress_json"]) if v["progress_json"] else None, "flags": _flags(conn, v["id"]),
            "status": v["status"]}


def _flags(conn, version_id: str) -> dict:
    """What reading the file found (S2.1): counts only, the details are in the version's marks."""
    kinds = dict(conn.execute("SELECT kind, COUNT(*) FROM legal_marks WHERE version_id=? GROUP BY kind", (version_id,)).fetchall())
    hidden_ai = conn.execute("SELECT COUNT(*) FROM legal_marks WHERE version_id=? AND kind='hidden_text' "
                             "AND json_extract(meta_json, '$.instructions') = 1", (version_id,)).fetchone()[0]
    row = conn.execute("SELECT pages_json FROM legal_version_text WHERE version_id=?", (version_id,)).fetchone()
    return {"hidden_text": kinds.get("hidden_text", 0), "instructions": kinds.get("instructions", 0) + hidden_ai,
            "tracked_changes": kinds.get("tracked_insert", 0) + kinds.get("tracked_delete", 0), "comments": kinds.get("comment", 0),
            "ocr_pages": sum(1 for p in json.loads(row[0]) if p["source"] == "ocr") if row else 0,
            "ocr_low_confidence": kinds.get("ocr_low_confidence", 0)}


def _findings(conn, version_id: Optional[str]) -> dict:
    if not version_id:
        return {}
    return {k: n for k, n in conn.execute("SELECT kind, COUNT(*) FROM legal_findings WHERE version_id=? GROUP BY kind",
                                          (version_id,))}


def _document_out(conn, row) -> dict:
    d = dict(zip(_D_COLS, row))
    cur = conn.execute(f"SELECT {', '.join(_V_COLS)} FROM legal_document_versions WHERE id=?",
                       (d["current_version_id"],)).fetchone() if d["current_version_id"] else None
    if d["connector_id"]:
        k = conn.execute("SELECT name FROM legal_connectors WHERE id=?", (d["connector_id"],)).fetchone()
        source = {"kind": "connector", "name": k[0] if k else None, "state": d["source_state"] or "current", "path": d["remote_id"]}
    else:
        source = {"kind": "upload"}
    return {"id": d["id"], "title": d["title"], "kind": d["kind"], "status": d["status"], "version": d["version"],
            "created_by": _username(conn, d["created_by"]), "created_at": d["created_at"], "updated_at": d["updated_at"],
            "current_version": _version_out(conn, cur) if cur else None,
            "findings": _findings(conn, d["current_version_id"]), "source": source, **_label(d["id"])}


def _label(document_id: str) -> dict:
    from ..records import labels
    st = labels.state("document", document_id)
    return {"label": st["current"]["label"] if st["current"] else None, "label_suggested": bool(st["suggestions"])}


_DSEL = ", ".join(f"d.{c}" for c in _D_COLS)


def list_documents(matter_id: str, uid: Optional[str] = None) -> list[dict]:
    where, args = _where(uid)
    with get_conn() as conn:
        rows = conn.execute(f"SELECT {_DSEL} FROM legal_documents d WHERE d.matter_id=? AND {where} "
                            "ORDER BY COALESCE(d.updated_at, d.created_at) DESC", (matter_id, *args)).fetchall()
        return [_document_out(conn, r) for r in rows]


def get_document(matter_id: str, document_id: str, uid: Optional[str] = None) -> Optional[dict]:
    where, args = _where(uid)
    with get_conn() as conn:
        row = conn.execute(f"SELECT {_DSEL} FROM legal_documents d WHERE d.id=? AND d.matter_id=? AND {where}",
                           (document_id, matter_id, *args)).fetchone()
        return _document_out(conn, row) if row else None


def visible(matter_id: str, document_id: str, uid: str) -> bool:
    where, args = _where(uid)
    with get_conn() as conn:
        return conn.execute(f"SELECT 1 FROM legal_documents d WHERE d.id=? AND d.matter_id=? AND {where}",
                            (document_id, matter_id, *args)).fetchone() is not None


def list_versions(matter_id: str, document_id: str, uid: Optional[str] = None) -> list[dict]:
    if not get_document(matter_id, document_id, uid):
        return []
    with get_conn() as conn:
        rows = conn.execute(f"SELECT {', '.join(_V_COLS)} FROM legal_document_versions WHERE document_id=? AND matter_id=? "
                            "ORDER BY number DESC", (document_id, matter_id)).fetchall()
        return [_version_out(conn, r) for r in rows]


def version_row(matter_id: str, document_id: str, version_id: str, uid: Optional[str] = None) -> Optional[dict]:
    ok = visible(matter_id, document_id, uid) if uid else get_document(matter_id, document_id) is not None
    if not ok:
        return None
    with get_conn() as conn:
        row = conn.execute(f"SELECT {', '.join(_V_COLS)} FROM legal_document_versions WHERE id=? AND document_id=? AND matter_id=?",
                           (version_id, document_id, matter_id)).fetchone()
    return dict(zip(_V_COLS, row)) if row else None


def reader(version_id: str) -> Optional[dict]:
    """A version's stored text and what reading it found (S2.1). None if it was read before S2.1."""
    with get_conn() as conn:
        row = conn.execute("SELECT text, pages_json, lines_json, blocks_json, engines_json FROM legal_version_text WHERE version_id=?",
                           (version_id,)).fetchone()
        if not row:
            return None
        clauses = [dict(zip(("number", "label", "heading", "level", "parent", "start", "end", "page"), r)) for r in conn.execute(
            "SELECT number, label, heading, level, parent, char_start, char_end, page FROM legal_clauses WHERE version_id=? "
            "ORDER BY char_start, level", (version_id,))]
        terms = [dict(zip(("term", "start", "end", "term_start", "term_end"), r)) for r in conn.execute(
            "SELECT term, char_start, char_end, term_start, term_end FROM legal_defined_terms WHERE version_id=? ORDER BY char_start",
            (version_id,))]
        marks = [{"kind": r[0], "start": r[1], "end": r[2], "page": r[3], "meta": json.loads(r[4] or "{}")} for r in conn.execute(
            "SELECT kind, char_start, char_end, page, meta_json FROM legal_marks WHERE version_id=? ORDER BY COALESCE(char_start, -1)",
            (version_id,))]
    blocks = [{k: b[k] for k in ("kind", "start", "end", "label", "path")} for b in json.loads(row[3])]
    return {"text": row[0], "pages": json.loads(row[1]), "lines": json.loads(row[2]), "blocks": blocks,
            "engines": json.loads(row[4]), "clauses": clauses, "terms": terms, "marks": marks}


def current_sha(matter_id: str, document_id: str) -> Optional[str]:
    with get_conn() as conn:
        row = conn.execute("SELECT v.sha256 FROM legal_documents d JOIN legal_document_versions v ON v.id = d.current_version_id "
                           "WHERE d.id=? AND d.matter_id=?", (document_id, matter_id)).fetchone()
    return row[0] if row else None


def same_content_in_matter(matter_id: str, sha256: str, exclude_document: Optional[str] = None) -> Optional[str]:
    """Title of another document in this matter whose current version has identical content."""
    with get_conn() as conn:
        row = conn.execute("SELECT d.title FROM legal_documents d JOIN legal_document_versions v ON v.id = d.current_version_id "
                           "WHERE d.matter_id=? AND v.sha256=? AND d.id IS NOT ? AND d.deleted_at IS NULL",
                           (matter_id, sha256, exclude_document)).fetchone()
    return row[0] if row else None


# ── Connector-synced documents (S1.11) ───────────────────────────────────────────
def find_remote(link_id: str) -> dict[str, dict]:
    """{remote_id: {id, remote_version, deleted}} for documents a connector link has synced."""
    with get_conn() as conn:
        rows = conn.execute("SELECT id, remote_id, remote_version, deleted_at, source_state FROM legal_documents WHERE link_id=?",
                            (link_id,)).fetchall()
    return {r[1]: {"id": r[0], "remote_version": r[2], "deleted": r[3] is not None, "state": r[4]} for r in rows}


def create_synced_document(org_id: str, matter_id: str, title: str, by: str, connector_id: str, link_id: str,
                           remote_id: str) -> str:
    did = new_id()
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_documents (id, org_id, matter_id, title, kind, status, created_by, created_at, connector_id, "
                     "link_id, remote_id, acl_restricted, source_state) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (did, org_id, matter_id, title, "other", "received", by, now_iso(), connector_id, link_id, remote_id, 1, "current"))
        conn.commit()
    return did


def set_remote_version(document_id: str, version: str, state: str = "current") -> None:
    with get_conn() as conn:
        conn.execute("UPDATE legal_documents SET remote_version=?, source_state=? WHERE id=?", (version, state, document_id))
        conn.commit()


def set_source_state(document_id: str, state: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE legal_documents SET source_state=? WHERE id=?", (state, document_id))
        conn.commit()


def acl_of(document_id: str) -> set[str]:
    with get_conn() as conn:
        return {r[0] for r in conn.execute("SELECT user_id FROM legal_document_acl WHERE document_id=?", (document_id,))}


def set_acl(document_id: str, user_ids: set[str]) -> bool:
    """Replace who the source says may see the document; True if it changed. An empty set means
    nobody (the document stays restricted)."""
    if acl_of(document_id) == set(user_ids):
        return False
    with get_conn() as conn:
        conn.execute("DELETE FROM legal_document_acl WHERE document_id=?", (document_id,))
        conn.executemany("INSERT INTO legal_document_acl (document_id, user_id) VALUES (?,?)", [(document_id, u) for u in user_ids])
        conn.execute("UPDATE legal_documents SET acl_restricted=1 WHERE id=?", (document_id,))
        conn.commit()
    return True
