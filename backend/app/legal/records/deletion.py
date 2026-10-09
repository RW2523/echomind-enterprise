"""The deletion service (S1.9, SEC-07; spec test T04). Every Law deletion goes through here.

1. **Request.** Refused if the matter is under a legal hold. A person's request soft-deletes at
   once: the item disappears from lists, search and research, but nothing is destroyed. A
   retention request is only proposed.
2. **Report.** A dry run counts everything the purge would remove.
3. **Approval** by a second person with records authority. Rejecting restores the item.
4. **Purge.** One database transaction that re-checks holds before deleting anything. Then the
   stored files and index vectors go.
5. **Verification.** Database, matter index, stored files and quotes in research answers are
   re-checked, and the result is stored with the request.

The audit trail is never purged. The deletion record stays, without the object's title.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from typing import Optional

from ...core.config import settings
from ...core.db import get_conn
from ..policy import now_iso
from ..retrieval import registry
from . import holds, labels

KEPT_DOCUMENT = ["The audit trail (identifiers only, never content)", "This deletion record, without the document's title"]
KEPT_TRANSCRIPT = ["Consent records and the recording timeline", "Audio receipts (hashes; the audio itself was never kept)",
                   "The audit trail (identifiers only, never content)", "This deletion record"]
PENDING = ("requested", "blocked")


class NotFound(Exception):
    pass


class Conflict(Exception):
    pass


_COLS = ("id", "org_id", "matter_id", "object_type", "object_id", "object_label", "object_label_sha256", "reason", "origin",
         "soft_deleted", "status", "requested_by", "requested_at", "decided_by", "decided_at", "decision_note", "purged_at",
         "report_json", "verification_json")


def _row(conn, deletion_id: str, org_id: Optional[str] = None) -> Optional[dict]:
    q = f"SELECT {', '.join(_COLS)} FROM legal_deletions WHERE id=?"
    args: list = [deletion_id]
    if org_id:
        q += " AND org_id=?"
        args.append(org_id)
    r = conn.execute(q, args).fetchone()
    return dict(zip(_COLS, r)) if r else None


# ── Reports ──────────────────────────────────────────────────────────────────────
def _blob(matter_id: str, version_id: str) -> str:
    return os.path.join(settings.DATA_DIR, "legal", "blobs", matter_id, f"{version_id}.bin")


def _citing_turns(conn, matter_id: str, document_id: str) -> list[tuple[str, list]]:
    out = []
    for tid, cj in conn.execute("SELECT id, citations_json FROM legal_research_turns WHERE matter_id=? AND citations_json LIKE ?",
                                (matter_id, f"%{document_id}%")).fetchall():
        cites = json.loads(cj or "[]")
        if any(c.get("document_id") == document_id for c in cites):
            out.append((tid, cites))
    return out


def report_document(conn, matter_id: str, document_id: str) -> dict:
    versions = [r[0] for r in conn.execute("SELECT id FROM legal_document_versions WHERE document_id=? AND matter_id=?",
                                           (document_id, matter_id))]
    chunks = [r[0] for r in conn.execute("SELECT id FROM legal_chunks WHERE document_id=?", (document_id,))]
    findings = conn.execute(f"SELECT COUNT(*) FROM legal_findings WHERE version_id IN ({','.join('?' * len(versions)) or 'NULL'})",
                            versions).fetchone()[0] if versions else 0
    files = [v for v in versions if os.path.exists(_blob(matter_id, v))]
    in_index = set(registry.for_matter(matter_id).chunk_ids) & set(chunks)
    from ..exports import service as exports
    return {"object": "document", "versions": len(versions), "passages": len(chunks), "findings": findings,
            "exports": exports.count_for(conn, "document", document_id),
            "labels": conn.execute("SELECT COUNT(*) FROM legal_labels WHERE object_type='document' AND object_id=?", (document_id,)).fetchone()[0],
            "files": len(files), "file_bytes": sum(os.path.getsize(_blob(matter_id, v)) for v in files),
            "index_vectors": len(in_index), "research_citations": len(_citing_turns(conn, matter_id, document_id)),
            "kept": KEPT_DOCUMENT}


def report_transcript(conn, matter_id: str, meeting_id: str) -> dict:
    n = conn.execute("SELECT COUNT(*) FROM legal_transcript_segments WHERE meeting_id=? AND matter_id=?", (meeting_id, matter_id)).fetchone()[0]
    from ..exports import service as exports
    return {"object": "meeting_transcript", "segments": n, "exports": exports.count_for(conn, "meeting_transcript", meeting_id),
            "kept": KEPT_TRANSCRIPT}


def preview(matter_id: str, object_type: str, object_id: str) -> dict:
    with get_conn() as conn:
        _target(conn, matter_id, object_type, object_id)
        return report_document(conn, matter_id, object_id) if object_type == "document" else report_transcript(conn, matter_id, object_id)


def _target(conn, matter_id: str, object_type: str, object_id: str) -> str:
    """The object's display label; NotFound unless it exists, in this matter, and is not already deleted."""
    if object_type == "document":
        r = conn.execute("SELECT title FROM legal_documents WHERE id=? AND matter_id=? AND deleted_at IS NULL",
                         (object_id, matter_id)).fetchone()
    else:
        r = conn.execute("SELECT title FROM legal_meetings WHERE id=? AND matter_id=? AND transcript_deleted_at IS NULL",
                         (object_id, matter_id)).fetchone()
        if r:
            r = (f"Transcript of {r[0]}",)
    if not r:
        raise NotFound(object_id)
    return r[0]


# ── Soft delete and restore ──────────────────────────────────────────────────────
def _soft_delete(conn, d: dict) -> list[str]:
    """Hide the object; returns the passage ids to take out of the matter's index."""
    if d["object_type"] == "document":
        conn.execute("UPDATE legal_documents SET deleted_at=?, deletion_id=? WHERE id=?", (now_iso(), d["id"], d["object_id"]))
        ids = [r[0] for r in conn.execute("SELECT id FROM legal_chunks WHERE document_id=? AND active=1", (d["object_id"],))]
        conn.execute("UPDATE legal_chunks SET active=0 WHERE document_id=?", (d["object_id"],))
        return ids
    conn.execute("UPDATE legal_meetings SET transcript_deleted_at=?, transcript_deletion_id=? WHERE id=?",
                 (now_iso(), d["id"], d["object_id"]))
    return []


def _undo_soft_delete(conn, d: dict) -> bool:
    if not d["soft_deleted"]:
        return False
    if d["object_type"] == "document":
        conn.execute("UPDATE legal_documents SET deleted_at=NULL, deletion_id=NULL WHERE id=? AND deletion_id=?", (d["object_id"], d["id"]))
        conn.execute("UPDATE legal_chunks SET active=1 WHERE version_id=(SELECT current_version_id FROM legal_documents WHERE id=?)",
                     (d["object_id"],))
        return True
    conn.execute("UPDATE legal_meetings SET transcript_deleted_at=NULL, transcript_deletion_id=NULL WHERE id=? AND transcript_deletion_id=?",
                 (d["object_id"], d["id"]))
    return False


def request(org_id: str, matter_id: str, object_type: str, object_id: str, reason: str, by: str,
            origin: str = "person") -> dict:
    """Raises holds.Held, NotFound or Conflict (already pending)."""
    soft = origin in ("person", "connector")      # a person's request, or removal at the source: hidden at once
    did = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            holds.assert_not_held(conn, matter_id)
            label = _target(conn, matter_id, object_type, object_id)
            if conn.execute("SELECT 1 FROM legal_deletions WHERE object_type=? AND object_id=? AND status IN ('requested','blocked')",
                            (object_type, object_id)).fetchone():
                raise Conflict("A deletion of this item is already waiting for approval.")
            rep = report_document(conn, matter_id, object_id) if object_type == "document" else report_transcript(conn, matter_id, object_id)
            conn.execute("INSERT INTO legal_deletions (id, org_id, matter_id, object_type, object_id, object_label, object_label_sha256, "
                         "reason, origin, soft_deleted, status, requested_by, requested_at, report_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (did, org_id, matter_id, object_type, object_id, label, hashlib.sha256(label.encode()).hexdigest(), reason,
                          origin, int(soft), "requested", by, now_iso(), json.dumps(rep)))
            hidden = _soft_delete(conn, {"id": did, "object_type": object_type, "object_id": object_id}) if soft else []
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    if hidden:
        registry.for_matter(matter_id).remove(hidden)
    return get(did)


def restore(deletion_id: str, by: str, status: str = "restored", note: Optional[str] = None) -> dict:
    with get_conn() as conn:
        d = _row(conn, deletion_id)
        if not d:
            raise NotFound(deletion_id)
        if d["status"] not in PENDING:
            raise Conflict(f"This request is already {d['status']}.")
        conn.execute("BEGIN IMMEDIATE")
        reindex = _undo_soft_delete(conn, d)
        conn.execute("UPDATE legal_deletions SET status=?, decided_by=?, decided_at=?, decision_note=? WHERE id=?",
                     (status, by, now_iso(), note, deletion_id))
        conn.commit()
    if reindex:
        registry.for_matter(d["matter_id"]).rebuild_from_db()
    return get(deletion_id)


# ── Purge and verification ───────────────────────────────────────────────────────
def purge(deletion_id: str, approved_by: str, note: Optional[str] = None) -> dict:
    """Destroy the item for good. Raises holds.Held (the request is marked blocked), NotFound or Conflict."""
    with get_conn() as conn:
        d = _row(conn, deletion_id)
        if not d:
            raise NotFound(deletion_id)
        if d["status"] not in PENDING:
            raise Conflict(f"This request is already {d['status']}.")
        conn.execute("BEGIN IMMEDIATE")
        try:
            if holds.matter_on_hold(conn, d["matter_id"]):
                raise holds.Held(d["matter_id"])
            from ..exports import service as exports
            if d["object_type"] == "document":
                gone = _purge_document(conn, d)
            else:
                gone = _purge_transcript(conn, d)
            gone["exports"] = exports.purge_for(conn, d["object_type"], d["object_id"])
            conn.execute("UPDATE legal_deletions SET status='purged', purged_at=?, decided_by=?, decided_at=?, decision_note=?, "
                         "object_label=NULL WHERE id=?", (now_iso(), approved_by, now_iso(), note, deletion_id))
            conn.commit()
        except holds.Held:
            conn.rollback()
            conn.execute("UPDATE legal_deletions SET status='blocked', decided_by=?, decided_at=? WHERE id=?",
                         (approved_by, now_iso(), deletion_id))
            conn.commit()
            raise
        except BaseException:
            conn.rollback()
            raise
    # Outside the database: stored files and index vectors (both re-checked below).
    for vid in gone.get("versions", []):
        try:
            os.remove(_blob(d["matter_id"], vid))
        except FileNotFoundError:
            pass
    if gone.get("chunks"):
        registry.for_matter(d["matter_id"]).remove(gone["chunks"])
    from ..documents import blobs
    for matter_id, export_id in gone.get("exports", []):
        blobs.remove(matter_id, export_id, "exports")
    verification = verify(d, gone)
    with get_conn() as conn:
        conn.execute("UPDATE legal_deletions SET verification_json=? WHERE id=?", (json.dumps(verification), deletion_id))
        conn.commit()
    return get(deletion_id)


def _purge_document(conn, d: dict) -> dict:
    doc = d["object_id"]
    versions = [r[0] for r in conn.execute("SELECT id FROM legal_document_versions WHERE document_id=?", (doc,))]
    chunks = [r[0] for r in conn.execute("SELECT id FROM legal_chunks WHERE document_id=?", (doc,))]
    if versions:
        conn.execute(f"DELETE FROM legal_findings WHERE version_id IN ({','.join('?' * len(versions))})", versions)
    conn.execute("DELETE FROM legal_chunks WHERE document_id=?", (doc,))
    conn.execute("DELETE FROM legal_document_versions WHERE document_id=?", (doc,))
    labels.purge(conn, "document", doc)
    conn.execute("DELETE FROM legal_documents WHERE id=?", (doc,))
    # Research answers keep their wording, but quotes from the purged document are removed.
    for tid, cites in _citing_turns(conn, d["matter_id"], doc):
        cleaned = [({**c, "text": "", "document_title": "", "removed": True} if c.get("document_id") == doc else c) for c in cites]
        conn.execute("UPDATE legal_research_turns SET citations_json=? WHERE id=?", (json.dumps(cleaned), tid))
    return {"versions": versions, "chunks": chunks}


def _purge_transcript(conn, d: dict) -> dict:
    conn.execute("DELETE FROM legal_transcript_segments WHERE meeting_id=? AND matter_id=?", (d["object_id"], d["matter_id"]))
    conn.execute("UPDATE legal_meetings SET transcript_deleted_at=NULL, transcript_deletion_id=NULL WHERE id=?", (d["object_id"],))
    return {}


def verify(d: dict, gone: dict) -> dict:
    with get_conn() as conn:
        if d["object_type"] == "document":
            doc = d["object_id"]
            versions = gone.get("versions", [])
            db_rows = sum(conn.execute(q, (doc,)).fetchone()[0] for q in (
                "SELECT COUNT(*) FROM legal_documents WHERE id=?", "SELECT COUNT(*) FROM legal_document_versions WHERE document_id=?",
                "SELECT COUNT(*) FROM legal_chunks WHERE document_id=?",
                "SELECT COUNT(*) FROM legal_labels WHERE object_type='document' AND object_id=?"))
            if versions:
                db_rows += conn.execute(f"SELECT COUNT(*) FROM legal_findings WHERE version_id IN ({','.join('?' * len(versions))})",
                                        versions).fetchone()[0]
            quotes = [c for _, cites in _citing_turns(conn, d["matter_id"], doc) for c in cites
                      if c.get("document_id") == doc and (c.get("text") or c.get("document_title"))]
            idx = registry.for_matter(d["matter_id"])
            checks = {"database": db_rows == 0,
                      "index": not (set(idx.chunk_ids) & set(gone.get("chunks", []))),
                      "files": not any(os.path.exists(_blob(d["matter_id"], v)) for v in versions),
                      "research_quotes": not quotes}
        else:
            n = conn.execute("SELECT COUNT(*) FROM legal_transcript_segments WHERE meeting_id=?", (d["object_id"],)).fetchone()[0]
            checks = {"database": n == 0}
        from ..exports import service as exports
        checks["exports"] = exports.count_for(conn, d["object_type"], d["object_id"]) == 0 and not any(
            exports.file_exists(m, e) for m, e in gone.get("exports", []))
    return {"ok": all(checks.values()), "checks": checks, "verified_at": now_iso()}


# ── Reading ──────────────────────────────────────────────────────────────────────
def _out(conn, d: dict) -> dict:
    def name(uid):
        r = conn.execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone() if uid else None
        return r[0] if r else None
    m = conn.execute("SELECT number, title FROM matters WHERE id=?", (d["matter_id"],)).fetchone()
    return {"id": d["id"], "object_type": d["object_type"], "object_id": d["object_id"], "object_label": d["object_label"],
            "reason": d["reason"], "origin": d["origin"], "status": d["status"], "requested_by": name(d["requested_by"]),
            "requested_at": d["requested_at"], "decided_by": name(d["decided_by"]), "decided_at": d["decided_at"],
            "decision_note": d["decision_note"], "purged_at": d["purged_at"], "report": json.loads(d["report_json"]),
            "verification": json.loads(d["verification_json"]) if d["verification_json"] else None,
            "matter": {"id": d["matter_id"], "number": m[0], "title": m[1]} if m else None}


def get(deletion_id: str) -> dict:
    with get_conn() as conn:
        d = _row(conn, deletion_id)
        return _out(conn, d) if d else None


def row(org_id: str, deletion_id: str) -> Optional[dict]:
    with get_conn() as conn:
        return _row(conn, deletion_id, org_id)


def listing(org_id: str, matter_id: Optional[str] = None, pending_only: bool = False) -> list[dict]:
    q = f"SELECT {', '.join(_COLS)} FROM legal_deletions WHERE org_id=?"
    args: list = [org_id]
    if matter_id:
        q += " AND matter_id=?"
        args.append(matter_id)
    if pending_only:
        q += " AND status IN ('requested','blocked')"
    with get_conn() as conn:
        return [_out(conn, dict(zip(_COLS, r))) for r in conn.execute(q + " ORDER BY requested_at DESC LIMIT 200", args).fetchall()]
