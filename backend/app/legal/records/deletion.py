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

KEPT_DOCUMENT = ["Research notes and drafts already filed as matter documents", "The audit trail (identifiers only, never content)",
                 "This deletion record, without the document's title"]
KEPT_TRANSCRIPT = ["Consent records and the recording timeline", "Audio receipts (hashes; the audio itself was never kept)",
                   "Minutes already filed as a matter document", "The audit trail (identifiers only, never content)", "This deletion record"]
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


def _evidence_turns(conn, matter_id: str, document_id: str) -> list[tuple[str, list, list]]:
    rows = conn.execute("SELECT id, evidence_json, conflicts_json FROM legal_research_turns WHERE matter_id=? "
                        "AND (evidence_json LIKE ? OR conflicts_json LIKE ?)", (matter_id, f"%{document_id}%", f"%{document_id}%")).fetchall()
    return [(r[0], json.loads(r[1] or "[]"), json.loads(r[2] or "[]")) for r in rows]


def _citing_turns(conn, matter_id: str, document_id: str) -> list[tuple[str, list]]:
    out = []
    for tid, cj in conn.execute("SELECT id, citations_json FROM legal_research_turns WHERE matter_id=? AND citations_json LIKE ?",
                                (matter_id, f"%{document_id}%")).fetchall():
        cites = json.loads(cj or "[]")
        if any(c.get("document_id") == document_id for c in cites):
            out.append((tid, cites))
    return out


def _notes(conn, matter_id: str, document_id: str, kind_: str = "research_note") -> list[str]:
    """Research notes (S2.7) or drafts (S3.1), every version of each, that quote the document: their subjects."""
    from .. import artifacts
    out = set()
    for aid, subject, kind, cj in conn.execute("SELECT id, subject_id, kind, content_json FROM legal_ai_artifacts WHERE matter_id=? "
                                               "AND kind=? AND content_json LIKE ?", (matter_id, kind_, f"%{document_id}%")).fetchall():
        if document_id in artifacts.cited_documents({"kind": kind, "content": json.loads(cj)}):
            out.add(subject)
    return sorted(out)


def _comparisons(conn, matter_id: str, document_id: str) -> list[str]:
    """Saved comparisons and conflict scans that quote the document (S2.3)."""
    return [r[0] for r in conn.execute("SELECT c.id FROM legal_comparisons c WHERE c.matter_id=? AND EXISTS "
                                       "(SELECT 1 FROM json_each(c.document_ids_json) j WHERE j.value=?)", (matter_id, document_id))]


def report_document(conn, matter_id: str, document_id: str) -> dict:
    versions = [r[0] for r in conn.execute("SELECT id FROM legal_document_versions WHERE document_id=? AND matter_id=?",
                                           (document_id, matter_id))]
    chunks = [r[0] for r in conn.execute("SELECT id FROM legal_chunks WHERE document_id=?", (document_id,))]
    findings = conn.execute(f"SELECT COUNT(*) FROM legal_findings WHERE version_id IN ({','.join('?' * len(versions)) or 'NULL'})",
                            versions).fetchone()[0] if versions else 0
    files = [v for v in versions if os.path.exists(_blob(matter_id, v))]
    in_index = set(registry.for_matter(matter_id).chunk_ids) & set(chunks)
    from ..exports import service as exports
    qs = ",".join("?" * len(versions)) or "NULL"
    stored = conn.execute(f"SELECT COUNT(*) FROM legal_version_text WHERE version_id IN ({qs})", versions).fetchone()[0] if versions else 0
    marks = conn.execute(f"SELECT COUNT(*) FROM legal_marks WHERE version_id IN ({qs})", versions).fetchone()[0] if versions else 0
    clause_rows = conn.execute(f"SELECT COUNT(*) FROM legal_clauses WHERE version_id IN ({qs})", versions).fetchone()[0] if versions else 0
    term_rows = conn.execute(f"SELECT COUNT(*) FROM legal_defined_terms WHERE version_id IN ({qs})", versions).fetchone()[0] if versions else 0
    return {"object": "document", "versions": len(versions), "passages": len(chunks), "findings": findings,
            "stored_text": stored, "marks": marks, "clauses": clause_rows, "defined_terms": term_rows,
            "exports": exports.count_for(conn, "document", document_id),
            "labels": conn.execute("SELECT COUNT(*) FROM legal_labels WHERE object_type='document' AND object_id=?", (document_id,)).fetchone()[0],
            "files": len(files), "file_bytes": sum(os.path.getsize(_blob(matter_id, v)) for v in files),
            "index_vectors": len(in_index), "research_citations": len(_citing_turns(conn, matter_id, document_id)),
            "comparisons": len(_comparisons(conn, matter_id, document_id)),
            "research_notes": len(_notes(conn, matter_id, document_id)),
            "drafts": len(_notes(conn, matter_id, document_id, "legal_draft")),
            "playbook_checks": conn.execute("SELECT COUNT(*) FROM legal_playbook_checks WHERE document_id=? AND matter_id=?",
                                            (document_id, matter_id)).fetchone()[0],
            "kept": KEPT_DOCUMENT}


def _transcript_document(conn, matter_id: str, meeting_id: str) -> Optional[str]:
    """The matter document a finalised transcript became (S2.4), if it still exists."""
    row = conn.execute("SELECT m.transcript_document_id FROM legal_meetings m JOIN legal_documents d ON d.id = m.transcript_document_id "
                       "WHERE m.id=? AND m.matter_id=?", (meeting_id, matter_id)).fetchone()
    return row[0] if row else None


def report_transcript(conn, matter_id: str, meeting_id: str) -> dict:
    n = conn.execute("SELECT COUNT(*) FROM legal_transcript_segments WHERE meeting_id=? AND matter_id=?", (meeting_id, matter_id)).fetchone()[0]
    revisions = conn.execute("SELECT COUNT(*) FROM legal_transcript_revisions WHERE meeting_id=? AND matter_id=?",
                             (meeting_id, matter_id)).fetchone()[0]
    from ..exports import service as exports
    doc = _transcript_document(conn, matter_id, meeting_id)
    alerts = conn.execute("SELECT COUNT(*) FROM legal_assist_alerts WHERE meeting_id=? AND matter_id=?", (meeting_id, matter_id)).fetchone()[0]
    minutes = conn.execute("SELECT COUNT(*) FROM legal_ai_artifacts WHERE subject_type='meeting' AND subject_id=? AND matter_id=?",
                           (meeting_id, matter_id)).fetchone()[0]
    return {"object": "meeting_transcript", "segments": n, "revisions": revisions, "alerts": alerts, "minutes_versions": minutes,
            "finalised_document": 1 if doc else 0,
            "exports": exports.count_for(conn, "meeting_transcript", meeting_id), "kept": KEPT_TRANSCRIPT}


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
    doc = _transcript_document(conn, d["matter_id"], d["object_id"]) if d.get("matter_id") else None
    if doc:                                       # the finalised copy is hidden with the transcript (S2.4)
        conn.execute("UPDATE legal_documents SET deleted_at=?, deletion_id=? WHERE id=? AND deleted_at IS NULL", (now_iso(), d["id"], doc))
        ids = [r[0] for r in conn.execute("SELECT id FROM legal_chunks WHERE document_id=? AND active=1", (doc,))]
        conn.execute("UPDATE legal_chunks SET active=0 WHERE document_id=?", (doc,))
        return ids
    return []


def _undo_soft_delete(conn, d: dict) -> bool:
    if not d["soft_deleted"]:
        return False
    if d["object_type"] == "document":
        conn.execute("UPDATE legal_documents SET deleted_at=NULL, deletion_id=NULL WHERE id=? AND deletion_id=?", (d["object_id"], d["id"]))
        from ..documents import versions
        versions.sync_active(conn, d["object_id"])           # current and latest signed versions (S2.2a); index rebuilt after
        return True
    conn.execute("UPDATE legal_meetings SET transcript_deleted_at=NULL, transcript_deletion_id=NULL WHERE id=? AND transcript_deletion_id=?",
                 (d["object_id"], d["id"]))
    doc = conn.execute("SELECT id FROM legal_documents WHERE deletion_id=?", (d["id"],)).fetchone()
    if doc:
        conn.execute("UPDATE legal_documents SET deleted_at=NULL, deletion_id=NULL WHERE id=?", (doc[0],))
        from ..documents import versions
        versions.sync_active(conn, doc[0])
        return True
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
            hidden = _soft_delete(conn, {"id": did, "object_type": object_type, "object_id": object_id, "matter_id": matter_id}) if soft else []
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
    for matter_id, export_id in gone.get("exports", []) + gone.get("document_exports", []):
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
        qs = ",".join("?" * len(versions))
        conn.execute(f"DELETE FROM legal_findings WHERE version_id IN ({qs})", versions)
        conn.execute(f"DELETE FROM legal_marks WHERE version_id IN ({qs})", versions)
        conn.execute(f"DELETE FROM legal_version_text WHERE version_id IN ({qs})", versions)
        conn.execute(f"DELETE FROM legal_clauses WHERE version_id IN ({qs})", versions)
        conn.execute(f"DELETE FROM legal_defined_terms WHERE version_id IN ({qs})", versions)
    conn.execute("DELETE FROM legal_chunks WHERE document_id=?", (doc,))
    conn.execute("DELETE FROM legal_document_versions WHERE document_id=?", (doc,))
    labels.purge(conn, "document", doc)
    conn.execute("DELETE FROM legal_documents WHERE id=?", (doc,))
    # Research answers keep their wording, but quotes from the purged document are removed.
    for tid, cites in _citing_turns(conn, d["matter_id"], doc):
        cleaned = [({**c, "text": "", "document_title": "", "removed": True} if c.get("document_id") == doc else c) for c in cites]
        conn.execute("UPDATE legal_research_turns SET citations_json=? WHERE id=?", (json.dumps(cleaned), tid))
    for tid, ej, cj in _evidence_turns(conn, d["matter_id"], doc):          # S2.2: saved evidence and conflicts too
        ev = [({**e, "quote": "", "document_title": "", "removed": True} if e.get("document_id") == doc else e) for e in ej]
        conn.execute("UPDATE legal_research_turns SET evidence_json=?, conflicts_json=? WHERE id=?",
                     (json.dumps(ev), json.dumps([c for c in cj if c.get("document_id") != doc]), tid))
    conn.execute("UPDATE legal_ai_artifacts SET document_id=NULL WHERE document_id=?", (doc,))   # S2.6: filed minutes purged
    conn.execute("DELETE FROM legal_artifact_filings WHERE document_id=?", (doc,))               # S2.7: the filing goes with it
    # S2.7: research notes quote the document; every version of each is removed with its decisions
    # (a note can be sent again from the answer, whose quotes are removed above). Filed notes stay.
    for kind, subject_type in (("research_note", "research_turn"), ("legal_draft", "draft")):     # S3.1: drafts too
        for subject in _notes(conn, d["matter_id"], doc, kind):
            conn.execute("DELETE FROM legal_review_approvals WHERE artifact_id IN (SELECT id FROM legal_ai_artifacts "
                         "WHERE subject_type=? AND subject_id=? AND matter_id=?)", (subject_type, subject, d["matter_id"]))
            conn.execute("DELETE FROM legal_ai_artifacts WHERE subject_type=? AND subject_id=? AND matter_id=?",
                         (subject_type, subject, d["matter_id"]))
    # S2.5: alerts keep their decision, but the document's quote and name are removed from their source.
    for (aid,) in conn.execute("SELECT a.id FROM legal_assist_alerts a WHERE a.matter_id=? AND EXISTS "
                               "(SELECT 1 FROM json_each(a.document_ids_json) j WHERE j.value=?)", (d["matter_id"], doc)).fetchall():
        conn.execute("UPDATE legal_assist_alerts SET source_json=?, document_ids_json='[]' WHERE id=?", (json.dumps({"removed": True}), aid))
    # S3.2: playbook checks quote the document; they go with their decisions (each can be run again).
    conn.execute("DELETE FROM legal_playbook_decisions WHERE check_id IN (SELECT id FROM legal_playbook_checks WHERE document_id=?)", (doc,))
    conn.execute("DELETE FROM legal_playbook_checks WHERE document_id=?", (doc,))
    # S2.3: comparisons and scans quote the document throughout; they are removed whole (each can be run again).
    for cid in _comparisons(conn, d["matter_id"], doc):
        conn.execute("DELETE FROM legal_comparisons WHERE id=?", (cid,))
    return {"versions": versions, "chunks": chunks}


def _purge_transcript(conn, d: dict) -> dict:
    doc = conn.execute("SELECT transcript_document_id FROM legal_meetings WHERE id=?", (d["object_id"],)).fetchone()
    doc = doc[0] if doc and doc[0] and conn.execute("SELECT 1 FROM legal_documents WHERE id=?", (doc[0],)).fetchone() else None
    conn.execute("DELETE FROM legal_assist_alerts WHERE meeting_id=? AND matter_id=?", (d["object_id"], d["matter_id"]))   # S2.5
    # S2.6: minutes drafts and their review decisions quote the transcript; they go with it. Minutes already
    # filed as a matter document are a record of their own and stay (P-015, minutes-only mode).
    conn.execute("DELETE FROM legal_review_approvals WHERE artifact_id IN (SELECT id FROM legal_ai_artifacts WHERE subject_type='meeting' "
                 "AND subject_id=? AND matter_id=?)", (d["object_id"], d["matter_id"]))
    conn.execute("DELETE FROM legal_ai_artifacts WHERE subject_type='meeting' AND subject_id=? AND matter_id=?", (d["object_id"], d["matter_id"]))
    conn.execute("DELETE FROM legal_transcript_revisions WHERE meeting_id=? AND matter_id=?", (d["object_id"], d["matter_id"]))
    conn.execute("DELETE FROM legal_transcript_segments WHERE meeting_id=? AND matter_id=?", (d["object_id"], d["matter_id"]))
    conn.execute("UPDATE legal_meetings SET transcript_deleted_at=NULL, transcript_deletion_id=NULL, transcript_document_id=NULL, "
                 "transcript_finalised_at=NULL, transcript_finalised_by=NULL WHERE id=?", (d["object_id"],))
    if not doc:
        return {}
    # The finalised copy is a matter document: purged the same way, with its files and index entries (S2.4).
    from ..exports import service as exports
    gone = _purge_document(conn, {**d, "object_id": doc})
    return {**gone, "document": doc, "document_exports": exports.purge_for(conn, "document", doc)}


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
                qs = ",".join("?" * len(versions))
                for table in ("legal_findings", "legal_marks", "legal_version_text", "legal_clauses", "legal_defined_terms"):
                    db_rows += conn.execute(f"SELECT COUNT(*) FROM {table} WHERE version_id IN ({qs})", versions).fetchone()[0]
            quotes = [c for _, cites in _citing_turns(conn, d["matter_id"], doc) for c in cites
                      if c.get("document_id") == doc and (c.get("text") or c.get("document_title"))]
            quotes += [e for _, ev, cf in _evidence_turns(conn, d["matter_id"], doc) for e in ev + cf
                       if e.get("document_id") == doc and (e.get("quote") or e.get("document_title") or e.get("clause") and "other" in e)]
            db_rows += len(_comparisons(conn, d["matter_id"], doc)) + len(_notes(conn, d["matter_id"], doc)) + \
                len(_notes(conn, d["matter_id"], doc, "legal_draft"))
            db_rows += conn.execute("SELECT COUNT(*) FROM legal_artifact_filings WHERE document_id=?", (doc,)).fetchone()[0]
            db_rows += conn.execute("SELECT COUNT(*) FROM legal_playbook_checks WHERE document_id=?", (doc,)).fetchone()[0]
            db_rows += conn.execute("SELECT COUNT(*) FROM legal_assist_alerts a WHERE a.matter_id=? AND EXISTS "
                                    "(SELECT 1 FROM json_each(a.document_ids_json) j WHERE j.value=?)", (d["matter_id"], doc)).fetchone()[0]
            idx = registry.for_matter(d["matter_id"])
            checks = {"database": db_rows == 0,
                      "index": not (set(idx.chunk_ids) & set(gone.get("chunks", []))),
                      "files": not any(os.path.exists(_blob(d["matter_id"], v)) for v in versions),
                      "research_quotes": not quotes}
        else:
            n = conn.execute("SELECT COUNT(*) FROM legal_transcript_segments WHERE meeting_id=?", (d["object_id"],)).fetchone()[0]
            n += conn.execute("SELECT COUNT(*) FROM legal_transcript_revisions WHERE meeting_id=?", (d["object_id"],)).fetchone()[0]
            n += conn.execute("SELECT COUNT(*) FROM legal_assist_alerts WHERE meeting_id=?", (d["object_id"],)).fetchone()[0]
            n += conn.execute("SELECT COUNT(*) FROM legal_ai_artifacts WHERE subject_type='meeting' AND subject_id=?", (d["object_id"],)).fetchone()[0]
            checks = {"database": n == 0}
            if gone.get("document"):
                doc, versions = gone["document"], gone.get("versions", [])
                rows = sum(conn.execute(q, (doc,)).fetchone()[0] for q in (
                    "SELECT COUNT(*) FROM legal_documents WHERE id=?", "SELECT COUNT(*) FROM legal_chunks WHERE document_id=?"))
                idx = registry.for_matter(d["matter_id"])
                checks["finalised_document"] = rows == 0 and not (set(idx.chunk_ids) & set(gone.get("chunks", []))) and \
                    not any(os.path.exists(_blob(d["matter_id"], v)) for v in versions)
        from ..exports import service as exports
        checks["exports"] = exports.count_for(conn, d["object_type"], d["object_id"]) == 0 and not any(
            exports.file_exists(m, e) for m, e in gone.get("exports", []) + gone.get("document_exports", [])) and \
            (not gone.get("document") or exports.count_for(conn, "document", gone["document"]) == 0)
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
