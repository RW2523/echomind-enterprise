"""AI artifacts and their review (S2.6 foundation, S2.7 framework; decisions D-042, D-043).

An artifact is something the model drafted from matter content: a meeting's minutes (S2.6) or a
research answer sent for review by the person who asked (S2.7):
- **Versioned and fixed.** Content is never edited in place; a person's change is a new version.
  Each version's content is hashed (SHA-256 of its canonical JSON).
- **Approved by a second person.** An approval needs `artifacts.approve` on the matter (a lead,
  attorney or reviewer) and must come from someone other than the author of that version. It binds
  the version's content hash: anything exported or filed from it must carry the same hash.
- **Decisions are append-only.** A new version supersedes the earlier drafts. An approved version
  stays approved, as the record of what was approved, until a later version is approved in its turn.
- **Shown only to those who can see its sources.** A research note quotes documents; a person who
  cannot see one of them (source permissions, S1.11) does not see the note (`visible_to`).
- **Filed with its approval.** Filing an approved version makes a matter document and records which
  version and body it became (`legal_artifact_filings`), so an export can carry the approval and
  refuse a copy changed after it.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Optional

from ..core.db import get_conn
from .policy import now_iso


class Conflict(Exception):
    pass


# kind → (subject type, what it is called, the title of the document it is filed as)
KINDS = {"minutes": ("meeting", "Minutes", "Minutes — {title}"),
         "research_note": ("research_turn", "Research note", "Research note — {title}"),
         "legal_draft": ("draft", "Draft", "{title}")}


def title(a: dict) -> str:
    c = a["content"]
    if a["kind"] == "minutes":
        return (c.get("meeting") or {}).get("title") or ""
    return c.get("title") or "" if a["kind"] == "legal_draft" else c.get("question") or ""


def cited_documents(a: dict) -> set[str]:
    """Documents an artifact quotes. Minutes quote the transcript only."""
    c = a["content"]
    if a["kind"] == "legal_draft":                       # S3.1: field quotes and written passages' sources
        return {s["document_id"] for item in (c.get("fields") or []) + (c.get("blocks") or []) for s in item.get("sources") or []
                if s.get("document_id")}
    if a["kind"] != "research_note":
        return set()
    return {x["document_id"] for key in ("evidence", "sources", "conflicts") for x in c.get(key) or [] if x.get("document_id")}


def visible_to(a: dict, uid: str) -> bool:
    """Every document it quotes is visible to this person (deleted ones count as not visible)."""
    from .documents import repo as documents
    return all(documents.visible(a["matter_id"], d, uid) for d in cited_documents(a))


def canonical(content: dict) -> str:
    return json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(content: dict) -> str:
    return hashlib.sha256(canonical(content).encode("utf-8")).hexdigest()


_COLS = ("id", "matter_id", "kind", "subject_type", "subject_id", "version", "status", "content_json", "content_sha256", "inputs_json",
         "created_by", "created_at", "origin", "document_id")


def _out(conn, r) -> dict:
    d = dict(zip(_COLS, r))
    d["content"], d["inputs"] = json.loads(d.pop("content_json")), json.loads(d.pop("inputs_json"))
    d["created_by_name"] = (conn.execute("SELECT username FROM users WHERE id=?", (d["created_by"],)).fetchone() or [None])[0]
    d["title"] = title(d)
    d["decisions"] = [{"decision": x[0], "content_sha256": x[1], "reason": x[2], "by": x[3], "at": x[4]} for x in conn.execute(
        "SELECT a.decision, a.content_sha256, a.reason, (SELECT username FROM users u WHERE u.id = a.by), a.at "
        "FROM legal_review_approvals a WHERE a.artifact_id=? ORDER BY a.at, a.rowid", (d["id"],))]
    return d


def create(matter_id: str, kind: str, subject_type: str, subject_id: str, content: dict, inputs: dict, by: str, origin: str) -> dict:
    """A new version; earlier drafts of the same artifact are superseded."""
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            n = conn.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM legal_ai_artifacts WHERE subject_type=? AND subject_id=? AND kind=?",
                             (subject_type, subject_id, kind)).fetchone()[0]
            conn.execute("UPDATE legal_ai_artifacts SET status='superseded' WHERE subject_type=? AND subject_id=? AND kind=? AND status='draft'",
                         (subject_type, subject_id, kind))
            aid = str(uuid.uuid4())
            conn.execute("INSERT INTO legal_ai_artifacts (id, matter_id, kind, subject_type, subject_id, version, status, content_json, "
                         "content_sha256, inputs_json, created_by, created_at, origin) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (aid, matter_id, kind, subject_type, subject_id, n, "draft", canonical(content), digest(content),
                          json.dumps(inputs, sort_keys=True), by, now_iso(), origin))
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    return get(matter_id, aid)


def get(matter_id: str, artifact_id: str) -> Optional[dict]:
    with get_conn() as conn:
        r = conn.execute(f"SELECT {', '.join(_COLS)} FROM legal_ai_artifacts WHERE id=? AND matter_id=?", (artifact_id, matter_id)).fetchone()
        return _out(conn, r) if r else None


def versions(matter_id: str, subject_type: str, subject_id: str, kind: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(f"SELECT {', '.join(_COLS)} FROM legal_ai_artifacts WHERE matter_id=? AND subject_type=? AND subject_id=? "
                            f"AND kind=? ORDER BY version DESC", (matter_id, subject_type, subject_id, kind)).fetchall()
        return [_out(conn, r) for r in rows]


def decide(matter_id: str, artifact_id: str, decision: str, by: str, content_sha256: str, reason: Optional[str]) -> dict:
    """Approve, or ask for changes to, the version whose content has this hash. Raises Conflict when
    it is not the latest draft, the hash does not match what was reviewed, or the author approves."""
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            r = conn.execute("SELECT status, content_sha256, created_by, subject_type, subject_id, kind, version FROM legal_ai_artifacts "
                             "WHERE id=? AND matter_id=?", (artifact_id, matter_id)).fetchone()
            if not r:
                raise LookupError(artifact_id)
            status, sha, author, st, sid, kind, version = r
            latest = conn.execute("SELECT MAX(version) FROM legal_ai_artifacts WHERE subject_type=? AND subject_id=? AND kind=?",
                                  (st, sid, kind)).fetchone()[0]
            if status != "draft" or version != latest:
                raise Conflict("Only the latest draft can be reviewed.")
            if content_sha256 != sha:
                raise Conflict("This draft changed since you opened it. Review the current version.")
            if decision == "approved" and by == author:
                raise Conflict("A draft is approved by someone other than its author.")
            if decision == "approved" and kind == "legal_draft":
                gaps = json.loads(conn.execute("SELECT content_json FROM legal_ai_artifacts WHERE id=?", (artifact_id,)).fetchone()[0]).get("gaps", 0)
                if gaps:
                    raise Conflict(f"Fill every [[NEEDS INPUT]] gap before approving ({gaps} left).")
            conn.execute("INSERT INTO legal_review_approvals (id, artifact_id, matter_id, decision, content_sha256, reason, by, at) "
                         "VALUES (?,?,?,?,?,?,?,?)", (str(uuid.uuid4()), artifact_id, matter_id, decision, sha, reason, by, now_iso()))
            if decision == "approved":
                conn.execute("UPDATE legal_ai_artifacts SET status='approved' WHERE id=?", (artifact_id,))
                conn.execute("UPDATE legal_ai_artifacts SET status='superseded' WHERE subject_type=? AND subject_id=? AND kind=? "
                             "AND status='approved' AND version < ?", (st, sid, kind, version))
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    return get(matter_id, artifact_id)


def approved_hash(artifact: dict) -> Optional[str]:
    """The content hash an approval bound, if this version is approved and unchanged."""
    if artifact["status"] != "approved":
        return None
    last = next((d for d in reversed(artifact["decisions"]) if d["decision"] == "approved"), None)
    return last["content_sha256"] if last and last["content_sha256"] == digest(artifact["content"]) else None


def approval_of(artifact: dict) -> Optional[dict]:
    """The approval in force on this version: who and when, and the hash it bound."""
    if not approved_hash(artifact):
        return None
    return next(d for d in reversed(artifact["decisions"]) if d["decision"] == "approved")


def latest(matter_id: str, subject_type: str, subject_id: str, kind: str) -> Optional[dict]:
    found = versions(matter_id, subject_type, subject_id, kind)
    return found[0] if found else None


def file(p: dict, artifact: dict, body: bytes, filename: str, mime: str = "text/markdown", doc_kind: str = "note") -> str:
    """An approved version becomes a matter document (status final) carrying its approval; the filing
    is recorded with the body's hash. Returns the document id."""
    import hashlib as _hashlib
    from .documents import blobs
    from .documents import repo as docs
    from .jobs import worker
    approval = approval_of(artifact)
    if not approval:
        raise Conflict("Only an approved version can be filed.")
    name = KINDS[artifact["kind"]][2].format(title=artifact["title"])[:300]
    doc_id = docs.create_document(p["org_id"], artifact["matter_id"], name, doc_kind, "final", p["sub"])
    version_id = docs.new_id()
    sha = _hashlib.sha256(body).hexdigest()
    wrapped = blobs.store(artifact["matter_id"], version_id, body)
    try:
        docs.add_version(version_id, doc_id, artifact["matter_id"], filename, mime, len(body), sha, None, wrapped,
                         p["sub"], status="final")
    except BaseException:
        blobs.remove(artifact["matter_id"], version_id)
        raise
    with get_conn() as conn:
        approver, approved_at = conn.execute("SELECT by, at FROM legal_review_approvals WHERE artifact_id=? AND decision='approved' "
                                             "ORDER BY at DESC, rowid DESC LIMIT 1", (artifact["id"],)).fetchone()
        conn.execute("UPDATE legal_ai_artifacts SET document_id=? WHERE id=?", (doc_id, artifact["id"]))
        conn.execute("INSERT INTO legal_artifact_filings (id, artifact_id, matter_id, kind, artifact_version, document_id, version_id, "
                     "body_sha256, content_sha256, approved_by, approved_at, filed_by, filed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (str(uuid.uuid4()), artifact["id"], artifact["matter_id"], artifact["kind"], artifact["version"], doc_id, version_id,
                      sha, artifact["content_sha256"], approver, approved_at, p["sub"], now_iso()))
        conn.commit()
    worker.wake()
    return doc_id


def filing_for_document(conn, document_id: str) -> Optional[dict]:
    """The approved draft a document was filed from, with the approval it carries."""
    r = conn.execute("SELECT f.artifact_id, f.version_id, f.body_sha256, f.content_sha256, f.kind, f.artifact_version, "
                     "(SELECT u.username FROM users u WHERE u.id = f.approved_by), f.approved_at FROM legal_artifact_filings f "
                     "WHERE f.document_id=? ORDER BY f.filed_at DESC LIMIT 1", (document_id,)).fetchone()
    if not r:
        return None
    return {"artifact_id": r[0], "version_id": r[1], "body_sha256": r[2], "content_sha256": r[3], "kind": r[4], "version": r[5],
            "approved_by": r[6], "approved_at": r[7]}
