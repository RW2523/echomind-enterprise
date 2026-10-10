"""AI artifacts and their review (S2.6 foundation, extended in S2.7; decision D-042).

An artifact is something the model drafted from matter content, such as a meeting's minutes:
- **Versioned and fixed.** Content is never edited in place; a person's change is a new version.
  Each version's content is hashed (SHA-256 of its canonical JSON).
- **Approved by a second person.** An approval needs `artifacts.approve` on the matter (a lead,
  attorney or reviewer) and must come from someone other than the author of that version. It binds
  the version's content hash: anything exported or filed from it must carry the same hash.
- **Decisions are append-only.** A new version supersedes the earlier drafts. An approved version
  stays approved, as the record of what was approved, until a later version is approved in its turn.
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
            conn.execute("INSERT INTO legal_review_approvals (id, artifact_id, matter_id, decision, content_sha256, reason, by, at) "
                         "VALUES (?,?,?,?,?,?,?,?)", (str(uuid.uuid4()), artifact_id, matter_id, decision, sha, reason, by, now_iso()))
            if decision == "approved":
                conn.execute("UPDATE legal_ai_artifacts SET status='approved' WHERE id=?", (artifact_id,))
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
