"""Retention (S1.9, SEC-07; policy P-002 safe default: no automatic deletion).

A records administrator drafts a policy ("documents, N days after the matter closed"); a managing
partner approves it. A run lists what approved policies select (dry run) and what drafts would.
Held matters are skipped and counted. With LEGAL_RETENTION_APPLY on, an "apply" run turns the
selection into deletion requests — each still needs a person's approval; nothing is purged here.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from ...core.config import settings
from ...core.db import get_conn
from ..policy import now_iso
from . import deletion, holds

_P_COLS = ("id", "org_id", "object_type", "days_after_close", "note", "status", "drafted_by", "drafted_at", "approved_by", "approved_at")


def _iso_days_ago(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def policies(org_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(f"SELECT {', '.join(_P_COLS)} FROM legal_retention_policies WHERE org_id=? ORDER BY drafted_at", (org_id,)).fetchall()
        out = []
        for r in rows:
            p = dict(zip(_P_COLS, r))
            for k in ("drafted_by", "approved_by"):
                u = conn.execute("SELECT username FROM users WHERE id=?", (p[k],)).fetchone() if p[k] else None
                p[k] = u[0] if u else None
            del p["org_id"]
            out.append(p)
    return out


def policy_row(org_id: str, policy_id: str) -> Optional[dict]:
    with get_conn() as conn:
        r = conn.execute(f"SELECT {', '.join(_P_COLS)} FROM legal_retention_policies WHERE id=? AND org_id=?", (policy_id, org_id)).fetchone()
    return dict(zip(_P_COLS, r)) if r else None


def draft(org_id: str, object_type: str, days: int, note: Optional[str], by: str) -> str:
    pid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_retention_policies (id, org_id, object_type, days_after_close, note, status, drafted_by, drafted_at) "
                     "VALUES (?,?,?,?,?,?,?,?)", (pid, org_id, object_type, days, note, "draft", by, now_iso()))
        conn.commit()
    return pid


def set_status(policy_id: str, status: str, by: Optional[str] = None) -> None:
    with get_conn() as conn:
        if status == "approved":
            conn.execute("UPDATE legal_retention_policies SET status=?, approved_by=?, approved_at=? WHERE id=?", (status, by, now_iso(), policy_id))
        else:
            conn.execute("UPDATE legal_retention_policies SET status=? WHERE id=?", (status, policy_id))
        conn.commit()


def _candidates(conn, org_id: str, p: dict) -> list[tuple[str, str]]:
    """(matter_id, object_id) selected by one policy, before the hold check."""
    cutoff = _iso_days_ago(p["days_after_close"])
    closed = ("m.org_id = ? AND m.status IN ('closed','archived') AND COALESCE(m.closed_at, m.updated_at, m.created_at) <= ?")
    if p["object_type"] == "document":
        q = (f"SELECT d.matter_id, d.id FROM legal_documents d JOIN matters m ON m.id = d.matter_id WHERE {closed} AND d.deleted_at IS NULL "
             "AND NOT EXISTS (SELECT 1 FROM legal_deletions x WHERE x.object_type='document' AND x.object_id=d.id AND x.status IN ('requested','blocked'))")
    else:
        q = (f"SELECT mt.matter_id, mt.id FROM legal_meetings mt JOIN matters m ON m.id = mt.matter_id WHERE {closed} "
             "AND mt.transcript_deleted_at IS NULL AND EXISTS (SELECT 1 FROM legal_transcript_segments s WHERE s.meeting_id = mt.id) "
             "AND NOT EXISTS (SELECT 1 FROM legal_deletions x WHERE x.object_type='meeting_transcript' AND x.object_id=mt.id "
             "AND x.status IN ('requested','blocked'))")
    return [(r[0], r[1]) for r in conn.execute(q, (org_id, cutoff)).fetchall()]


def run(org_id: str, mode: str, by: Optional[str]) -> dict:
    """mode "dry_run" or "apply" (callers check LEGAL_RETENTION_APPLY and an approved policy)."""
    report = {"candidates": 0, "held_skipped": 0, "draft_preview": 0, "requests_created": 0, "by_policy": []}
    created: list[tuple[str, str, str, str]] = []
    with get_conn() as conn:
        held = {r[0] for r in conn.execute("SELECT DISTINCT s.object_id FROM legal_hold_scopes s JOIN legal_holds h ON h.id = s.hold_id "
                                           "WHERE h.status IN ('active','release_requested') AND s.object_type='matter'")}
        for r in conn.execute(f"SELECT {', '.join(_P_COLS)} FROM legal_retention_policies WHERE org_id=? AND status IN ('draft','approved')",
                              (org_id,)).fetchall():
            p = dict(zip(_P_COLS, r))
            found = _candidates(conn, org_id, p)
            free = [(m, o) for m, o in found if m not in held]
            entry = {"policy_id": p["id"], "status": p["status"], "object_type": p["object_type"], "selected": len(free),
                     "held_skipped": len(found) - len(free)}
            report["by_policy"].append(entry)
            if p["status"] != "approved":
                report["draft_preview"] += len(free)
                continue
            report["candidates"] += len(free)
            report["held_skipped"] += len(found) - len(free)
            created += [(m, p["object_type"], o, p["id"]) for m, o in free]
    if mode == "apply":
        for matter_id, object_type, object_id, pid in created:
            try:
                deletion.request(org_id, matter_id, object_type, object_id, f"Retention policy {pid[:8]}", by or "system", origin="retention")
                report["requests_created"] += 1
            except (holds.Held, deletion.Conflict, deletion.NotFound):
                pass                                          # held or changed since it was selected
    rid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_retention_runs (id, org_id, mode, run_by, run_at, report_json) VALUES (?,?,?,?,?,?)",
                     (rid, org_id, mode, by, now_iso(), json.dumps(report)))
        conn.commit()
    return {"id": rid, "mode": mode, "run_at": now_iso(), "report": report}


def runs(org_id: str, limit: int = 10) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT id, mode, (SELECT username FROM users WHERE id=r.run_by), run_at, report_json FROM legal_retention_runs r "
                            "WHERE org_id=? ORDER BY run_at DESC LIMIT ?", (org_id, limit)).fetchall()
    return [{"id": r[0], "mode": r[1], "run_by": r[2] or "scheduled", "run_at": r[3], "report": json.loads(r[4])} for r in rows]


def daily_dry_run(org_id: str) -> Optional[dict]:
    """Called by the job worker: one scheduled dry run per day, so the records team sees what would
    become due without anything happening to it."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with get_conn() as conn:
        done = conn.execute("SELECT 1 FROM legal_retention_runs WHERE org_id=? AND run_by IS NULL AND substr(run_at,1,10)=?",
                            (org_id, today)).fetchone()
        any_policy = conn.execute("SELECT 1 FROM legal_retention_policies WHERE org_id=? AND status IN ('draft','approved')", (org_id,)).fetchone()
    if done or not any_policy:
        return None
    return run(org_id, "dry_run", None)


def apply_allowed(org_id: str) -> Optional[str]:
    """None when an apply run may go ahead, else the reason it may not."""
    if not settings.LEGAL_RETENTION_APPLY:
        return "Applying retention is switched off for this firm (policy P-002 awaits counsel). Dry runs are available."
    with get_conn() as conn:
        if not conn.execute("SELECT 1 FROM legal_retention_policies WHERE org_id=? AND status='approved'", (org_id,)).fetchone():
            return "No retention policy has been approved."
    return None
