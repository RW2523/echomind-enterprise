"""Consent rules (S1.8; policy P-001 — safe defaults until counsel decides).

All-party, explicit, per participant, before any capture: every current participant's latest
consent event must be a grant made after they joined and no older than LEGAL_CONSENT_MAX_AGE_HOURS.
A participant who leaves and rejoins is a new participant and must consent again. Withdrawal is
recorded by any member of the matter and stops capture at once (capture.py).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

from ...core.config import settings
from ...core.db import get_conn

METHODS = ("verbal_on_record", "written", "host_attestation")
# P-001: wording to be approved by counsel; {number} is the matter number.
STATEMENT = ("This meeting will be recorded and transcribed by EchoMind for the firm, for matter {number} only. "
             "You can withdraw your consent at any time and recording stops at once. Do you consent to being recorded?")
POLICY_SOURCE = "P-001 safe default (awaiting counsel)"


def policy() -> dict:
    return {"mode": "all_party", "methods": list(METHODS), "max_age_hours": settings.LEGAL_CONSENT_MAX_AGE_HOURS,
            "statement": STATEMENT, "source": POLICY_SOURCE}


def statement_for(number: str) -> str:
    return STATEMENT.format(number=number)


def _parse(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)


def participants(conn, meeting_id: str) -> list[dict]:
    """Current participants with their latest consent event."""
    rows = conn.execute(
        "SELECT p.id, p.name, p.role, p.added_at, e.id, e.kind, e.method, e.recorded_at, "
        "(SELECT username FROM users WHERE id = e.recorded_by) FROM legal_meeting_participants p "
        "LEFT JOIN legal_consent_events e ON e.id = (SELECT id FROM legal_consent_events x WHERE x.participant_id = p.id "
        "ORDER BY x.recorded_at DESC, x.rowid DESC LIMIT 1) WHERE p.meeting_id = ? AND p.removed_at IS NULL ORDER BY p.added_at",
        (meeting_id,)).fetchall()
    now = datetime.now(timezone.utc)
    max_age = timedelta(hours=settings.LEGAL_CONSENT_MAX_AGE_HOURS)
    out = []
    for pid, name, role, added_at, eid, kind, method, at, by in rows:
        if kind != "granted":
            state = "withdrawn" if kind == "withdrawn" else "missing"
        elif at < added_at:
            state = "missing"
        elif now - _parse(at) > max_age:
            state = "stale"
        else:
            state = "granted"
        out.append({"id": pid, "name": name, "role": role, "added_at": added_at,
                    "consent": {"state": state, "method": method, "at": at, "by": by, "event_id": eid}})
    return out


def check(meeting_id: str) -> tuple[bool, list[str], str]:
    """(everyone has consented, ids still needing consent, snapshot hash of the consent relied on)."""
    with get_conn() as conn:
        people = participants(conn, meeting_id)
    missing = [p["id"] for p in people if p["consent"]["state"] != "granted"]
    basis = sorted((p["id"], p["consent"]["event_id"], p["consent"]["at"]) for p in people)
    snapshot = hashlib.sha256(json.dumps(basis, separators=(",", ":")).encode()).hexdigest()
    return bool(people) and not missing, missing, snapshot
