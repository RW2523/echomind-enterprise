"""Suggested time entries from a person's own audited work (S4.3, OPT-03; decision D-056).

Rules, no model:
- **What counts:** only the person's own actions that are work on a matter, and only those that were allowed. Opening
  a matter, reading lists and refused requests are not work.
- **Sessions:** per matter and per day in the firm's timezone (LEGAL_TIMEZONE), events closer than 15 minutes run
  together; a session counts the whole minutes from its first event to its last, plus 6 minutes for the last piece of work.
- **Totals:** rounded up to 6-minute units (0.1 hour).
- **Narrative:** the kinds of work with their counts, in the order first done. It never holds content: no titles,
  quotes or questions.
- **Matters:** only those the person can still open.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from ...core.config import settings
from ...core.db import get_conn

GAP_MINUTES = 15
TAIL_MINUTES = 6
UNIT_MINUTES = 6
MAX_DAYS = 31

# kind of work → the audited actions that are that work (labels as they read in a narrative)
KINDS: "OrderedDict[str, frozenset]" = OrderedDict([
    ("document review", frozenset({"document.read", "document.read_text", "documents.search", "label.set"})),
    ("legal research", frozenset({"research.query", "research.note.send"})),
    ("drafting", frozenset({"draft.create", "draft.edit", "draft.preview"})),
    ("redlining", frozenset({"redline.create", "redline.edit", "redline.word"})),
    ("playbook review", frozenset({"playbook.check", "playbook.decision"})),
    ("citation checking", frozenset({"check.citations", "check.citations.note"})),
    ("definitions checking", frozenset({"check.definitions", "check.definitions.note"})),
    ("comparing versions", frozenset({"compare.run", "compare.read", "compare.explain", "compare.scan"})),
    ("obligations review", frozenset({"obligation.extract", "obligation.update", "obligation.create"})),
    ("due diligence", frozenset({"diligence.run.start", "diligence.issue.update"})),
    ("meetings and minutes", frozenset({"meeting.create", "meeting.end", "meeting.assist", "meeting.agenda.add", "minutes.draft", "minutes.edit",
                                        "minutes.file", "transcript.revise", "transcript.finalise", "consent.grant"})),
    ("review and approval", frozenset({"artifact.review", "artifact.file", "minutes.review", "export.approve"})),
    ("summaries", frozenset({"summary.create", "summary.edit"})),
    ("document management", frozenset({"document.import", "document.update", "document.file", "matter.party.add"})),
    ("exports", frozenset({"export.request", "export.plan", "export.submit"})),
])
_KIND_OF = {a: k for k, actions in KINDS.items() for a in actions}


class Invalid(ValueError):
    pass


def _zone() -> ZoneInfo:
    return ZoneInfo(settings.LEGAL_TIMEZONE or "UTC")


def _utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def bounds(frm: str, to: str) -> tuple[date, date, str, str]:
    """The days asked for, and the UTC instants that bound them in the firm's timezone."""
    try:
        a, b = date.fromisoformat(frm), date.fromisoformat(to)
    except ValueError:
        raise Invalid("Use dates in the form 2026-10-05.") from None
    if b < a:
        raise Invalid("The end date is before the start date.")
    if (b - a).days + 1 > MAX_DAYS:
        raise Invalid(f"Ask for at most {MAX_DAYS} days at a time.")
    z = _zone()
    start = datetime(a.year, a.month, a.day, tzinfo=z)
    end = datetime(b.year, b.month, b.day, tzinfo=z) + timedelta(days=1)
    return a, b, _iso(start), _iso(end)


def narrative(counts: "OrderedDict[str, int]") -> str:
    text = "; ".join(f"{k} ({n})" for k, n in counts.items())
    return text[:1].upper() + text[1:]


def rounded(minutes: float) -> int:
    return max(UNIT_MINUTES, int(math.ceil(minutes / UNIT_MINUTES - 1e-9)) * UNIT_MINUTES)


def sessions(times: list[datetime]) -> list[tuple[datetime, datetime]]:
    out: list[list[datetime]] = []
    for t in sorted(times):
        if out and (t - out[-1][1]).total_seconds() <= GAP_MINUTES * 60:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [(a, b) for a, b in out]


def work(user_id: str, start: str, end: str, can_open) -> dict[tuple[str, str], dict]:
    """(matter id, local day) → its sessions, kinds of work and minutes, from the person's own allowed actions."""
    with get_conn() as conn:
        rows = conn.execute("SELECT ts, action, matter_id FROM audit_events WHERE actor_user_id=? AND outcome='allow' AND matter_id IS NOT NULL "
                            "AND ts >= ? AND ts < ? ORDER BY ts, seq", (user_id, start, end)).fetchall()
    z = _zone()
    grouped: dict[tuple[str, str], dict] = {}
    allowed: dict[str, bool] = {}
    for ts, action, matter_id in rows:
        kind = _KIND_OF.get(action)
        if not kind:
            continue
        if matter_id not in allowed:
            allowed[matter_id] = bool(can_open(matter_id))
        if not allowed[matter_id]:
            continue
        at = _utc(ts)
        day = at.astimezone(z).date().isoformat()
        g = grouped.setdefault((matter_id, day), {"times": [], "counts": OrderedDict()})
        g["times"].append(at)
        g["counts"][kind] = g["counts"].get(kind, 0) + 1
    out = {}
    for key, g in grouped.items():
        ss = sessions(g["times"])
        total = sum((b - a).total_seconds() // 60 + TAIL_MINUTES for a, b in ss)       # whole minutes: seconds never round a unit up
        out[key] = {"minutes": rounded(total), "narrative": narrative(g["counts"]), "counts": dict(g["counts"]),
                    "sessions": [{"from": a.astimezone(z).strftime("%H:%M"), "to": b.astimezone(z).strftime("%H:%M")} for a, b in ss]}
    return out


def hours(minutes: Optional[int]) -> str:
    return f"{(minutes or 0) / 60:.1f}"
