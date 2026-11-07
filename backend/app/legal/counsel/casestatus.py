"""Where a matter stands, for the chat (case management, 2026-11-07): its stage in its track, the latest updates its
people posted, and the next open tasks of its plan. Built from the records by rule, never by the model: the words shown
are the ones a person wrote, shortened, with who wrote them and when."""
from __future__ import annotations

from datetime import date, datetime, timezone
from urllib.parse import quote as urlquote

from .. import policy
from ..matters import casework, pipeline

KIND_LABEL = {"note": "Note", "call": "Call", "meeting": "Meeting", "filing": "Filing", "hearing": "Hearing", "order": "Court order",
              "instruction": "Client instruction", "milestone": "Milestone", "stage": "Stage"}
_TEXT = 600


def build(principal: dict, matter: dict) -> dict:
    s = casework.status_summary(matter["id"])
    m = s["matter"]
    track = pipeline.TRACKS[m["track"]]
    keys = [st.key for st in track.stages]
    index = keys.index(m["stage"]) if m["stage"] in keys else 0
    since = (m.get("stage_since") or m.get("created_at") or "")[:10]
    try:
        days = max(0, (datetime.now(timezone.utc).date() - date.fromisoformat(since)).days)
    except ValueError:
        days = None
    updates = [{"kind": u["kind"], "kind_label": KIND_LABEL[u["kind"]], "occurred_on": u["occurred_on"], "author": (u["author"] or "")[:80],
                "text": u["body"][:_TEXT - 1] + "…" if len(u["body"]) > _TEXT else u["body"],
                "document_title": (u["document_title"] or None) and u["document_title"][:300]} for u in s["updates"]]
    tasks = [{"title": t["title"][:300], "kind": t["kind"], "due_on": t["due_on"], "overdue": t["overdue"],
              "assignee": (t["assignee"] or None) and t["assignee"][:80]} for t in s["tasks"]]
    stage_label = track.stages[index].label
    parts = [f"{m['number']} is at {stage_label} ({track.label.lower()})"]
    if days is not None:
        parts[0] += f", {days} day{'s' if days != 1 else ''} in this stage"
    if updates:
        last = updates[0]
        parts.append(f"the latest update is a {last['kind_label'].lower()} on {last['occurred_on']}")
    else:
        parts.append("no updates have been posted yet")
    parts.append(f"{s['open_tasks']} open task{'s' if s['open_tasks'] != 1 else ''}" + (f", {s['overdue']} overdue" if s["overdue"] else ""))
    summary = "; ".join(parts) + "."
    return {"summary": summary[:400], "number": m["number"], "title": m["title"][:200], "track_label": track.label,
            "stages": [st.label for st in track.stages], "stage_index": index, "days_in_stage": days, "updates": updates, "tasks": tasks,
            "open_tasks": s["open_tasks"], "overdue": s["overdue"],
            "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "can_post": policy.authorize(principal, "updates.post", m["id"]).allowed,
            "path": f"/m/{urlquote(m['number'], safe='')}/overview"}


def has_plan(matter_id: str) -> bool:
    return any(t["status"] == "open" for t in casework.list_tasks(matter_id))
