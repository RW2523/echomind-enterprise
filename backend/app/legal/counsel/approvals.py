"""What waits for this person's decision in the conversation's matters, as a card they can decide from
(docs/echomind-legal/03 `approval_request`; owner's request 2026-11-02).

The items are the review queue's own (reviews/queue.py: the latest draft of each, on a matter where the person may
approve, not their own, whose sources they can all see, not sent back), limited to the conversation's matters. A
decision goes through the review route's own checks (reviews/routes.decide_review): the right to approve, not the
author, only the latest draft, the content hash of the version shown, no gap left in a legal draft, a partner for a
panel. Approving is never filing or sending anything; the card says so. The person also confirms they read it.

Only references (artifact, matter, the hash shown) are stored; the card is rebuilt from the records each time.
"""
from __future__ import annotations

from typing import Optional

from .. import artifacts, policy
from ..reviews import queue

KIND_LABEL = {"legal_draft": "Legal draft", "minutes": "Minutes", "research_note": "Research note", "redline": "Redline",
              "summary": "Summary", "panel": "Panel review"}
CONSEQUENCE = {
    "legal_draft": "Approving makes this version the approved draft. Nothing is filed or sent: filing it as a matter document is a separate step.",
    "minutes": "Approving makes these the approved minutes of the meeting. Nothing is sent.",
    "research_note": "Approving marks this research note as reviewed. Nothing is filed or sent.",
    "redline": "Approving marks this redline as reviewed. Filing it as the document's next version is a separate step.",
    "summary": "Approving marks this summary as reviewed. Nothing is filed or sent.",
    "panel": "Approving accepts the panel's summary as reviewed by a partner. It gives options to consider, never a decision.",
}
CAUTION = "Open each item and read it before deciding. A decision is recorded with your name and the exact version you approved."


def _preview(a: dict) -> list[str]:
    c = a.get("content") or {}
    lines: list[str] = []
    if a["kind"] == "legal_draft":
        lines = [b.get("text") or "" for b in c.get("blocks") or [] if b.get("type") == "paragraph"]
    else:
        for key in ("summary", "answer", "overview", "text"):
            if isinstance(c.get(key), str):
                lines.append(c[key])
        for key in ("blocks", "sections", "items", "decisions"):
            for x in c.get(key) or []:
                if isinstance(x, dict) and isinstance(x.get("text"), str):
                    lines.append(x["text"])
    return [" ".join(t.split())[:600] for t in lines if t and t.strip()][:3]


def _can(principal: dict, a: dict) -> bool:
    if a["created_by"] == principal["sub"] or not policy.authorize(principal, "artifacts.approve", a["matter_id"]).allowed:
        return False
    if a["kind"] == "panel":
        from ..panel.run import is_partner
        return not (a.get("content") or {}).get("stopped") and is_partner(principal, a["matter_id"])
    return True


def _item(principal: dict, ref: dict, matters: dict[str, dict]) -> dict:
    a = artifacts.get(ref["matter_id"], ref["artifact_id"])
    m = matters.get(ref["matter_id"]) or {"number": "?"}
    if not a or not artifacts.visible_to(a, principal["sub"]):
        return {"artifact_id": ref["artifact_id"], "kind": "Item", "title": "No longer available", "version": 1, "author": "", "created_at": "",
                "consequence": "", "path": "/drafts", "state": "gone", "can_decide": False}
    last = a["decisions"][-1] if a["decisions"] else None
    newest = artifacts.latest(a["matter_id"], a["subject_type"], a["subject_id"], a["kind"])
    if a["status"] == "approved":
        state = "approved"
    elif last and last["decision"] == "changes_requested":
        state = "changes_requested"
    elif a["content_sha256"] != ref["sha"] or (newest and newest["id"] != a["id"]) or a["status"] != "draft":
        state = "changed"
    else:
        state = "waiting"
    path = queue._KIND_PATH[a["kind"]].format(number=m["number"], subject=a["subject_id"], id=a["id"])
    return {"artifact_id": a["id"], "kind": KIND_LABEL.get(a["kind"], a["kind"]), "title": (a["title"] or KIND_LABEL.get(a["kind"], "Draft"))[:200],
            "version": a["version"], "author": a["created_by_name"] or "someone", "created_at": a["created_at"], "preview": _preview(a),
            "gaps": int((a.get("content") or {}).get("gaps") or 0) if a["kind"] == "legal_draft" else 0,
            "consequence": CONSEQUENCE.get(a["kind"], "Approving marks this as reviewed. Nothing is filed or sent."), "path": path,
            "state": state, "decided_by": last["by"] if last and state in ("approved", "changes_requested") else None,
            "can_decide": state == "waiting" and _can(principal, a)}


def build(principal: dict, matters: list[dict], stored: Optional[list[dict]] = None) -> tuple[dict, list[dict]]:
    """(approval_request props, the references to store). `stored`: rebuild those references as they stand now."""
    by_id = {m["id"]: m for m in matters}
    if stored is None:
        waiting: list = []
        queue._drafts(principal, {m["id"]: {"id": m["id"], "number": m["number"], "title": m["title"]} for m in matters}, waiting, [])
        refs = []
        for w in waiting:
            a = artifacts.get(w["matter"]["id"], w["id"])
            if a:
                refs.append({"artifact_id": a["id"], "matter_id": a["matter_id"], "sha": a["content_sha256"]})
        total = len(refs)
        refs = refs[:3]
    else:
        refs, total = stored, len(stored)
    items = [_item(principal, r, by_id) for r in refs]
    names = " and ".join(m["number"] for m in matters) if len(matters) <= 2 else f"these {len(matters)} matters"
    if stored is None:
        summary = (f"{total} item{'s' if total != 1 else ''} in {names} wait{'s' if total == 1 else ''} for your decision." if total
                   else f"Nothing in {names} waits for your decision.")
    else:
        left = sum(1 for i in items if i["state"] == "waiting")
        summary = f"{left} of {len(items)} still wait{'s' if left == 1 else ''} for your decision." if left else "Decided."
    return {"summary": summary[:400], "items": items, "more": max(0, total - len(refs)), "caution": CAUTION}, refs


def decide(principal: dict, ref: dict, decision: str, reason: Optional[str]) -> dict:
    """The decision through the review route's own checks, on the version the card showed (raises its HTTP errors)."""
    from ..reviews.routes import ReviewIn, decide_review
    return decide_review(principal, ref["matter_id"], ref["artifact_id"],
                         ReviewIn(decision=decision, content_sha256=ref["sha"], reason=reason))
