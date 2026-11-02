"""The matter's open obligations as a checklist (docs/echomind-legal/03 `checklist`; owner's request 2026-11-02).

Built from the obligations register, never from the model: each item is a register row the person may see (its document
visible to them), with the document's own words one click away. Ticking an item done, reopening it, or confirming a
deadline goes through the register's own change (obligations/routes.change_obligation): the same roles, the same
version check, the same history. A date read from a document counts only once a lawyer confirms it; nothing here sets one.

Only the obligation ids are stored with the component; it is rebuilt from the register each time it is shown, so a
ticked item stays ticked and a change made elsewhere shows.
"""
from __future__ import annotations

from typing import Optional
from urllib.parse import quote as urlquote

from .. import policy
from ..obligations import routes as register
from . import citations

MAX_ITEMS = 10
CAUTION = ("From the matter's obligations register. A deadline read from a document counts only once a lawyer confirms it; "
           "check the document's words before relying on a date.")


def _order(o: dict) -> tuple:
    return (not o["overdue"], o["due_date"] is None, o["due_date"] or "", o["clause"] or "", o["action"] or "")


def _item(o: dict, number: str) -> dict:
    ev = o.get("evidence") or {}
    return {"id": o["id"], "title": (o["action"] or "Obligation")[:300], "party": (o.get("party") or None) and o["party"][:80],
            "when": (o.get("when") or None) and o["when"][:300], "due_date": o.get("due_date"), "status": o["status"],
            "overdue": bool(o.get("overdue")), "owner": o.get("owner"), "document_title": (o.get("document_title") or "A document")[:300],
            "clause": o.get("clause"), "version": o["version"],
            "path": citations.reader_path(number, o["document_id"], o["version_id"], ev.get("start"), ev.get("end"))}


def build(principal: dict, matter: dict, only: Optional[list[str]] = None) -> tuple[dict, list[str], list[str]]:
    """(checklist props, document ids it names, obligation ids shown). `only`: rebuild exactly those (a stored checklist),
    whatever their state now; otherwise the open ones (proposed or confirmed)."""
    rows = register._visible_rows(principal, matter["id"])
    if only is not None:
        keep = set(only)
        rows = [o for o in rows if o["id"] in keep]
        rows.sort(key=lambda o: only.index(o["id"]))
    else:
        rows = sorted((o for o in rows if o["status"] in ("proposed", "confirmed")), key=_order)
    shown = rows[:MAX_ITEMS]
    open_ = [o for o in rows if o["status"] in ("proposed", "confirmed")]
    overdue = sum(1 for o in open_ if o["overdue"])
    unconfirmed = sum(1 for o in open_ if o["status"] == "proposed" and o["due_date"])
    if only is None:
        if not rows:
            summary = f"No open obligations are recorded for {matter['number']}."
        else:
            summary = (f"{len(rows)} open obligation{'s' if len(rows) != 1 else ''} in {matter['number']}"
                       + (f", {overdue} overdue" if overdue else "")
                       + (f", {unconfirmed} with a date not yet confirmed" if unconfirmed else "") + ".")
    else:
        done = sum(1 for o in rows if o["status"] == "done")
        summary = f"{matter['number']}: {len(open_)} open, {done} done" + (f", {overdue} overdue" if overdue else "") + "."
    props = {"summary": summary[:400], "items": [_item(o, matter["number"]) for o in shown], "more": max(0, len(rows) - len(shown)),
             "can_done": policy.authorize(principal, "ai.draft", matter["id"]).allowed,
             "can_confirm": policy.authorize(principal, "artifacts.approve", matter["id"]).allowed,
             "caution": CAUTION, "register_path": f"/m/{urlquote(matter['number'], safe='')}/obligations"}
    return props, sorted({o["document_id"] for o in shown}), [o["id"] for o in shown]


def change(principal: dict, matter_id: str, obligation_id: str, action: str, expected_version: int) -> dict:
    """Tick done, reopen or confirm one obligation through the register (raises its HTTP errors as they are)."""
    return register.change_obligation(principal, matter_id, obligation_id,
                                      register.PatchIn(expected_version=expected_version, action=action))
