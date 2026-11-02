"""The review panel from the conversation (docs/echomind-legal/03 `guided_questions` and `council_summary`; owner's
request 2026-11-02). Only when the firm has the panel on (LEGAL_PANEL) and the person may start one (`ai.draft`).

"Challenge our position on the delivery date" asks two questions (the question for the panel, and the specialist area),
then queues the panel exactly as its page does (panel/routes.start: four perspectives and a moderator, each citing the
matter's documents, run by the job worker). The card stores only the review's reference; it shows the run's state and,
when done, the issues with each perspective's position and where they disagree, from the panel's own stored result.
A partner approves the summary on the panel page; the panel gives options to consider, never a decision.
"""
from __future__ import annotations

from typing import Optional
from urllib.parse import quote as urlquote

from ...core.db import get_conn
from .. import artifacts

CAUTION = "The panel gives options to consider, never a decision. A partner approves its summary on the panel page."
_GUESS = {"employment": ("employ", "dismiss", "worker"), "ip": ("patent", "trademark", "copyright", "licen"), "data": ("data", "privacy", "gdpr"),
          "tax": ("tax", "vat"), "real_estate": ("lease", "landlord", "tenant", "property"), "competition": ("competition", "antitrust", "cartel"),
          "finance": ("loan", "facility", "lender", "bank"), "regulatory": ("regulator", "fca", "sec "), "corporate": ("shareholder", "board", "merger")}


def form(matter: dict, request: str, document_ids: list[str]) -> tuple[dict, dict]:
    from ..panel.run import SPECIALISMS
    low = request.lower()
    area = next((k for k, words in _GUESS.items() if any(w in low for w in words)), "commercial")
    using = (f"Only the {len(document_ids)} document{'s' if len(document_ids) != 1 else ''} you chose." if document_ids
             else f"All the documents of {matter['number']} you may read.")
    props = {"purpose": "panel", "title": "Ask the review panel",
             "intro": f"Four perspectives (deal, risk, litigation and a specialist) each answer from the documents, and a moderator "
                      f"sets out where they agree and differ. Sources: {using}"[:400],
             "questions": [
                 {"id": "question", "label": "The question for the panel", "kind": "text", "default": " ".join(request.split())[:1000],
                  "max_length": 1000},
                 {"id": "specialist", "label": "Specialist area", "kind": "choice", "default": area,
                  "options": [{"value": k, "label": v[:1].upper() + v[1:]} for k, v in SPECIALISMS.items()]}],
             "submit_label": "Ask the panel",
             "note": "It runs in the background, about a minute; you can keep asking here meanwhile."}
    return props, {"purpose": "panel", "document_ids": list(document_ids)}


def start(principal: dict, matter: dict, values: dict, stored: dict) -> str:
    """Queue the review (every check of the panel route) and return its id."""
    from ..panel.routes import AskIn
    from ..panel.routes import start as start_panel
    out = start_panel(principal, matter["id"], AskIn(question=values["question"], specialist=values["specialist"],
                                                      documents=(stored.get("document_ids") or [])[:20]))
    from ..audit.record import record
    record("panel.ask", "allow", principal=principal, object_type="matter", object_id=matter["id"], matter_id=matter["id"],
           details={"kind": values["specialist"], "documents": len(stored.get("document_ids") or []), "mode": "chat"})
    return out["review"]["id"]


def summary(principal: dict, matter_id: str, review_id: str) -> Optional[dict]:
    """The council_summary props as the person may see the review now, or None when it is gone or not theirs to see."""
    from ..panel.run import ROLES, SPECIALISMS
    with get_conn() as conn:
        r = conn.execute("SELECT question, specialist, status FROM legal_panel_reviews WHERE id=? AND matter_id=?", (review_id, matter_id)).fetchone()
        number = (conn.execute("SELECT number FROM matters WHERE id=?", (matter_id,)).fetchone() or ["?"])[0]
    if not r:
        return None
    question, specialist, status = r
    versions = artifacts.versions(matter_id, "panel_review", review_id, "panel")
    a = versions[0] if versions else None
    if a and not artifacts.visible_to(a, principal["sub"]):
        return None
    area = SPECIALISMS.get(specialist, specialist)
    labels = {role: label.format(area=area) for role, label, _ in ROLES}
    issues, missing, approval = [], [], None
    if a:
        c = a["content"]
        for row in (c.get("matrix") or [])[:6]:
            positions = [{"role": labels.get(k, k)[:60], "text": " ".join(str(v).split())[:600]}
                         for k, v in (row.get("positions") or {}).items() if v]
            issues.append({"issue": (row.get("issue") or "")[:300], "positions": positions[:5], "disagreement": bool(row.get("disagreement"))})
        missing = [" ".join(str(m.get("question") if isinstance(m, dict) else m).split())[:300] for m in (c.get("missing_facts") or [])][:6]
        approval = {"approved": "Approved by a partner", "draft": "Waiting for a partner's approval",
                    "superseded": "A newer version exists"}.get(a["status"])
    return {"review_id": review_id, "question": question[:2000], "area": area[:80],
            "status": status if status in ("queued", "running", "done", "needs_facts", "failed", "not_allowed") else "failed",
            "issues": issues, "missing_facts": [m for m in missing if m], "approval": approval,
            "path": f"/m/{urlquote(number, safe='')}/panel/{review_id}", "caution": CAUTION}

