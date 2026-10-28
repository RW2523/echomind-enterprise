"""Several matters side by side, from their records (the `case_table` component; owner's request 2026-10-28).

One row per matter of the conversation, each authorized again by the caller: client, status, sensitivity, how many
documents the person may read, how many potential inconsistencies are still open, the next dated deadline from the
obligations register with its review state (a deadline is confirmed only by a lawyer, never here), and the last time a
readable document changed. Nothing is generated or inferred; a row opens that matter's own overview.
"""
from __future__ import annotations

from urllib.parse import quote as urlquote

from ...core.db import get_conn
from ..documents import repo as doc_repo
from . import discrepancies

_STATE = {"proposed": "Not confirmed", "confirmed": "Confirmed by a lawyer"}


def row(principal: dict, matter_id: str) -> dict:
    uid = principal["sub"]
    with get_conn() as conn:
        m = conn.execute("SELECT m.id, m.number, m.title, c.name, m.status, m.sensitivity FROM matters m "
                         "JOIN clients c ON c.id = m.client_id WHERE m.id=?", (matter_id,)).fetchone()
        docs = conn.execute(f"SELECT COUNT(*), MAX(COALESCE(d.updated_at, d.created_at)) FROM legal_documents d "
                            f"WHERE d.matter_id = ? AND {doc_repo.VISIBLE_TO}", (matter_id, uid)).fetchone()
        due = conn.execute(
            "SELECT o.due_date, o.action_text, o.status FROM legal_obligations o JOIN legal_documents d ON d.id = o.document_id "
            f"WHERE o.matter_id = ? AND o.status IN ('proposed','confirmed') AND o.due_date IS NOT NULL AND {doc_repo.VISIBLE_TO} "
            "ORDER BY o.due_date LIMIT 1", (matter_id, uid)).fetchone()
    try:
        open_items = len(discrepancies.listed(principal, matter_id, discrepancies.OPEN))
    except Exception:                                      # the register is optional for a row; the rest still stands
        open_items = None
    out = {"matter_id": m[0], "number": m[1], "title": m[2][:200], "client": m[3][:200], "status": m[4], "sensitivity": m[5],
           "documents": int(docs[0] or 0), "open_inconsistencies": open_items,
           "path": f"/m/{urlquote(m[1], safe='')}/overview"}
    if docs[1]:
        out["last_activity"] = str(docs[1])[:19]
    if due:
        out.update({"next_deadline": str(due[0])[:10], "next_deadline_text": (due[1] or "")[:200], "deadline_state": _STATE[due[2]]})
    return out


def build(principal: dict, matter_ids: list[str]) -> dict:
    rows = [row(principal, mid) for mid in matter_ids]
    n = len(rows)
    clients = {r["client"] for r in rows}
    summary = (f"{n} matter{'s' if n != 1 else ''} side by side" + (f", {len(clients)} clients" if len(clients) > 1 else "") + ".")
    return {"summary": summary, "rows": rows,
            "caution": "Counts and dates come from each matter's records as they stand now, only what you may read. A deadline "
                       "counts only once a lawyer confirms it."}
