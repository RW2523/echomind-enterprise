"""Privilege labels (S1.9, SEC-06; policy P-004 safe default: advisory labels).

Labels advise; they do not change who can see a document (membership and walls do that). A lead,
attorney or reviewer sets the label; a paralegal's choice is recorded as a suggestion. When a
document is indexed, a small detector suggests "privileged" or "work product" from explicit
markers in its text. Every change is kept, and audited.
"""
from __future__ import annotations

import re
import uuid
from typing import Optional

from ...core.db import get_conn
from ..policy import now_iso

LABELS = ("privileged", "work_product", "confidential", "not_privileged", "unreviewed")
OBJECT_TYPES = ("document", "meeting")

# (pattern, suggested label, reason code). Explicit markers only: "confidential" alone is in most
# contracts' confidentiality clauses and would suggest a label on almost everything.
_MARKERS = [
    (re.compile(r"\battorney[- ]client privileged?\b|\battorney[- ]client communication\b", re.I), "privileged", "attorney_client"),
    (re.compile(r"\bprivileged\s*(?:&|and)\s*confidential\b", re.I), "privileged", "privileged_confidential"),
    (re.compile(r"\blegally privileged\b|\bsubject to legal (?:advice )?privilege\b", re.I), "privileged", "legal_privilege"),
    (re.compile(r"\bwithout prejudice\b", re.I), "privileged", "without_prejudice"),
    (re.compile(r"\b(?:attorney )?work product\b|\bprepared in anticipation of litigation\b", re.I), "work_product", "work_product"),
]


def detect(text: str) -> list[tuple[str, str]]:
    found: dict[str, str] = {}
    for pat, label, code in _MARKERS:
        if label not in found and pat.search(text or ""):
            found[label] = code
    return list(found.items())


def add(org_id: str, matter_id: str, object_type: str, object_id: str, label: str, kind: str, source: str,
        by: Optional[str], reason_code: Optional[str] = None) -> None:
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_labels (id, org_id, matter_id, object_type, object_id, label, kind, source, reason_code, "
                     "set_by, set_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                     (str(uuid.uuid4()), org_id, matter_id, object_type, object_id, label, kind, source, reason_code, by, now_iso()))
        conn.commit()


def suggest_from_text(org_id: str, matter_id: str, document_id: str, text: str) -> int:
    """Detector suggestions for a newly indexed document, unless a person already labelled it."""
    st = state("document", document_id)
    if st["current"] is not None:
        return 0
    open_labels = {s["label"] for s in st["suggestions"]}
    n = 0
    for label, code in detect(text):
        if label not in open_labels:
            add(org_id, matter_id, "document", document_id, label, "suggested", "detector", None, code)
            n += 1
    return n


def state(object_type: str, object_id: str) -> dict:
    with get_conn() as conn:
        rows = conn.execute("SELECT l.label, l.kind, l.source, l.reason_code, (SELECT username FROM users WHERE id=l.set_by), l.set_at "
                            "FROM legal_labels l WHERE l.object_type=? AND l.object_id=? ORDER BY l.set_at, l.rowid",
                            (object_type, object_id)).fetchall()
    history = [{"label": r[0], "kind": r[1], "source": r[2], "reason_code": r[3], "by": r[4], "at": r[5]} for r in rows]
    current, suggestions = None, []
    for h in history:
        if h["kind"] == "set":
            current, suggestions = h, []
        else:
            suggestions = [s for s in suggestions if s["label"] != h["label"]] + [h]
    suggestions = [s for s in suggestions if not current or s["label"] != current["label"]]
    return {"current": current, "suggestions": list(reversed(suggestions)), "history": list(reversed(history))}


def current_labels(object_type: str, ids: list[str]) -> dict[str, dict]:
    """{object_id: {"label": current or None, "suggested": bool}} for lists."""
    return {i: {"label": (st := state(object_type, i))["current"]["label"] if st["current"] else None,
                "suggested": bool(st["suggestions"])} for i in ids}


def purge(conn, object_type: str, object_id: str) -> int:
    return conn.execute("DELETE FROM legal_labels WHERE object_type=? AND object_id=?", (object_type, object_id)).rowcount
