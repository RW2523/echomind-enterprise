"""The redaction plan's suggestions (S1.10; P-006 safe defaults).

Redacted by default: personal data (emails, phone, card, social-security and long account
numbers, IBANs), credentials and keys, and references to OTHER matters (they would reveal that
those matters exist). Flagged but kept by default: this matter's own number (it is in the
watermark), party names, and privilege markers (a warning that the document may be privileged).
A person can switch any item and add their own.
"""
from __future__ import annotations

import hashlib
import re

from ...core.db import get_conn
from ...core.export_gateway import _DETECTORS, _luhn_ok
from ...rag.chunking.sanitize import find_pii
from ..records import labels

_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,4})?\b")
_PII_REASON = {"email": "Email address", "phone": "Phone number", "long_number": "Account or reference number",
               "card": "Card number", "ssn": "Social security number", "id_number": "Identity document number"}
_SECRET_KINDS = {"api_key", "aws_secret", "private_key", "jwt"}


def item_id(kind: str, text: str) -> str:
    return hashlib.sha1(f"{kind}\x00{text}".encode()).hexdigest()[:12]


def _item(kind: str, text: str, redact: bool, reason: str, count: int = 1, source: str = "detector") -> dict:
    return {"id": item_id(kind, text), "kind": kind, "text": text, "count": count, "redact": redact, "reason": reason, "source": source}


def detect(text: str, matter_id: str, org_id: str, parties: list[str]) -> list[dict]:
    found: dict[tuple, dict] = {}

    def add(kind, value, redact, reason):
        value = value.strip()
        if len(value) < 3:
            return
        key = (kind, value)
        if key in found:
            found[key]["count"] += 1
        else:
            found[key] = _item(kind, value, redact, reason)

    for kind, a, b in find_pii(text):
        add(kind, text[a:b], True, _PII_REASON.get(kind, "Personal data"))
    for m in _IBAN.finditer(text):
        add("iban", m.group(), True, "Bank account (IBAN)")
    for typ, _sev, rx in _DETECTORS:
        if typ in _SECRET_KINDS:
            for m in rx.finditer(text):
                add("credential", m.group(), True, "Credential or key")
    for m in re.finditer(r"\b(?:\d[ -]?){13,19}\b", text):
        if _luhn_ok(m.group()):
            add("card", m.group(), True, "Card number")
    with get_conn() as conn:
        own = conn.execute("SELECT number FROM matters WHERE id=?", (matter_id,)).fetchone()[0]
        numbers = [r[0] for r in conn.execute("SELECT number FROM matters WHERE org_id=?", (org_id,))]
    for num in sorted(numbers, key=len, reverse=True):
        for _ in re.finditer(rf"(?<![\w-]){re.escape(num)}(?![\w-])", text):
            if num == own:
                add("matter_number", num, False, "This matter's number (it is in the watermark)")
            else:
                add("matter_number", num, True, "Another matter's reference")
    for name in parties:
        for _ in re.finditer(rf"\b{re.escape(name)}\b", text, re.I):
            add("party", name, False, "Party name (kept by default)")
    for label, code in labels.detect(text):
        add("privilege_marker", {"attorney_client": "attorney-client privilege", "privileged_confidential": "privileged and confidential",
                                 "legal_privilege": "legally privileged", "without_prejudice": "without prejudice",
                                 "work_product": "work product"}.get(code, code), False,
            "Privilege marker: this document may be privileged")
    order = {"credential": 0, "card": 1, "ssn": 2, "iban": 3, "long_number": 4, "email": 5, "phone": 6, "id_number": 7,
             "matter_number": 8, "party": 9, "privilege_marker": 10}
    # Overlapping finds: keep the longest (a phone pattern inside an IBAN is the IBAN).
    items = sorted(found.values(), key=lambda i: (order.get(i["kind"], 99), -len(i["text"])))
    kept: list[dict] = []
    for i in items:
        if i["kind"] in ("party", "privilege_marker", "matter_number") or not any(i["text"] in k["text"] for k in kept if k["redact"]):
            kept.append(i)
    return kept


def manual(text: str) -> dict:
    return _item("manual", text, True, "Added by hand", source="person")


def parties_for(matter_id: str, object_type: str, object_id: str) -> list[str]:
    with get_conn() as conn:
        names = [r[0] for r in conn.execute("SELECT c.name FROM matters m JOIN clients c ON c.id = m.client_id WHERE m.id=?", (matter_id,))]
        if object_type == "meeting_transcript":
            names += [r[0] for r in conn.execute("SELECT name FROM legal_meeting_participants WHERE meeting_id=? AND removed_at IS NULL", (object_id,))]
    return [n for n in dict.fromkeys(names) if n]
