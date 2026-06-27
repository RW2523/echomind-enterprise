"""Secure offline->online export gateway (#5).

Evaluate content for sensitive data BEFORE it leaves the box and produce a redacted version.
Fully offline + deterministic (regex + Luhn) — no GPU, no network — so it works air-gapped and is
safe to run on every export. The premium "data never leaves unchecked" differentiator.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# (type, severity, compiled regex)
_DETECTORS: List[Tuple[str, str, "re.Pattern"]] = [
    ("ssn", "high", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("api_key", "high", re.compile(r"\b(?:sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,})\b")),
    ("aws_secret", "high", re.compile(r"(?i)aws_secret_access_key\s*[=:]\s*\S+")),
    ("private_key", "high", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("jwt", "medium", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("email", "low", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone", "low", re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("ip", "low", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]

_CC = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_SEV_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


def _luhn_ok(s: str) -> bool:
    digits = [int(c) for c in s if c.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _mask(s: str) -> str:
    s = s.strip()
    if len(s) <= 4:
        return "*" * len(s)
    return s[:2] + "*" * (len(s) - 4) + s[-2:]


def evaluate_export(text: str) -> dict:
    """Return {risk_level, finding_count, findings[], redacted_text, safe_to_export}."""
    text = text or ""
    findings: List[Dict] = []
    spans: List[Tuple[int, int, str]] = []

    for typ, sev, rx in _DETECTORS:
        for m in rx.finditer(text):
            findings.append({"type": typ, "severity": sev, "preview": _mask(m.group()), "start": m.start(), "end": m.end()})
            spans.append((m.start(), m.end(), typ))

    for m in _CC.finditer(text):
        if _luhn_ok(m.group()):
            findings.append({"type": "credit_card", "severity": "high", "preview": _mask(m.group()), "start": m.start(), "end": m.end()})
            spans.append((m.start(), m.end(), "credit_card"))

    # Redact right-to-left so earlier offsets stay valid.
    redacted = text
    for start, end, typ in sorted(spans, key=lambda s: s[0], reverse=True):
        redacted = redacted[:start] + f"[REDACTED:{typ}]" + redacted[end:]

    risk = "none"
    for f in findings:
        if _SEV_RANK[f["severity"]] > _SEV_RANK[risk]:
            risk = f["severity"]

    return {
        "risk_level": risk,
        "finding_count": len(findings),
        "findings": [{"type": f["type"], "severity": f["severity"], "preview": f["preview"]} for f in findings],
        "redacted_text": redacted,
        "safe_to_export": risk in ("none", "low"),
    }
