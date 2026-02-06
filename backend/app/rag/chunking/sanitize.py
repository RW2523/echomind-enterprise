"""
PII detection and redaction. No model calls; regex-based for speed and auditability.
"""
from __future__ import annotations
import re
from typing import Tuple

from .models import SensitivityLevel


# Patterns: (compiled_regex, replacement_label)
_PII_PATTERNS: list[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b"), "[REDACTED_PHONE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"), "[REDACTED_CARD]"),
    (re.compile(r"\b[A-Z]{2}\d{6,8}\b"), "[REDACTED_ID]"),
    (re.compile(r"\b\d{10,}\b"), "[REDACTED_NUM]"),
]


def _pii_density(text: str) -> float:
    """Fraction of characters that are part of a PII match (approx)."""
    if not text:
        return 0.0
    total_matched = 0
    for pat, _ in _PII_PATTERNS:
        for m in pat.finditer(text):
            total_matched += len(m.group(0))
    return total_matched / len(text)


def sanitize_text(text: str) -> Tuple[str, bool, SensitivityLevel]:
    """
    Redact PII and return (clean_text, redacted_flag, sensitivity_level).
    sensitivity: HIGH if any redaction, else MEDIUM if PII pattern present but not matched, else LOW.
    """
    if not text:
        return "", False, SensitivityLevel.LOW

    redacted = False
    out = text
    for pat, repl in _PII_PATTERNS:
        new_out, n = pat.subn(repl, out)
        if n > 0:
            out = new_out
            redacted = True

    if redacted:
        level = SensitivityLevel.HIGH
    elif _pii_density(text) > 0.005:
        level = SensitivityLevel.MEDIUM
    else:
        level = SensitivityLevel.LOW

    return out, redacted, level
