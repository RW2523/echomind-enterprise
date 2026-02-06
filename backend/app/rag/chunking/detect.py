"""
Lightweight document-type detection using heuristics (no model calls).
"""
from __future__ import annotations
import re

from .models import DocType
from .sanitize import _pii_density


def detect_document_type(text: str) -> DocType:
    """
    Detect document type from content for adaptive chunking.
    Order: sensitive (PII-heavy) -> faq -> book (long-form) -> user (default).
    """
    if not (text or "").strip():
        return DocType.USER

    t = text.strip()
    length = len(t)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]

    if _pii_density(t) >= 0.015:
        return DocType.SENSITIVE

    if _looks_like_faq(t, lines):
        return DocType.FAQ

    if _looks_like_long_form(t, lines, length):
        return DocType.BOOK

    return DocType.USER


def _looks_like_faq(text: str, lines: list[str]) -> bool:
    """FAQ: many Q/A pairs (question lines followed by short answers)."""
    if len(lines) < 4:
        return False
    q_pattern = re.compile(
        r"^(?:\d+[.)]\s*)?(?:Q(?:uestion)?\s*[:.]?\s*)?(.+\?)",
        re.IGNORECASE,
    )
    a_pattern = re.compile(
        r"^(?:\d+[.)]\s*)?(?:A(?:nswer)?\s*[:.]?\s*)?(.+)",
        re.IGNORECASE,
    )
    q_count = sum(1 for ln in lines if q_pattern.match(ln))
    if q_count < 2:
        return False
    if q_count >= max(3, len(lines) // 10):
        return True
    if "FAQ" in text[:2000] or "frequently asked" in text[:2000].lower():
        return True
    return False


def _looks_like_long_form(text: str, lines: list[str], length: int) -> bool:
    """Long-form: substantial length, paragraphs, optional headings/chapters."""
    if length < 8000:
        return False
    paragraph_breaks = text.count("\n\n")
    if paragraph_breaks < 5:
        return False
    heading_like = sum(
        1
        for ln in lines
        if len(ln) < 120
        and re.match(r"^(?:Chapter|Part|Section|\d+[.)])\s+.+", ln, re.IGNORECASE)
    )
    if heading_like >= 2 or (paragraph_breaks >= 20 and length > 20000):
        return True
    avg_line = sum(len(ln) for ln in lines) / max(1, len(lines))
    if avg_line > 80 and length > 15000:
        return True
    return False
