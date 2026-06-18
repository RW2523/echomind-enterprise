"""
Canonical section-id extraction and normalization for DoD-style regulatory books.

Handles: "010201 PURPOSE", "Section 0301", "paragraph 030201", "030201", "0301 vs 0402".
Ensures consistent matching between cross-refs, TOC paths, and chunk metadata.
"""
from __future__ import annotations

import re
from typing import Optional

# DoD-style 4–6 digit codes (e.g. 030201, 7001, 0301.2)
_DOD_CODE_RE = re.compile(r"\b(\d{4,6}(?:\.\d{1,4}){0,3})\b")

# "Section 0301", "paragraph 030201", "030201 PURPOSE"
_SECTION_LABEL_RE = re.compile(
    r"\b(?:Section|Sec\.?|Paragraph|Para\.?|Subparagraph|Subpara\.?)\s+([\d]{3,6}(?:\.\d{1,4}){0,3})\b",
    re.I,
)

# "Chapter 5", "Volume 15", "Volume 2B"
_CHAPTER_RE = re.compile(r"\b(?:Chapter|Chap\.?)\s+(\d+[A-Za-z]?)\b", re.I)
_VOLUME_RE = re.compile(r"\b(?:Volume|Vol\.?)\s+(\d+[A-Za-z]?)\b", re.I)


def canonicalize_section_id(sid: str) -> str:
    """Normalize section id: strip punctuation, normalize leading zeros, lowercase for consistency.

    Examples: "030201" -> "030201", "0301.2" -> "0301.2", " 030201 " -> "030201"
    """
    if not sid or not isinstance(sid, str):
        return ""
    s = sid.strip()
    # Remove trailing punctuation
    s = re.sub(r"[.,;:]+$", "", s)
    s = s.strip()
    return s


def extract_section_id(line: str) -> Optional[str]:
    """Extract a canonical section id from a line (heading, reference, etc.).

    Handles: "010201 PURPOSE", "Section 0301", "paragraph 030201", "030201", "0301.2 Purpose".
    Returns None if no match.
    """
    if not (line or "").strip():
        return None
    line = line.strip()

    # "Section 0301" / "paragraph 030201"
    m = _SECTION_LABEL_RE.search(line)
    if m:
        return canonicalize_section_id(m.group(1))

    # "010201 PURPOSE" / "030201" at start or as standalone
    m = _DOD_CODE_RE.search(line)
    if m:
        return canonicalize_section_id(m.group(1))

    return None


def section_id_from_path(section_path: str) -> Optional[str]:
    """Extract the most specific (leaf) section_id from a hierarchical section_path.

    Examples:
      "Volume 1 > Chapter 3 > 030201 PURPOSE" -> "030201"
      "0301 PURPOSE" -> "0301"
      "Volume 1 > Chapter 3 > Section 0301" -> "0301"
    """
    if not (section_path or "").strip():
        return None
    path = section_path.strip()
    segments = [s.strip() for s in path.split(">")]
    # Look for DoD-style code in last segment first (most specific)
    for seg in reversed(segments):
        if not seg:
            continue
        m = _DOD_CODE_RE.search(seg)
        if m:
            return canonicalize_section_id(m.group(1))
        m = _SECTION_LABEL_RE.search(seg)
        if m:
            return canonicalize_section_id(m.group(1))
    return None


def _looks_like_year_or_amount(raw: str) -> bool:
    """Heuristic: a bare 4-digit number in the calendar-year range (1900-2099) with no decimal is
    almost certainly a year (or amount), not a DoD section id — exclude it from code extraction so
    body text like 'fiscal year 2024' doesn't pollute section routing. (M6)"""
    if "." in raw:
        return False
    if len(raw) == 4 and raw.isdigit():
        return 1900 <= int(raw) <= 2099
    return False


def extract_all_codes(text: str) -> list[str]:
    """Extract all DoD-style codes from text (for comparison queries like "0301 vs 0402")."""
    if not (text or "").strip():
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _DOD_CODE_RE.finditer(text):
        raw = m.group(1)
        if _looks_like_year_or_amount(raw):
            continue
        c = canonicalize_section_id(raw)
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out
