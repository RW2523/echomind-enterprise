"""
Heading normalization for DoD-style regulatory documents.

Supports variants: 030201 PURPOSE, 030201. Purpose, Section 0705, Chapter 5, Volume 2A.
Extracts volume/chapter/section from headings and parent context.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

from .section_id import canonicalize_section_id, extract_section_id

# Malformed patterns to reject
SEGMENT_RE = re.compile(r"^Segment\s+\d+\s*$", re.I)
MALFORMED_RE = re.compile(
    r"^(?:2BDoD|DoD\s*\d+|Volume\s+X\b|Chapter\s+\d{3,})\s*$",
    re.I,
)
VOLUME_RE = re.compile(r"\b(?:Volume|Vol\.?)\s+(\d+[A-Za-z]?)\b", re.I)
CHAPTER_RE = re.compile(r"\b(?:Chapter|Chap\.?)\s+(\d+[A-Za-z]?)\b", re.I)
SECTION_LABEL_RE = re.compile(
    r"\b(?:Section|Sec\.?|Paragraph|Para\.?)\s+([\d]{3,6}(?:\.\d{1,4}){0,3})\b",
    re.I,
)
DOD_CODE_RE = re.compile(r"\b(\d{4,6}(?:\.\d{1,3})?)\b")


def normalize_heading_to_metadata(
    text: str,
    parent_section_path: Optional[str] = None,
    parent_volume_id: Optional[str] = None,
    parent_chapter_id: Optional[str] = None,
) -> Dict:
    """Normalize heading text to structured metadata.

    Strips punctuation noise, canonicalizes section ids, extracts volume/chapter
    from text or parent. Returns dict with volume_id, chapter_id, section_id,
    section_title, section_path (partial). Prevents malformed labels.
    """
    if not (text or "").strip():
        return {}
    t = str(text).strip()
    # Strip trailing punctuation noise
    t = re.sub(r"[.,;:\-\s]+$", "", t).strip()
    if not t:
        return {}

    # Reject known malformed patterns
    if SEGMENT_RE.match(t) or MALFORMED_RE.match(t):
        return {}

    out: Dict = {}

    # Volume
    vol_m = VOLUME_RE.search(t)
    if vol_m:
        out["volume_id"] = vol_m.group(1)
    elif parent_volume_id:
        out["volume_id"] = parent_volume_id

    # Chapter
    ch_m = CHAPTER_RE.search(t)
    if ch_m:
        out["chapter_id"] = ch_m.group(1)
    elif parent_chapter_id:
        out["chapter_id"] = parent_chapter_id

    # Section id (DoD-style)
    sid = extract_section_id(t)
    if sid:
        out["section_id"] = canonicalize_section_id(sid)
    else:
        m = DOD_CODE_RE.search(t)
        if m:
            out["section_id"] = canonicalize_section_id(m.group(1))

    # Section title: text after the code/label
    if out.get("section_id"):
        # Remove "030201 PURPOSE" -> keep "PURPOSE" or "Purpose"
        rest = re.sub(r"^\d{4,6}(?:\.\d{1,3})?\.?\s*", "", t, flags=re.I).strip()
        rest = re.sub(r"^(?:Section|Paragraph|Para\.?)\s+[\d.]+\s*", "", rest, flags=re.I).strip()
        if rest:
            out["section_title"] = rest[:200]

    # Build section_path segment
    parts = []
    if out.get("volume_id"):
        parts.append(f"Volume {out['volume_id']}")
    if out.get("chapter_id"):
        parts.append(f"Chapter {out['chapter_id']}")
    if out.get("section_id") or out.get("section_title"):
        seg = out.get("section_id", "") or ""
        if out.get("section_title"):
            seg = f"{seg} {out['section_title']}".strip() if seg else out["section_title"]
        if seg:
            parts.append(seg)
    if parts:
        out["section_path_segment"] = " > ".join(parts)

    # Inherit from parent for full path
    if parent_section_path and parts:
        out["section_path"] = f"{parent_section_path} > {parts[-1]}"
    elif parts:
        out["section_path"] = " > ".join(parts)

    return out


def is_valid_section_path(path: Optional[str]) -> bool:
    """True if path has Volume and Chapter and is not malformed."""
    if not (path or "").strip():
        return False
    p = str(path).strip()
    if SEGMENT_RE.match(p) or "2BDoD" in p or "Segment " in p:
        return False
    if not VOLUME_RE.search(p) or not CHAPTER_RE.search(p):
        return False
    return True
