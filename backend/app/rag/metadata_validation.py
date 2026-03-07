"""
Metadata validation for BOOK chunk ingestion.

Validates section_id, volume, chapter, page_number. Rejects or corrects malformed metadata.
Prevents fallback "Segment N" citations by requiring proper section_path (Volume + Chapter).
"""
from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# DoD-style: 4-6 digits, optional .1-3 sub (e.g. 030201, 0705.12)
SECTION_ID_RE = re.compile(r"^\d{4,6}(\.\d{1,3})?$")

# Reject "Segment 465", "Segment 1326" etc. — must have Volume and Chapter
SEGMENT_FALLBACK_RE = re.compile(r"^Segment\s+\d+\s*$", re.I)
# Malformed: "2BDoD 7000 14-R", "Chapter 469" (when chapter 469 doesn't exist), etc.
MALFORMED_PATH_RE = re.compile(
    r"(?:Segment\s+\d+|2BDoD\s*\d+|Volume\s+X\b|Chapter\s+\d{3,}(?!\s))",
    re.I,
)


def validate_section_id(section_id: Optional[str]) -> Tuple[Optional[str], bool]:
    """Validate section_id format. Returns (corrected_value, is_valid)."""
    if section_id is None or section_id == "":
        return None, True  # Missing is ok for non-BOOK
    s = str(section_id).strip()
    if not s:
        return None, True
    if SECTION_ID_RE.match(s):
        return s, True
    # Try to extract valid part
    m = re.search(r"(\d{4,6}(?:\.\d{1,3})?)", s)
    if m:
        return m.group(1), True
    logger.warning("metadata_validation: invalid section_id=%r (expected ^\\d{4,6}(\\.\\d{1,3})?$)", section_id)
    return None, False


def validate_volume(volume: Optional) -> Tuple[Optional[int], bool]:
    """Ensure volume is integer. Returns (value, is_valid)."""
    if volume is None:
        return None, True
    try:
        v = int(volume) if not isinstance(volume, int) else volume
        return v, True
    except (ValueError, TypeError):
        logger.warning("metadata_validation: invalid volume=%r (expected integer)", volume)
        return None, False


def validate_chapter(chapter: Optional) -> Tuple[Optional[int], bool]:
    """Ensure chapter is integer. Returns (value, is_valid)."""
    if chapter is None:
        return None, True
    try:
        c = int(chapter) if not isinstance(chapter, int) else chapter
        return c, True
    except (ValueError, TypeError):
        logger.warning("metadata_validation: invalid chapter=%r (expected integer)", chapter)
        return None, False


def validate_page_number(page_number: Optional, required_for_book: bool = False) -> Tuple[Optional[int], bool]:
    """Ensure page_number exists and is int. Returns (value, is_valid)."""
    if page_number is None:
        if required_for_book:
            logger.warning("metadata_validation: page_number missing (required for BOOK)")
            return None, False
        return None, True
    try:
        p = int(page_number) if not isinstance(page_number, int) else page_number
        return p if p > 0 else None, True
    except (ValueError, TypeError):
        logger.warning("metadata_validation: invalid page_number=%r (expected positive integer)", page_number)
        return None, False


def validate_and_fix_source(source: dict, doc_type: str = "") -> Tuple[dict, bool]:
    """Validate and correct chunk source metadata. Returns (corrected_source, is_valid).

    - section_id: must match ^\\d{4,6}(\\.\\d{1,3})?$ or is corrected/dropped
    - volume, chapter: must be integers if present
    - page_number: must exist for BOOK doc_type
    """
    if not source:
        return source, True
    src = dict(source)
    is_book = (doc_type or src.get("doc_type") or "").lower() == "book"
    valid = True

    # section_id
    sid = src.get("section_id") or src.get("section")
    if sid is not None or "section_id" in src or "section" in src:
        corrected, ok = validate_section_id(sid)
        valid = valid and ok
        if corrected is not None:
            src["section_id"] = corrected
            if "section" in src:
                src["section"] = corrected
        elif sid and not ok:
            src.pop("section_id", None)
            src.pop("section", None)

    # volume (parse from section_path if missing for BOOK)
    vol = src.get("volume")
    if vol is None and is_book and src.get("section_path"):
        m = re.search(r"Volume\s+(\d+)", str(src.get("section_path") or ""), re.I)
        if m:
            vol = m.group(1)
    if vol is not None:
        corrected, ok = validate_volume(vol)
        valid = valid and ok
        if corrected is not None:
            src["volume"] = corrected
        elif not ok:
            src.pop("volume", None)

    # chapter (parse from section_path if missing for BOOK)
    ch = src.get("chapter")
    if ch is None and is_book and src.get("section_path"):
        m = re.search(r"Chapter\s+(\d+)", str(src.get("section_path") or ""), re.I)
        if m:
            ch = m.group(1)
    if ch is not None:
        corrected, ok = validate_chapter(ch)
        valid = valid and ok
        if corrected is not None:
            src["chapter"] = corrected
        elif not ok:
            src.pop("chapter", None)

    # page_number (required for BOOK; missing is warned but not rejected to avoid breaking ingestion)
    pn = src.get("page_number")
    corrected, ok = validate_page_number(pn, required_for_book=False)
    valid = valid and ok
    if corrected is not None:
        src["page_number"] = corrected
    if is_book and pn is None:
        logger.debug("metadata_validation: BOOK chunk missing page_number (consider providing page_offsets)")

    return src, valid


def validate_section_path(section_path: Optional[str], reject_segment_fallback: bool = True) -> Tuple[bool, Optional[str]]:
    """Validate section_path for BOOK chunks.

    Strict pass: path contains both "Volume N" and "Chapter N" markers (ideal for full DoD FMR).
    Relaxed pass: path contains a DoD-style numeric code (4-6 digits, e.g. "010101 GENERAL")
      OR at least one of Volume/Chapter — handles PDFs that split by DoD code without full breadcrumbs.
    Always rejects: empty paths, bare "Segment N" fallback tokens.
    Returns (is_valid, reason).
    """
    if not (section_path or "").strip():
        return False, "section_path missing"
    sp = str(section_path).strip()
    if reject_segment_fallback and SEGMENT_FALLBACK_RE.match(sp):
        logger.warning("metadata_validation: rejecting Segment fallback path=%r", sp)
        return False, "Segment N fallback not allowed"
    has_volume = bool(re.search(r"Volume\s+\d+", sp, re.I))
    has_chapter = bool(re.search(r"Chapter\s+\d+", sp, re.I))
    if has_volume and has_chapter:
        return True, None
    # Accept DoD-style 4-6 digit section codes ("010101 GENERAL", "030201.A PURPOSE")
    has_dod_code = bool(re.search(r"\b\d{4,6}(?:\.\w+)?\b", sp))
    if has_dod_code:
        return True, None
    # Accept partial hierarchy (Volume-only or Chapter-only with at least one word)
    if (has_volume or has_chapter) and len(sp.split()) >= 2:
        return True, None
    logger.warning(
        "metadata_validation: section_path has no Volume/Chapter/DoD code: %r",
        sp[:80],
    )
    return False, "section_path must contain Volume/Chapter markers or a DoD section code"


def validate_book_chunk_for_indexing(source: dict) -> Tuple[dict, bool]:
    """Strict validation for BOOK chunks before indexing. Returns (corrected_source, is_valid).

    If invalid after normalization, chunk should be skipped (not indexed).
    Required fields for BOOK: doc_id, section_path, section_id, page_number, parent_chunk_id (for children).
    """
    if not source:
        return source, False
    src = dict(source)
    doc_type = (src.get("doc_type") or "").lower()
    if doc_type != "book":
        return validate_and_fix_source(src, doc_type)

    # Try canonicalize_section_id first
    sid = src.get("section_id") or src.get("section")
    if sid:
        try:
            from .book.section_id import canonicalize_section_id
            canonical = canonicalize_section_id(str(sid))
            if canonical:
                corrected, ok = validate_section_id(canonical)
                if ok and corrected:
                    src["section_id"] = corrected
                    if "section" in src:
                        src["section"] = corrected
                elif not ok:
                    valid_sid, _ = validate_section_id(sid)
                    if valid_sid:
                        src["section_id"] = valid_sid
        except Exception:
            pass

    # section_path: must contain Volume and Chapter; reject Segment fallback
    sp = src.get("section_path") or ""
    path_ok, _ = validate_section_path(sp, reject_segment_fallback=True)
    if not path_ok:
        logger.warning("metadata_validation: BOOK chunk invalid section_path, skipping: %s", sp[:60])
        return src, False

    # page_number: required for BOOK
    pn = src.get("page_number")
    _, pn_ok = validate_page_number(pn, required_for_book=True)
    if not pn_ok or pn is None:
        logger.warning("metadata_validation: BOOK chunk missing page_number, skipping")
        return src, False

    # section_id: must match regex (after canonicalize attempt)
    sid = src.get("section_id") or src.get("section")
    if sid:
        corrected, ok = validate_section_id(sid)
        if not ok:
            logger.warning("metadata_validation: BOOK chunk invalid section_id=%r, skipping", sid)
            return src, False
        if corrected:
            src["section_id"] = corrected

    # Require doc_id; parent_chunk_id required for child chunks only
    if not src.get("doc_id"):
        logger.warning("metadata_validation: BOOK chunk missing doc_id, skipping")
        return src, False
    if not src.get("is_parent") and not src.get("parent_chunk_id"):
        logger.warning("metadata_validation: BOOK child chunk missing parent_chunk_id, skipping")
        return src, False

    src, _ = validate_and_fix_source(src, "book")
    return src, True


def validate_chunk_metadata(chunk: dict) -> Tuple[dict, bool]:
    """Validate and repair chunk metadata. Returns (corrected_chunk, metadata_valid).

    For BOOK chunks, requires: doc_id, volume_id, chapter_id, section_id, section_path,
    page_number, parent_chunk_id (children), is_parent.

    Rules:
    - section_id must match DoD pattern or be null only when truly impossible
    - volume_id and chapter_id must not be malformed
    - page_number must exist
    - section_path must not contain: Segment N, 2BDoD 7000 14-R, Volume X, malformed Chapter

    If invalid after repair attempt: metadata_valid=False. Chunk may be indexed as
    low-confidence background only, never as primary evidence or citation.
    """
    if not chunk:
        return chunk, False
    src = dict(chunk)
    doc_type = (src.get("doc_type") or "").lower()
    if doc_type != "book":
        repaired, ok = validate_and_fix_source(src, doc_type)
        repaired["metadata_valid"] = ok
        return repaired, ok

    # Attempt repair from section_path
    sp = (src.get("section_path") or "").strip()
    if sp and MALFORMED_PATH_RE.search(sp):
        # Try to extract valid volume/chapter from path
        vol_m = re.search(r"Volume\s+(\d+[A-Za-z]?)", sp, re.I)
        ch_m = re.search(r"Chapter\s+(\d+[A-Za-z]?)", sp, re.I)
        if vol_m:
            v = vol_m.group(1)
            if len(v) <= 3:  # "5", "14", "2A" ok; "469" suspicious for chapter
                src["volume_id"] = v
                if "volume" not in src:
                    try:
                        src["volume"] = int(re.search(r"\d+", v).group()) if re.search(r"\d+", v) else None
                    except Exception:
                        pass
        if ch_m:
            c = ch_m.group(1)
            if len(c) <= 3:  # Chapter 5, 11 ok; Chapter 469 may be malformed
                src["chapter_id"] = c
                if "chapter" not in src:
                    try:
                        src["chapter"] = int(re.search(r"\d+", c).group()) if re.search(r"\d+", c) else None
                    except Exception:
                        pass
        # If path still looks malformed, mark invalid
        if SEGMENT_FALLBACK_RE.match(sp) or "2BDoD" in sp or "Segment " in sp:
            src["metadata_valid"] = False
            return src, False

    # Ensure volume_id, chapter_id as strings
    vol = src.get("volume") or src.get("volume_id")
    if vol is not None:
        src["volume_id"] = str(vol)
        src["volume"] = int(vol) if str(vol).isdigit() else vol
    ch = src.get("chapter") or src.get("chapter_id")
    if ch is not None:
        src["chapter_id"] = str(ch)
        src["chapter"] = int(ch) if str(ch).isdigit() else ch

    # section_id
    sid = src.get("section_id") or src.get("section")
    if sid:
        try:
            from .book.section_id import canonicalize_section_id
            canonical = canonicalize_section_id(str(sid))
            corrected, ok = validate_section_id(canonical or sid)
            if ok and corrected:
                src["section_id"] = corrected
                src["section"] = corrected
            elif not ok:
                m = re.search(r"(\d{4,6}(?:\.\d{1,3})?)", str(sid))
                if m:
                    src["section_id"] = m.group(1)
                    src["section"] = m.group(1)
        except Exception:
            pass

    # section_path validation
    sp = src.get("section_path") or ""
    path_ok, _ = validate_section_path(sp, reject_segment_fallback=True)
    if not path_ok:
        src["metadata_valid"] = False
        return src, False

    # page_number required
    pn = src.get("page_number")
    _, pn_ok = validate_page_number(pn, required_for_book=True)
    if not pn_ok or pn is None:
        src["metadata_valid"] = False
        return src, False

    # Required fields
    if not src.get("doc_id"):
        src["metadata_valid"] = False
        return src, False
    if not src.get("is_parent") and not src.get("parent_chunk_id"):
        src["metadata_valid"] = False
        return src, False

    # volume_id and chapter_id from section_path if missing
    if not src.get("volume_id") and sp:
        vol_m = re.search(r"Volume\s+(\d+[A-Za-z]?)", sp, re.I)
        if vol_m:
            src["volume_id"] = vol_m.group(1)
    if not src.get("chapter_id") and sp:
        ch_m = re.search(r"Chapter\s+(\d+[A-Za-z]?)", sp, re.I)
        if ch_m:
            src["chapter_id"] = ch_m.group(1)

    # Titles from section_path
    if not src.get("volume_title") and src.get("volume_id"):
        src["volume_title"] = f"Volume {src['volume_id']}"
    if not src.get("chapter_title") and src.get("chapter_id"):
        src["chapter_title"] = f"Chapter {src['chapter_id']}"
    if not src.get("section_title") and src.get("section_id"):
        src["section_title"] = src.get("section_id")

    src, _ = validate_and_fix_source(src, "book")
    src["metadata_valid"] = True
    return src, True
