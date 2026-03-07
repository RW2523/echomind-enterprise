"""
Orchestrator: detect type, sanitize, dispatch chunker, assign IDs.
Propagates section_path (hierarchical breadcrumb) through all BOOK chunks for citation.
Supports true page-number assignment via page_offsets from parse_pdf.
Metadata validation: rejects chunks with malformed section_path (e.g. "Segment N").

BookRAG-lite++ additions:
  - Assigns clause_id (detected DoD clause code) to each BOOK child chunk.
  - Builds retrieval_text (heading path + clause label prepended) for enriched embeddings.
  - Assigns canonical_id (deterministic vol_XX_ch_XX_sec_XXXXXX identifier).
  - Links prev_chunk_id / next_chunk_id between sibling children within the same section.
  - Marks has_table when heuristic detects table content in chunk text.

Fallback policy:
  When a BOOK document produces zero valid hierarchical sections (no Volume + Chapter markers
  detected in section titles), the pipeline logs a warning and falls back to unstructured
  chunking so the document still gets indexed and queried — without hierarchical citations.
"""
from __future__ import annotations
import bisect
import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from ...utils.ids import new_id

from ..normalize import normalize_extracted_text
from .models import Chunk, DocType
from .detect import detect_document_type
from .sanitize import sanitize_text
from .chunkers import (
    chunk_faq,
    chunk_long_form,
    chunk_sensitive,
    chunk_unstructured,
    _split_book_into_sections,
)
from ..book.section_id import extract_section_id, section_id_from_path
from ..book.clause_parser import (
    dominant_clause_id,
    build_retrieval_text,
    build_canonical_id,
    has_table_heuristic,
)


def _is_valid_book_section_path(section_path: Optional[str]) -> bool:
    """Accept meaningful section paths; reject noise tokens.

    Strict pass (ideal): path contains both Volume N and Chapter N markers.
    Relaxed pass: path contains a DoD-style numeric code (4+ digit section code, e.g.
      "010101 ADMINISTRATION" or "Chapter 2" with at least one code).  These come from
      PDFs that embed volume/chapter info in running text rather than section headings.
    Rejected: empty, bare "Segment N" fallback tokens, single-word non-code titles.
    """
    if not (section_path or "").strip():
        return False
    sp = str(section_path).strip()
    # Always reject bare "Segment N" fallback
    if re.match(r"^Segment\s+\d+\s*$", sp, re.I):
        return False
    has_volume = bool(re.search(r"Volume\s+\d+", sp, re.I))
    has_chapter = bool(re.search(r"Chapter\s+\d+", sp, re.I))
    if has_volume and has_chapter:
        return True
    # Relaxed: accept DoD-style 4-6 digit section codes ("010101", "030201.A")
    has_dod_code = bool(re.search(r"\b\d{4,6}(?:\.\w+)?\b", sp))
    # Relaxed: accept if at least Volume or Chapter is present (partial hierarchy)
    if has_dod_code or has_volume or has_chapter:
        return True
    # Reject very short or purely alphabetic titles with no structural marker
    words = sp.split()
    if len(words) <= 2 and not re.search(r"\d", sp):
        return False
    return True


def _build_section_path(title: Optional[str], parent_path: Optional[str] = None) -> Optional[str]:
    """Build hierarchical section path like 'Volume 1 > Chapter 3 > Section 0301'."""
    if not title:
        return parent_path
    clean = re.sub(r"\s+", " ", title.strip())
    if parent_path:
        return f"{parent_path} > {clean}"
    return clean


def _estimate_page_for_offset(offset: int, total_chars: int, estimated_pages: int) -> Optional[int]:
    """Estimate page number from character offset (linear approximation, fallback when no page_offsets)."""
    if estimated_pages <= 0 or total_chars <= 0:
        return None
    return max(1, int((offset / total_chars) * estimated_pages) + 1)


def _true_page_for_offset(
    offset: int, page_offsets: List[Tuple[int, int]]
) -> Optional[int]:
    """Binary-search page_offsets to find the 1-indexed page number for a character offset.

    page_offsets: list of (start_char_offset, page_number_1indexed) sorted ascending.
    Returns the page whose start_offset is the largest value <= offset.
    """
    if not page_offsets:
        return None
    # Extract just the offsets for bisect
    starts = [po[0] for po in page_offsets]
    pos = bisect.bisect_right(starts, offset) - 1
    if pos < 0:
        pos = 0
    return page_offsets[pos][1]


def _page_for_offset(
    offset: int,
    total_chars: int,
    estimated_pages: int,
    page_offsets: Optional[List[Tuple[int, int]]],
) -> Optional[int]:
    """Return the best available page number for a given char offset."""
    if page_offsets:
        return _true_page_for_offset(offset, page_offsets)
    return _estimate_page_for_offset(offset, total_chars, estimated_pages)


def _get_book_flags() -> tuple[bool, bool, bool]:
    """Return (heading_path_in_embed, clause_chunking, table_extraction) from config."""
    try:
        from ...core.config import settings
        return (
            getattr(settings, "BOOK_HEADING_PATH_IN_EMBED", True),
            getattr(settings, "BOOK_CLAUSE_CHUNKING_ENABLED", True),
            getattr(settings, "BOOK_TABLE_EXTRACTION", True),
        )
    except Exception:
        return (True, True, True)


def chunk_document(
    text: str,
    doc_id: str,
    estimated_pages: int = 0,
    page_offsets: Optional[List[Tuple[int, int]]] = None,
) -> List[Chunk]:
    """Full pipeline: detect document type, sanitize, chunk by strategy, assign IDs.

    Returns a flat list of Chunk (for long-form: parent + children; only children are
    used for retrieval).

    Args:
        estimated_pages: total page count from parser (for linear-interpolation fallback).
        page_offsets: list of (start_char_offset, page_number_1indexed) from parse_pdf.
            When provided, used for exact page assignment instead of linear estimation.
    """
    if not (text or "").strip():
        return []

    text = normalize_extracted_text(text or "")
    doc_type = detect_document_type(text)
    clean_text, redacted, sensitivity_level = sanitize_text(text)
    total_chars = len(clean_text)

    if doc_type == DocType.FAQ:
        chunks = chunk_faq(clean_text, sensitivity_level, redacted)
    elif doc_type == DocType.BOOK:
        use_heading_embed, use_clause, use_table = _get_book_flags()
        sections = _split_book_into_sections(clean_text)
        chunks = []
        _rejected_section_titles: List[str] = []
        section_char_offset = 0
        for section_title, section_text in sections:
            section_path = _build_section_path(section_title)
            if not _is_valid_book_section_path(section_path):
                _rejected_section_titles.append(str(section_title or ""))
                continue  # Skip sections with malformed path (e.g. "Segment N")
            canonical_sid = section_id_from_path(section_path or "") or extract_section_id(section_title or "")
            pc_list = chunk_long_form(
                section_text,
                sensitivity_level,
                redacted,
                section=section_title,
                section_path=section_path,
            )
            for pc in pc_list:
                pc.parent.section_id = canonical_sid
                for c in pc.children:
                    c.section_id = canonical_sid
            for pc in pc_list:
                parent = pc.parent
                parent.chunk_id = new_id("chk")
                parent.doc_id = doc_id
                parent.section_title = section_title
                parent.evidence_type = "parent"
                page = _page_for_offset(
                    section_char_offset, total_chars, estimated_pages, page_offsets
                )
                if page is not None:
                    parent.page_number = page
                    parent.page_start = page

                # Assign canonical_id to parent
                if use_clause or use_heading_embed:
                    parent.canonical_id = build_canonical_id(
                        section_path, page=page, chunk_index=parent.chunk_index
                    )

                # Assign table heuristic to parent
                if use_table:
                    parent.has_table = has_table_heuristic(parent.text)

                chunks.append(parent)

                # Build children list for this parent (for prev/next linking)
                section_children: List[Chunk] = []
                for c in pc.children:
                    c.parent_chunk_id = parent.chunk_id
                    c.doc_id = doc_id
                    c.chunk_id = new_id("chk")
                    c.section_title = section_title
                    c.evidence_type = "child"

                    # Page attribution: use per-child char offset when page_offsets available,
                    # otherwise inherit parent page (better than wrong linear estimate)
                    if page_offsets and c.char_start is not None:
                        child_page = _true_page_for_offset(c.char_start, page_offsets)
                        c.page_number = child_page if child_page is not None else parent.page_number
                    else:
                        c.page_number = parent.page_number
                    c.page_start = c.page_number

                    # BookRAG-lite++: clause detection
                    if use_clause:
                        c.clause_id = dominant_clause_id(c.text)

                    # BookRAG-lite++: heading path in retrieval_text
                    if use_heading_embed:
                        c.retrieval_text = build_retrieval_text(
                            c.text,
                            section_path=section_path,
                            clause_id=c.clause_id,
                            section_title=section_title,
                            page_number=c.page_number,
                            doc_type="BOOK",
                        )

                    # BookRAG-lite++: canonical_id
                    if use_clause or use_heading_embed:
                        c.canonical_id = build_canonical_id(
                            section_path,
                            page=c.page_number,
                            clause_id=c.clause_id,
                        )

                    # BookRAG-lite++: table heuristic
                    if use_table:
                        c.has_table = has_table_heuristic(c.text)

                    section_children.append(c)

                # Assign prev/next links within this parent's children
                for i_c, child in enumerate(section_children):
                    if i_c > 0:
                        child.prev_chunk_id = section_children[i_c - 1].chunk_id
                    if i_c < len(section_children) - 1:
                        child.next_chunk_id = section_children[i_c + 1].chunk_id
                    # Propagate new fields into source metadata via chunk_id
                    # (IDs are already set above; prev/next assigned here)
                    chunks.append(child)

            section_char_offset += len(section_text)

        if not chunks:
            # No valid hierarchical sections found. Log details so the user/dev can understand why.
            sample = _rejected_section_titles[:8]
            logger.warning(
                "chunk_document: BOOK doc '%s' — %d section(s) found but ALL rejected by path validation "
                "(no Volume/Chapter markers or DoD codes detected). Rejected samples: %s. "
                "Falling back to unstructured chunking so the document still gets indexed.",
                doc_id, len(_rejected_section_titles), sample,
            )
            chunks = chunk_unstructured(clean_text, sensitivity_level, redacted)
            for i, c in enumerate(chunks):
                c.doc_id = doc_id
                c.chunk_id = new_id("chk")
                c.chunk_index = i
            return chunks

        _assign_indices(chunks)
        return chunks
    elif doc_type == DocType.SENSITIVE:
        chunks = chunk_sensitive(clean_text, sensitivity_level, redacted)
    else:
        chunks = chunk_unstructured(clean_text, sensitivity_level, redacted)

    for i, c in enumerate(chunks):
        c.doc_id = doc_id
        c.chunk_id = new_id("chk")
        c.chunk_index = i
    return chunks


def _assign_indices(chunks: List[Chunk]) -> None:
    """Set chunk_index by order (parents then children per parent)."""
    for i, c in enumerate(chunks):
        c.chunk_index = i
