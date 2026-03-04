"""
Orchestrator: detect type, sanitize, dispatch chunker, assign IDs.
Propagates section_path (hierarchical breadcrumb) through all BOOK chunks for citation.
Supports true page-number assignment via page_offsets from parse_pdf.
"""
from __future__ import annotations
import bisect
import re
from typing import Dict, List, Optional, Tuple

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
        sections = _split_book_into_sections(clean_text)
        chunks = []
        section_char_offset = 0
        for section_title, section_text in sections:
            section_path = _build_section_path(section_title)
            pc_list = chunk_long_form(
                section_text,
                sensitivity_level,
                redacted,
                section=section_title,
                section_path=section_path,
            )
            for pc in pc_list:
                parent = pc.parent
                parent.chunk_id = new_id("chk")
                parent.doc_id = doc_id
                page = _page_for_offset(
                    section_char_offset, total_chars, estimated_pages, page_offsets
                )
                if page is not None:
                    parent.page_number = page
                chunks.append(parent)
                for c in pc.children:
                    c.parent_chunk_id = parent.chunk_id
                    c.doc_id = doc_id
                    c.chunk_id = new_id("chk")
                    c.page_number = parent.page_number
                    chunks.append(c)
            section_char_offset += len(section_text)
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
