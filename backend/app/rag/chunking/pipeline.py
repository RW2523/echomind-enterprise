"""
Orchestrator: detect type, sanitize, dispatch chunker, assign IDs.
Propagates section_path (hierarchical breadcrumb) through all BOOK chunks for citation.
Supports true page-number assignment via page_offsets from parse_pdf.
Metadata validation: rejects chunks with malformed section_path (e.g. "Segment N").
"""
from __future__ import annotations
import bisect
import logging
import os
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
    handle_large_sections,
    _split_book_into_sections,
)
from ..book.section_id import extract_section_id, section_id_from_path


logger = logging.getLogger(__name__)

def _is_valid_book_section_path(section_path: Optional[str]) -> bool:
    """Reject generic 'Segment N' fallback but accept DoD-style numbered sections.

    Valid paths:
      - Contains Volume and Chapter (e.g. "Volume 5 > Chapter 3 > Section 0301")
      - DoD numbered section (e.g. "0101 GENERAL", "010201 PURPOSE")
      - Contains Chapter or Section keywords
    Invalid:
      - Empty or whitespace-only
      - Generic "Segment N" fallback from paragraph splitter
    """
    if not (section_path or "").strip():
        return False
    sp = str(section_path).strip()
    if re.match(r"^Segment\s+\d+\s*$", sp, re.I):
        return False
    # DoD numbered sections: 4-6 digit code followed by title (e.g. "0101 GENERAL")
    if re.match(r"^\d{4,6}\s+", sp):
        return True
    # Decimal-numbered sections used throughout the FMR chapters ("2.0 POLICY",
    # "1.1 Purpose", "4.5 Incorporated References"). These are real section labels;
    # rejecting them silently dropped every FMR chapter to flat chunking.
    if re.match(r"^\d+(?:\.\d+)*\s+\S", sp):
        return True
    has_volume = bool(re.search(r"Volume\s+\d+", sp, re.I))
    has_chapter = bool(re.search(r"Chapter\s+\d+", sp, re.I))
    if has_volume and has_chapter:
        return True
    # Accept paths with Section, Chapter, or Appendix keywords
    if re.search(r"(?:Section|Chapter|Appendix)\s+[\dA-Z]", sp, re.I):
        return True
    return False


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



def _coverage_ratio(source: str, chunks: List[Chunk]) -> float:
    """Fraction of the source represented by the chunks, measured on word shingles.

    Substring matching is too brittle here: chunkers re-join sentences and normalize
    whitespace, so identical content fails an exact `find`. Comparing sets of 8-word
    shingles is robust to whitespace/rejoining while still detecting genuinely dropped
    passages. Summing chunk lengths is NOT a substitute — overlap double-counts and can
    read >100% while a third of the document is missing (observed).
    """
    N = 8
    src_words = (source or "").split()
    if len(src_words) < N:
        return 1.0
    src = {tuple(src_words[i:i + N]) for i in range(len(src_words) - N + 1)}
    if not src:
        return 1.0
    got: set = set()
    for c in chunks:
        w = ((getattr(c, "text", "") or "")).split()
        for i in range(max(0, len(w) - N + 1)):
            sh = tuple(w[i:i + N])
            if sh in src:
                got.add(sh)
    return len(got) / len(src)


def chunk_document(
    text: str,
    doc_id: str,
    estimated_pages: int = 0,
    page_offsets: Optional[List[Tuple[int, int]]] = None,
    already_normalized: bool = False,
) -> List[Chunk]:
    """Full pipeline: detect document type, sanitize, chunk by strategy, assign IDs.

    Returns a flat list of Chunk (for long-form: parent + children; only children are
    used for retrieval).

    Args:
        estimated_pages: total page count from parser (for linear-interpolation fallback).
        page_offsets: list of (start_char_offset, page_number_1indexed) from parse_pdf.
            When provided, used for exact page assignment instead of linear estimation.
        already_normalized: when True (PDFs, normalized per-page in parse_pdf so page_offsets
            stay aligned), skip re-normalization here — re-running it would shift every offset
            relative to page_offsets and corrupt page citations. (H3)
    """
    if not (text or "").strip():
        return []

    if not already_normalized:
        text = normalize_extracted_text(text or "")
    doc_type = detect_document_type(text)
    # Record the routing decision: doc_type selects the entire chunking strategy, and it
    # was previously never logged at any level — a misdetection was invisible.
    logger.info(
        "chunking: doc_id=%s doc_type=%s chars=%d paragraph_breaks=%d",
        doc_id, getattr(doc_type, "value", doc_type), len(text), text.count("\n\n"),
    )
    # BookRAG (parent/child + section paths + contextual headers) is OPT-IN. The chunking
    # fixes below are correct, but on the current FMR corpus the structured path measured
    # WORSE end-to-end than flat chunking: the golden eval dropped 48/52 -> 43/52 with
    # doc-precision 0.98 -> 0.90 (facts land in parent-context chunks that rank differently,
    # and several questions started citing the wrong chapter). Flat chunking stays the
    # default until that is understood; set RAG_ENABLE_BOOKRAG=1 to evaluate the structured
    # path. See eval/test_chunk_coverage.py before changing this.
    if doc_type == DocType.BOOK and os.getenv("RAG_ENABLE_BOOKRAG", "0").lower() not in ("1", "true", "yes"):
        logger.info(
            "chunking: doc_id=%s detected BOOK but RAG_ENABLE_BOOKRAG is off — using flat chunking",
            doc_id,
        )
        doc_type = DocType.USER

    clean_text, redacted, sensitivity_level = sanitize_text(text)
    total_chars = len(clean_text)

    if doc_type == DocType.FAQ:
        chunks = chunk_faq(clean_text, sensitivity_level, redacted)
    elif doc_type == DocType.BOOK:
        sections = _split_book_into_sections(clean_text)
        chunks = []
        section_char_offset = 0
        salvaged_sections = 0
        salvaged_chars = 0
        for section_title, section_text in sections:
            section_path = _build_section_path(section_title)
            if not _is_valid_book_section_path(section_path):
                # SALVAGE, don't discard. This used to `continue`, dropping the whole
                # section's text — never chunked, never embedded, never retrievable —
                # while ingestion still reported success. On the FMR corpus that lost
                # 21-95% of a document (one kept only 5%). Sections without a usable
                # section_path still get indexed via flat chunking; they just carry no
                # section metadata. Only reachable since BOOK detection was repaired.
                salvaged = chunk_unstructured(section_text, sensitivity_level, redacted)
                for c in salvaged:
                    c.doc_id = doc_id
                    c.chunk_id = new_id("chk")
                    c.section_title = section_title or None
                    page = _page_for_offset(
                        section_char_offset, total_chars, estimated_pages, page_offsets
                    )
                    if page is not None:
                        c.page_number = page
                    chunks.append(c)
                salvaged_sections += 1
                salvaged_chars += len(section_text)
                section_char_offset += len(section_title or "") + len(section_text)
                continue
            canonical_sid = section_id_from_path(section_path or "") or extract_section_id(section_title or "")
            # Split oversized sections at paragraph boundaries before chunking
            sub_texts = handle_large_sections(section_text)
            sub_char_offset = section_char_offset
            for sub_idx, sub_text in enumerate(sub_texts):
                effective_title = section_title if len(sub_texts) == 1 else f"{section_title} (part {sub_idx + 1})"
                effective_path = section_path if len(sub_texts) == 1 else f"{section_path} (part {sub_idx + 1})"
                pc_list = chunk_long_form(
                    sub_text,
                    sensitivity_level,
                    redacted,
                    section=effective_title,
                    section_path=effective_path,
                )
                for pc in pc_list:
                    pc.parent.section_id = canonical_sid
                    for c in pc.children:
                        c.section_id = canonical_sid
                for pc in pc_list:
                    parent = pc.parent
                    parent.chunk_id = new_id("chk")
                    parent.doc_id = doc_id
                    parent.section_title = effective_title
                    page = _page_for_offset(
                        sub_char_offset, total_chars, estimated_pages, page_offsets
                    )
                    if page is not None:
                        parent.page_number = page
                    chunks.append(parent)
                    child_count = len(pc.children)
                    for ci, c in enumerate(pc.children):
                        c.parent_chunk_id = parent.chunk_id
                        c.doc_id = doc_id
                        c.chunk_id = new_id("chk")
                        c.section_title = effective_title
                        if child_count > 0 and (page_offsets or estimated_pages > 0):
                            child_offset = sub_char_offset + int((ci / max(child_count, 1)) * len(sub_text))
                            child_page = _page_for_offset(child_offset, total_chars, estimated_pages, page_offsets)
                            c.page_number = child_page if child_page is not None else parent.page_number
                        else:
                            c.page_number = parent.page_number
                        chunks.append(c)
                sub_char_offset += len(sub_text)
            section_char_offset += len(section_title or "") + len(section_text)
        if salvaged_sections:
            logger.info(
                "chunking: doc_id=%s salvaged %d section(s) (%d chars) via flat chunking — "
                "no valid section_path, but content is indexed",
                doc_id, salvaged_sections, salvaged_chars,
            )
        if not chunks:
            # No valid Volume/Chapter sections found — fall back to unstructured chunking.
            # LOUD: this silently disables the whole BookRAG path (parent/child chunks,
            # section index, TOC routing, contextual headers). It hid two real bugs for
            # months because the chunking package had no logging at all.
            logger.warning(
                "chunking: doc_id=%s detected as BOOK but produced no valid sections — "
                "falling back to flat chunking (no parent/child, no section_path, "
                "no contextual retrieval)",
                doc_id,
            )
            chunks = chunk_unstructured(clean_text, sensitivity_level, redacted)
            n_chunks = len(chunks)
            for i, c in enumerate(chunks):
                c.doc_id = doc_id
                c.chunk_id = new_id("chk")
                c.chunk_index = i
                # Estimate page from chunk position for document preview
                if n_chunks > 0 and (estimated_pages > 0 or page_offsets):
                    offset = int((i / n_chunks) * total_chars) if total_chars > 0 else 0
                    c.page_number = _page_for_offset(offset, total_chars, estimated_pages, page_offsets)
            return chunks
        # CONTENT-LOSS GUARD: structured chunking must not lose the document. Sub-paths
        # (section split, parent/child sizing, large-section handling) have each dropped
        # text at some point; rather than trust them, verify and fall back to the
        # proven-lossless flat chunker when coverage is short. Never silent.
        min_cov = float(os.getenv("CHUNK_MIN_COVERAGE", "0.98"))
        cov = _coverage_ratio(clean_text, chunks)
        if cov < min_cov:
            logger.warning(
                "chunking: doc_id=%s BOOK chunking covered only %.0f%% of the document "
                "(< %.0f%% required) — falling back to flat chunking to preserve content",
                doc_id, cov * 100, min_cov * 100,
            )
            chunks = chunk_unstructured(clean_text, sensitivity_level, redacted)
            n_chunks = len(chunks)
            for i, c in enumerate(chunks):
                c.doc_id = doc_id
                c.chunk_id = new_id("chk")
                c.chunk_index = i
                if n_chunks > 0 and (estimated_pages > 0 or page_offsets):
                    offset = int((i / n_chunks) * total_chars) if total_chars > 0 else 0
                    c.page_number = _page_for_offset(offset, total_chars, estimated_pages, page_offsets)
            return chunks
        logger.info("chunking: doc_id=%s BOOK coverage %.0f%%", doc_id, cov * 100)
        _assign_indices(chunks)
        return chunks
    elif doc_type == DocType.SENSITIVE:
        chunks = chunk_sensitive(clean_text, sensitivity_level, redacted)
    else:
        chunks = chunk_unstructured(clean_text, sensitivity_level, redacted)

    n_chunks = len(chunks)
    for i, c in enumerate(chunks):
        c.doc_id = doc_id
        c.chunk_id = new_id("chk")
        c.chunk_index = i
        # Estimate page from chunk position for document preview (USER/SENSITIVE)
        if n_chunks > 0 and (estimated_pages > 0 or page_offsets) and c.page_number is None:
            offset = int((i / n_chunks) * total_chars) if total_chars > 0 else 0
            c.page_number = _page_for_offset(offset, total_chars, estimated_pages, page_offsets)
    return chunks


def _assign_indices(chunks: List[Chunk]) -> None:
    """Set chunk_index by order (parents then children per parent)."""
    for i, c in enumerate(chunks):
        c.chunk_index = i
