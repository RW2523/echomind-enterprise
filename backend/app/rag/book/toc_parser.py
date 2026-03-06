"""
Dedicated TOC PDF parser for DoD Financial Management Regulation.

Parses a Table-of-Contents PDF into a structured hierarchy:
  Volume → Chapter → Section → Paragraph

The parsed structure is stored in book_sections and used to build the FAISS TOC
index for section-level routing.

Usage:
    from app.rag.book.toc_parser import parse_toc_pdf, ingest_toc

    # Parse raw PDF bytes into a tree
    tree = parse_toc_pdf(pdf_bytes)

    # Ingest into book_sections table + rebuild TOC index
    await ingest_toc(tree, doc_id="toc_001")
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .section_id import canonicalize_section_id, extract_section_id

logger = logging.getLogger(__name__)

# ── Volume / Chapter / Section regex for TOC lines ──────────────────────────

_VOL_RE = re.compile(
    r"^\s*Volume\s+(\d+[A-Za-z]?)\s*[-–:,]?\s*(.*?)\s*(?:\.{2,}|\s{3,}|\d+\s*$|$)",
    re.IGNORECASE,
)
_CHAP_RE = re.compile(
    r"^\s*Chapter\s+(\d+[A-Za-z]?)\s*[-–:,]?\s*(.*?)\s*(?:\.{2,}|\s{3,}|\d+\s*$|$)",
    re.IGNORECASE,
)
_APPENDIX_RE = re.compile(
    r"^\s*Appendix\s+([A-Z\d]+)\s*[-–:,]?\s*(.*?)\s*(?:\.{2,}|\s{3,}|\d+\s*$|$)",
    re.IGNORECASE,
)
_SECTION_RE = re.compile(
    r"^\s*(?:Section\s+)?(0[1-9]\d{2,4})\s+(.*?)\s*(?:\.{2,}|\s{3,}|\d+\s*$|$)",
    re.IGNORECASE,
)
_PARA_RE = re.compile(
    r"^\s*(\d{6})\s+(.*?)\s*(?:\.{2,}|\s{3,}|\d+\s*$|$)",
)
_PAGE_NUM_RE = re.compile(r"\.{2,}\s*(\d+)\s*$|(?:\s{3,})(\d+)\s*$")


# ── TOC node data class ─────────────────────────────────────────────────────

class TocEntry:
    """A single entry in the parsed TOC hierarchy."""

    __slots__ = ("level", "kind", "number", "title", "page", "section_path", "children")

    def __init__(
        self,
        level: int,
        kind: str,
        number: str,
        title: str,
        page: Optional[int] = None,
    ):
        self.level = level
        self.kind = kind            # "volume", "chapter", "appendix", "section", "paragraph"
        self.number = number        # e.g. "5", "2B", "0505", "050501"
        self.title = title.strip()
        self.page = page
        self.section_path = ""
        self.children: List[TocEntry] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "kind": self.kind,
            "number": self.number,
            "title": self.title,
            "page": self.page,
            "section_path": self.section_path,
            "children": [c.to_dict() for c in self.children],
        }


# ── Extract text from TOC PDF ───────────────────────────────────────────────

def _extract_toc_text(data: bytes) -> str:
    """Extract text from a TOC PDF using PyMuPDF (fitz) or pypdf fallback."""
    try:
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        pages = [page.get_text("text") or "" for page in doc]
        doc.close()
        return "\n".join(pages)
    except ImportError:
        pass
    from pypdf import PdfReader
    from io import BytesIO
    r = PdfReader(BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in r.pages)


def _extract_page_number(line: str) -> Optional[int]:
    """Extract trailing page number from a TOC line (after dots or whitespace)."""
    m = _PAGE_NUM_RE.search(line)
    if m:
        num_str = m.group(1) or m.group(2)
        if num_str:
            try:
                return int(num_str)
            except ValueError:
                pass
    return None


# ── Parse TOC lines into hierarchy ───────────────────────────────────────────

def _parse_toc_lines(lines: List[str]) -> List[TocEntry]:
    """Parse cleaned TOC lines into a flat list of TocEntry with level info."""
    entries: List[TocEntry] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or len(line) < 4:
            continue

        page = _extract_page_number(line)

        m = _VOL_RE.match(line)
        if m:
            entries.append(TocEntry(
                level=1, kind="volume", number=m.group(1),
                title=m.group(2).strip() or f"Volume {m.group(1)}", page=page,
            ))
            continue

        m = _CHAP_RE.match(line)
        if m:
            entries.append(TocEntry(
                level=2, kind="chapter", number=m.group(1),
                title=m.group(2).strip() or f"Chapter {m.group(1)}", page=page,
            ))
            continue

        m = _APPENDIX_RE.match(line)
        if m:
            entries.append(TocEntry(
                level=2, kind="appendix", number=m.group(1),
                title=m.group(2).strip() or f"Appendix {m.group(1)}", page=page,
            ))
            continue

        m = _SECTION_RE.match(line)
        if m:
            code = m.group(1)
            level = 3 if len(code) <= 4 else 4
            entries.append(TocEntry(
                level=level, kind="section", number=code,
                title=m.group(2).strip() or code, page=page,
            ))
            continue

        m = _PARA_RE.match(line)
        if m:
            entries.append(TocEntry(
                level=4, kind="paragraph", number=m.group(1),
                title=m.group(2).strip() or m.group(1), page=page,
            ))
            continue

    return entries


def _build_tree(entries: List[TocEntry]) -> List[TocEntry]:
    """Build a nested tree from flat entries using level-based nesting."""
    if not entries:
        return []

    root_entries: List[TocEntry] = []
    stack: List[TocEntry] = []

    for entry in entries:
        while stack and stack[-1].level >= entry.level:
            stack.pop()

        if stack:
            parent = stack[-1]
            parent.children.append(entry)
        else:
            root_entries.append(entry)

        stack.append(entry)

    return root_entries


def _assign_paths(entries: List[TocEntry], parent_path: str = "") -> None:
    """Recursively assign section_path to each entry."""
    for entry in entries:
        if entry.kind == "volume":
            entry.section_path = f"Volume {entry.number}"
            if entry.title and entry.title != f"Volume {entry.number}":
                entry.section_path += f" - {entry.title}"
        elif entry.kind == "chapter":
            prefix = parent_path
            label = f"Chapter {entry.number}"
            if entry.title and entry.title != label:
                label += f" - {entry.title}"
            entry.section_path = f"{prefix} > {label}" if prefix else label
        elif entry.kind == "appendix":
            prefix = parent_path
            label = f"Appendix {entry.number}"
            if entry.title and entry.title != label:
                label += f" - {entry.title}"
            entry.section_path = f"{prefix} > {label}" if prefix else label
        elif entry.kind == "section":
            prefix = parent_path
            label = f"{entry.number} {entry.title}" if entry.title != entry.number else entry.number
            entry.section_path = f"{prefix} > {label}" if prefix else label
        elif entry.kind == "paragraph":
            prefix = parent_path
            label = f"{entry.number} {entry.title}" if entry.title != entry.number else entry.number
            entry.section_path = f"{prefix} > {label}" if prefix else label
        else:
            entry.section_path = parent_path

        _assign_paths(entry.children, entry.section_path)


# ── Public API ───────────────────────────────────────────────────────────────

def parse_toc_pdf(data: bytes) -> List[TocEntry]:
    """Parse a TOC PDF into a structured hierarchy of TocEntry objects.

    Returns a list of root-level entries (typically Volumes), each with nested
    children representing Chapters, Sections, and Paragraphs.
    """
    text = _extract_toc_text(data)
    lines = text.split("\n")
    logger.info("toc_parser: extracted %d lines from TOC PDF", len(lines))

    entries = _parse_toc_lines(lines)
    logger.info("toc_parser: parsed %d TOC entries", len(entries))

    tree = _build_tree(entries)
    _assign_paths(tree)

    total = _count_entries(tree)
    logger.info(
        "toc_parser: built tree with %d root(s), %d total entries",
        len(tree), total,
    )
    return tree


def _count_entries(entries: List[TocEntry]) -> int:
    count = len(entries)
    for e in entries:
        count += _count_entries(e.children)
    return count


def flatten_toc(tree: List[TocEntry]) -> List[TocEntry]:
    """Flatten the nested tree into a list (pre-order traversal)."""
    flat: List[TocEntry] = []

    def _walk(entries: List[TocEntry]) -> None:
        for e in entries:
            flat.append(e)
            _walk(e.children)

    _walk(tree)
    return flat


async def ingest_toc(
    tree: List[TocEntry],
    doc_id: str,
) -> Dict[str, int]:
    """Store parsed TOC entries into book_sections table and rebuild TOC index.

    Returns {"sections_added": N, "toc_nodes_built": M}.
    """
    from ...core.db import get_conn
    from ...utils.ids import new_id, now_iso

    flat = flatten_toc(tree)
    if not flat:
        return {"sections_added": 0, "toc_nodes_built": 0}

    added = 0
    with get_conn() as conn:
        for entry in flat:
            sid = canonicalize_section_id(entry.number) or extract_section_id(entry.title) or ""
            section_id = new_id("sec")
            title_text = f"{entry.number} {entry.title}".strip() if entry.title else entry.number
            text_for_section = (
                f"{entry.kind.title()}: {title_text}. "
                f"Path: {entry.section_path}."
            )
            conn.execute(
                "INSERT OR REPLACE INTO book_sections "
                "(section_id, doc_id, section_title, section_path, full_section_text, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    section_id,
                    doc_id,
                    title_text,
                    entry.section_path,
                    text_for_section,
                    now_iso(),
                ),
            )
            added += 1
        conn.commit()
    logger.info("toc_parser: ingested %d TOC entries into book_sections (doc_id=%s)", added, doc_id)

    toc_nodes = 0
    try:
        from .toc_builder import build_toc_from_sections, save_toc_nodes
        nodes = build_toc_from_sections()
        save_toc_nodes(nodes)
        toc_nodes = len(nodes)
        logger.info("toc_parser: rebuilt TOC index with %d nodes", toc_nodes)
    except Exception as e:
        logger.warning("toc_parser: TOC index rebuild failed: %s", e)

    return {"sections_added": added, "toc_nodes_built": toc_nodes}
