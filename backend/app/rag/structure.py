"""
Lightweight structure index: extract headings/section markers from text.
Used for deterministic structure fallback (TOC/chapters/sections) without LLM.
All list items returned to the user must be verbatim from this index or retrieved context.
"""
from __future__ import annotations
import re
from typing import List, Tuple, Dict

from ..core.db import get_conn

# Heading patterns (generic): lines that look like section markers
# ALL CAPS line (short), Title Case line, CHAPTER N, PART N, SECTION N, INTRODUCTION, CONCLUSION, APPENDIX, numbered 1. / 1.1 / I. / A)
_HEADING_PATTERNS = [
    re.compile(r"^(?:CHAPTER|PART|SECTION|INTRODUCTION|CONCLUSION|APPENDIX|PREFACE)\s*(?:[IVXLC\d]+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)?\s*[.:]?\s*(.*)$", re.I),
    re.compile(r"^([A-Z][A-Z\s]{2,50})$"),  # ALL CAPS line, 2–50 chars
    re.compile(r"^(?:Table of )?Contents\s*$", re.I),
    re.compile(r"^\s*(\d{1,2}\.\s*\d{0,2}\s*[A-Za-z].*)$"),   # 1. or 1.1 Title
    re.compile(r"^\s*([IVXLC]+\.?\s+[A-Za-z].*)$"),            # I. or II. Title
    re.compile(r"^\s*([A-Za-z]\)\s+[A-Za-z].*)$"),             # A) Title
    re.compile(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,5})\s*$"),  # Title Case short line (single line, no trailing content)
]


def _extract_headings_from_text(text: str) -> List[str]:
    """Extract heading-like lines from a single text block. Returns list of verbatim heading strings."""
    if not (text or "").strip():
        return []
    lines = text.split("\n")
    headings: List[str] = []
    for line in lines:
        raw = line.strip()
        if len(raw) < 2 or len(raw) > 200:
            continue
        for pat in _HEADING_PATTERNS:
            m = pat.match(raw)
            if m:
                if pat.groups:
                    h = (m.group(1) or raw).strip()
                else:
                    h = raw
                if h and h not in headings:
                    headings.append(h)
                break
    return headings


def extract_headings_from_chunks(
    doc_id: str,
    chunks: List[Tuple[str, int, str]],
) -> List[Tuple[str, int, str, str, int]]:
    """
    Extract headings from a list of (chunk_id, chunk_index, text).
    Returns list of (doc_id, idx, heading_text, chunk_id, chunk_index).
    idx is global order (chunk_index * 1000 + local_idx) for stable ordering.
    """
    out: List[Tuple[str, int, str, str, int]] = []
    for chunk_id, chunk_index, text in chunks:
        hs = _extract_headings_from_text(text or "")
        for local_idx, h in enumerate(hs):
            idx = chunk_index * 1000 + local_idx
            out.append((doc_id, idx, h, chunk_id, chunk_index))
    return out


def store_doc_headings(doc_id: str, rows: List[Tuple[str, int, str, str, int]]) -> None:
    """Persist headings for a document. rows: (doc_id, idx, heading_text, chunk_id, chunk_index)."""
    if not doc_id or not rows:
        return
    with get_conn() as conn:
        conn.execute("DELETE FROM doc_headings WHERE doc_id = ?", (doc_id,))
        for r in rows:
            conn.execute(
                "INSERT INTO doc_headings (doc_id, idx, heading_text, chunk_id, chunk_index) VALUES (?,?,?,?,?)",
                (r[0], r[1], r[2], r[3], r[4]),
            )
        conn.commit()


def get_doc_headings_for_docs(
    doc_ids: List[str],
    context_window: str = "all",
) -> List[Tuple[str, str]]:
    """
    Return (doc_id, heading_text) for the given doc_ids, ordered by doc_id, idx.
    If context_window != 'all', filter by document created_at (caller can pass filtered doc_ids).
    """
    if not doc_ids:
        return []
    placeholders = ",".join("?" * len(doc_ids))
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT doc_id, heading_text FROM doc_headings WHERE doc_id IN ({placeholders}) ORDER BY doc_id, idx",
            doc_ids,
        ).fetchall()
    return [(r[0], r[1]) for r in rows]


def get_headings_verbatim_list(doc_ids: List[str]) -> List[str]:
    """Return a single ordered list of heading text (verbatim) for the given docs. For STRUCTURE fallback response."""
    pairs = get_doc_headings_for_docs(doc_ids)
    return [h for _, h in pairs]


def backfill_doc_headings() -> int:
    """
    Backfill doc_headings for all existing documents (e.g. after adding the table).
    Returns number of documents processed.
    """
    with get_conn() as conn:
        doc_ids = [r[0] for r in conn.execute("SELECT id FROM documents").fetchall()]
    count = 0
    for doc_id in doc_ids:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT id, chunk_index, text FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
                (doc_id,),
            ).fetchall()
        chunks = [(r[0], r[1], r[2]) for r in rows]
        if not chunks:
            continue
        heading_rows = extract_headings_from_chunks(doc_id, chunks)
        if heading_rows:
            store_doc_headings(doc_id, heading_rows)
        count += 1
    return count
