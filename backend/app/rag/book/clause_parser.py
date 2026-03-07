"""
Clause-level detection and retrieval-text construction for DoD regulatory documents.

DoD FMR uses a structured clause numbering system:
  030201       → Section-level code (6 digits)
  030201.A     → First-level subclause (letter suffix)
  030201.A.1   → Second-level subclause (digit after letter)
  030201.1     → Alternate numeric subclause

This module:
  1. Detects clause IDs present in a text block
  2. Returns the dominant (most prominent) clause ID
  3. Builds retrieval_text = heading path + clause label + raw text
     (used for embedding; raw text is stored separately as display_text)
  4. Generates deterministic canonical IDs for BOOK chunks

Activated via BOOK_HEADING_PATH_IN_EMBED and BOOK_CLAUSE_CHUNKING_ENABLED config flags.
Does NOT depend on any external libraries — safe for offline deployment.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# DoD clause pattern: 4–6 digit base code, optionally followed by:
#   .A, .A.1, .a.1, .1, .1.a style suffixes.
# Examples: 030201, 030201.A, 030201.A.1, 7001.B.2
_CLAUSE_RE = re.compile(
    r"\b(\d{4,6}(?:\.[A-Za-z]\d*(?:\.\d+)?|\.\d+(?:\.[A-Za-z])?)?)\b"
)

# Section code only (no subclause suffix) — used to detect section titles
_SECTION_CODE_RE = re.compile(r"^\s*(\d{4,6})\s+[A-Z]")

# DoD volume marker in section path: "Volume 5", "Volume 05", etc.
_VOL_RE = re.compile(r"Volume\s+(\d+)", re.I)
_CHAP_RE = re.compile(r"Chapter\s+(\d+)", re.I)
_SEC_RE = re.compile(r"(?:Section\s+)?(\d{4,6})", re.I)


# ---------------------------------------------------------------------------
# Clause ID detection
# ---------------------------------------------------------------------------

def detect_clause_ids(text: str) -> List[str]:
    """Return unique DoD clause IDs found in the first 3000 chars of text, in order of appearance."""
    if not (text or "").strip():
        return []
    seen_set: set = set()
    seen: List[str] = []
    for m in _CLAUSE_RE.finditer(text[:3000]):
        cid = m.group(1)
        if cid not in seen_set:
            seen_set.add(cid)
            seen.append(cid)
    return seen


def dominant_clause_id(text: str) -> Optional[str]:
    """Return the most prominent clause ID (highest frequency in first 3000 chars).

    Falls back to first-found when all frequencies are equal.
    Ignores plain section codes that appear as headings (no subclause suffix)
    to avoid over-tagging with the section number already stored in section_id.
    """
    if not (text or "").strip():
        return None
    all_ids = [m.group(1) for m in _CLAUSE_RE.finditer(text[:3000])]
    if not all_ids:
        return None
    c = Counter(all_ids)
    # Prefer IDs with a subclause suffix (.A, .A.1, .1) when present
    subclauses = [cid for cid in c if "." in cid]
    if subclauses:
        sub_counter = Counter({k: v for k, v in c.items() if "." in k})
        return sub_counter.most_common(1)[0][0]
    return c.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Retrieval-text builder (heading path + clause label + content)
# ---------------------------------------------------------------------------

def build_retrieval_text(
    text: str,
    section_path: Optional[str] = None,
    clause_id: Optional[str] = None,
    section_title: Optional[str] = None,
    page_number: Optional[int] = None,
    doc_type: str = "BOOK",
) -> str:
    """Build enriched embedding text: prepend document type, heading path, clause, page.

    Format:
        [Document Type: BOOK]
        [Volume 5 > Chapter 3 > Section 030201 PURPOSE]
        [Clause 030201.A]
        [Page 142]
        <raw chunk text>

    The returned text is used ONLY for embedding/BM25 indexing.
    The original `text` is stored separately as display_text for evidence output.
    """
    parts: List[str] = []
    if doc_type and str(doc_type).strip():
        parts.append(f"[Document Type: {str(doc_type).strip()}]")
    if section_path and section_path.strip():
        label = section_path.strip()
        if section_title and section_title.strip() and section_title not in label:
            label = f"{label} – {section_title.strip()}"
        parts.append(f"[{label}]")
    if clause_id and clause_id.strip():
        parts.append(f"[Clause {clause_id.strip()}]")
    if page_number is not None and page_number > 0:
        parts.append(f"[Page {page_number}]")
    if parts:
        return "\n".join(parts) + "\n" + (text or "")
    return text or ""


# ---------------------------------------------------------------------------
# Canonical ID generator
# ---------------------------------------------------------------------------

def build_canonical_id(
    section_path: Optional[str],
    page: Optional[int] = None,
    chunk_index: Optional[int] = None,
    clause_id: Optional[str] = None,
) -> Optional[str]:
    """Build a deterministic canonical ID such as:
        vol_05_ch_03_sec_030201_page_0142_chunk_02
        vol_05_ch_03_sec_030201_clause_A
        vol_05_ch_03_sec_030201_clause_A_page_0142

    Returns None when section_path lacks both Volume and Chapter (prevents false IDs).
    """
    if not section_path:
        return None

    vol_m = _VOL_RE.search(section_path)
    chap_m = _CHAP_RE.search(section_path)
    if not (vol_m and chap_m):
        return None

    vol_num = vol_m.group(1).zfill(2)
    chap_num = chap_m.group(1).zfill(2)

    # Extract section code — prefer last distinct 4–6-digit code in path
    sec_codes = _SEC_RE.findall(section_path)
    sec_num = sec_codes[-1].zfill(6) if sec_codes else "000000"

    parts = [f"vol_{vol_num}", f"ch_{chap_num}", f"sec_{sec_num}"]

    if clause_id and clause_id.strip():
        # Normalize: "030201.A" → "clause_A"; "030201.A.1" → "clause_A_1"
        suffix = clause_id.split(".", 1)[-1] if "." in clause_id else clause_id
        safe = re.sub(r"[^A-Za-z0-9]", "_", suffix)
        parts.append(f"clause_{safe}")

    if page is not None:
        parts.append(f"page_{str(page).zfill(4)}")

    if chunk_index is not None:
        parts.append(f"chunk_{str(chunk_index).zfill(2)}")

    return "_".join(parts)


# ---------------------------------------------------------------------------
# Table / list heuristics (used by parse.py and page_index.py)
# ---------------------------------------------------------------------------

# Lines that look like table rows: multiple tab/pipe separators or repeated spaces with short cells
_TABLE_ROW_RE = re.compile(r"(\t|\s{3,}|\|).*(\t|\s{3,}|\|)", re.MULTILINE)
_TABLE_HEADER_RE = re.compile(
    r"\b(table|fig(?:ure)?|exhibit|schedule|matrix|appendix)\s+[A-Z0-9\-]{1,10}\b",
    re.I,
)

# Bullet / numbered list line
_LIST_LINE_RE = re.compile(r"^\s*(?:[•\-\*]|\d+[.)]\s|[a-z][.)]\s)", re.MULTILINE)


def has_table_heuristic(text: str) -> bool:
    """True when text likely contains a table (tab-separated cells, pipe chars, or table header)."""
    if not text:
        return False
    if _TABLE_HEADER_RE.search(text[:500]):
        return True
    sample = text[:2000]
    tab_lines = sum(1 for line in sample.splitlines() if "\t" in line)
    pipe_lines = sum(1 for line in sample.splitlines() if "|" in line)
    return tab_lines >= 3 or pipe_lines >= 3


def has_list_heuristic(text: str) -> bool:
    """True when text contains a bullet or numbered list."""
    return bool(_LIST_LINE_RE.search((text or "")[:2000]))
