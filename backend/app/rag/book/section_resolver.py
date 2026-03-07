"""
SectionResolver: deterministic navigation for explicit refs (Volume/Chapter/Section/paragraph codes).

Input: query text
Output: explicit_section_ids, explicit_volume/chapter, comparison_pair, resolved section_paths.

Used before retrieval to anchor search to requested sections when user mentions codes.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ...core.db import get_conn
from .section_id import (
    canonicalize_section_id,
    extract_all_codes,
    extract_section_id,
    section_id_from_path,
)
from .clause_parser import detect_clause_ids

logger = logging.getLogger(__name__)


def extract_explicit_refs(query: str) -> Dict[str, Any]:
    """Extract explicit section references from query.

    Returns:
        section_ids: list of canonical section ids (e.g. 030201, 0705)
        volumes: list of volume refs (e.g. 2, "2B")
        chapters: list of chapter refs (e.g. 8)
        is_comparison: True if query compares two sections
        pair: (section_a, section_b) when is_comparison, else []
    """
    q = (query or "").strip()
    out: Dict[str, Any] = {
        "section_ids": [],
        "volumes": [],
        "chapters": [],
        "is_comparison": False,
        "pair": [],
    }
    if not q:
        return out

    # Comparison: "0301 vs 0402", "compare 0301 and 0402"
    for pat in _COMPARISON_PATTERNS:
        m = pat.search(q)
        if m:
            a, b = canonicalize_section_id(m.group(1)), canonicalize_section_id(m.group(2))
            if a and b:
                out["is_comparison"] = True
                out["pair"] = [a, b]
                out["section_ids"] = list(dict.fromkeys([a, b]))
                return out

    # DoD codes (4-6 digit)
    for c in extract_all_codes(q):
        if c and c not in out["section_ids"]:
            out["section_ids"].append(c)

    # "Section 0705", "paragraph 030201"
    for m in _SECTION_LABEL_RE.finditer(q):
        sid = canonicalize_section_id(m.group(1))
        if sid and sid not in out["section_ids"]:
            out["section_ids"].append(sid)

    # Volume 2B, Chapter 8
    for m in _VOL_CH_RE.finditer(q):
        v = m.group(1)
        if v and v not in out["volumes"]:
            out["volumes"].append(v)
    for m in _CHAP_RE.finditer(q):
        c = m.group(1)
        if c and c not in out["chapters"]:
            out["chapters"].append(c)

    return out


# "Section 0705", "paragraph 030201", "030201.A", "030201.B.1"
_SECTION_LABEL_RE = re.compile(
    r"\b(?:Section|Sec\.?|Paragraph|Para\.?|Subparagraph|Subpara\.?)\s+([\d]{3,6}(?:\.\d{1,4}|\.\w[\d.]*)?)\b",
    re.I,
)

# "0301 vs 0402", "compare 0301 and 0402", "difference between 0301 and 0402"
_COMPARISON_PATTERNS = [
    re.compile(r"(\d{3,6}(?:\.\d{1,4}){0,3})\s+(?:vs\.?|versus)\s+(\d{3,6}(?:\.\d{1,4}){0,3})", re.I),
    re.compile(
        r"(?:compare|comparison|difference\s+between)\s+(\d{3,6}(?:\.\d{1,4}){0,3})\s+and\s+(\d{3,6}(?:\.\d{1,4}){0,3})",
        re.I,
    ),
]


# Volume/Chapter extraction (used in extract_explicit_refs and _build_section_id_to_paths)
_VOL_CH_RE = re.compile(r"\b(?:Volume|Vol\.?)\s+(\d+[A-Za-z]?)\b", re.I)
_CHAP_RE = re.compile(r"\b(?:Chapter|Chap\.?)\s+(\d+[A-Za-z]?)\b", re.I)


def _build_section_id_to_paths() -> Dict[str, List[str]]:
    """Build lookup: section_id -> [section_path, ...] from book_sections and chunks.

    Uses book_sections (section_path, section_title, full_section_text) and chunks (source_json)
    for coverage. Indexes DoD codes from full_section_text so "Segment 64" containing "030201"
    maps 030201 -> "Segment 64". Also indexes Volume/Chapter (e.g. vol2B, ch8, vol2B_ch8).
    """
    mapping: Dict[str, List[str]] = {}
    seen: Dict[str, set] = {}

    def add(sid: str, path: str) -> None:
        if not sid or not path:
            return
        path = path.strip()
        if sid not in seen:
            seen[sid] = set()
        if path not in seen[sid]:
            seen[sid].add(path)
            mapping.setdefault(sid, []).append(path)

    try:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT section_path, section_title, full_section_text FROM book_sections WHERE section_path IS NOT NULL"
            ).fetchall()
        for row in rows:
            path, title, full_text = (row[0] or "").strip(), row[1] or "", row[2] or ""
            sid = section_id_from_path(path) or extract_section_id(title) or extract_section_id(path)
            if sid:
                add(sid, path)
            # Index by path segments (e.g. "030201" in "Volume 1 > Chapter 3 > 030201 PURPOSE")
            for code in extract_all_codes(path):
                add(code, path)
            for code in extract_all_codes(title):
                add(code, path)
            # Index DoD codes from full_section_text (critical for "Segment N" docs that mention codes in body)
            for code in extract_all_codes(full_text):
                add(code, path)
            # Index Volume/Chapter for "Volume 2B", "Chapter 8" resolution
            vol_m = _VOL_CH_RE.search(path)
            chap_m = _CHAP_RE.search(path)
            if vol_m:
                v = vol_m.group(1)
                add(f"vol{v}", path)
            if chap_m:
                c = chap_m.group(1)
                add(f"ch{c}", path)
            if vol_m and chap_m:
                add(f"vol{vol_m.group(1)}_ch{chap_m.group(1)}", path)
    except Exception as e:
        logger.warning("SectionResolver: failed to build id->path from book_sections: %s", e)

    try:
        import json
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT source_json FROM chunks WHERE source_json IS NOT NULL"
            ).fetchall()
        for row in rows:
            try:
                src = json.loads(row[0]) if isinstance(row[0], str) else row[0]
            except Exception:
                continue
            path = (src.get("section_path") or "").strip()
            if not path:
                continue
            sid = section_id_from_path(path) or extract_section_id(path)
            if sid:
                add(sid, path)
            for code in extract_all_codes(path):
                add(code, path)
    except Exception as e:
        logger.warning("SectionResolver: failed to build id->path from chunks: %s", e)

    return mapping


class SectionResolver:
    """Resolve explicit section references in queries to canonical section_paths."""

    def __init__(self) -> None:
        self._id_to_paths: Dict[str, List[str]] = {}
        self._refresh()

    def _refresh(self) -> None:
        self._id_to_paths = _build_section_id_to_paths()
        logger.info("SectionResolver: loaded %d section_id -> path(s) mappings", len(self._id_to_paths))

    def resolve(self, query: str) -> "ResolveResult":
        """Parse query and resolve explicit refs. Call before retrieval."""
        return resolve_section_refs(query, self._id_to_paths)

    def resolve_to_paths(self, section_ids: List[str]) -> List[str]:
        """Resolve list of section_ids to section_paths (deduplicated)."""
        paths: List[str] = []
        seen: set = set()
        for sid in section_ids:
            sid = canonicalize_section_id(sid)
            for p in self._id_to_paths.get(sid, []):
                if p and p not in seen:
                    seen.add(p)
                    paths.append(p)
        return paths


class ResolveResult:
    """Result of resolving explicit refs in a query."""

    def __init__(
        self,
        explicit_section_ids: List[str],
        explicit_volume: Optional[int],
        explicit_chapter: Optional[int],
        comparison_pair: Optional[Tuple[str, str]],
        resolved_paths: List[str],
        has_explicit_refs: bool,
    ) -> None:
        self.explicit_section_ids = explicit_section_ids
        self.explicit_volume = explicit_volume
        self.explicit_chapter = explicit_chapter
        self.comparison_pair = comparison_pair
        self.resolved_paths = resolved_paths
        self.has_explicit_refs = has_explicit_refs


def resolve_section_refs(
    query: str,
    id_to_paths: Optional[Dict[str, List[str]]] = None,
) -> ResolveResult:
    """Parse query for explicit refs and resolve to section_paths.

    Returns ResolveResult with explicit_section_ids, comparison_pair, resolved_paths.
    """
    q = (query or "").strip()
    mapping = id_to_paths or _build_section_id_to_paths()

    explicit_ids: List[str] = []
    explicit_volume: Optional[int] = None
    explicit_chapter: Optional[int] = None
    comparison_pair: Optional[Tuple[str, str]] = None

    # Comparison: "0301 vs 0402" or "compare 0301 and 0402"
    for pat in _COMPARISON_PATTERNS:
        m = pat.search(q)
        if m:
            a, b = canonicalize_section_id(m.group(1)), canonicalize_section_id(m.group(2))
            if a and b:
                comparison_pair = (a, b)
                explicit_ids.extend([a, b])
                logger.info("SectionResolver: comparison pair %s vs %s", a, b)
            break

    # Extract all DoD codes (section-level: 030201, 030202)
    codes = extract_all_codes(q)
    for c in codes:
        if c and c not in explicit_ids:
            explicit_ids.append(c)

    # Extract clause-level refs (030201.A, 030201.B.1) and add base section
    clause_ids = detect_clause_ids(q)
    for cid in clause_ids:
        if cid and cid not in explicit_ids:
            explicit_ids.append(cid)
        base = cid.split(".")[0] if cid and "." in cid else None
        if base and base not in explicit_ids:
            explicit_ids.append(base)

    # Volume/Chapter if present ("Volume 2B", "Chapter 8")
    vol_m = re.search(r"\b(?:Volume|Vol\.?)\s+(\d+[A-Za-z]?)\b", q, re.I)
    if vol_m:
        try:
            explicit_volume = int(vol_m.group(1))
        except ValueError:
            explicit_volume = vol_m.group(1)  # e.g. "2B"
    chap_m = re.search(r"\b(?:Chapter|Chap\.?)\s+(\d+[A-Za-z]?)\b", q, re.I)
    if chap_m:
        try:
            explicit_chapter = int(chap_m.group(1))
        except ValueError:
            explicit_chapter = chap_m.group(1)  # e.g. "8"

    # Resolve ids to paths (section + clause; clause 030201.A falls back to base 030201)
    resolved: List[str] = []
    seen: set = set()
    for sid in explicit_ids:
        sid = canonicalize_section_id(sid)
        for p in mapping.get(sid, []):
            if p and p not in seen:
                seen.add(p)
                resolved.append(p)
        if "." in sid:
            base = sid.split(".")[0]
            for p in mapping.get(base, []):
                if p and p not in seen:
                    seen.add(p)
                    resolved.append(p)

    # Resolve Volume/Chapter to paths
    if explicit_volume is not None or explicit_chapter is not None:
        vol_key = f"vol{explicit_volume}" if explicit_volume is not None else None
        chap_key = f"ch{explicit_chapter}" if explicit_chapter is not None else None
        combined_key = f"vol{explicit_volume}_ch{explicit_chapter}" if vol_key and chap_key else None
        for key in [combined_key, vol_key, chap_key]:
            if key and key in mapping:
                for p in mapping[key]:
                    if p and p not in seen:
                        seen.add(p)
                        resolved.append(p)

    has_explicit = bool(explicit_ids or explicit_volume is not None or explicit_chapter is not None)
    return ResolveResult(
        explicit_section_ids=explicit_ids,
        explicit_volume=explicit_volume,
        explicit_chapter=explicit_chapter,
        comparison_pair=comparison_pair,
        resolved_paths=resolved,
        has_explicit_refs=has_explicit,
    )
