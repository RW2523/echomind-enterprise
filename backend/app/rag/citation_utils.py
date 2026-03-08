"""
Citation accuracy, inference transparency, and response structuring utilities for RAG.

Provides:
- get_section_content: Retrieve content from a section by path/code
- suggest_related_section: Find related sections for a query
- improve_citation_accuracy: Ensure responses include explicit section references
- improve_inference_transparency: Clarify when information is inferred
- compare_sections: Format side-by-side comparison of two sections
- structure_procedural_steps: Format steps as numbered list
- handle_missing_information: Suggest alternatives when info is not found
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

from ..core.db import get_conn
from .index import index
from .book.section_id import extract_all_codes, canonicalize_section_id


def _section_path_matches(ref_path: str, chunk_path: str) -> bool:
    """True if ref_path matches chunk_path (exact or normalized)."""
    if not chunk_path or not ref_path:
        return False
    ref_path = ref_path.strip()
    chunk_path = chunk_path.strip()
    if chunk_path == ref_path:
        return True
    segments = [s.strip() for s in chunk_path.split(">")]
    for seg in segments:
        if seg == ref_path:
            return True
        if seg.startswith(ref_path + " "):
            return True
        if seg.endswith(" " + ref_path) or seg.endswith(ref_path):
            return True
    if len(ref_path) >= 3 and re.search(r"\b" + re.escape(ref_path) + r"\b", chunk_path):
        return True
    return False


def get_section_content(
    section_reference: str,
    doc_ids: Optional[List[str]] = None,
    max_chars: int = 4000,
) -> Optional[str]:
    """
    Retrieve content from a section by path or code (e.g. "0301", "Volume 1 > Chapter 3 > 0301").

    Tries book_sections.full_section_text first, then falls back to chunk text.
    Returns None if section not found.
    """
    if not section_reference or not section_reference.strip():
        return None
    ref = section_reference.strip()

    # 1. Try book_sections table (full section text)
    try:
        with get_conn() as conn:
            if doc_ids:
                placeholders = ",".join("?" for _ in doc_ids)
                rows = conn.execute(
                    f"SELECT section_path, full_section_text FROM book_sections "
                    f"WHERE full_section_text IS NOT NULL AND full_section_text != '' "
                    f"AND doc_id IN ({placeholders})",
                    (*doc_ids,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT section_path, full_section_text FROM book_sections "
                    "WHERE full_section_text IS NOT NULL AND full_section_text != ''",
                ).fetchall()
    except Exception:
        rows = []

    ref_canon = canonicalize_section_id(ref)
    for row in rows:
        sp, text = (row[0] or "").strip(), (row[1] or "").strip()
        if not sp or not text:
            continue
        if _section_path_matches(ref, sp):
            return (text[:max_chars] + "…") if len(text) > max_chars else text
        # Match DoD code in section text (e.g. ref "030201" when section_path is "Segment 64")
        if ref_canon and ref_canon in extract_all_codes(text):
            return (text[:max_chars] + "…") if len(text) > max_chars else text

    # 2. Fallback: chunks from DB (avoids circular import with advanced)
    try:
        with get_conn() as conn:
            if doc_ids:
                placeholders = ",".join("?" for _ in doc_ids)
                rows = conn.execute(
                    f"SELECT c.text, c.source_json FROM chunks c "
                    f"WHERE c.doc_id IN ({placeholders}) "
                    f"AND json_extract(c.source_json, '$.section_path') IS NOT NULL",
                    (*doc_ids,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT c.text, c.source_json FROM chunks c "
                    "WHERE json_extract(c.source_json, '$.section_path') IS NOT NULL",
                ).fetchall()
    except Exception:
        return None
    ref_canon = canonicalize_section_id(ref)
    for row in rows:
        src = json.loads(row[1]) if isinstance(row[1], str) else row[1]
        chunk_path = (src or {}).get("section_path") or ""
        chunk_text = (row[0] or "").strip()
        if _section_path_matches(ref, chunk_path):
            if chunk_text:
                return (chunk_text[:max_chars] + "…") if len(chunk_text) > max_chars else chunk_text
        # Match DoD code in chunk text when path is "Segment N"
        if ref_canon and chunk_text and ref_canon in extract_all_codes(chunk_text):
            return (chunk_text[:max_chars] + "…") if len(chunk_text) > max_chars else chunk_text
    return None


async def suggest_related_section(
    query: str,
    k: int = 3,
    doc_ids: Optional[List[str]] = None,
) -> List[str]:
    """
    Suggest related section paths for a query using section-level semantic search.

    Returns list of section_path strings, e.g. ["Volume 1 > Chapter 3 > 0301", ...].
    """
    if not query or not query.strip():
        return []
    try:
        results = await index.section_index.search(query, k=k, doc_ids=doc_ids)
        paths = []
        seen = set()
        for r in results:
            sp = (r.get("section_path") or "").strip()
            if sp and sp not in seen:
                paths.append(sp)
                seen.add(sp)
        return paths
    except Exception:
        return []


def improve_citation_accuracy(
    response: str,
    section_reference: str,
    doc_ids: Optional[List[str]] = None,
) -> str:
    """
    Ensure the response explicitly references the section. If section_reference
    appears in the response, prepend with quoted content. Otherwise, append
    inferred notice and suggest the section.
    """
    if not response or not response.strip():
        return response
    if not section_reference or not section_reference.strip():
        return response

    content = get_section_content(section_reference, doc_ids=doc_ids, max_chars=800)
    ref = section_reference.strip()

    if ref in response or re.search(r"\b" + re.escape(ref) + r"\b", response):
        return response

    if content:
        return f"{response}\n\n*See also Section {ref}: {content[:400]}*"
    return f"{response}\n\n*See also: Section {ref}.*"


def is_inferred(response: str) -> bool:
    """Heuristic: True if the response *opens* with a hedging/uncertainty disclaimer.

    Only triggers on phrases that appear in the first ~300 chars (the opening statement).
    Avoids false positives on well-cited answers that naturally say "based on Section X" or
    "suggest" mid-answer.
    """
    if not response or not response.strip():
        return False
    opening = response.strip()[:300].lower()
    strong_signals = [
        "i couldn't find a direct",
        "i could not find",
        "no direct definition",
        "not directly found in",
        "unable to find the requested",
        "the information is not available",
        "i don't have enough context",
        "the provided context does not contain",
        "this information is inferred",
    ]
    return any(p in opening for p in strong_signals)


def _build_inference_notice(
    response: str,
    related: List[str],
    query: Optional[str] = None,
) -> str:
    """Append a brief notice pointing to related sections when the answer is uncertain."""
    if related:
        refs = ", ".join(related[:3])
        notice = f"\n\n---\n*For more detail, see: {refs}.*"
    else:
        notice = ""
    return response + notice


def improve_inference_transparency(
    response: str,
    related_sections: Optional[List[str]] = None,
    query: Optional[str] = None,
) -> str:
    """
    When the response is inferred, prepend a notice and suggest related sections.

    Pass related_sections from the caller (e.g. from enriched hits). For async
    section lookup, use improve_inference_transparency_async.
    """
    if not is_inferred(response):
        return response
    related = related_sections or []
    return _build_inference_notice(response, related, query=query)


async def improve_inference_transparency_async(
    query: str,
    response: str,
    doc_ids: Optional[List[str]] = None,
) -> str:
    """Async version: fetches related sections and prepends inference notice with query context."""
    if not is_inferred(response):
        return response
    related = await suggest_related_section(query, k=3, doc_ids=doc_ids)
    return _build_inference_notice(response, related, query=query)


def structure_comparison_response(
    section_1_label: str,
    section_1_content: str,
    section_2_label: str,
    section_2_content: str,
    differences: Optional[str] = None,
) -> str:
    """
    Format a side-by-side comparison with optional key differences.

    Use for cross-section queries. differences can be bullet points or a summary.
    """
    d = differences or "Compare the above sections for specific differences."
    return (
        f"**Section {section_1_label}:**\n{section_1_content}\n\n"
        f"**Section {section_2_label}:**\n{section_2_content}\n\n"
        f"**Key Differences:**\n{d}"
    )


def compare_sections(
    section_1: str,
    section_2: str,
    doc_ids: Optional[List[str]] = None,
    max_chars_per_section: int = 2000,
    differences: Optional[str] = None,
) -> str:
    """
    Return a formatted comparison of two sections.

    Retrieves content via get_section_content and formats with structure_comparison_response.
    For key differences, pass differences or use compare_sections_async with an extractor.
    """
    c1 = get_section_content(section_1, doc_ids=doc_ids, max_chars=max_chars_per_section)
    c2 = get_section_content(section_2, doc_ids=doc_ids, max_chars=max_chars_per_section)
    c1 = c1 or "(No content retrieved for this section)"
    c2 = c2 or "(No content retrieved for this section)"
    return structure_comparison_response(
        section_1, c1, section_2, c2, differences=differences
    )


def structure_complex_query(
    query_type: str,
    response_data: dict,
    doc_ids: Optional[List[str]] = None,
) -> str:
    """
    Route complex queries to appropriate formatters.

    query_type: "procedural" | "comparison" | "citation" | "definition" | "conceptual"
    response_data: dict with keys per type:
      - procedural: steps (list), section_reference (optional)
      - comparison: section_1, section_2
      - else: returns response_data.get("response", "") as-is
    """
    if query_type == "procedural":
        steps = response_data.get("steps") or []
        section_ref = response_data.get("section_reference")
        return structure_procedural_steps(steps, section_reference=section_ref)
    if query_type == "comparison":
        s1 = response_data.get("section_1") or ""
        s2 = response_data.get("section_2") or ""
        if s1 and s2:
            return compare_sections(s1, s2, doc_ids=doc_ids)
    return response_data.get("response", "")


def structure_procedural_steps(
    steps: List[str],
    section_reference: Optional[str] = None,
    step_section_refs: Optional[List[str]] = None,
) -> str:
    """
    Format a list of steps as a numbered list with section citations.

    Example: ["Do X", "Do Y"] -> "Step 1: Do X\nStep 2: Do Y"
    When section_reference is provided, appends " (see Section X for details)" to each step.
    When step_section_refs is provided (list of section refs per step), uses per-step citations.
    """
    if not steps:
        return ""
    refs = step_section_refs if step_section_refs and len(step_section_refs) >= len(steps) else None
    out = []
    for i, s in enumerate(steps):
        ref = (refs[i] if refs else section_reference) if section_reference or refs else None
        suffix = f" (see Section {ref} for details)" if ref else ""
        out.append(f"Step {i + 1}: {s}{suffix}")
    return "\n".join(out)


def information_missing(response: str) -> bool:
    """Heuristic: True if the response *opens* with an explicit statement that info is missing.

    Only checks the first ~300 chars to avoid false positives on answers that mention
    "not found" in a different context mid-answer.
    """
    if not response or not response.strip():
        return False
    opening = response.strip()[:300].lower()
    phrases = [
        "insufficient context",
        "i couldn't find",
        "i could not find",
        "the context does not contain",
        "not in the provided context",
        "the retrieved passages do not",
        "no relevant information was found",
    ]
    return any(p in opening for p in phrases)


def answer_grounding(response: str, section_path: str) -> str:
    """
    Ensure answers are grounded. If the response indicates inference, append a notice
    guiding the user to the section for full details.
    """
    if not response or not section_path:
        return response
    r = response.strip().lower()
    if "inferred" in r or "infer" in r or "not directly" in r or "not found" in r:
        return f"{response}\n\nThis information is inferred. For full details, refer to Section {section_path}."
    return response


async def handle_missing_information(
    query: str,
    doc_ids: Optional[List[str]] = None,
) -> str:
    """
    When information cannot be found, suggest related sections.

    Returns a fallback message with suggested sections.
    """
    related = await suggest_related_section(query, k=3, doc_ids=doc_ids)
    if related:
        refs = ", ".join(related[:3])
        return (
            f"Information not found directly in the queried section. "
            f"Please refer to {refs} for more details."
        )
    return (
        "Information not found directly in the queried section. "
        "Please refer to the document table of contents or index for related sections."
    )
