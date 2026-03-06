"""
AnswerGating: verify context + evidence before answering.

Two-stage gate:
  1. Evidence gate (evidence_gate.py): composite confidence scoring + concept coverage
  2. Context gate: explicit section id presence + key token count

Stops inferred/hallucinated answers and returns structured fallback with top sections.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _key_query_tokens(question: str, min_len: int = 3) -> List[str]:
    """Extract meaningful query tokens (exclude stopwords)."""
    if not (question or "").strip():
        return []
    stop = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below",
            "what", "which", "who", "whom", "whose", "where", "when", "why",
            "how", "that", "this", "these", "those", "it", "its",
            "about", "between", "under", "over", "not", "and", "or", "but"}
    tokens = re.findall(r"[a-z0-9]{2,}", (question or "").lower())
    return [t for t in tokens if t not in stop and len(t) >= min_len][:10]


def _context_contains_section_id(context: str, section_ids: List[str]) -> bool:
    if not section_ids or not context:
        return True
    ctx_lower = context.lower()
    for sid in section_ids:
        if sid and sid in ctx_lower:
            return True
        if re.search(r"\b" + re.escape(sid) + r"\b", ctx_lower):
            return True
    return False


def _context_contains_query_tokens(context: str, tokens: List[str], min_count: int = 2) -> bool:
    if not tokens or min_count <= 0:
        return True
    ctx_lower = context.lower()
    found = sum(1 for t in tokens if t in ctx_lower)
    return found >= min_count


def _build_fallback_msg(hits: List[Dict], base_msg: str) -> Tuple[str, List[Dict]]:
    """Build fallback message with top 3 closest sections and citations."""
    top3 = hits[:3] if hits else []
    if not top3:
        return (base_msg, top3)
    parts = [base_msg, "\nClosest sections found:"]
    for h in top3:
        src = h.get("source") or {}
        sp = src.get("section_path") or "(no section)"
        pn = src.get("page_number")
        vol = src.get("volume")
        ch = src.get("chapter")

        cite_parts = []
        if vol is not None:
            cite_parts.append(f"Volume {vol}")
        if ch is not None:
            cite_parts.append(f"Chapter {ch}")
        cite_parts.append(sp)
        if pn is not None:
            cite_parts.append(f"page {pn}")
        parts.append(f"  - {' → '.join(cite_parts)}")
    return ("\n".join(parts), top3)


def gate_context(
    question: str,
    context: str,
    hits: List[Dict],
    explicit_section_ids: Optional[List[str]] = None,
    min_query_tokens: int = 2,
) -> Tuple[bool, Optional[str], List[Dict]]:
    """Verify context before answering.

    Returns (pass_gate, fallback_message, top_sections_for_citation).
    """
    if not context or not context.strip():
        msg, top3 = _build_fallback_msg(hits, "Not found in retrieved passages.")
        return (False, msg, top3)

    # Check explicit section id (when present)
    if explicit_section_ids and not _context_contains_section_id(context, explicit_section_ids):
        msg, top3 = _build_fallback_msg(
            hits,
            "Relevant section not confidently identified in the regulation.",
        )
        logger.info(
            "AnswerGate: FAIL section_ids=%s not in context, q=%s",
            explicit_section_ids[:3], question[:80],
        )
        return (False, msg, top3)

    # Check concept coverage: require at least min_query_tokens key terms in context
    tokens = _key_query_tokens(question)
    if tokens and not _context_contains_query_tokens(context, tokens, min_count=min_query_tokens):
        msg, top3 = _build_fallback_msg(
            hits,
            "Unable to find strong supporting evidence in the regulation.",
        )
        ctx_lower = context.lower()
        found_tokens = [t for t in tokens if t in ctx_lower]
        logger.info(
            "AnswerGate: FAIL tokens (%d/%d) found=%s q=%s",
            len(found_tokens), len(tokens), found_tokens, question[:80],
        )
        return (False, msg, top3)

    return (True, None, [])
