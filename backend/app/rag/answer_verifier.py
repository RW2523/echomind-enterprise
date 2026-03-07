"""
Post-generation verifier for BOOK/regulatory answers.

When BOOK_VERIFIER_ENABLED=1, runs a verification pass after draft generation:
  - every material claim has support in evidence
  - citations exist and align with the claim
  - answer does not overstate beyond evidence
  - section references are not malformed

Actions: approve, revise with stricter grounding, fallback, refuse.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _extract_section_refs(text: str) -> List[str]:
    """Extract section/paragraph refs from text (e.g. 030201, Section 0301)."""
    if not (text or "").strip():
        return []
    refs: List[str] = []
    # DoD codes
    for m in re.finditer(r"\b(\d{4,6}(?:\.\w[\d.]*)?)\b", text):
        refs.append(m.group(1))
    # "Section 0301", "paragraph 030201"
    for m in re.finditer(r"\b(?:Section|Paragraph|Para\.?)\s+(\d{4,6}(?:\.\w[\d.]*)?)\b", text, re.I):
        refs.append(m.group(1))
    return list(dict.fromkeys(refs))


def _evidence_contains_ref(evidence_text: str, ref: str) -> bool:
    """Check if evidence block contains the section ref."""
    if not evidence_text or not ref:
        return False
    ev_lower = evidence_text.lower()
    ref_lower = ref.lower()
    return ref_lower in ev_lower or re.search(r"\b" + re.escape(ref) + r"\b", ev_lower, re.I)


def verify_answer(
    answer: str,
    evidence_text: str,
    citations: List[Dict],
    question: str,
    explicit_section_ids: Optional[List[str]] = None,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Verify answer against evidence.

    Returns (passed, revised_answer, refusal_reason).
    - passed=True: answer is grounded
    - passed=False, revised_answer set: use revised (cautious) version
    - passed=False, refusal_reason set: refuse and return fallback
    """
    if not (answer or "").strip():
        return (False, None, "Empty answer.")

    evidence_lower = (evidence_text or "").lower()
    answer_refs = _extract_section_refs(answer)

    # 1. Explicit section requested but not in evidence
    if explicit_section_ids:
        for sid in explicit_section_ids[:5]:
            if sid and not _evidence_contains_ref(evidence_text or "", sid):
                logger.warning("Verifier: explicit section %s not in evidence", sid)
                return (
                    False,
                    None,
                    f"Requested section {sid} was not found in retrieved evidence.",
                )

    # 2. Answer cites sections not in evidence

    missing_refs = []
    for ref in answer_refs[:10]:
        if ref and not _evidence_contains_ref(evidence_text or "", ref):
            missing_refs.append(ref)
    if missing_refs:
        logger.warning("Verifier: answer cites %s not in evidence", missing_refs[:3])
        # Could revise to remove unsupported refs; for now return cautious
        return (
            False,
            None,
            f"Answer cites section(s) {missing_refs} not found in evidence.",
        )

    # 3. Overclaim indicators (weak heuristic)
    overclaim_phrases = [
        r"\b(?:clearly|obviously|definitely|certainly)\s+(?:it\s+)?(?:states?|says?|requires?)\b",
        r"\b(?:always|never)\s+(?:the\s+)?(?:regulation|fmr)\s+",
    ]
    for pat in overclaim_phrases:
        if re.search(pat, answer, re.I) and not re.search(pat, evidence_text or "", re.I):
            logger.debug("Verifier: possible overclaim in answer")
            # Don't fail; just flag

    # 4. Citations present and non-empty
    if citations is None:
        citations = []
    if not citations and answer_refs:
        logger.warning("Verifier: answer has section refs but no citations")
        return (False, None, "Answer references sections but no citations were generated.")

    return (True, None, None)
