"""
Deterministic query-time routing: TRANSCRIPT_FIRST, DOCUMENT_FIRST, SUMMARIZE_DOC, GENERAL.
Rule set (production-friendly, no LLM for classification).
"""
from __future__ import annotations
import re
from typing import Literal

Intent = Literal["TRANSCRIPT_FIRST", "DOCUMENT_FIRST", "SUMMARIZE_DOC", "GENERAL"]

# Time/date/location language -> transcript first
TIME_LOC_PATTERNS = [
    r"last\s+\d+\s*(?:min(?:ute)?s?|hours?|days?)",
    r"today",
    r"yesterday",
    r"between\s+.+?\s+and\s+",
    r"recent\s+(?:transcript|conversation|meeting)",
    r"latest\s+(?:transcript|conversation)",
    r"this\s+(?:morning|afternoon|week)",
]

# Clear document intent
DOC_INTENT_PATTERNS = [
    r"\bin\s+the\s+book\b",
    r"\bchapter\b",
    r"\bpage\s+\d+",
    r"\bsummarize\s+(?:the\s+)?(?:book|document)\b",
    r"\bsearch\s+in\s+(?:document|book)\b",
    r"\b(?:from\s+)?(?:the\s+)?document\b",
    r"\b(?:from\s+)?(?:the\s+)?pdf\b",
]

# Topic phrase (not a question) -> transcript first
QUESTION_END = re.compile(r"\?\s*$")
WH_WORDS = re.compile(r"\b(what|which|when|where|who|how|why|can|does|is|are|do)\b", re.I)


def classify_intent(query: str) -> Intent:
    """
    Deterministic classification:
    1. Time/date/location language -> TRANSCRIPT_FIRST
    2. Else clear doc intent -> DOCUMENT_FIRST
    3. Else not a question / topic phrase -> TRANSCRIPT_FIRST
    4. Else -> DOCUMENT_FIRST (with transcript fallback in orchestrator)
    """
    q = (query or "").strip()
    if not q:
        return "GENERAL"
    q_lower = q.lower()

    # 1. Time/date/location
    for pat in TIME_LOC_PATTERNS:
        if re.search(pat, q_lower):
            return "TRANSCRIPT_FIRST"

    # 2. Clear document intent
    for pat in DOC_INTENT_PATTERNS:
        if re.search(pat, q_lower):
            return "DOCUMENT_FIRST"

    # 3. Summarize doc
    if re.search(r"\bsummarize\s+(?:the\s+)?(?:book|document)\b", q_lower):
        return "SUMMARIZE_DOC"

    # 4. Not a question / topic phrase -> transcript first
    if not QUESTION_END.search(q) and not WH_WORDS.search(q):
        return "TRANSCRIPT_FIRST"

    # 5. Default: document first (orchestrator will fallback to transcript if needed)
    return "DOCUMENT_FIRST"
