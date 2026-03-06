"""
Query-type classifier for adaptive dense/sparse RRF weighting.

Classifies queries as:
  citation    – asks for a specific paragraph/section/code (prefer sparse/BM25)
  definition  – asks what something means (prefer dense/semantic)
  procedural  – asks how to do something (moderately dense)
  conceptual  – general content question (most dense)

Supports multi-type queries (e.g. citation + procedural): uses blended weights.

Weight tuning for DoD FMR (regulatory text):
  citation:   BM25 wins — exact codes/numbers appear verbatim.
  definition: Dense wins — meaning distributed across sentences.
  procedural: BM25-leaning — regulatory phrases ("shall submit", "must certify").
  conceptual: Strongly dense — topic captured by embedding similarity.
"""
from __future__ import annotations

import re
from typing import List, Tuple

# 4-6 digit code (e.g. 030201, 7001) typical for DoD paragraph numbering
_DOD_CODE_RE = re.compile(r"\b\d{4,6}(?:\.\d{1,4}){0,3}\b")

# Quoted phrase (user wraps exact text in quotes → keyword match is critical)
_QUOTED_RE = re.compile(r'"[^"]{3,}"')

# Procedural verbs / nouns — expanded for regulatory queries
_PROCEDURAL_RE = re.compile(
    r"\b(how\s+to|how\s+do\s+i|how\s+should|procedure|procedures|process|"
    r"steps?\s+(?:to|for)|requirements?\s+for|requirements?\s+to|"
    r"must\s+(?:be\s+)?submitted?|submit|report(?:ing)?|authorize|authorization|"
    r"certif(?:y|ication)|approve|approval|compute|calculate|"
    r"list\s+(?:the\s+)?steps?|walk\s+me\s+through|checklist\s+for|"
    r"what\s+is\s+required\s+for|what\s+are\s+the\s+requirements|"
    r"required\s+to|required\s+for|compliance\s+with|"
    r"obtain|obtaining|request|requesting|file|filing)\b",
    re.I,
)

# Citation markers (paragraph, section, volume, etc.)
_CITATION_RE = re.compile(
    r"\b(paragraph|section|subparagraph|subsection|volume)\s+\d",
    re.I,
)

# Definition markers
_DEFINITION_RE = re.compile(
    r"\b(define|definition|what\s+is\s+(?:a\s+|an\s+|the\s+)?meaning\s+of|"
    r"meaning\s+of|what\s+does\s+\w[\w\s]{0,30}mean|explain\s+the\s+term|"
    r"what\s+is\s+a\b|what\s+is\s+an\b|what\s+are\s+the\s+terms|"
    r"glossary\s+of|defined\s+as)\b",
    re.I,
)

# "What is required for..." → procedural (not definition)
_WHAT_IS_REQUIRED_RE = re.compile(
    r"\bwhat\s+is\s+required\s+(?:for|to)\b",
    re.I,
)


def classify_query_type(query: str) -> str:
    """Return one of 'citation' | 'definition' | 'procedural' | 'conceptual'.

    For multi-type queries (citation + procedural), returns the dominant type
    that drives retrieval; get_rrf_weights() handles blending.
    """
    t = (query or "").strip()
    t_lower = t.lower()

    if not t:
        return "conceptual"

    has_citation = bool(
        _DOD_CODE_RE.search(t)
        or _CITATION_RE.search(t_lower)
        or _QUOTED_RE.search(t)
    )
    has_procedural = bool(_PROCEDURAL_RE.search(t))
    has_definition = bool(_DEFINITION_RE.search(t))

    # "What is required for..." → procedural (ambiguous "what is" but procedural intent)
    if _WHAT_IS_REQUIRED_RE.search(t_lower):
        has_procedural = True

    # Multi-type: citation + procedural → return primary for logging; get_rrf_weights blends
    if has_citation and has_procedural:
        return "citation"  # citation drives primary; weights are blended in get_rrf_weights
    if has_citation:
        return "citation"

    # Definition (exclude procedural "what is required")
    if has_definition and not has_procedural:
        _DEFINITION_WHAT_IS_EXCLUSIONS = re.compile(
            r"\b(purpose|role|impact|difference|effect|implication|goal|objective|"
            r"process|procedure|requirement|use|used|used\s+for|meant|responsible|covered)\b",
            re.I,
        )
        if (
            re.match(r"^what\s+is\s+(?:a|an)\s+\w", t_lower)
            and not _DEFINITION_WHAT_IS_EXCLUSIONS.search(t_lower)
        ):
            return "definition"
        if _DEFINITION_RE.search(t) and not _WHAT_IS_REQUIRED_RE.search(t_lower):
            return "definition"

    # Procedural
    if has_procedural:
        return "procedural"

    return "conceptual"


def classify_query_types(query: str) -> List[str]:
    """Return all applicable types for multi-type blending. Used by get_rrf_weights."""
    t = (query or "").strip()
    t_lower = t.lower()
    types: List[str] = []

    if not t:
        return ["conceptual"]

    if _DOD_CODE_RE.search(t) or _CITATION_RE.search(t_lower) or _QUOTED_RE.search(t):
        types.append("citation")
    if _PROCEDURAL_RE.search(t) or _WHAT_IS_REQUIRED_RE.search(t_lower):
        types.append("procedural")
    if _DEFINITION_RE.search(t) and not _WHAT_IS_REQUIRED_RE.search(t_lower):
        _DEFINITION_WHAT_IS_EXCLUSIONS = re.compile(
            r"\b(purpose|role|impact|difference|effect|implication|goal|objective|"
            r"process|procedure|requirement|use|used|used\s+for|meant|responsible|covered)\b",
            re.I,
        )
        if (
            re.match(r"^what\s+is\s+(?:a|an)\s+\w", t_lower)
            and not _DEFINITION_WHAT_IS_EXCLUSIONS.search(t_lower)
        ) or (_DEFINITION_RE.search(t)):
            types.append("definition")

    if not types:
        types.append("conceptual")
    return types


# Weight mapping: (dense_weight, sparse_weight)
# Tuned for DoD 7000.14-R: citation/procedural favor BM25 for regulatory phrases.
_QUERY_WEIGHTS: dict[str, Tuple[float, float]] = {
    "citation":   (0.30, 0.70),  # Strong BM25 for exact DoD codes
    "definition": (0.70, 0.30),
    "procedural": (0.60, 0.40),  # BM25-leaning for "steps", "submit", "filing"
    "conceptual": (0.75, 0.25),
}


def get_rrf_weights(query: str) -> Tuple[float, float]:
    """Return (dense_weight, sparse_weight) appropriate for the query type.

    Truly multi-label: always blend dense/sparse weights across all detected types.
    No single-type priority; averaging ensures balanced retrieval for mixed queries.
    """
    types = classify_query_types(query)
    if not types:
        return _QUERY_WEIGHTS["conceptual"]
    # Always blend across all detected types (no special-case priority)
    dense_sum = 0.0
    sparse_sum = 0.0
    for t in types:
        w = _QUERY_WEIGHTS.get(t, _QUERY_WEIGHTS["conceptual"])
        dense_sum += w[0]
        sparse_sum += w[1]
    n = len(types)
    blended = (dense_sum / n, sparse_sum / n)
    total = blended[0] + blended[1]
    if total > 0:
        return (blended[0] / total, blended[1] / total)
    return _QUERY_WEIGHTS["conceptual"]
