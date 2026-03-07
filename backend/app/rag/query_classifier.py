"""
Query-type classifier for adaptive dense/sparse RRF weighting.

Classifies queries as:
  citation    – asks for a specific paragraph/section/code (prefer sparse/BM25)
  definition  – asks what something means (prefer dense/semantic)
  procedural  – asks how to do something (moderately dense)
  conceptual  – general content question (most dense)
  threshold   – asks about a numeric threshold, limit, minimum, maximum, deadline, exception
  table       – asks about a table, matrix, schedule, rates, or structured numeric list
  comparison  – asks to compare two sections/rules/policies

Supports multi-type queries (e.g. citation + procedural): uses blended weights.

Weight tuning for DoD FMR (regulatory text):
  citation:   BM25 wins — exact codes/numbers appear verbatim.
  definition: Dense wins — meaning distributed across sentences.
  procedural: BM25-leaning — regulatory phrases ("shall submit", "must certify").
  conceptual: Strongly dense — topic captured by embedding similarity.
  threshold:  Heavy BM25 — numeric limits usually appear verbatim in clauses.
  table:      Balanced, slightly BM25-leaning — table titles and captions are exact.
  comparison: Dense — conceptual matching across multiple sections.

BookRAG-lite++ additions (threshold, table, comparison types):
  - threshold: biases retrieval toward clause chunks and exact sparse matches.
  - table: biases toward table chunks; also used in advanced.py to boost has_table chunks.
  - comparison: used in advanced.py to relax MAX_SECTIONS_PER_ANSWER limit.
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

# Threshold / exception queries: numeric limits, deadlines, minimums, maximums, exceptions
_THRESHOLD_RE = re.compile(
    r"\b(threshold|minimum|maximum|limit|deadline|ceiling|floor|cap|rate|percent(?:age)?|"
    r"days?\s+(?:after|within|before|to|from)|how\s+many\s+days?|"
    r"how\s+much|at\s+least|no\s+more\s+than|not\s+to\s+exceed|"
    r"exception(?:s)?|waiver|penalty|fine|interest\s+rate|surcharge|"
    r"over\s+(?:age|years|days)|under\s+(?:age|years|days)|"
    r"allowable|deductible|reimburs(?:e|able|ment))\b",
    re.I,
)

# Table lookup queries
_TABLE_RE = re.compile(
    r"\b(table|schedule|matrix|exhibit|appendix|rate\s+table|fee\s+schedule|"
    r"pay\s+table|salary\s+table|grade\s+table|rates?|chart|grid|listing|"
    r"in\s+(?:table|exhibit|appendix)\s+[A-Z0-9]|table\s+[A-Z0-9\-]{1,5})\b",
    re.I,
)

# Comparison query markers
_COMPARISON_RE = re.compile(
    r"\b(compare|comparison|differ(?:ence)?|versus|vs\.?|contrast|"
    r"what\s+is\s+the\s+difference|how\s+(?:does|do|is|are)\s+.{0,40}differ|"
    r"distinguish|distinction)\b",
    re.I,
)

# Phrase-style / exact-wording queries: BM25-heavy for verbatim match
_PHRASE_RE = re.compile(
    r"\b(where\s+does\s+(?:it\s+)?say|exact\s+word(?:ing)?|exact\s+sentence|"
    r"show\s+(?:the\s+)?exact|mention(?:s|ed)?\s+(?:that|the)|"
    r"word(?:ing|ed)\s+(?:in|of)|verbatim|quote|quoted|"
    r"in\s+(?:the\s+)?(?:exact\s+)?words?|"
    r"what\s+(?:are\s+)?the\s+exact\s+words)\b",
    re.I,
)


def classify_query_type(query: str) -> str:
    """Return the primary query type for logging/routing.

    Priority order: citation > threshold > table > comparison > definition > procedural > conceptual.
    Multi-type blending is handled separately by get_rrf_weights().
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
    has_threshold = bool(_THRESHOLD_RE.search(t))
    has_table = bool(_TABLE_RE.search(t))
    has_comparison = bool(_COMPARISON_RE.search(t))
    has_phrase = bool(_PHRASE_RE.search(t) or _QUOTED_RE.search(t))

    # "What is required for..." → procedural (ambiguous "what is" but procedural intent)
    if _WHAT_IS_REQUIRED_RE.search(t_lower):
        has_procedural = True

    # Explicit citation/code always wins primary
    if has_citation:
        return "citation"

    # Phrase-style: "where does it say", quoted text, exact wording → BM25-heavy
    if has_phrase:
        return "phrase"

    # Threshold and table are high-precision regulatory types
    if has_threshold:
        return "threshold"
    if has_table:
        return "table"
    if has_comparison:
        return "comparison"

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
    # BookRAG-lite++ types
    if _THRESHOLD_RE.search(t):
        types.append("threshold")
    if _TABLE_RE.search(t):
        types.append("table")
    if _COMPARISON_RE.search(t):
        types.append("comparison")
    if _PHRASE_RE.search(t) or _QUOTED_RE.search(t):
        types.append("phrase")

    if not types:
        types.append("conceptual")
    return types


def is_threshold_query(query: str) -> bool:
    """True when query is asking for a numeric threshold, limit, exception, or deadline."""
    return bool(_THRESHOLD_RE.search(query or ""))


def is_table_query(query: str) -> bool:
    """True when query explicitly references a table, schedule, or matrix."""
    return bool(_TABLE_RE.search(query or ""))


def is_comparison_query(query: str) -> bool:
    """True when query compares two sections, policies, or rules."""
    return bool(_COMPARISON_RE.search(query or ""))


def is_phrase_query(query: str) -> bool:
    """True when query asks for exact wording, quoted text, or 'where does it say'."""
    return bool(_PHRASE_RE.search(query or "") or _QUOTED_RE.search(query or ""))


# Weight mapping: (dense_weight, sparse_weight)
# Tuned for DoD 7000.14-R: citation/procedural favor BM25 for regulatory phrases.
_QUERY_WEIGHTS: dict[str, Tuple[float, float]] = {
    "citation":   (0.30, 0.70),  # Strong BM25 for exact DoD codes
    "definition": (0.70, 0.30),
    "procedural": (0.60, 0.40),  # BM25-leaning for "steps", "submit", "filing"
    "conceptual": (0.75, 0.25),
    # BookRAG-lite++ types
    "threshold":  (0.25, 0.75),  # Very strong BM25: numeric limits are verbatim in clauses
    "table":      (0.40, 0.60),  # BM25-leaning: table titles/captions are exact keywords
    "comparison": (0.70, 0.30),  # Dense: conceptual similarity across multiple sections
    "phrase":     (0.20, 0.80),  # Very strong BM25: "where does it say", quoted text, exact wording
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
