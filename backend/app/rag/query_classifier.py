"""
Query-type classifier for adaptive dense/sparse RRF weighting.

Classifies queries as:
  citation    – asks for a specific paragraph/section/code (prefer sparse/BM25)
  definition  – asks what something means (prefer dense/semantic)
  procedural  – asks how to do something (moderately dense)
  conceptual  – general content question (most dense)

Weight tuning rationale for DoD FMR (regulatory text):
  citation:   BM25 wins because the user gives exact codes/numbers that appear verbatim.
  definition: Dense wins because meaning is distributed across sentences/paragraphs.
  procedural: Slightly dense-leaning; steps are often scattered across nearby paragraphs.
  conceptual: Strongly dense; topic is best captured by embedding similarity.
"""
from __future__ import annotations

import re
from typing import Tuple

# 4-6 digit code (e.g. 030201, 7001) typical for DoD paragraph numbering
_DOD_CODE_RE = re.compile(r"\b\d{4,6}(?:\.\d{1,4}){0,3}\b")

# Quoted phrase (user wraps exact text in quotes → keyword match is critical)
_QUOTED_RE = re.compile(r'"[^"]{3,}"')

# Procedural verbs / nouns
_PROCEDURAL_RE = re.compile(
    r"\b(how\s+to|how\s+do\s+i|how\s+should|procedure|procedures|process|"
    r"steps?\s+(?:to|for)|requirements?\s+for|must\s+(?:be\s+)?submitted?|"
    r"submit|report(?:ing)?|authorize|authorization|certif(?:y|ication)|"
    r"approve|approval|compute|calculate|compute)\b",
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


def classify_query_type(query: str) -> str:
    """Return one of 'citation' | 'definition' | 'procedural' | 'conceptual'."""
    t = (query or "").strip()
    t_lower = t.lower()

    if not t:
        return "conceptual"

    # --- Citation ---
    # DoD numeric codes (4–6 digits) appear verbatim → BM25 excels
    if _DOD_CODE_RE.search(t):
        return "citation"
    # Explicit "paragraph N", "section N.N", "subparagraph N"
    if re.search(r"\b(paragraph|section|subparagraph|subsection|volume)\s+\d", t_lower):
        return "citation"
    # User quoted exact phrase → treat as citation (keyword-first)
    if _QUOTED_RE.search(t):
        return "citation"

    # --- Definition ---
    if _DEFINITION_RE.search(t):
        return "definition"
    # "what is [a/an] <term>" (asking for a definition of a noun, not purpose/role/difference)
    # Exclude: "what is the purpose", "what is the role", "what is the difference", "what is the impact", etc.
    _DEFINITION_WHAT_IS_EXCLUSIONS = re.compile(
        r"\b(purpose|role|impact|difference|effect|implication|goal|objective|"
        r"process|procedure|requirement|use|used|used\s+for|meant|responsible|covered)\b",
        re.I,
    )
    if (
        re.match(r"^what\s+is\s+(?:a|an)\s+\w", t_lower)
        and not _DEFINITION_WHAT_IS_EXCLUSIONS.search(t_lower)
        and not _PROCEDURAL_RE.search(t)
    ):
        return "definition"

    # --- Procedural ---
    if _PROCEDURAL_RE.search(t):
        return "procedural"

    # --- Default: conceptual ---
    return "conceptual"


# Weight mapping: (dense_weight, sparse_weight)
_QUERY_WEIGHTS: dict[str, Tuple[float, float]] = {
    "citation":   (0.40, 0.60),
    "definition": (0.70, 0.30),
    "procedural": (0.65, 0.35),
    "conceptual": (0.75, 0.25),
}


def get_rrf_weights(query: str) -> Tuple[float, float]:
    """Return (dense_weight, sparse_weight) appropriate for the query type."""
    qt = classify_query_type(query)
    return _QUERY_WEIGHTS[qt]
