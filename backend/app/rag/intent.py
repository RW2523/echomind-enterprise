"""
Generic query intent classification and retrieval profiles.
Deterministic heuristics first; optional LLM refinement.
"""
from __future__ import annotations
import re
from typing import Dict, Any, List

# Intent labels (generic, not document-specific)
STRUCTURE = "structure"
FACTUAL = "factual"
PROCEDURAL = "procedural"
EXPLORATORY = "exploratory"

INTENTS = (STRUCTURE, FACTUAL, PROCEDURAL, EXPLORATORY)


# Deterministic patterns for STRUCTURE: TOC, headings, lists, counts
_STRUCTURE_PHRASES = (
    "table of contents", "toc", "list of chapters", "chapter list", "list the chapters",
    "what are the chapters", "chapters in", "contents of the", "book structure",
    "section list", "list sections", "list of sections", "chapter titles", "section titles",
    "list the sections", "list all headings", "list headings", "what sections",
    "how many chapters", "how many parts", "how many sections", "number of chapters",
    "number of sections", "list the parts", "parts of the", "headings in",
)
_STRUCTURE_WORDS = frozenset({
    "chapters", "sections", "headings", "contents", "toc", "parts", "structure",
    "list", "table", "index", "outline",
})
_STRUCTURE_QUESTION_STARTS = ("list", "how many", "what are the", "which chapters", "which sections", "name the")

# FACTUAL: specific fact, number, date, definition
_FACTUAL_PHRASES = ("what is ", "what was ", "define ", "definition of", "exact date", "exact number", "how much", "how many")
_FACTUAL_PATTERNS = (r"\bdate\b", r"\bnumber\b", r"\bwhen did\b", r"\bwho wrote\b", r"\bwho said\b")

# PROCEDURAL: how-to, steps
_PROCEDURAL_PHRASES = ("how do i", "how to", "how can i", "steps to", "steps for", "process for", "procedure", "walk me through")


def classify_query_intent(question: str) -> str:
    """
    Classify query intent using deterministic heuristics.
    Returns one of: structure, factual, procedural, exploratory.
    """
    if not (question or "").strip():
        return EXPLORATORY
    q = (question or "").strip().lower()
    words = set(re.findall(r"[a-z0-9]+", q))

    # STRUCTURE: TOC, chapters, sections, list, count
    if any(p in q for p in _STRUCTURE_PHRASES):
        return STRUCTURE
    if "list" in words and (words & {"chapter", "section", "heading", "part", "content"}):
        return STRUCTURE
    if "how many" in q and (words & {"chapter", "section", "part"}):
        return STRUCTURE
    if any(q.startswith(s) for s in _STRUCTURE_QUESTION_STARTS) and (words & _STRUCTURE_WORDS):
        return STRUCTURE

    # PROCEDURAL
    if any(p in q for p in _PROCEDURAL_PHRASES):
        return PROCEDURAL

    # FACTUAL: definition, specific fact, number, date
    if any(p in q for p in _FACTUAL_PHRASES):
        return FACTUAL
    if any(re.search(pat, q) for pat in _FACTUAL_PATTERNS):
        return FACTUAL

    return EXPLORATORY


# Retrieval profile defaults per intent (can be overridden by config)
def get_retrieval_profile(intent: str, base_k: int) -> Dict[str, Any]:
    """
    Return retrieval profile for the given intent.
    Keys: k_per_query, top_k, dense_weight, sparse_weight, rerank_candidates, rerank_top_n, relevance_threshold.
    """
    profiles = {
        STRUCTURE: {
            "k_per_query": 24,
            "top_k": 12,
            "dense_weight": 0.45,
            "sparse_weight": 0.55,
            "rerank_candidates": 16,
            "rerank_top_n": 10,
            "relevance_threshold": 0.35,
        },
        FACTUAL: {
            "k_per_query": 12,
            "top_k": 8,
            "dense_weight": 0.55,
            "sparse_weight": 0.45,
            "rerank_candidates": 12,
            "rerank_top_n": 8,
            "relevance_threshold": 0.45,
        },
        PROCEDURAL: {
            "k_per_query": 14,
            "top_k": 8,
            "dense_weight": 0.6,
            "sparse_weight": 0.4,
            "rerank_candidates": 12,
            "rerank_top_n": 8,
            "relevance_threshold": 0.45,
        },
        EXPLORATORY: {
            "k_per_query": max(base_k, 8),
            "top_k": base_k,
            "dense_weight": 0.6,
            "sparse_weight": 0.4,
            "rerank_candidates": max(base_k, 12),
            "rerank_top_n": base_k,
            "relevance_threshold": 0.45,
        },
    }
    return profiles.get(intent, profiles[EXPLORATORY]).copy()
