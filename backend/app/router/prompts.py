"""
Prompts for query routing, clarification, and answer generation.
"""
from __future__ import annotations

SYSTEM_ANSWER_FROM_CONTEXT = (
    "Answer based only on the following context. "
    "If the context does not contain the answer, say so briefly. "
    "Cite sources when relevant."
)

USER_QUERY_TEMPLATE = "Context:\n{context}\n\nQuestion: {query}"

GENERAL_FALLBACK = "I couldn't find relevant information to answer that question."
