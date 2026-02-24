"""
Query intent classification: TRANSCRIPT_FIRST, DOCUMENT_FIRST, SUMMARIZE_DOC, GENERAL.
Deterministic rules: time/date/location -> TRANSCRIPT_FIRST; doc references -> DOCUMENT_FIRST; topic phrases -> TRANSCRIPT_FIRST; else DOCUMENT_FIRST.
"""
from __future__ import annotations
import re
from enum import Enum
from typing import Optional


class QueryIntent(str, Enum):
    TRANSCRIPT_FIRST = "transcript_first"
    DOCUMENT_FIRST = "document_first"
    SUMMARIZE_DOC = "summarize_doc"
    GENERAL = "general"


# Phrases that suggest transcript/temporal/location query
TRANSCRIPT_INDICATORS = [
    "when did", "when did we", "what did we say", "what was said",
    "at 2pm", "at 3pm", "yesterday", "last week", "last meeting",
    "in the meeting", "during the call", "in the transcript",
    "location", "where did we", "recent", "recently", "last hour",
    "transcript", "discussion", "meeting", "call", "conversation",
]

# Phrases that suggest document query
DOCUMENT_INDICATORS = [
    "document", "doc", "pdf", "file", "uploaded", "in the doc",
    "in the document", "chapter", "section", "page", "book",
    "summary of the doc", "summarize the document", "what does the document say",
]

# Summarize doc intent
SUMMARIZE_INDICATORS = [
    "summarize", "summary of", "overview of", "summarise",
]


def classify_intent(query: str) -> QueryIntent:
    """
    Classify query into TRANSCRIPT_FIRST, DOCUMENT_FIRST, SUMMARIZE_DOC, or GENERAL.
    """
    if not query or not query.strip():
        return QueryIntent.GENERAL
    q = query.strip().lower()
    if any(p in q for p in SUMMARIZE_INDICATORS) and any(p in q for p in DOCUMENT_INDICATORS):
        return QueryIntent.SUMMARIZE_DOC
    if any(p in q for p in TRANSCRIPT_INDICATORS):
        return QueryIntent.TRANSCRIPT_FIRST
    if any(p in q for p in DOCUMENT_INDICATORS):
        return QueryIntent.DOCUMENT_FIRST
    # Default: try documents first
    return QueryIntent.DOCUMENT_FIRST
