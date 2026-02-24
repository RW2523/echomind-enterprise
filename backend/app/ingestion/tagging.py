"""
Tagging: generate topics/keywords per chunk for documents and transcripts.
Uses existing tagging heuristics; can be extended with LLM.
"""
from __future__ import annotations
from typing import List
from ..tagging import get_tags as _get_tags, get_conversation_type


def tag_chunk(text: str, max_tags: int = 12) -> List[str]:
    """Generate topic/keyword tags for a chunk. Returns list of strings."""
    return _get_tags(text or "", max_tags=max_tags)


def get_conversation_type_for_transcript(text: str) -> str:
    """Infer conversation type (meeting, lecture, interview, etc.) for transcript chunk."""
    return get_conversation_type(text or "")
