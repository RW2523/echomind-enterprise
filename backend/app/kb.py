"""
Knowledge base: FAISS CPU + embeddings. Incremental add and search.
Wraps existing RAG index for transcript storage with metadata (tags, conversation_type, paragraph_id).
"""
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

from .core.config import settings
from .utils.ids import new_id, now_iso

# Use existing RAG index (FAISS CPU + Ollama embeddings)
from .rag.index import index as faiss_index


class KnowledgeBase:
    """
    Add text chunks with metadata; search returns (text, metadata, score).
    Incremental: only embed/store when paragraph is closed or user requests store.
    """

    def __init__(self):
        self._index = faiss_index

    async def add_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        title_prefix: str = "transcript",
    ) -> str:
        """
        Store text as a single chunk in FAISS. Returns item_id (doc/chunk id from index).
        metadata can include: paragraph_id, tags, conversation_type, ts, session_id, kind (raw|polished).
        """
        if not text or not text.strip():
            raise ValueError("Cannot add empty text")
        meta = dict(metadata or {})
        meta.setdefault("created_at", now_iso())
        meta.setdefault("source", "realtime_transcript")
        title = f"{title_prefix}_{new_id('x')}"
        try:
            res = await self._index.add_text(title, text.strip(), meta)
            return res.get("doc_id", title)
        except Exception as e:
            raise RuntimeError(f"KB add_text failed: {e}") from e

    async def search(self, query: str, top_k: int = 10) -> List[tuple]:
        """
        Search FAISS. Returns list of (text, metadata, score).
        """
        try:
            hits = await self._index.search(query, top_k)
            return [(h["text"], h["source"], h["score"]) for h in hits]
        except Exception as e:
            return []


# Singleton for use in WS handler
_kb: Optional[KnowledgeBase] = None


def get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb
