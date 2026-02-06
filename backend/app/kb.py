"""
Knowledge base storage for transcripts: FAISS CPU + embeddings.
Incremental: embed and store only when paragraph is closed or user requests store.
Uses existing RAG index (Ollama/Nomic) for add_text and search.
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional
from .utils.ids import new_id, now_iso
from .rag.index import index as faiss_index


async def kb_add_text(text: str, metadata: Dict[str, Any]) -> str:
    """
    Add a text chunk to the knowledge base. Returns item_id.
    Metadata typically includes: session_id, paragraph_id, kind (raw|polished), tags, ts, conversation_type.
    """
    if not text or not text.strip():
        raise ValueError("Cannot add empty text to KB")
    item_id = new_id("kb")
    meta = {**metadata, "kb_id": item_id, "created_at": now_iso()}
    await faiss_index.add_text(f"transcript_{item_id}", text.strip(), meta)
    return item_id


async def kb_search(query: str, top_k: int = 8) -> List[tuple]:
    """
    Search the knowledge base. Returns list of (text, metadata, score).
    For later RAG use (citations).
    """
    hits = await faiss_index.search(query, top_k)
    return [(h["text"], h["source"], h["score"]) for h in hits]
