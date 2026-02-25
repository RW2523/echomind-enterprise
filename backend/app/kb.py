"""
Knowledge base storage for transcripts.
When RAG_PLATFORM_URL is set: sends chunks to RAG platform (transcript ingest).
Otherwise: uses existing FAISS index (Ollama/Nomic) for add_text and search.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Any, Optional

from .utils.ids import new_id, now_iso
from .rag.index import index as faiss_index
from .rag_platform_client import is_configured as rag_platform_configured, ingest_transcript_chunk as rag_ingest_chunk

logger = logging.getLogger(__name__)


async def kb_add_text(text: str, metadata: Dict[str, Any]) -> str:
    """
    Add a text chunk to the knowledge base. Returns item_id.
    Metadata typically includes: session_id, paragraph_id, kind (raw|refined), tags, ts, conversation_type.
    When RAG_PLATFORM_URL is set, chunks are sent to RAG platform under transcript_id ws_{session_id}.
    """
    if not text or not text.strip():
        raise ValueError("Cannot add empty text to KB")
    item_id = new_id("kb")

    if rag_platform_configured():
        try:
            session_id = (metadata.get("session_id") or "").strip() or new_id("ws")
            transcript_id = f"ws_{session_id}"
            tags = metadata.get("tags")
            if isinstance(tags, list):
                tags = [str(t) for t in tags][:20]
            else:
                tags = None
            await rag_ingest_chunk(
                transcript_id=transcript_id,
                text=text.strip(),
                ts=0,
                location=metadata.get("location"),
                tags=tags,
            )
        except Exception as e:
            logger.warning("RAG platform ingest failed for kb_add_text: %s", e)
        return item_id

    meta = {**metadata, "kb_id": item_id, "created_at": now_iso()}
    await faiss_index.add_text(f"transcript_{item_id}", text.strip(), meta)
    return item_id


async def kb_search(query: str, top_k: int = 8) -> List[tuple]:
    """
    Search the knowledge base. Returns list of (text, metadata, score).
    Only used when not RAG platform (RAG platform does retrieval inside /query).
    """
    if rag_platform_configured():
        return []  # No direct search; use /query for RAG
    hits = await faiss_index.search(query, top_k)
    return [(h["text"], h["source"], h["score"]) for h in hits]
