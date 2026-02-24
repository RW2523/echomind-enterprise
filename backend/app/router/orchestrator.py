"""
Orchestrator: fallback logic for RAG query.
1. Search transcript first (top_k=15, threshold=T1) when intent is TRANSCRIPT_FIRST.
2. If results < 15 or best_score < T_low -> search documents (top_k=15, threshold=T2).
3. If no results -> general response.
Collects evidence for citations.
"""
from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional
from ..core.config import settings
from .intent import classify_intent, QueryIntent
from .prompts import GENERAL_FALLBACK

logger = logging.getLogger(__name__)

# Thresholds
T1 = getattr(settings, "RAG_RELEVANCE_THRESHOLD", 0.45)
T_LOW = 0.35
TOP_K = getattr(settings, "TOP_K", 20)
TOP_K_TRANSCRIPT = 15
TOP_K_DOC = 15


async def search_transcript(query: str, filters: Optional[Dict] = None):
    """Search transcripts collection (Qdrant) or existing transcript index (FAISS)."""
    if filters is None:
        filters = {}
    try:
        from ..qdrant.client import is_qdrant_enabled
        from ..qdrant.search import transcript_search as qdrant_transcript_search
        if is_qdrant_enabled():
            return await qdrant_transcript_search(query, filters, top_k=TOP_K_TRANSCRIPT, threshold=T1)
    except Exception as e:
        logger.warning("Qdrant transcript search failed: %s", e)
    from ..rag.index import index
    hits = await index.search_transcript_only(query, TOP_K_TRANSCRIPT)
    return [{"score": h["score"], "text": h.get("text", ""), "payload": h.get("source", {}), "chunk_id": h.get("chunk_id")} for h in hits]


async def search_documents(query: str, filters: Optional[Dict] = None):
    """Search documents collection (Qdrant) or main FAISS index."""
    if filters is None:
        filters = {}
    try:
        from ..qdrant.client import is_qdrant_enabled
        from ..qdrant.search import document_search as qdrant_document_search
        if is_qdrant_enabled():
            return await qdrant_document_search(query, filters, top_k=TOP_K_DOC, threshold=T1)
    except Exception as e:
        logger.warning("Qdrant document search failed: %s", e)
    from ..rag.index import index
    hits = await index.search(query, TOP_K_DOC)
    return [{"score": h["score"], "text": h.get("text", ""), "payload": h.get("source", {}), "chunk_id": h.get("chunk_id")} for h in hits]


async def orchestrate(
    query: str,
    intent: Optional[QueryIntent] = None,
    filters: Optional[Dict] = None,
) -> tuple[List[Dict[str, Any]], str]:
    """
    Run retrieval with fallback. Returns (evidence_list, source_used).
    source_used: "transcript" | "document" | "both" | "none"
    """
    intent = intent or classify_intent(query)
    evidence = []
    source_used = "none"

    if intent == QueryIntent.TRANSCRIPT_FIRST:
        evidence = await search_transcript(query, filters)
        source_used = "transcript"
        best = max([e["score"] for e in evidence], default=0)
        if len(evidence) < 5 or best < T_LOW:
            doc_hits = await search_documents(query, filters)
            if doc_hits:
                evidence.extend(doc_hits)
                source_used = "both"
    else:
        evidence = await search_documents(query, filters)
        source_used = "document"
        best = max([e["score"] for e in evidence], default=0)
        if len(evidence) < 5 or best < T_LOW:
            trans_hits = await search_transcript(query, filters)
            if trans_hits:
                evidence.extend(trans_hits)
                source_used = "both"

    # Sort by score and cap
    evidence.sort(key=lambda x: -x["score"])
    evidence = evidence[: TOP_K]
    return evidence, source_used


async def answer_with_evidence(
    query: str,
    filters: Optional[Dict] = None,
    max_context_chunks: int = 10,
) -> Dict[str, Any]:
    """
    Full flow: orchestrate retrieval -> generate answer. Returns { answer, evidence[], source_used }.
    """
    evidence, source_used = await orchestrate(query, filters=filters)
    chunks_for_context = [{"text": e.get("text", ""), "source": e.get("payload", {})} for e in evidence]
    if not chunks_for_context or not any(c.get("text") for c in chunks_for_context):
        return {"answer": GENERAL_FALLBACK, "evidence": [], "source_used": "none"}
    from ..models.generator import generate_answer
    answer = await generate_answer(chunks_for_context, query, max_context_chunks=max_context_chunks)
    expose = getattr(settings, "RAG_EXPOSE_SOURCES", False)
    return {
        "answer": answer,
        "evidence": evidence if expose else [],
        "source_used": source_used,
    }
