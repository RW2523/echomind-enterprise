"""
Cross-encoder re-ranker for hybrid RAG results (Step 2).

Priority:
  1. sentence-transformers CrossEncoder (zero LLM latency, runs locally on CPU).
     Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~85MB, auto-downloaded on first use).
  2. LLM fallback (existing _rerank_hits logic passed in as llm_fallback_fn).

To disable the cross-encoder and always use the LLM fallback, set:
  RAG_USE_CE_RERANKER=0

The cross-encoder is loaded lazily once (singleton) to avoid model-load latency on
every rerank call.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)

_USE_CE = os.getenv("RAG_USE_CE_RERANKER", "1").lower() in ("1", "true", "yes")
_CE_MODEL = os.getenv("RAG_CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Lazy singleton: None = not loaded yet, False = unavailable
_cross_encoder: Any = None
_ce_attempted: bool = False


def _load_cross_encoder() -> Any:
    """Load (or return cached) sentence-transformers CrossEncoder.  Returns None on failure."""
    global _cross_encoder, _ce_attempted
    if _ce_attempted:
        return _cross_encoder
    _ce_attempted = True
    if not _USE_CE:
        logger.info("Reranker: cross-encoder disabled via RAG_USE_CE_RERANKER=0")
        return None
    try:
        from sentence_transformers import CrossEncoder  # type: ignore[import]
        _cross_encoder = CrossEncoder(_CE_MODEL)
        logger.info("Reranker: CrossEncoder loaded (%s)", _CE_MODEL)
    except Exception as e:
        logger.info("Reranker: CrossEncoder not available (%s) — LLM fallback will be used", e)
        _cross_encoder = None
    return _cross_encoder


async def rerank_hits(
    query: str,
    hits: List[Dict],
    top_k: int,
    llm_fallback_fn: Optional[Callable[..., Coroutine]] = None,
) -> List[Dict]:
    """Re-rank hits by (query, chunk_text) relevance and return top_k.

    Uses cross-encoder if available, otherwise calls llm_fallback_fn(query, hits, top_k).
    If neither is available, returns hits[:top_k] unchanged.
    """
    if not hits:
        return hits
    top_k = max(1, top_k)

    encoder = await asyncio.to_thread(_load_cross_encoder)
    if encoder is not None:
        logger.debug("Reranker: using cross-encoder for %d hits → top %d", len(hits), top_k)
        return await _rerank_ce(encoder, query, hits, top_k)

    if llm_fallback_fn is not None:
        logger.debug("Reranker: using LLM fallback for %d hits → top %d", len(hits), top_k)
        return await llm_fallback_fn(query, hits, top_k)

    return hits[:top_k]


async def _rerank_ce(encoder: Any, query: str, hits: List[Dict], top_k: int) -> List[Dict]:
    """Score (query, passage) pairs with cross-encoder; sort descending; return top_k."""
    # Truncate to 512 chars to stay within typical cross-encoder token limits
    pairs = [(query, (h.get("text") or "")[:512]) for h in hits]
    try:
        scores = await asyncio.to_thread(encoder.predict, pairs)
        scored = sorted(zip(scores, hits), key=lambda x: float(x[0]), reverse=True)
        return [
            {
                "chunk_id": h["chunk_id"],
                "score": float(s),
                "text": h["text"],
                "source": h["source"],
            }
            for s, h in scored[:top_k]
        ]
    except Exception as e:
        logger.warning("Reranker cross-encoder predict failed: %s — returning unranked top_k", e)
        return hits[:top_k]
