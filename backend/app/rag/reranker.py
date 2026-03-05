"""
Cross-encoder re-ranker for hybrid RAG results (Step 2).

Priority:
  1. sentence-transformers CrossEncoder (zero LLM latency, runs locally on CPU).
     Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~85MB, auto-downloaded on first use).
  2. LLM fallback (existing _rerank_hits logic passed in as llm_fallback_fn).

To disable the cross-encoder and always use the LLM fallback, set:
  RAG_USE_CE_RERANKER=0

Chunk text truncation: RAG_CE_MAX_CHARS (default 1024, up to 1500) for regulatory docs.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)

_USE_CE = os.getenv("RAG_USE_CE_RERANKER", "1").lower() in ("1", "true", "yes")


def _get_ce_model() -> str:
    """Cross-encoder model; supports RAG_CE_MODEL for domain-specific fine-tuned models."""
    try:
        from ..core.config import settings
        return getattr(settings, "RAG_CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        return os.getenv("RAG_CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

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
        model = _get_ce_model()
        _cross_encoder = CrossEncoder(model)
        logger.info("Reranker: CrossEncoder loaded (%s)", model)
    except Exception as e:
        logger.info("Reranker: CrossEncoder not available (%s) — LLM fallback will be used", e)
        _cross_encoder = None
    return _cross_encoder


def _get_ce_max_chars() -> int:
    """Max chars per chunk for cross-encoder input (configurable for regulatory docs)."""
    try:
        from ..core.config import settings
        return max(256, min(1500, getattr(settings, "RAG_CE_MAX_CHARS", 1024)))
    except Exception:
        return 1024


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
    max_chars = _get_ce_max_chars()
    pairs = [(query, (h.get("text") or "")[:max_chars]) for h in hits]
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
