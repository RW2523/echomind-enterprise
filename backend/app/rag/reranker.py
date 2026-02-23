"""
Optional cross-encoder reranker for RAG. Uses sentence-transformers CrossEncoder when available.
Reranks top candidates only; preserves hit dict shape (chunk_id, score, text, source).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Cap candidates passed to reranker to avoid OOM / slow inference
RERANK_MAX_CANDIDATES = 20

_CROSS_ENCODER = None
_CROSS_ENCODER_LOAD_ERROR = None


def _get_device() -> str:
    """Resolve RERANK_DEVICE: cuda if available else cpu."""
    try:
        from ..core.config import settings
        dev = (getattr(settings, "RERANK_DEVICE", None) or "") or ""
        if isinstance(dev, str):
            dev = dev.strip()
        if dev and dev.lower() in ("cuda", "cpu"):
            return dev.lower()
    except Exception:
        pass
    try:
        import torch
        return "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_cross_encoder():
    """Lazy-load CrossEncoder. Returns (model, None) or (None, error_msg)."""
    global _CROSS_ENCODER, _CROSS_ENCODER_LOAD_ERROR
    if _CROSS_ENCODER_LOAD_ERROR is not None:
        return None, _CROSS_ENCODER_LOAD_ERROR
    if _CROSS_ENCODER is not None:
        return _CROSS_ENCODER, None
    try:
        from sentence_transformers import CrossEncoder
        from ..core.config import settings
        model_name = getattr(settings, "RERANK_MODEL_NAME", "BAAI/bge-reranker-base") or "BAAI/bge-reranker-base"
        device = _get_device()
        _CROSS_ENCODER = CrossEncoder(model_name, device=device)
        logger.info("RAG cross-encoder reranker loaded: model=%s device=%s", model_name, device)
        return _CROSS_ENCODER, None
    except ImportError as e:
        _CROSS_ENCODER_LOAD_ERROR = str(e)
        logger.warning("RAG cross-encoder not available (install sentence-transformers): %s", e)
        return None, _CROSS_ENCODER_LOAD_ERROR
    except Exception as e:
        _CROSS_ENCODER_LOAD_ERROR = str(e)
        logger.warning("RAG cross-encoder load failed: %s", e)
        return None, _CROSS_ENCODER_LOAD_ERROR


def _normalize_score(raw: float) -> float:
    """Clamp score to [0,1] for consistency with other RAG scores. Preserves order."""
    try:
        return max(0.0, min(1.0, float(raw)))
    except (ValueError, TypeError):
        return 0.0


async def rerank(question: str, hits: List[Dict], top_n: int) -> List[Dict]:
    """
    Rerank hits by relevance to question using a local cross-encoder.
    Preserves hit dict shape (chunk_id, score, text, source); score is replaced with reranker score (0-1).
    If cross-encoder is unavailable or fails, returns hits[:top_n] unchanged.
    """
    if not hits or top_n <= 0:
        return hits
    # Only rerank top candidates (cap to avoid OOM)
    candidates = hits[: min(len(hits), RERANK_MAX_CANDIDATES)]
    if len(candidates) == 0:
        return hits[:top_n]
    model, err = _load_cross_encoder()
    if model is None:
        logger.debug("RAG rerank: cross-encoder unavailable (%s), returning order unchanged", err)
        return hits[:top_n]
    pairs = [(question, (h.get("text") or "")[:2000]) for h in candidates]
    loop = asyncio.get_running_loop()
    try:
        scores = await loop.run_in_executor(None, lambda: model.predict(pairs))
    except Exception as e:
        logger.warning("RAG cross-encoder predict failed: %s", e)
        return hits[:top_n]
    if scores is None or len(scores) != len(candidates):
        return hits[:top_n]
    # Normalize to 0-1 and sort by score descending
    scored = [(candidates[i], _normalize_score(float(scores[i]))) for i in range(len(candidates))]
    scored.sort(key=lambda x: -x[1])
    out = [
        {"chunk_id": h["chunk_id"], "score": s, "text": h["text"], "source": h["source"]}
        for h, s in scored[:top_n]
    ]
    return out
