from __future__ import annotations
import asyncio
import re
import logging
import time
from collections import OrderedDict

import numpy as np
import httpx

from ..core.config import settings

logger = logging.getLogger(__name__)


def _normalize_query_text(text: str) -> str:
    """Normalize query for cache key: strip and collapse whitespace to single spaces."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", (text or "").strip())


def _truncate_for_embed(text: str, max_chars: int | None = None) -> str:
    """Truncate at word boundary so embedding API never exceeds context length."""
    limit = max_chars if max_chars is not None else settings.EMBED_MAX_CHARS
    if limit <= 0:
        limit = 2000
    if not text or len(text) <= limit:
        return text or ""
    truncated = text[:limit]
    last_space = truncated.rfind(" ")
    if last_space > limit // 2:
        return truncated[:last_space]
    return truncated


def _parse_embed_response(out: dict, index: int = 0) -> list[float]:
    """Parse Ollama embed response: 'embeddings' (list) or 'embedding' (single). Returns one vector as list of floats."""
    embeds = out.get("embeddings")
    if isinstance(embeds, list) and len(embeds) > index:
        vec = embeds[index]
        return list(vec) if vec is not None else []
    vec = out.get("embedding")
    if vec is not None and index == 0:
        return list(vec)
    return []


class OllamaEmbeddings:
    """
    Ollama embedding client. Single-query embeds use an in-memory LRU cache
    (key: OLLAMA_EMBED_MODEL + normalized query text) to reduce latency; chunk/batch
    embeds are not cached.
    """

    def __init__(self) -> None:
        self._cache: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
        self._cache_max = max(0, getattr(settings, "EMBED_QUERY_CACHE_SIZE", 2048))
        self._embed_http_calls = 0
        self._embed_cache_hits = 0
        self._lock = asyncio.Lock()

    async def embed(self, texts: list[str]) -> np.ndarray:
        # Single query path: use LRU cache when enabled (used by index.search and index.search_transcript_only).
        if len(texts) == 1 and self._cache_max > 0:
            return await self._embed_single_cached(texts[0])
        # Single query, cache disabled: one request (avoid batch path which would try batch + fallback = 2 requests).
        if len(texts) == 1:
            return await self._embed_single_uncached(texts[0])
        # Batch path: batched/concurrent requests (document ingestion, rebuilds). Preserves order; logs total time and emb/s.
        return await self._embed_batch(texts)

    async def _embed_single_uncached(self, text: str) -> np.ndarray:
        """One HTTP request for a single query when cache is disabled. Uses /api/embed with input."""
        safe = _truncate_for_embed(text)
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                settings.OLLAMA_EMBED_URL,
                json={"model": settings.OLLAMA_EMBED_MODEL, "input": safe},
            )
            r.raise_for_status()
            self._embed_http_calls += 1
            out = r.json()
            vec = _parse_embed_response(out, index=0)
        return np.array([vec], dtype=np.float32)

    async def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts in batches: try true batch API (input array), else concurrent single requests with semaphore."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        batch_size = max(1, getattr(settings, "EMBED_BATCH_SIZE", 64))
        concurrency = max(1, getattr(settings, "EMBED_CONCURRENCY", 4))
        safes = [_truncate_for_embed(t) for t in texts]
        start = time.perf_counter()
        all_vecs: list[list[float]] = []
        async with httpx.AsyncClient(timeout=120) as client:
            for i in range(0, len(safes), batch_size):
                batch_safes = safes[i : i + batch_size]
                batch_vecs = await self._embed_one_batch(client, batch_safes, concurrency)
                all_vecs.extend(batch_vecs)
        elapsed = time.perf_counter() - start
        n = len(all_vecs)
        rate = n / elapsed if elapsed > 0 else 0
        logger.info(
            "embed batch: total_texts=%d batch_size=%d total_time_sec=%.2f embeddings_per_sec=%.1f",
            n,
            batch_size,
            elapsed,
            rate,
        )
        return np.array(all_vecs, dtype=np.float32)

    async def _embed_one_batch(
        self,
        client: httpx.AsyncClient,
        batch_safes: list[str],
        concurrency: int,
    ) -> list[list[float]]:
        """Process one batch: try single request with input=[...]; on failure fall back to per-text requests. Order preserved."""
        # Try batch: /api/embed with {"input": [s1, s2, ...]} returns {"embeddings": [[...], [...]]}.
        try:
            t0 = time.perf_counter()
            r = await client.post(
                settings.OLLAMA_EMBED_URL,
                json={"model": settings.OLLAMA_EMBED_MODEL, "input": batch_safes},
            )
            r.raise_for_status()
            data = r.json()
            if "embeddings" in data and isinstance(data["embeddings"], list) and len(data["embeddings"]) == len(batch_safes):
                self._embed_http_calls += 1
                latency = time.perf_counter() - t0
                n = len(batch_safes)
                rate = n / latency if latency > 0 else 0
                logger.info(
                    "embed batch request: batch_size=%d latency_sec=%.3f embeddings_per_sec=%.1f",
                    n,
                    latency,
                    rate,
                )
                return [list(v) for v in data["embeddings"]]
        except (httpx.HTTPStatusError, KeyError, TypeError) as e:
            logger.debug("embed batch API failed (%s), falling back to per-text requests", e)
        # Fallback: per-text requests with "input" (string) for compatibility with servers that don't support batch.
        sem = asyncio.Semaphore(concurrency)

        async def one(safe: str) -> list[float]:
            async with sem:
                r = await client.post(
                    settings.OLLAMA_EMBED_URL,
                    json={"model": settings.OLLAMA_EMBED_MODEL, "input": safe},
                )
                r.raise_for_status()
                self._embed_http_calls += 1
                out = r.json()
                return _parse_embed_response(out, 0)

        results = await asyncio.gather(*[one(s) for s in batch_safes])
        return list(results)

    async def _embed_single_cached(self, text: str) -> np.ndarray:
        norm = _normalize_query_text(text)
        key = (settings.OLLAMA_EMBED_MODEL, norm)
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._embed_cache_hits += 1
                vec = self._cache[key].copy()
                logger.debug(
                    "embed query cache hit (embed_http_calls=%d embed_cache_hits=%d)",
                    self._embed_http_calls,
                    self._embed_cache_hits,
                )
                return np.array([vec], dtype=np.float32)
        safe = _truncate_for_embed(text)
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                settings.OLLAMA_EMBED_URL,
                json={"model": settings.OLLAMA_EMBED_MODEL, "input": safe},
            )
            r.raise_for_status()
            out = r.json()
            parsed = _parse_embed_response(out, 0)
        vec = np.array(parsed, dtype=np.float32) if parsed else np.zeros(0, dtype=np.float32)
        async with self._lock:
            self._embed_http_calls += 1
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = vec
            else:
                while len(self._cache) >= self._cache_max:
                    self._cache.popitem(last=False)
                self._cache[key] = vec
            logger.debug(
                "embed query cache miss (embed_http_calls=%d embed_cache_hits=%d)",
                self._embed_http_calls,
                self._embed_cache_hits,
            )
        return np.array([vec], dtype=np.float32)


async def run_embedding_sanity_check(emb: OllamaEmbeddings) -> bool:
    """Startup self-check: determinism, dimension, norms, endpoint/model, and optional batch [A,B,C]. Returns True if all checks pass."""
    tolerance = 1e-5
    test_str = "RAG sanity check"
    logger.info(
        "Embedding sanity check: endpoint=%s model=%s",
        getattr(settings, "OLLAMA_EMBED_URL", ""),
        getattr(settings, "OLLAMA_EMBED_MODEL", ""),
    )
    try:
        v1 = await emb.embed([test_str])
        v2 = await emb.embed([test_str])
        if v1.size == 0 or v2.size == 0:
            logger.warning("Embedding sanity check: empty vector returned")
            return False
        if v1.shape != v2.shape or not np.allclose(v1, v2, rtol=tolerance, atol=tolerance):
            logger.warning("Embedding sanity check FAILED: same-string vectors differ (determinism)")
            return False
        dim = v1.shape[1]
        norms = np.linalg.norm(v1, axis=1)
        nmin, navg, nmax = float(norms.min()), float(norms.mean()), float(norms.max())
        logger.info(
            "Embedding sanity check: dim=%d norm min=%.4f avg=%.4f max=%.4f",
            dim,
            nmin,
            navg,
            nmax,
        )
        batch_size = max(1, getattr(settings, "EMBED_BATCH_SIZE", 64))
        if batch_size >= 3:
            batch_vecs = await emb.embed(["A", "B", "C"])
            if batch_vecs.shape[0] != 3:
                logger.warning("Embedding sanity check FAILED: batch [A,B,C] returned count=%s (expected 3)", batch_vecs.shape[0])
                return False
            batch_norms = np.linalg.norm(batch_vecs, axis=1)
            logger.info(
                "Embedding sanity check: batch [A,B,C] count=3 norms=%s",
                [round(float(n), 4) for n in batch_norms],
            )
        logger.info("Embedding sanity check: passed")
        return True
    except Exception as e:
        logger.warning("Embedding sanity check FAILED: %s", e)
        return False
