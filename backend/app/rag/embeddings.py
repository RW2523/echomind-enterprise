from __future__ import annotations
import collections
import logging
import numpy as np, httpx
from ..core.config import settings

logger = logging.getLogger(__name__)

_MIN_EMBED_CHARS = 500

# Query-embedding cache: single-text embeds (queries) are deterministic for a fixed model,
# so caching is safe (no staleness). Batch embeds (ingestion) are unique-heavy and skipped.
_QUERY_EMBED_CACHE: "collections.OrderedDict[str, list]" = collections.OrderedDict()
_QUERY_EMBED_CACHE_MAX = 4096


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


class OllamaEmbeddings:
    async def embed(self, texts: list[str]) -> np.ndarray:
        # Cache single-text (query) embeds; return a fresh array so callers can normalize in place
        # without mutating the cached vector.
        if len(texts) == 1:
            cached = _QUERY_EMBED_CACHE.get(texts[0])
            if cached is not None:
                _QUERY_EMBED_CACHE.move_to_end(texts[0])
                return np.array([cached], dtype=np.float32)
        async with httpx.AsyncClient(timeout=120) as client:
            vecs = []
            for t in texts:
                safe = _truncate_for_embed(t)
                vec = await self._embed_one(client, safe, len(t))
                vecs.append(vec)
        if len(texts) == 1:
            _QUERY_EMBED_CACHE[texts[0]] = vecs[0]
            _QUERY_EMBED_CACHE.move_to_end(texts[0])
            while len(_QUERY_EMBED_CACHE) > _QUERY_EMBED_CACHE_MAX:
                _QUERY_EMBED_CACHE.popitem(last=False)
        return np.array(vecs, dtype=np.float32)

    async def _embed_one(self, client: httpx.AsyncClient, text: str, original_len: int) -> list[float]:
        """Embed a single text, retrying with progressively shorter truncation on context overflow."""
        limit = len(text)
        for attempt in range(4):
            r = await client.post(
                settings.OLLAMA_EMBED_URL,
                json={"model": settings.OLLAMA_EMBED_MODEL, "prompt": text},
            )
            if r.status_code != 500 or "input length exceeds" not in (r.text or ""):
                r.raise_for_status()
                return r.json()["embedding"]
            limit = max(_MIN_EMBED_CHARS, limit // 2)
            text = _truncate_for_embed(text, max_chars=limit)
            logger.warning(
                "Embedding context overflow (original %d chars); retry %d with %d chars",
                original_len, attempt + 1, limit,
            )
        r.raise_for_status()
        return r.json()["embedding"]
