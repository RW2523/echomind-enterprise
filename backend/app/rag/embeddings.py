from __future__ import annotations
import asyncio
import numpy as np
import httpx
from ..core.config import settings

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


# Module-level shared client so connection pooling is reused across requests.
# httpx.AsyncClient is thread-safe for concurrent async usage.
_embed_client: httpx.AsyncClient | None = None


def _get_embed_client() -> httpx.AsyncClient:
    global _embed_client
    if _embed_client is None or _embed_client.is_closed:
        _embed_client = httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_connections=20, max_keepalive_connections=10))
    return _embed_client


class OllamaEmbeddings:
    async def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts concurrently (all requests fired in parallel)."""
        client = _get_embed_client()
        safe_texts = [_truncate_for_embed(t) for t in texts]

        async def _embed_one(text: str) -> list[float]:
            r = await client.post(
                settings.OLLAMA_EMBED_URL,
                json={"model": settings.OLLAMA_EMBED_MODEL, "prompt": text},
            )
            r.raise_for_status()
            return r.json()["embedding"]

        # Fire all embedding requests concurrently instead of serially
        results = await asyncio.gather(*(_embed_one(t) for t in safe_texts))
        return np.array(results, dtype=np.float32)
