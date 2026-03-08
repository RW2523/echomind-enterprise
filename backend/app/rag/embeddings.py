from __future__ import annotations
import logging
import numpy as np, httpx
from ..core.config import settings

logger = logging.getLogger(__name__)

_MIN_EMBED_CHARS = 500


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
        async with httpx.AsyncClient(timeout=120) as client:
            vecs = []
            for t in texts:
                safe = _truncate_for_embed(t)
                vec = await self._embed_one(client, safe, len(t))
                vecs.append(vec)
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
