from __future__ import annotations
import numpy as np, httpx
from ..core.config import settings

def _truncate_for_embed(text: str, max_chars: int | None = None) -> str:
    """Truncate at word boundary so embedding API never exceeds context length."""
    limit = max_chars if max_chars is not None else settings.EMBED_MAX_CHARS
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
                r = await client.post(
                    settings.OLLAMA_EMBED_URL,
                    json={"model": settings.OLLAMA_EMBED_MODEL, "prompt": safe},
                )
                r.raise_for_status()
                vecs.append(r.json()["embedding"])
        return np.array(vecs, dtype=np.float32)
