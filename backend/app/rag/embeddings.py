"""
Embeddings: supports Ollama (prompt/embedding) and SGLang/OpenAI (input/data[0].embedding).
"""
from __future__ import annotations
import numpy as np, httpx
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


class OpenAICompatEmbeddings:
    """Unified embeddings: Ollama (prompt/embedding) or OpenAI (input/data[0].embedding)."""

    async def embed(self, texts: list[str]) -> np.ndarray:
        fmt = (settings.EMBED_FORMAT or "ollama").lower()
        async with httpx.AsyncClient(timeout=120) as client:
            vecs = []
            for t in texts:
                safe = _truncate_for_embed(t)
                if fmt == "ollama":
                    r = await client.post(
                        settings.EMBED_URL,
                        json={"model": settings.EMBED_MODEL, "prompt": safe},
                    )
                    r.raise_for_status()
                    vecs.append(r.json()["embedding"])
                else:
                    r = await client.post(
                        settings.EMBED_URL,
                        json={"model": settings.EMBED_MODEL, "input": safe},
                    )
                    r.raise_for_status()
                    data = r.json()
                    vecs.append(data["data"][0]["embedding"])
        return np.array(vecs, dtype=np.float32)
