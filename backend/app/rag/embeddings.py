from __future__ import annotations
import numpy as np, httpx
from ..core.config import settings

class OllamaEmbeddings:
    async def embed(self, texts: list[str]) -> np.ndarray:
        async with httpx.AsyncClient(timeout=120) as client:
            vecs=[]
            for t in texts:
                r = await client.post(settings.OLLAMA_EMBED_URL, json={"model": settings.OLLAMA_EMBED_MODEL, "prompt": t})
                r.raise_for_status()
                vecs.append(r.json()["embedding"])
        return np.array(vecs, dtype=np.float32)
