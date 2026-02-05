from __future__ import annotations
import httpx
from ..core.config import settings

class OpenAICompatChat:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def chat(self, messages, temperature: float, max_tokens: int) -> str:
        payload={"model":self.model,"messages":messages,"temperature":temperature,"max_tokens":max_tokens,"stream":False}
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(f"{self.base_url}/chat/completions", json=payload)
            r.raise_for_status()
            j=r.json()
            return (j["choices"][0]["message"]["content"] or "").strip()
