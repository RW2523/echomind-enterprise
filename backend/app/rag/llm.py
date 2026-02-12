from __future__ import annotations
import json
import logging
import httpx
from ..core.config import settings
from typing import AsyncIterator

logger = logging.getLogger(__name__)


def _log_chat_request(url: str, payload: dict, stream: bool) -> None:
    """Log full prompt before sending chat/completions request (no content cut)."""
    logger.info(
        "LLM request %s -> %s/chat/completions full_payload=%s",
        "stream" if stream else "sync",
        url,
        json.dumps(payload, ensure_ascii=False),
    )


class OpenAICompatChat:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def chat(self, messages, temperature: float, max_tokens: int) -> str:
        payload={"model":self.model,"messages":messages,"temperature":temperature,"max_tokens":max_tokens,"stream":False}
        _log_chat_request(self.base_url, payload, stream=False)
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(f"{self.base_url}/chat/completions", json=payload)
            r.raise_for_status()
            j=r.json()
            return (j["choices"][0]["message"]["content"] or "").strip()

    async def chat_stream(self, messages, temperature: float, max_tokens: int) -> AsyncIterator[str]:
        """Stream LLM response token-by-token (Ollama SSE). Yields content deltas."""
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": True}
        _log_chat_request(self.base_url, payload, stream=True)
        async with httpx.AsyncClient(timeout=180) as client:
            async with client.stream("POST", f"{self.base_url}/chat/completions", json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line or line.strip() != line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            j = json.loads(data)
                            delta = (j.get("choices") or [{}])[0].get("delta") or {}
                            content = delta.get("content")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass
