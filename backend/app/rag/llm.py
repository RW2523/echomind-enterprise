from __future__ import annotations
import json
import logging
import time
import httpx
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
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(f"{self.base_url}/chat/completions", json=payload)
            r.raise_for_status()
            j=r.json()
            out = (j["choices"][0]["message"]["content"] or "").strip()
        elapsed_ms = (time.monotonic() - t0) * 1000
        usage = j.get("usage") or {}
        logger.info(
            "LLM sync done model=%s total_ms=%.1f http_status=%s prompt_msgs=%d completion_chars=%d "
            "usage_prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            self.model,
            elapsed_ms,
            getattr(r, "status_code", "?"),
            len(messages),
            len(out),
            usage.get("prompt_tokens", "n/a"),
            usage.get("completion_tokens", "n/a"),
            usage.get("total_tokens", "n/a"),
        )
        return out

    async def chat_stream(self, messages, temperature: float, max_tokens: int) -> AsyncIterator[str]:
        """Stream LLM response token-by-token (SSE). Yields content deltas."""
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": True}
        _log_chat_request(self.base_url, payload, stream=True)
        t0 = time.monotonic()
        ttft_mono: float | None = None
        sse_data_events = 0
        content_deltas = 0
        total_chars = 0

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
                        sse_data_events += 1
                        try:
                            j = json.loads(data)
                            delta = (j.get("choices") or [{}])[0].get("delta") or {}
                            content = delta.get("content")
                            if content:
                                if ttft_mono is None:
                                    ttft_mono = time.monotonic()
                                content_deltas += 1
                                total_chars += len(content)
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass

        t1 = time.monotonic()
        ttft_ms = (ttft_mono - t0) * 1000 if ttft_mono is not None else 0.0
        logger.info(
            "LLM stream done model=%s ttft_ms=%.1f stream_total_ms=%.1f sse_data_events=%d "
            "content_deltas=%d output_chars=%d prompt_msgs=%d",
            self.model,
            ttft_ms,
            (t1 - t0) * 1000,
            sse_data_events,
            content_deltas,
            total_chars,
            len(messages),
        )
