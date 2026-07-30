from __future__ import annotations
import json
import logging
import os
import re
import time
import httpx
from typing import AsyncIterator

logger = logging.getLogger(__name__)

# Logging full prompts means writing every uploaded-document / transcript excerpt to logs
# in plaintext (PII / confidential content). Off by default; opt in only for debugging.
_LOG_FULL_PROMPTS = os.getenv("ECHOMIND_LOG_FULL_PROMPTS", "0").lower() in ("1", "true", "yes")


def _log_chat_request(url: str, payload: dict, stream: bool) -> None:
    """Log chat/completions request. Logs only metadata at INFO; the full prompt (which
    contains retrieved document/transcript content) is logged at DEBUG and only when
    ECHOMIND_LOG_FULL_PROMPTS is enabled, to avoid leaking sensitive content into logs."""
    messages = payload.get("messages") or []
    total_chars = sum(len(str(m.get("content") or "")) for m in messages)
    logger.info(
        "LLM request %s -> %s/chat/completions model=%s msgs=%d prompt_chars=%d max_tokens=%s",
        "stream" if stream else "sync",
        url,
        payload.get("model"),
        len(messages),
        total_chars,
        payload.get("max_tokens"),
    )
    if _LOG_FULL_PROMPTS and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "LLM full_payload (%s) %s",
            "stream" if stream else "sync",
            json.dumps(payload, ensure_ascii=False),
        )


# Reasoning models (e.g. Qwen3) emit <think> blocks by default — slow + noisy for chat/voice.
# Disabled by default; set LLM_ENABLE_THINKING=1 to re-enable (e.g. for batch report quality).
_ENABLE_THINKING = (__import__("os").getenv("LLM_ENABLE_THINKING", "0").strip().lower() in ("1", "true", "yes"))
_EXTRA = {} if _ENABLE_THINKING else {"chat_template_kwargs": {"enable_thinking": False}}


# ── Reasoning-leak guard ────────────────────────────────────────────────────────
# Qwen3-30B on TRT-LLM intermittently (~6% of short answers, sampling-dependent) fails to
# emit EOS: it produces a complete answer, then a stray token ("RefreshLayout", "移除",
# "Feedback", "<think>"), then chain-of-thought about the user's request. thinking is
# already disabled via chat_template_kwargs and finish_reason is usually "stop", so this is
# a decode-time artifact, not a prompt bug. Cut the answer at the leak boundary.
_LEAK_RE = re.compile(
    r"(?:<think>|"
    r"(?:^|\n)\s*(?:Okay|Alright|Hmm|So)\s*,?\s+(?:the user|I need to|I should|let me|we need)\b|"
    r"(?:^|\n)\s*The user (?:asked|wants|is asking)\b)",
    re.IGNORECASE,
)


def strip_reasoning_leak(text: str) -> str:
    """Truncate an answer at leaked chain-of-thought. Returns the clean prefix."""
    if not text:
        return text
    m = _LEAK_RE.search(text)
    if not m or m.start() == 0:
        # start==0 means the WHOLE response is reasoning — keep it rather than blank the answer
        return text
    cleaned = text[: m.start()].rstrip()
    # Drop a trailing stray token left on the last line (e.g. "... 😊\nRefreshLayout")
    lines = cleaned.split("\n")
    if len(lines) > 1 and 0 < len(lines[-1].strip()) <= 24 and " " not in lines[-1].strip():
        lines = lines[:-1]
        cleaned = "\n".join(lines).rstrip()
    logger.warning(
        "LLM: reasoning leak detected — truncated %d trailing chars", len(text) - len(cleaned)
    )
    return cleaned


class OpenAICompatChat:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def chat(self, messages, temperature: float, max_tokens: int) -> str:
        payload={"model":self.model,"messages":messages,"temperature":temperature,"max_tokens":max_tokens,"stream":False, **_EXTRA}
        _log_chat_request(self.base_url, payload, stream=False)
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(f"{self.base_url}/chat/completions", json=payload)
            r.raise_for_status()
            j=r.json()
            out = strip_reasoning_leak((j["choices"][0]["message"]["content"] or "").strip())
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
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": True, **_EXTRA}
        _log_chat_request(self.base_url, payload, stream=True)
        t0 = time.monotonic()
        ttft_mono: float | None = None
        sse_data_events = 0
        content_deltas = 0
        _acc = ""
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
                                # Reasoning-leak guard (streaming): once chain-of-thought
                                # starts leaking, stop the stream rather than emit it.
                                _acc += content
                                if _LEAK_RE.search(_acc[-400:]):
                                    logger.warning("LLM stream: reasoning leak detected — cutting stream")
                                    break
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
