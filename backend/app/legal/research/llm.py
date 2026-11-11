"""The answer model client for research. Tests replace `_override` with a local stand-in.

Model calls wait their turn (2026-11-12, evaluation fix list item 15): at most LEGAL_MODEL_CONCURRENCY at once from this
backend, the model server being shared, and `ahead()` says how many are waiting, so a person is told the model is busy
instead of watching one progress line for half a minute.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Optional

_override: Optional[Any] = None
_gate: Optional[asyncio.Semaphore] = None
_waiting = 0
_running = 0


def _limit() -> int:
    from ...core.config import settings
    return max(1, settings.LEGAL_MODEL_CONCURRENCY)


@asynccontextmanager
async def _slot():
    global _gate, _waiting, _running
    if _gate is None:
        _gate = asyncio.Semaphore(_limit())
    _waiting += 1
    try:
        await _gate.acquire()
    finally:
        _waiting -= 1
    _running += 1
    try:
        yield
    finally:
        _running -= 1
        _gate.release()


def ahead() -> int:
    """How many model calls would be served before a new one: those waiting, when every slot is taken."""
    return _waiting + 1 if _running >= _limit() else 0


class _Gated:
    """The client, each call waiting for a slot; a streamed answer holds its slot until it ends."""

    def __init__(self, inner):
        self._inner = inner
        if hasattr(inner, "chat_raw"):
            async def chat_raw(*a, **k):
                async with _slot():
                    return await inner.chat_raw(*a, **k)
            self.chat_raw = chat_raw

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def chat(self, *a, **k):
        async with _slot():
            return await self._inner.chat(*a, **k)

    async def chat_stream(self, *a, **k):
        async with _slot():
            async for piece in self._inner.chat_stream(*a, **k):
                yield piece


def get():
    if _override is not None:
        return _override
    from ...core.config import settings
    from ...rag.llm import OpenAICompatChat
    return _Gated(OpenAICompatChat(settings.LLM_BASE_URL, settings.LLM_MODEL))


async def complete(messages, temperature: float, max_tokens: int) -> str:
    """For structured answers (JSON): the model's reply as written. The client's prose tidy-up trims a
    reply that does not end on a sentence back to its last sentence, which cut the final JSON object
    (S3.1 found it cutting the last item of every minutes batch)."""
    client = get()
    call = getattr(client, "chat_raw", None) or client.chat
    return await call(messages, temperature=temperature, max_tokens=max_tokens)
