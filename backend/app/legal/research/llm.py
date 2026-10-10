"""The answer model client for research. Tests replace `_override` with a local stand-in."""
from __future__ import annotations

from typing import Any, Optional

_override: Optional[Any] = None


def get():
    if _override is not None:
        return _override
    from ...core.config import settings
    from ...rag.llm import OpenAICompatChat
    return OpenAICompatChat(settings.LLM_BASE_URL, settings.LLM_MODEL)


async def complete(messages, temperature: float, max_tokens: int) -> str:
    """For structured answers (JSON): the model's reply as written. The client's prose tidy-up trims a
    reply that does not end on a sentence back to its last sentence, which cut the final JSON object
    (S3.1 found it cutting the last item of every minutes batch)."""
    client = get()
    call = getattr(client, "chat_raw", None) or client.chat
    return await call(messages, temperature=temperature, max_tokens=max_tokens)
