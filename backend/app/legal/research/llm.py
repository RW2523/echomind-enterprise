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
