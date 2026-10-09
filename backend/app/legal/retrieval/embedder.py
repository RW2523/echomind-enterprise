"""The embedding client used for Law documents. Tests replace `_override` with a local stand-in."""
from __future__ import annotations

from typing import Any, Optional

_override: Optional[Any] = None


def get():
    if _override is not None:
        return _override
    from ...rag.embeddings import OllamaEmbeddings
    return OllamaEmbeddings()
