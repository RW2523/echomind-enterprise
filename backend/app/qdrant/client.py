"""
Qdrant client connection for the RAG platform.
Connects to local or Docker Qdrant instance; used when VECTOR_BACKEND=qdrant and QDRANT_URL is set.
"""
from __future__ import annotations
from typing import Optional
from ..core.config import settings

_client = None


def get_qdrant_client():
    """Return Qdrant client singleton. Returns None if Qdrant is not configured."""
    global _client
    if not (settings.QDRANT_URL and settings.VECTOR_BACKEND == "qdrant"):
        return None
    if _client is None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import exceptions as qdrant_exc
            url = settings.QDRANT_URL.strip().rstrip("/")
            api_key = (settings.QDRANT_API_KEY or "").strip() or None
            _client = QdrantClient(url=url, api_key=api_key)
        except Exception:
            _client = False  # mark as attempted but failed
    return _client if _client is not False else None


def is_qdrant_enabled() -> bool:
    """True when Qdrant URL is set and vector backend is qdrant."""
    return bool(settings.QDRANT_URL and settings.VECTOR_BACKEND == "qdrant")
