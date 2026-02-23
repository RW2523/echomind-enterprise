"""Minimal tests for query embedding LRU cache: cache hits reduce HTTP calls."""
from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

os.environ.setdefault("ECHOMIND_DATA_DIR", tempfile.mkdtemp(prefix="echomind_test_"))

try:
    from app.rag.embeddings import OllamaEmbeddings, _normalize_query_text
except ModuleNotFoundError as e:
    pytest.skip("app.rag.embeddings not available: " + str(e), allow_module_level=True)


def test_normalize_query_text():
    assert _normalize_query_text("") == ""
    assert _normalize_query_text("  foo  bar  ") == "foo bar"
    assert _normalize_query_text("a\t\nb") == "a b"


@pytest.mark.asyncio
async def test_query_embed_cache_reduces_http_calls():
    """Embedding the same query twice uses cache on second call: only one HTTP request."""
    fake_embedding = [0.1] * 64  # arbitrary dim
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": fake_embedding}
    mock_response.raise_for_status = MagicMock()

    with patch("app.rag.embeddings.httpx.AsyncClient") as client_cls:
        async def post(*args, **kwargs):
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        client_cls.return_value = mock_client

        with patch("app.rag.embeddings.settings") as mock_settings:
            mock_settings.OLLAMA_EMBED_URL = "http://ollama:11434/api/embed"
            mock_settings.OLLAMA_EMBED_MODEL = "nomic-embed-text"
            mock_settings.EMBED_MAX_CHARS = 2000
            mock_settings.EMBED_QUERY_CACHE_SIZE = 2048

            emb = OllamaEmbeddings()
            query = "what did we discuss in the transcript?"

            out1 = await emb.embed([query])
            out2 = await emb.embed([query])

            assert out1.shape == (1, 64)
            assert out2.shape == (1, 64)
            assert emb._embed_http_calls == 1, "cache hit on second call should avoid second HTTP request"
            assert emb._embed_cache_hits == 1


@pytest.mark.asyncio
async def test_query_embed_cache_disabled_zero_size():
    """When EMBED_QUERY_CACHE_SIZE is 0, no cache: every single-query embed hits HTTP."""
    fake_embedding = [0.2] * 32
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": fake_embedding}
    mock_response.raise_for_status = MagicMock()

    with patch("app.rag.embeddings.httpx.AsyncClient") as client_cls:
        async def post(*args, **kwargs):
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        client_cls.return_value = mock_client

        with patch("app.rag.embeddings.settings") as mock_settings:
            mock_settings.OLLAMA_EMBED_URL = "http://ollama:11434/api/embed"
            mock_settings.OLLAMA_EMBED_MODEL = "nomic-embed-text"
            mock_settings.EMBED_MAX_CHARS = 2000
            mock_settings.EMBED_QUERY_CACHE_SIZE = 0
            mock_settings.EMBED_BATCH_SIZE = 64
            mock_settings.EMBED_CONCURRENCY = 4

            emb = OllamaEmbeddings()
            await emb.embed(["same query"])
            await emb.embed(["same query"])

            assert emb._embed_http_calls == 2, "cache disabled: each embed should trigger HTTP"
