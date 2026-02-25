"""
HTTP client for RAG Platform. Used when RAG_PLATFORM_URL is set; backend proxies docs, query, transcript ingest.
"""
from __future__ import annotations
import asyncio
import logging
from typing import List, Optional

import httpx

from .core.config import settings

logger = logging.getLogger(__name__)

BASE = (settings.RAG_PLATFORM_URL or "").rstrip("/")

# Retry connection errors (e.g. RAG platform still starting after restart). Rag-platform can take 1–2 min to load the model.
_QUERY_RETRIES = 15
_QUERY_RETRY_DELAY = 6.0


def _url(path: str) -> str:
    return f"{BASE}{path}" if BASE else ""


def is_configured() -> bool:
    return bool(BASE)


async def upload_doc(file_content: bytes, filename: str, content_type: str = "application/octet-stream") -> dict:
    """POST /docs/upload. Returns { ok, doc_id, chunks }."""
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            _url("/docs/upload"),
            files={"file": (filename, file_content, content_type)},
        )
        r.raise_for_status()
        return r.json()


async def list_docs() -> dict:
    """GET /docs/list. Returns { documents: [{ id, filename, filetype, created_at }] }."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(_url("/docs/list"))
        r.raise_for_status()
        return r.json()


async def delete_doc(doc_id: str) -> dict:
    """DELETE /docs/{doc_id}. Returns { ok, deleted }."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.delete(_url(f"/docs/{doc_id}"))
        r.raise_for_status()
        return r.json()


async def get_usage() -> dict:
    """GET /docs/usage. Returns { usage_bytes, capacity_bytes }."""
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(_url("/docs/usage"))
        r.raise_for_status()
        return r.json()


async def get_data_preview() -> dict:
    """GET /docs/data-preview. Returns { documents, chunks, transcripts }."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(_url("/docs/data-preview"))
        r.raise_for_status()
        return r.json()


async def delete_all_docs() -> dict:
    """POST /docs/delete-all. Returns { ok, message }."""
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(_url("/docs/delete-all"))
        r.raise_for_status()
        return r.json()


async def query(user_query: str, mode: Optional[str] = None, doc_id: Optional[str] = None) -> dict:
    """
    POST /query. Returns { answer, evidence, source_used, from_sources }.
    Backend maps evidence -> citations for frontend.
    Retries on connection errors (e.g. RAG platform still starting).
    """
    body = {"user_query": user_query}
    if mode is not None:
        body["mode"] = mode
    if doc_id is not None:
        body["doc_id"] = doc_id
    last_exc = None
    for attempt in range(_QUERY_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(_url("/query"), json=body)
                r.raise_for_status()
                return r.json()
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            last_exc = e
            if attempt < _QUERY_RETRIES - 1:
                logger.warning("RAG platform connection failed (attempt %s/%s), retrying in %ss: %s",
                               attempt + 1, _QUERY_RETRIES, _QUERY_RETRY_DELAY, e)
                await asyncio.sleep(_QUERY_RETRY_DELAY)
            else:
                raise last_exc from last_exc


async def ingest_transcript_chunk(
    transcript_id: str,
    text: str,
    ts: float = 0,
    location: Optional[str] = None,
    tags: Optional[List[str]] = None,
    timezone: Optional[str] = None,
) -> dict:
    """POST /transcripts/ingest. Ingest one chunk (e.g. live transcript paragraph)."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            _url("/transcripts/ingest"),
            json={
                "transcript_id": transcript_id,
                "ts": ts,
                "text": text,
                "location": location,
                "tags": tags,
                "timezone": timezone,
            },
        )
        r.raise_for_status()
        return r.json()


async def ingest_transcript_batch(
    transcript_id: str,
    lines: List[dict],
    location: Optional[str] = None,
    timezone: Optional[str] = None,
) -> dict:
    """POST /transcripts/ingest_batch. lines: [{"text": "...", "ts": epoch_float}, ...]."""
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            _url("/transcripts/ingest_batch"),
            json={
                "transcript_id": transcript_id,
                "lines": lines,
                "location": location,
                "timezone": timezone,
            },
        )
        r.raise_for_status()
        return r.json()
