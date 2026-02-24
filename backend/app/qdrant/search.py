"""
Qdrant search: transcript_search and document_search with filters (time, location, tags, doc_id, etc.).
Returns vector similarity scores and chunk payload.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from .client import get_qdrant_client, is_qdrant_enabled
from ..models.embedder import embed_texts

TRANSCRIPTS_COLLECTION = "transcripts"
DOCUMENTS_COLLECTION = "documents"


def _build_transcript_filter(
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    location: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Optional[Any]:
    """Build Qdrant filter for transcripts."""
    if not any([start_ts is not None, end_ts is not None, location, tags]):
        return None
    from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
    must = []
    if start_ts is not None:
        must.append(FieldCondition(key="start_ts", range=Range(gte=start_ts)))
    if end_ts is not None:
        must.append(FieldCondition(key="end_ts", range=Range(lte=end_ts)))
    if location:
        must.append(FieldCondition(key="location", match=MatchValue(value=location)))
    if tags:
        from qdrant_client.models import MatchAny
        must.append(FieldCondition(key="tags", match=MatchAny(any=tags)))
    return Filter(must=must) if must else None


def _build_document_filter(
    doc_id: Optional[str] = None,
    doc_type: Optional[str] = None,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
) -> Optional[Any]:
    """Build Qdrant filter for documents."""
    if not any([doc_id, doc_type, page_start is not None, page_end is not None]):
        return None
    from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
    must = []
    if doc_id:
        must.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
    if doc_type:
        must.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))
    if page_start is not None:
        must.append(FieldCondition(key="page_start", range=Range(gte=page_start)))
    if page_end is not None:
        must.append(FieldCondition(key="page_end", range=Range(lte=page_end)))
    return Filter(must=must) if must else None


async def transcript_search(
    query_text: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 15,
    threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Search transcripts collection; returns list of {score, payload, text_preview}."""
    client = get_qdrant_client()
    if not client:
        return []
    q = await embed_texts([query_text])
    qv = q[0].tolist()
    f = None
    if filters:
        f = _build_transcript_filter(
            start_ts=filters.get("start_ts"),
            end_ts=filters.get("end_ts"),
            location=filters.get("location"),
            tags=filters.get("tags"),
        )
    try:
        results = client.search(
            collection_name=TRANSCRIPTS_COLLECTION,
            query_vector=qv,
            query_filter=f,
            limit=top_k,
            score_threshold=threshold,
        )
        out = []
        for r in results:
            payload = r.payload or {}
            if threshold is not None and r.score < threshold:
                continue
            out.append({
                "score": r.score,
                "payload": payload,
                "text": payload.get("text_preview", ""),
                "chunk_id": payload.get("chunk_id"),
            })
        return out
    except Exception:
        return []


async def document_search(
    query_text: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 15,
    threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Search documents collection; returns list of {score, payload, text}."""
    client = get_qdrant_client()
    if not client:
        return []
    q = await embed_texts([query_text])
    qv = q[0].tolist()
    f = None
    if filters:
        f = _build_document_filter(
            doc_id=filters.get("doc_id"),
            doc_type=filters.get("doc_type"),
            page_start=filters.get("page_start"),
            page_end=filters.get("page_end"),
        )
    try:
        results = client.search(
            collection_name=DOCUMENTS_COLLECTION,
            query_vector=qv,
            query_filter=f,
            limit=top_k,
            score_threshold=threshold,
        )
        out = []
        for r in results:
            payload = r.payload or {}
            if threshold is not None and r.score < threshold:
                continue
            out.append({
                "score": r.score,
                "payload": payload,
                "text": payload.get("text_preview", payload.get("text", "")),
                "chunk_id": payload.get("chunk_id"),
            })
        return out
    except Exception:
        return []
