"""
Transcript and document search with filters (time, location, tags) and score thresholds.
"""
from __future__ import annotations
import logging
from typing import Any, List, Optional

from qdrant_client.http import models as qmodels

from app.qdrant.client import get_qdrant_client
from app.qdrant.collections import ensure_transcripts_collection, ensure_documents_collection
from app.core.config import settings
from app.models.embedder import get_embedder

logger = logging.getLogger(__name__)


def _build_transcript_filter(
    start_ts_min: Optional[int] = None,
    end_ts_max: Optional[int] = None,
    location: Optional[str] = None,
    tags_any: Optional[List[str]] = None,
) -> Optional[qmodels.Filter]:
    conditions = []
    if start_ts_min is not None:
        conditions.append(qmodels.FieldCondition(key="start_ts", range=qmodels.Range(gte=start_ts_min)))
    if end_ts_max is not None:
        conditions.append(qmodels.FieldCondition(key="end_ts", range=qmodels.Range(lte=end_ts_max)))
    if location:
        conditions.append(qmodels.FieldCondition(key="location", match=qmodels.MatchValue(value=location)))
    if tags_any:
        conditions.append(
            qmodels.FieldCondition(key="tags", match=qmodels.MatchAny(any=tags_any))
        )
    if not conditions:
        return None
    return qmodels.Filter(must=conditions)


def search_transcripts(
    vector: List[float],
    top_k: int = 15,
    score_threshold: Optional[float] = None,
    start_ts_min: Optional[int] = None,
    end_ts_max: Optional[int] = None,
    location: Optional[str] = None,
    tags_any: Optional[List[str]] = None,
) -> List[dict]:
    ensure_transcripts_collection()
    client = get_qdrant_client()
    name = settings.QDRANT_COLLECTION_TRANSCRIPTS
    threshold = score_threshold if score_threshold is not None else settings.T1_TRANSCRIPT
    qfilter = _build_transcript_filter(start_ts_min, end_ts_max, location, tags_any)
    response = client.query_points(
        collection_name=name,
        query=vector,
        limit=top_k,
        query_filter=qfilter,
        score_threshold=threshold,
        with_payload=True,
    )
    results = response.points
    out = []
    for r in results:
        pl = r.payload or {}
        out.append({
            "id": str(r.id) if r.id else None,
            "score": r.score or 0.0,
            "payload": pl,
            "text": pl.get("text") or pl.get("text_preview", ""),
        })
    return out


def search_documents(
    vector: List[float],
    top_k: int = 15,
    score_threshold: Optional[float] = None,
    doc_id: Optional[str] = None,
    doc_type: Optional[str] = None,
) -> List[dict]:
    ensure_documents_collection()
    client = get_qdrant_client()
    name = settings.QDRANT_COLLECTION_DOCUMENTS
    threshold = score_threshold if score_threshold is not None else settings.T2_DOCUMENT
    conditions = []
    if doc_id:
        conditions.append(qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id)))
    if doc_type:
        conditions.append(qmodels.FieldCondition(key="doc_type", match=qmodels.MatchValue(value=doc_type)))
    qfilter = qmodels.Filter(must=conditions) if conditions else None
    response = client.query_points(
        collection_name=name,
        query=vector,
        limit=top_k,
        query_filter=qfilter,
        score_threshold=threshold,
        with_payload=True,
    )
    results = response.points
    out = []
    for r in results:
        pl = r.payload or {}
        out.append({
            "id": str(r.id) if r.id else None,
            "score": r.score or 0.0,
            "payload": pl,
            "text": pl.get("text") or pl.get("text_preview", pl.get("section_path", "")),
        })
    return out


def get_most_recent_transcript(
    limit: int = 1,
    location: Optional[str] = None,
) -> List[dict]:
    """No embedding: filter + sort by ingested_at desc. Returns latest chunk(s)."""
    ensure_transcripts_collection()
    client = get_qdrant_client()
    name = settings.QDRANT_COLLECTION_TRANSCRIPTS
    conditions = []
    if location:
        conditions.append(qmodels.FieldCondition(key="location", match=qmodels.MatchValue(value=location)))
    qfilter = qmodels.Filter(must=conditions) if conditions else None
    # Scroll by ingested_at; we don't have native sort-by-payload in all Qdrant versions, so scroll and sort in memory
    results, _ = client.scroll(
        collection_name=name,
        scroll_filter=qfilter,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )
    points = results  # list of Record
    points.sort(key=lambda p: (p.payload or {}).get("ingested_at") or 0, reverse=True)
    out = []
    for p in points[:limit]:
        pl = p.payload or {}
        out.append({
            "id": str(p.id) if p.id else None,
            "payload": pl,
            "text": pl.get("text") or pl.get("text_preview", ""),
        })
    return out
