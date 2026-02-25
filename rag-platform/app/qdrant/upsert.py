"""
Batch upsert points to transcripts and documents collections.
"""
from __future__ import annotations
import logging
import uuid
from typing import Any, List

from qdrant_client.http import models as qmodels

from app.qdrant.client import get_qdrant_client
from app.qdrant.collections import ensure_transcripts_collection, ensure_documents_collection
from app.core.config import settings
from app.models.embedder import get_embedder

logger = logging.getLogger(__name__)


def _points_transcripts(
    texts: List[str],
    payloads: List[dict],
    vectors: List[List[float]],
) -> List[qmodels.PointStruct]:
    ensure_transcripts_collection()
    points = []
    for i, (text, payload, vec) in enumerate(zip(texts, payloads, vectors)):
        pid = payload.get("chunk_id") or str(uuid.uuid4())
        p = dict(payload)
        p["source_type"] = "transcript"
        if "text_preview" not in p:
            p["text_preview"] = (text or "")[:300]
        p["text"] = text or ""  # full text for context building
        points.append(qmodels.PointStruct(id=pid, vector=vec, payload=p))
    return points


def _points_documents(
    texts: List[str],
    payloads: List[dict],
    vectors: List[List[float]],
) -> List[qmodels.PointStruct]:
    ensure_documents_collection()
    points = []
    for i, (text, payload, vec) in enumerate(zip(texts, payloads, vectors)):
        pid = payload.get("chunk_id") or str(uuid.uuid4())
        p = dict(payload)
        p["source_type"] = "document"
        if "text_preview" not in p:
            p["text_preview"] = (text or "")[:300]
        p["text"] = text or ""  # full text for context building
        points.append(qmodels.PointStruct(id=pid, vector=vec, payload=p))
    return points


def upsert_transcript_points(
    texts: List[str],
    payloads: List[dict],
    batch_size: int = 64,
) -> None:
    embedder = get_embedder()
    vectors = embedder.encode(texts, is_query=False, batch_size=batch_size)
    points = _points_transcripts(texts, payloads, vectors)
    client = get_qdrant_client()
    name = settings.QDRANT_COLLECTION_TRANSCRIPTS
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=name, points=batch)
    logger.info("Upserted %s points to %s", len(points), name)


def upsert_document_points(
    texts: List[str],
    payloads: List[dict],
    batch_size: int = 64,
) -> None:
    embedder = get_embedder()
    vectors = embedder.encode(texts, is_query=False, batch_size=batch_size)
    points = _points_documents(texts, payloads, vectors)
    client = get_qdrant_client()
    name = settings.QDRANT_COLLECTION_DOCUMENTS
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=name, points=batch)
    logger.info("Upserted %s points to %s", len(points), name)
