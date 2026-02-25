"""
Delete document or transcript points from Qdrant (for doc delete and delete-all).
"""
from __future__ import annotations
import logging
from qdrant_client.http import models as qmodels

from app.qdrant.client import get_qdrant_client
from app.qdrant.collections import ensure_documents_collection, ensure_transcripts_collection
from app.core.config import settings

logger = logging.getLogger(__name__)


def delete_document_points(doc_id: str) -> None:
    """Delete all points in the documents collection whose payload doc_id matches."""
    ensure_documents_collection()
    client = get_qdrant_client()
    name = settings.QDRANT_COLLECTION_DOCUMENTS
    client.delete(
        collection_name=name,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))]
            )
        ),
    )
    logger.info("Deleted document points for doc_id=%s", doc_id)


def clear_documents_collection() -> None:
    """Remove all points from the documents collection."""
    ensure_documents_collection()
    client = get_qdrant_client()
    name = settings.QDRANT_COLLECTION_DOCUMENTS
    client.delete_collection(name)
    ensure_documents_collection()
    logger.info("Cleared collection %s", name)


def clear_transcripts_collection() -> None:
    """Remove all points from the transcripts collection."""
    ensure_transcripts_collection()
    client = get_qdrant_client()
    name = settings.QDRANT_COLLECTION_TRANSCRIPTS
    client.delete_collection(name)
    ensure_transcripts_collection()
    logger.info("Cleared collection %s", name)
