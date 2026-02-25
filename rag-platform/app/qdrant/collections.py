"""
Create Qdrant collections (transcripts, documents) and payload indexes.
Vector size = embedding dim from Qwen3 (1024).
"""
from __future__ import annotations
import logging
from qdrant_client.http import models as qmodels

from app.qdrant.client import get_qdrant_client
from app.core.config import settings
from app.models.embedder import get_embedder

logger = logging.getLogger(__name__)


def _vector_size() -> int:
    try:
        return get_embedder().embedding_dim
    except RuntimeError:
        # Before lifespan: use known Qwen3-Embedding-0.6B dim
        return 1024


def _transcript_payload_schema() -> dict:
    return {
        "source_type": "keyword",
        "transcript_id": "keyword",
        "chunk_id": "keyword",
        "text_preview": "text",
        "start_ts": "integer",
        "end_ts": "integer",
        "ingested_at": "integer",
        "location": "keyword",
        "tags": "keyword",
        "timezone": "keyword",
    }


def _document_payload_schema() -> dict:
    return {
        "source_type": "keyword",
        "doc_id": "keyword",
        "chunk_id": "keyword",
        "doc_title": "text",
        "doc_type": "keyword",
        "file_type": "keyword",
        "section_path": "text",
        "page_start": "integer",
        "page_end": "integer",
        "row_start": "integer",
        "row_end": "integer",
        "tags": "keyword",
        "ingested_at": "integer",
        "version": "keyword",
    }


def ensure_transcripts_collection() -> None:
    name = settings.QDRANT_COLLECTION_TRANSCRIPTS
    client = get_qdrant_client()
    size = _vector_size()
    if client.collection_exists(name):
        logger.info("Collection %s already exists", name)
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(size=size, distance=qmodels.Distance.COSINE),
        optimizers_config=qmodels.OptimizersConfigDiff(
            indexing_threshold=10_000,
        ),
    )
    # Payload indexes for fast filtering
    client.create_payload_index(name, "start_ts", qmodels.PayloadSchemaType.INTEGER)
    client.create_payload_index(name, "end_ts", qmodels.PayloadSchemaType.INTEGER)
    client.create_payload_index(name, "ingested_at", qmodels.PayloadSchemaType.INTEGER)
    client.create_payload_index(name, "location", qmodels.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "tags", qmodels.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "transcript_id", qmodels.PayloadSchemaType.KEYWORD)
    logger.info("Created collection %s with vector size %s", name, size)


def ensure_documents_collection() -> None:
    name = settings.QDRANT_COLLECTION_DOCUMENTS
    client = get_qdrant_client()
    size = _vector_size()
    if client.collection_exists(name):
        logger.info("Collection %s already exists", name)
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(size=size, distance=qmodels.Distance.COSINE),
        optimizers_config=qmodels.OptimizersConfigDiff(
            indexing_threshold=10_000,
        ),
    )
    client.create_payload_index(name, "doc_id", qmodels.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "doc_type", qmodels.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "file_type", qmodels.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "page_start", qmodels.PayloadSchemaType.INTEGER)
    client.create_payload_index(name, "page_end", qmodels.PayloadSchemaType.INTEGER)
    client.create_payload_index(name, "row_start", qmodels.PayloadSchemaType.INTEGER)
    client.create_payload_index(name, "row_end", qmodels.PayloadSchemaType.INTEGER)
    client.create_payload_index(name, "tags", qmodels.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "ingested_at", qmodels.PayloadSchemaType.INTEGER)
    logger.info("Created collection %s with vector size %s", name, size)


def ensure_all_collections() -> None:
    ensure_transcripts_collection()
    ensure_documents_collection()
