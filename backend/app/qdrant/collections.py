"""
Qdrant collection definitions and payload schemas for transcripts and documents.
Creates collections with vector size from embedder; ensures indexes for fast filtering.
"""
from __future__ import annotations
import logging
from .client import get_qdrant_client, is_qdrant_enabled

logger = logging.getLogger(__name__)

TRANSCRIPTS_COLLECTION = "transcripts"
DOCUMENTS_COLLECTION = "documents"

# Payload keys for transcripts (as per design)
TRANSCRIPT_PAYLOAD_KEYS = [
    "source_type", "transcript_id", "chunk_id", "text_preview",
    "start_ts", "end_ts", "ingested_at", "location", "tags", "timezone",
]

# Payload keys for documents
DOCUMENT_PAYLOAD_KEYS = [
    "source_type", "doc_id", "chunk_id", "doc_title", "doc_type", "file_type",
    "section_path", "page_start", "page_end", "row_start", "row_end",
    "tags", "ingested_at", "version",
]


def ensure_collections(vector_size: int) -> None:
    """
    Create transcripts and documents collections if they do not exist.
    vector_size: embedding dimension (e.g. from embedder).
    """
    client = get_qdrant_client()
    if not client:
        return
    from qdrant_client.models import Distance, VectorParams
    for name in (TRANSCRIPTS_COLLECTION, DOCUMENTS_COLLECTION):
        try:
            if not client.collection_exists(name):
                client.create_collection(
                    name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
                logger.info("Created Qdrant collection: %s", name)
        except Exception as e:
            logger.warning("Could not ensure collection %s: %s", name, e)
