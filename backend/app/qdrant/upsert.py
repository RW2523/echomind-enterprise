"""
Qdrant upsert: add transcript and document points to their collections.
Used by ingestion pipelines after chunking and embedding.
"""
from __future__ import annotations
from typing import List, Dict, Any
from .client import get_qdrant_client
from .collections import TRANSCRIPTS_COLLECTION, DOCUMENTS_COLLECTION
from ..models.embedder import embed_texts


async def upsert_transcript_points(
    points: List[Dict[str, Any]],
    vectors: List[List[float]] | None = None,
) -> None:
    """
    Upsert points into transcripts collection.
    points: list of {id, payload}; payload must include source_type="transcript", chunk_id, text_preview, start_ts, end_ts, etc.
    vectors: optional precomputed; if None, will embed text_preview from each payload.
    """
    client = get_qdrant_client()
    if not client or not points:
        return
    if vectors is None:
        texts = [p.get("payload", {}).get("text_preview", "") for p in points]
        vecs = await embed_texts(texts)
        vectors = [vec.tolist() for vec in vecs]
    from qdrant_client.models import PointStruct
    structs = []
    for i, p in enumerate(points):
        vid = p.get("id")
        payload = p.get("payload", p) if "payload" in p else p
        vec = vectors[i] if i < len(vectors) else vectors[0]
        structs.append(PointStruct(id=vid, vector=vec, payload=payload))
    client.upsert(collection_name=TRANSCRIPTS_COLLECTION, points=structs)


async def upsert_document_points(
    points: List[Dict[str, Any]],
    vectors: List[List[float]] | None = None,
) -> None:
    """
    Upsert points into documents collection.
    points: list of {id, payload}; payload must include source_type="document", doc_id, chunk_id, text_preview, etc.
    """
    client = get_qdrant_client()
    if not client or not points:
        return
    if vectors is None:
        texts = [p.get("payload", {}).get("text_preview", "") for p in points]
        vecs = await embed_texts(texts)
        vectors = [vec.tolist() for vec in vecs]
    from qdrant_client.models import PointStruct
    structs = []
    for i, p in enumerate(points):
        vid = p.get("id")
        payload = p.get("payload", p) if "payload" in p else p
        vec = vectors[i] if i < len(vectors) else vectors[0]
        structs.append(PointStruct(id=vid, vector=vec, payload=payload))
    client.upsert(collection_name=DOCUMENTS_COLLECTION, points=structs)
