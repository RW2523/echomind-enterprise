"""
Transcripts ingestion pipeline: receive lines or batch -> 60s chunks -> tag -> embed -> upsert to Qdrant (or existing index).
"""
from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional
from ..core.config import settings
from ..core.timeutils import now_utc_ts
from ..utils.ids import new_id
from . import chunking as ing_chunking
from . import tagging as ing_tagging
from ..models.embedder import embed_texts
from ..qdrant.client import is_qdrant_enabled
from ..qdrant.upsert import upsert_transcript_points
from ..qdrant.collections import ensure_collections, TRANSCRIPTS_COLLECTION

logger = logging.getLogger(__name__)


async def run_pipeline_transcript(
    raw_text: str,
    transcript_id: str,
    location: str = "default",
    timezone: str = "UTC",
    ingested_at: Optional[int] = None,
    tags_override: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run transcript pipeline: 60s chunks, tag, embed, upsert.
    Returns {"transcript_id", "chunks_count"}.
    When Qdrant is disabled, caller should use existing store_to_db + index.add_text path.
    """
    ingested_at = ingested_at or now_utc_ts()
    chunks = ing_chunking.chunk_transcript_60s(
        raw_text, transcript_id, location=location, timezone=timezone, ingested_at=ingested_at
    )
    if not chunks:
        return {"transcript_id": transcript_id, "chunks_count": 0}

    texts = [c["text"] for c in chunks]
    chunk_tags = tags_override if tags_override is not None else []
    if not chunk_tags:
        chunk_tags = ing_tagging.tag_chunk(raw_text[:5000], max_tags=10)

    if is_qdrant_enabled():
        vecs = await embed_texts(texts)
        vector_size = vecs.shape[1]
        ensure_collections(vector_size)
        points = []
        for i, ch in enumerate(chunks):
            payload = {
                "source_type": "transcript",
                "transcript_id": transcript_id,
                "chunk_id": ch["chunk_id"],
                "text_preview": (ch["text"] or "")[:2000],
                "start_ts": ch["start_ts"],
                "end_ts": ch["end_ts"],
                "ingested_at": ch["ingested_at"],
                "location": ch["location"],
                "tags": chunk_tags if isinstance(chunk_tags, list) else [],
                "timezone": ch["timezone"],
            }
            try:
                point_id = hash(ch["chunk_id"]) & 0x7FFFFFFFFFFFFFFF
            except Exception:
                point_id = i
            points.append({"id": point_id, "payload": payload})
        await upsert_transcript_points(points, vectors=[v.tolist() for v in vecs])
        logger.info("Upserted %s transcript chunks to Qdrant", len(points))
        return {"transcript_id": transcript_id, "chunks_count": len(points)}

    return {"transcript_id": transcript_id, "chunks_count": len(chunks), "use_faiss": True}
