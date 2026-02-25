"""
Transcript ingestion: live stream -> 60s fixed chunks -> tag -> embed -> Qdrant.
Payload: transcript_id, start_ts, end_ts, ingested_at, location, tags, timezone.
"""
from __future__ import annotations
import time
import uuid
import logging
from typing import List, Optional

from app.core.config import settings
from app.ingestion.tagging import get_tags
from app.qdrant.upsert import upsert_transcript_points

logger = logging.getLogger(__name__)

TRANSCRIPT_CHUNK_SEC = getattr(settings, "TRANSCRIPT_CHUNK_SEC", 60)


def _segment_into_60s(
    lines: List[dict],
) -> List[dict]:
    """
    lines: [{"text": "...", "ts": epoch_float}, ...]
    Output: [{"text": "...", "start_ts": int, "end_ts": int}, ...] with end_ts - start_ts <= 60.
    """
    if not lines:
        return []
    chunks = []
    current_text = []
    current_start = None
    current_end = None
    for line in lines:
        ts = int(float(line.get("ts") or 0))
        text = (line.get("text") or "").strip()
        if current_start is None:
            current_start = ts
            current_end = ts
        if ts - current_start >= TRANSCRIPT_CHUNK_SEC and current_text:
            chunks.append({
                "text": " ".join(current_text),
                "start_ts": current_start,
                "end_ts": current_end,
            })
            current_start = ts
            current_end = ts
            current_text = []
        current_text.append(text)
        current_end = ts
    if current_text:
        chunks.append({
            "text": " ".join(current_text),
            "start_ts": current_start,
            "end_ts": current_end or current_start,
        })
    return chunks


def ingest_transcript_chunk(
    transcript_id: str,
    text: str,
    ts: float,
    location: Optional[str] = None,
    tags: Optional[List[str]] = None,
    timezone: Optional[str] = None,
) -> None:
    """
    Ingest one chunk (e.g. one 60s segment). If text is empty (silence), optionally skip.
    """
    if not (text or "").strip():
        return
    ingested_at = int(time.time())
    start_ts = int(ts)
    end_ts = start_ts + TRANSCRIPT_CHUNK_SEC
    auto_tags = get_tags(text)
    all_tags = list(dict.fromkeys((tags or []) + auto_tags))[:20]
    chunk_id = str(uuid.uuid4())
    payload = {
        "transcript_id": transcript_id,
        "chunk_id": chunk_id,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "ingested_at": ingested_at,
        "location": location or "",
        "tags": all_tags,
        "timezone": timezone or "",
    }
    upsert_transcript_points([text], [payload])


def ingest_transcript_batch(
    transcript_id: str,
    lines: List[dict],
    location: Optional[str] = None,
    timezone: Optional[str] = None,
) -> None:
    """
    lines: [{"text": "...", "ts": epoch_float}, ...]. Segment into 60s chunks, tag, embed, upsert.
    """
    chunks = _segment_into_60s(lines)
    ingested_at = int(time.time())
    texts = []
    payloads = []
    for c in chunks:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        start_ts = c["start_ts"]
        end_ts = c["end_ts"]
        auto_tags = get_tags(text)
        chunk_id = str(uuid.uuid4())
        payloads.append({
            "transcript_id": transcript_id,
            "chunk_id": chunk_id,
            "text_preview": text[:300],
            "start_ts": start_ts,
            "end_ts": end_ts,
            "ingested_at": ingested_at,
            "location": location or "",
            "tags": auto_tags[:20],
            "timezone": timezone or "",
        })
        texts.append(text)
    if texts:
        upsert_transcript_points(texts, payloads)
    logger.info("Ingested transcript %s: %s chunks", transcript_id, len(texts))
