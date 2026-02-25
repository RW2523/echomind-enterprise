"""Transcript ingest API."""
from __future__ import annotations
import logging
from typing import List, Optional

from fastapi import APIRouter, Body

from app.ingestion.pipeline_transcript import ingest_transcript_chunk, ingest_transcript_batch

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/transcripts", tags=["transcripts"])


@router.post("/ingest")
async def ingest_transcript(
    transcript_id: str = Body(..., embed=True),
    ts: float = Body(..., embed=True),
    location: Optional[str] = Body(None, embed=True),
    text: str = Body("", embed=True),
    tags: Optional[List[str]] = Body(None, embed=True),
    timezone: Optional[str] = Body(None, embed=True),
) -> dict:
    """Ingest one transcript chunk (e.g. one 60s segment)."""
    ingest_transcript_chunk(transcript_id=transcript_id, text=text, ts=ts, location=location, tags=tags, timezone=timezone)
    return {"status": "ok", "transcript_id": transcript_id}


@router.post("/ingest_batch")
async def ingest_transcript_batch_route(
    transcript_id: str = Body(..., embed=True),
    lines: List[dict] = Body(..., embed=True),
    location: Optional[str] = Body(None, embed=True),
    timezone: Optional[str] = Body(None, embed=True),
) -> dict:
    """Ingest batch of lines: [{"text": "...", "ts": epoch_float}, ...]. Segmented into 60s chunks."""
    ingest_transcript_batch(transcript_id=transcript_id, lines=lines, location=location, timezone=timezone)
    return {"status": "ok", "transcript_id": transcript_id}
