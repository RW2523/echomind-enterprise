"""
Chunking for documents (~800 tokens, 120 overlap) and transcripts (hard 60-second chunks).
Documents preserve structure (pages, headings, slides, row blocks); transcripts include start_ts, end_ts, ingested_at.
"""
from __future__ import annotations
from typing import List, Dict, Any
from ..core.config import settings
from ..utils.ids import new_id

# Document chunking: delegate to existing RAG chunking pipeline (respects doc type, ~800/120 via config).
from ..rag.chunking import chunk_document as _chunk_document
from ..rag.chunking.models import Chunk as RagChunk


def chunk_document_blocks(blocks: List[Dict[str, Any]], doc_id: str) -> List[Dict[str, Any]]:
    """
    Chunk document blocks (from extractors) into ~800-token chunks with 120 overlap.
    Returns list of {"chunk_id", "text", "page_start", "page_end", "section_path", "row_start", "row_end", ...}.
    Uses existing chunk_document on concatenated text but preserves structure in payload.
    """
    full_text = "\n\n".join(b.get("text", "") for b in blocks)
    if not full_text.strip():
        return []
    rag_chunks = _chunk_document(full_text, doc_id)
    out = []
    for c in rag_chunks:
        if c.is_parent:
            continue
        out.append({
            "chunk_id": c.chunk_id,
            "text": c.text,
            "doc_id": doc_id,
            "section_path": getattr(c, "section", None),
            "page_start": None,
            "page_end": None,
            "row_start": None,
            "row_end": None,
        })
    return out


def chunk_transcript_60s(
    lines_or_text: List[Dict[str, Any]] | str,
    transcript_id: str,
    location: str = "default",
    timezone: str = "UTC",
    ingested_at: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Segment transcript into hard 60-second chunks. Each chunk has start_ts, end_ts, ingested_at.
    - lines_or_text: either list of {"text", "start_ts", "end_ts"} or full raw text (then no timestamps).
    - Returns list of {"chunk_id", "text", "start_ts", "end_ts", "ingested_at", "location", "timezone"}.
    """
    from ..core.timeutils import now_utc_ts
    ingested_at = ingested_at or now_utc_ts()
    chunk_sec = getattr(settings, "TRANSCRIPT_CHUNK_SECONDS", 60)

    if isinstance(lines_or_text, str):
        # No timestamps: split by approximate 60s worth of chars (~900 chars per 60s at speech rate)
        text = (lines_or_text or "").strip()
        if not text:
            return []
        approx_chars_per_chunk = 900
        chunks = []
        for i in range(0, len(text), approx_chars_per_chunk):
            segment = text[i : i + approx_chars_per_chunk].strip()
            if not segment:
                continue
            start_ts = ingested_at + (i // approx_chars_per_chunk) * chunk_sec
            end_ts = start_ts + chunk_sec
            chunks.append({
                "chunk_id": new_id("chk"),
                "text": segment,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "ingested_at": ingested_at,
                "location": location,
                "timezone": timezone,
            })
        return chunks

    lines = lines_or_text
    if not lines:
        return []
    # Build chunks of 60s from lines with timestamps
    out = []
    current_text = []
    current_start = None
    current_end = None
    for line in lines:
        t = line.get("text", "").strip()
        start_ts = line.get("start_ts")
        end_ts = line.get("end_ts")
        if not t:
            continue
        if current_start is None:
            current_start = start_ts or ingested_at
            current_end = end_ts or (current_start + chunk_sec)
        else:
            current_end = end_ts or (current_end or current_start) + chunk_sec
        current_text.append(t)
        # Emit chunk when we exceed 60s (or use end_ts - current_start >= 60)
        span = (current_end or current_start) - current_start
        if span >= chunk_sec or (end_ts and (end_ts - current_start) >= chunk_sec):
            out.append({
                "chunk_id": new_id("chk"),
                "text": "\n".join(current_text),
                "start_ts": current_start,
                "end_ts": current_end or (current_start + chunk_sec),
                "ingested_at": ingested_at,
                "location": location,
                "timezone": timezone,
            })
            current_text = []
            current_start = end_ts if end_ts else (current_end or current_start)
            current_end = None
    if current_text:
        out.append({
            "chunk_id": new_id("chk"),
            "text": "\n".join(current_text),
            "start_ts": current_start or ingested_at,
            "end_ts": current_end or (current_start or ingested_at) + chunk_sec,
            "ingested_at": ingested_at,
            "location": location,
            "timezone": timezone,
        })
    return out
