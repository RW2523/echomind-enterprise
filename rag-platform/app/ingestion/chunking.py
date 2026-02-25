"""
Token-aware chunking for documents: ~800 tokens, ~120 overlap.
Preserves structure: PDF page windows, DOCX by headings, PPTX per slide, CSV/XLSX by row blocks.
"""
from __future__ import annotations
import re
import uuid
from typing import Any, List, Optional, Tuple

from app.core.config import settings

CHUNK_SIZE = getattr(settings, "CHUNK_SIZE", 800)
CHUNK_OVERLAP = getattr(settings, "CHUNK_OVERLAP", 120)


def token_len(text: str) -> int:
    if not (text or "").strip():
        return 0
    t = text.strip()
    return max((len(t) + 3) // 4, len(t.split()) // 2)


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _group_to_chunks(sentences: List[str], target_size: int, overlap: int) -> List[str]:
    if not sentences:
        return []
    chunks = []
    current = []
    current_tokens = 0
    for s in sentences:
        st = token_len(s)
        if current_tokens + st > target_size and current:
            chunk_text = " ".join(current)
            chunks.append(chunk_text)
            overlap_count = max(1, min(len(current), overlap // 50))
            current = current[-overlap_count:]
            current_tokens = token_len(" ".join(current))
        current.append(s)
        current_tokens += token_len(" ") + st if current_tokens else st
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_text_with_structure(
    full_text: str,
    structure: List[dict],
    file_type: str,
    doc_title: str,
) -> List[dict]:
    """
    Produce retrieval chunks with metadata: text, section_path, page_start/end, row_start/end.
    structure: from extractors (e.g. [{"page": 1, "text": "..."}] or [{"section": "Ch 3", "text": "..."}]).
    Each chunk: {text, section_path, page_start, page_end, row_start, row_end, chunk_id}.
    """
    out = []
    chunk_index = 0
    if file_type.lower() in ("csv", "xlsx"):
        # Row blocks: 50–200 rows per chunk, include headers
        for block in structure:
            rows = block.get("rows") or []
            header = block.get("header") or []
            if not rows:
                if block.get("text"):
                    out.append({
                        "text": block["text"],
                        "section_path": block.get("section", ""),
                        "page_start": None,
                        "page_end": None,
                        "row_start": None,
                        "row_end": None,
                        "chunk_id": str(uuid.uuid4()),
                    })
                continue
            block_size = min(200, max(50, len(rows) // 4))
            for i in range(0, len(rows), block_size):
                batch = rows[i : i + block_size]
                text = "\n".join([", ".join(str(v) for v in r.values()) for r in batch])
                if header:
                    text = ", ".join(header) + "\n" + text
                out.append({
                    "text": text,
                    "section_path": block.get("section", ""),
                    "page_start": None,
                    "page_end": None,
                    "row_start": i + 1,
                    "row_end": i + len(batch),
                    "chunk_id": str(uuid.uuid4()),
                })
        return out

    # PDF/DOCX/PPTX/TXT: merge small pages/sections, then token-based split with overlap
    for item in structure:
        section_path = item.get("section") or item.get("section_path") or ""
        if file_type.lower() == "pdf":
            page = item.get("page", 0)
            section_path = f"Page {page}" if page else section_path
        if file_type.lower() == "pptx":
            slide = item.get("slide", 0)
            section_path = f"Slide {slide}" if slide else section_path

        text = item.get("text", "").strip()
        if not text:
            continue
        sentences = _sentences(text)
        target = CHUNK_SIZE
        overlap_tokens = min(CHUNK_OVERLAP, target // 4)
        sub_chunks = _group_to_chunks(sentences, target, overlap_tokens)
        page_start = item.get("page") or item.get("slide")
        page_end = page_start
        for sc in sub_chunks:
            out.append({
                "text": sc,
                "section_path": section_path,
                "page_start": page_start,
                "page_end": page_end,
                "row_start": None,
                "row_end": None,
                "chunk_id": str(uuid.uuid4()),
            })
    return out


def chunk_plain_text(text: str, doc_id: str) -> List[dict]:
    """Fallback: no structure, just token-based chunking."""
    structure = [{"section": "", "text": text}]
    return chunk_text_with_structure(text, structure, "txt", doc_id)
