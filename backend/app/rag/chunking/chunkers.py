"""
Type-specific chunkers. Sentence-aware; no mid-sentence splits.
"""
from __future__ import annotations
import re
from typing import List

from .models import Chunk, DocType, ParentChildChunk, SensitivityLevel

# Sentence boundary: . ! ? followed by space or end
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n\n+")
# Min/max child size for long-form
_PARENT_MIN, _PARENT_MAX = 2000, 3500
_CHILD_MIN, _CHILD_MAX = 400, 700
_CHILD_OVERLAP = 80
# Sensitive: small, minimal overlap
_SENSITIVE_SIZE = 450
_SENSITIVE_OVERLAP = 40
# Unstructured (user): medium, sentence-aware
_USER_SIZE = 800
_USER_OVERLAP = 120
# FAQ: whole Q+A per chunk
# No fixed size; one chunk per Q&A pair


def _sentences(text: str) -> List[str]:
    """Split into sentences (preserving boundaries)."""
    if not text.strip():
        return []
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _group_sentences_to_size(
    sentences: List[str],
    target_min: int,
    target_max: int,
    overlap_sentences: int = 0,
) -> List[str]:
    """Group sentences into chunks of target size; never split a sentence."""
    if not sentences:
        return []
    chunks = []
    current: List[str] = []
    current_len = 0
    for s in sentences:
        if current_len + len(s) + 1 > target_max and current:
            chunk_text = " ".join(current)
            chunks.append(chunk_text)
            if overlap_sentences > 0 and len(current) > overlap_sentences:
                current = current[-overlap_sentences:]
                current_len = sum(len(x) for x in current) + len(current) - 1
            else:
                current = []
                current_len = 0
        current.append(s)
        current_len += len(s) + (1 if current else 0)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_faq(
    text: str,
    sensitivity_level: SensitivityLevel,
    redacted: bool,
) -> List[Chunk]:
    """
    One chunk per Q&A pair. Never split question from answer.
    Splits on start of next question (e.g. "Q:", "1. Q", "Question 2:").
    """
    chunks_out: List[Chunk] = []
    text = (text or "").strip()
    if not text:
        return []

    next_q = re.compile(r"\n\s*(?:\d+[.)]\s*)?(?:Q(?:uestion)?\s*[:.]?\s*)", re.IGNORECASE)
    parts = next_q.split(text)
    if len(parts) <= 1:
        parts = [text]
    idx = 0
    for block in parts:
        block = block.strip()
        if not block or len(block) < 5:
            continue
        chunks_out.append(
            Chunk(
                doc_id="",
                chunk_id="",
                text=block[:12000],
                doc_type=DocType.FAQ,
                sensitivity_level=sensitivity_level,
                redacted=redacted,
                section=None,
                parent_chunk_id=None,
                is_parent=False,
                chunk_index=idx,
            )
        )
        idx += 1
    if not chunks_out:
        chunks_out.append(
            Chunk(
                doc_id="",
                chunk_id="",
                text=text[:12000],
                doc_type=DocType.FAQ,
                sensitivity_level=sensitivity_level,
                redacted=redacted,
                section=None,
                parent_chunk_id=None,
                is_parent=False,
                chunk_index=0,
            )
        )
    return chunks_out


def chunk_long_form(
    text: str,
    sensitivity_level: SensitivityLevel,
    redacted: bool,
) -> List[ParentChildChunk]:
    """
    Parent chunks (2k–3k chars) for context; child chunks (400–700 chars) for retrieval.
    Children reference parent_chunk_id. Sentence-boundary aware.
    """
    text = (text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    parent_chunks: List[str] = []
    current = []
    current_len = 0
    for p in paragraphs:
        if current_len + len(p) + 2 > _PARENT_MAX and current:
            parent_chunks.append("\n\n".join(current))
            overlap_paras = [current[-1]] if len(current) >= 1 else []
            current = overlap_paras + [p] if len(p) <= _PARENT_MAX else [p]
            current_len = sum(len(x) for x in current) + 2 * (len(current) - 1)
        else:
            current.append(p)
            current_len += len(p) + (2 if len(current) > 1 else 0)
    if current:
        parent_chunks.append("\n\n".join(current))

    out: List[ParentChildChunk] = []
    for pi, parent_text in enumerate(parent_chunks):
        if len(parent_text) < _PARENT_MIN and pi < len(parent_chunks) - 1:
            continue
        sentences = _sentences(parent_text)
        child_texts = _group_sentences_to_size(
            sentences,
            _CHILD_MIN,
            _CHILD_MAX,
            overlap_sentences=2,
        )
        if not child_texts:
            child_texts = [parent_text[: _CHILD_MAX]] if parent_text else []

        parent_chunk = Chunk(
            doc_id="",
            chunk_id="",
            text=parent_text,
            doc_type=DocType.BOOK,
            sensitivity_level=sensitivity_level,
            redacted=redacted,
            section=None,
            parent_chunk_id=None,
            is_parent=True,
            chunk_index=pi,
        )
        children = [
            Chunk(
                doc_id="",
                chunk_id="",
                text=ct,
                doc_type=DocType.BOOK,
                sensitivity_level=sensitivity_level,
                redacted=redacted,
                section=None,
                parent_chunk_id="",  # set in pipeline
                is_parent=False,
                chunk_index=pi * 100 + ji,
            )
            for ji, ct in enumerate(child_texts)
        ]
        out.append(ParentChildChunk(parent=parent_chunk, children=children))
    return out


def chunk_sensitive(
    text: str,
    sensitivity_level: SensitivityLevel,
    redacted: bool,
) -> List[Chunk]:
    """Small chunks, minimal overlap; sentence-boundary only."""
    text = (text or "").strip()
    if not text:
        return []
    sentences = _sentences(text)
    chunk_texts = _group_sentences_to_size(
        sentences,
        target_min=_SENSITIVE_SIZE // 2,
        target_max=_SENSITIVE_SIZE,
        overlap_sentences=0,
    )
    if not chunk_texts:
        chunk_texts = [text[: _SENSITIVE_SIZE]]
    return [
        Chunk(
            doc_id="",
            chunk_id="",
            text=ct,
            doc_type=DocType.SENSITIVE,
            sensitivity_level=sensitivity_level,
            redacted=redacted,
            section=None,
            parent_chunk_id=None,
            is_parent=False,
            chunk_index=i,
        )
        for i, ct in enumerate(chunk_texts)
    ]


def chunk_unstructured(
    text: str,
    sensitivity_level: SensitivityLevel,
    redacted: bool,
) -> List[Chunk]:
    """Robust chunking for poor/mixed formatting; sentence-aware, medium size."""
    text = (text or "").strip()
    if not text:
        return []
    sentences = _sentences(text)
    chunk_texts = _group_sentences_to_size(
        sentences,
        target_min=_USER_SIZE // 2,
        target_max=_USER_SIZE,
        overlap_sentences=1,
    )
    if not chunk_texts:
        chunk_texts = [text[: _USER_SIZE * 2]]
    return [
        Chunk(
            doc_id="",
            chunk_id="",
            text=ct,
            doc_type=DocType.USER,
            sensitivity_level=sensitivity_level,
            redacted=redacted,
            section=None,
            parent_chunk_id=None,
            is_parent=False,
            chunk_index=i,
        )
        for i, ct in enumerate(chunk_texts)
    ]
