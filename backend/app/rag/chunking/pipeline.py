"""
Orchestrator: detect type, sanitize, dispatch chunker, assign IDs.
"""
from __future__ import annotations
from typing import List

from ...utils.ids import new_id

from .models import Chunk, DocType
from .detect import detect_document_type
from .sanitize import sanitize_text
from .chunkers import (
    chunk_faq,
    chunk_long_form,
    chunk_sensitive,
    chunk_unstructured,
)


def chunk_document(text: str, doc_id: str) -> List[Chunk]:
    """
    Full pipeline: detect document type, sanitize (PII redaction), chunk by strategy, assign IDs.
    Returns a flat list of Chunk (for long-form: parent + children; only children are used for retrieval).
    """
    if not (text or "").strip():
        return []

    doc_type = detect_document_type(text)
    clean_text, redacted, sensitivity_level = sanitize_text(text)

    if doc_type == DocType.FAQ:
        chunks = chunk_faq(clean_text, sensitivity_level, redacted)
    elif doc_type == DocType.BOOK:
        pc_list = chunk_long_form(clean_text, sensitivity_level, redacted)
        chunks = []
        for pc in pc_list:
            parent = pc.parent
            parent.chunk_id = new_id("chk")
            parent.doc_id = doc_id
            chunks.append(parent)
            for c in pc.children:
                c.parent_chunk_id = parent.chunk_id
                c.doc_id = doc_id
                c.chunk_id = new_id("chk")
                chunks.append(c)
        _assign_indices(chunks)
        return chunks
    elif doc_type == DocType.SENSITIVE:
        chunks = chunk_sensitive(clean_text, sensitivity_level, redacted)
    else:
        chunks = chunk_unstructured(clean_text, sensitivity_level, redacted)

    for i, c in enumerate(chunks):
        c.doc_id = doc_id
        c.chunk_id = new_id("chk")
        c.chunk_index = i
    return chunks


def _assign_indices(chunks: List[Chunk]) -> None:
    """Set chunk_index by order (parents then children per parent)."""
    for i, c in enumerate(chunks):
        c.chunk_index = i
