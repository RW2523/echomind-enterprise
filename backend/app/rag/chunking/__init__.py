"""
Production-grade adaptive chunking pipeline for RAG.
- DocType detection, PII sanitization, type-specific chunkers, hierarchical long-form.
"""
from .models import DocType, Chunk, SensitivityLevel
from .detect import detect_document_type
from .sanitize import sanitize_text
from .chunkers import (
    chunk_faq,
    chunk_long_form,
    chunk_sensitive,
    chunk_unstructured,
)
from .pipeline import chunk_document

__all__ = [
    "DocType",
    "Chunk",
    "SensitivityLevel",
    "detect_document_type",
    "sanitize_text",
    "chunk_faq",
    "chunk_long_form",
    "chunk_sensitive",
    "chunk_unstructured",
    "chunk_document",
]
