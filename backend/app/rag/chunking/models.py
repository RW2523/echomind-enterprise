"""
Chunk and document-type models for the chunking pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DocType(str, Enum):
    BOOK = "book"
    FAQ = "faq"
    USER = "user"
    SENSITIVE = "sensitive"


class SensitivityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Chunk:
    """Single chunk for embedding and retrieval. Parent chunks (long-form) are not embedded; children are."""

    doc_id: str
    chunk_id: str
    text: str
    doc_type: DocType
    sensitivity_level: SensitivityLevel
    redacted: bool
    section: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    is_parent: bool = False
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    chunk_index: int = 0

    def to_source_dict(self, filename: str, filetype: str) -> dict:
        """Serialize for source_json in DB (backward-compatible + extended metadata). doc_id and chunk_index used for structure fallback and adjacent expansion."""
        d = {
            "doc_id": self.doc_id,
            "filename": filename,
            "chunk_index": self.chunk_index,
            "filetype": filetype,
            "doc_type": self.doc_type.value,
            "section": self.section,
            "sensitivity_level": self.sensitivity_level.value,
            "redacted": self.redacted,
            "is_parent": self.is_parent,
        }
        if self.parent_chunk_id is not None:
            d["parent_chunk_id"] = self.parent_chunk_id
        return d


@dataclass
class ParentChildChunk:
    """Long-form: one parent (large context) and multiple children (for retrieval)."""

    parent: Chunk
    children: list[Chunk] = field(default_factory=list)
