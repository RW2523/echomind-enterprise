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
    section_path: Optional[str] = None  # hierarchical path e.g. "Volume 1 > Chapter 3 > Section 0301"
    section_title: Optional[str] = None  # section heading/title for citation
    section_id: Optional[str] = None  # canonical DoD code e.g. "030201" for deterministic matching
    toc_node_id: Optional[str] = None  # optional TOC node for routing
    page_number: Optional[int] = None

    # ── BookRAG-lite++ additions (all Optional; safe for non-BOOK docs) ──────
    clause_id: Optional[str] = None       # detected clause code e.g. "030201.A" or "030201.B.1"
    prev_chunk_id: Optional[str] = None   # previous sibling chunk within same section
    next_chunk_id: Optional[str] = None   # next sibling chunk within same section
    # retrieval_text: heading path + clause label prepended to chunk text for embedding.
    # None = use self.text. Stored in chunks.retrieval_text; raw text always in chunks.text.
    retrieval_text: Optional[str] = None
    canonical_id: Optional[str] = None    # deterministic ID: vol_05_ch_03_sec_030201_page_0142_chunk_02
    page_start: Optional[int] = None      # first page spanned by this chunk (for multi-page chunks)
    page_end: Optional[int] = None        # last page spanned by this chunk
    # evidence_type: "child" | "parent" | "clause" | "table" | "page" | "section_summary"
    evidence_type: Optional[str] = None
    has_table: bool = False               # heuristic: chunk contains a table

    def to_source_dict(self, filename: str, filetype: str) -> dict:
        """Serialize for source_json in DB (backward-compatible + extended metadata)."""
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
        if self.section_path is not None:
            d["section_path"] = self.section_path
        if self.section_title is not None:
            d["section_title"] = self.section_title
        if self.section_id is not None:
            d["section_id"] = self.section_id
        if self.toc_node_id is not None:
            d["toc_node_id"] = self.toc_node_id
        if self.page_number is not None:
            d["page_number"] = self.page_number
        # BookRAG-lite++ fields
        if self.clause_id is not None:
            d["clause_id"] = self.clause_id
        if self.prev_chunk_id is not None:
            d["prev_chunk_id"] = self.prev_chunk_id
        if self.next_chunk_id is not None:
            d["next_chunk_id"] = self.next_chunk_id
        if self.canonical_id is not None:
            d["canonical_id"] = self.canonical_id
        if self.page_start is not None:
            d["page_start"] = self.page_start
        if self.page_end is not None:
            d["page_end"] = self.page_end
        if self.evidence_type is not None:
            d["evidence_type"] = self.evidence_type
        if self.has_table:
            d["has_table"] = True
        return d


@dataclass
class ParentChildChunk:
    """Long-form: one parent (large context) and multiple children (for retrieval)."""

    parent: Chunk
    children: list[Chunk] = field(default_factory=list)
