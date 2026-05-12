"""Request/response models for unified KB-only transcript analysis (Assistant + Silent Assistant)."""
from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class AnalyzeTranscriptMode(str, Enum):
    assistant = "assistant"
    silent_assistant = "silent_assistant"


class KbFindingLabel(str, Enum):
    supported = "Supported"
    contradicted = "Contradicted"
    related = "Related"
    unverified = "Unverified"
    needs_review = "Needs Review"


class AssistantSourceOut(BaseModel):
    document_id: Optional[str] = None
    document_name: str = ""
    page: Optional[int] = None
    snippet: str = ""
    score: float = 0.0


class AssistantAnalysisItemOut(BaseModel):
    id: str
    text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    label: KbFindingLabel
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_status: Literal["grounded", "partial", "weak", "none"]
    explanation: str = ""
    feedback: str = ""
    speak_text: str = ""
    sources: List[AssistantSourceOut] = Field(default_factory=list)
    persisted_id: Optional[str] = Field(
        default=None,
        description="SQLite id when this item was stored (assistant_suggestion or silent_finding).",
    )


class AnalyzeTranscriptIn(BaseModel):
    session_id: str = Field(..., min_length=4, max_length=128)
    mode: Literal["assistant", "silent_assistant"]
    transcript_text: str = Field(..., min_length=8, max_length=120_000)
    full_transcript: Optional[str] = Field(
        default=None,
        max_length=120_000,
        description="Full transcript for global char offsets; if omitted, offsets are relative to transcript_text.",
    )
    transcript_offset: int = Field(
        default=0,
        ge=0,
        description="Index in full_transcript where transcript_text begins (when sending an append-only slice).",
    )
    since_last_analysis: bool = True
    knowledge_base_enabled: bool = True
    context_window: str = Field(default="all", max_length=32)
    persist_results: bool = Field(
        default=True,
        description="When true, inserts assistant suggestions or silent findings for qualifying items.",
    )


class AnalyzeTranscriptOut(BaseModel):
    items: List[AssistantAnalysisItemOut] = Field(default_factory=list)
    skipped_reason: Optional[str] = None
