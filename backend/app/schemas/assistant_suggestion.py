"""Pydantic models for Assistant Mode suggestions (aligned with frontend `Suggestion`)."""
from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class SuggestionMode(str, Enum):
    ASSISTANT = "ASSISTANT"


class SuggestionCategory(str, Enum):
    fact_check = "fact_check"
    contradiction = "contradiction"
    relevant_knowledge = "relevant_knowledge"
    action_reminder = "action_reminder"
    follow_up_question = "follow_up_question"
    clarification = "clarification"
    summary_help = "summary_help"
    missing_context = "missing_context"


class SuggestionStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    dismissed = "dismissed"
    ignored = "ignored"
    saved = "saved"
    spoken = "spoken"


class SourceOrigin(str, Enum):
    transcript = "transcript"
    rag = "rag"
    rules = "rules"
    rules_plus_rag = "rules_plus_rag"
    transcript_plus_rag = "transcript_plus_rag"
    notes = "notes"
    transcript_plus_notes = "transcript_plus_notes"
    notes_plus_rag = "notes_plus_rag"
    none = "none"


class EvidenceStatus(str, Enum):
    grounded = "grounded"
    partial = "partial"
    weak = "weak"
    none = "none"


class SuggestionBase(BaseModel):
    mode: SuggestionMode = SuggestionMode.ASSISTANT
    title: str = Field(..., min_length=1, max_length=500)
    short_text: str = Field(..., min_length=1, max_length=2000)
    speak_text: str = Field(..., min_length=1, max_length=8000)
    reason: str = Field(default="", max_length=4000)
    category: SuggestionCategory
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    source_origin: SourceOrigin
    evidence_status: EvidenceStatus
    citations: List[dict] = Field(default_factory=list)


class SuggestionCreate(SuggestionBase):
    session_id: str = Field(..., min_length=4, max_length=128)


class SuggestionOut(SuggestionBase):
    id: str
    session_id: str
    created_at: str
    status: SuggestionStatus
    influencing_rule_set_id: Optional[str] = None
    influencing_rule_set_name: Optional[str] = None
    influencing_rule_id: Optional[str] = None
    influencing_rule_title: Optional[str] = None
    # Substring of transcript used for retrieval / UI highlight.
    trigger_excerpt: Optional[str] = None

    model_config = {"from_attributes": True}


class GenerateSuggestionsIn(BaseModel):
    recent_transcript: str = Field(..., max_length=120_000)
    use_knowledge_base: bool = False
    context_window: str = Field(default="all", max_length=32)


class GenerateSuggestionsOut(BaseModel):
    suggestions: List[SuggestionOut]
    skipped_reason: Optional[str] = None


def row_to_suggestion_out(row: tuple) -> SuggestionOut:
    """Map SQLite row (full column order) to SuggestionOut."""
    import json

    influencing_rule_set_id: Optional[str] = None
    influencing_rule_set_name: Optional[str] = None
    influencing_rule_id: Optional[str] = None
    influencing_rule_title: Optional[str] = None
    trigger_excerpt: Optional[str] = None
    if len(row) >= 19:
        (
            sid,
            session_id,
            mode,
            title,
            short_text,
            speak_text,
            reason,
            category,
            confidence,
            source_origin,
            evidence_status,
            citations_json,
            created_at,
            status,
            influencing_rule_set_id,
            influencing_rule_set_name,
            influencing_rule_id,
            influencing_rule_title,
            trigger_excerpt,
        ) = row
    elif len(row) >= 18:
        (
            sid,
            session_id,
            mode,
            title,
            short_text,
            speak_text,
            reason,
            category,
            confidence,
            source_origin,
            evidence_status,
            citations_json,
            created_at,
            status,
            influencing_rule_set_id,
            influencing_rule_set_name,
            influencing_rule_id,
            influencing_rule_title,
        ) = row
    else:
        (
            sid,
            session_id,
            mode,
            title,
            short_text,
            speak_text,
            reason,
            category,
            confidence,
            source_origin,
            evidence_status,
            citations_json,
            created_at,
            status,
        ) = row
    cites: List[Any] = []
    if citations_json:
        try:
            cites = json.loads(citations_json)
            if not isinstance(cites, list):
                cites = []
        except Exception:
            cites = []
    return SuggestionOut(
        id=sid,
        session_id=session_id,
        mode=SuggestionMode(mode),
        title=title,
        short_text=short_text,
        speak_text=speak_text,
        reason=reason or "",
        category=SuggestionCategory(category),
        confidence=float(confidence),
        source_origin=SourceOrigin(source_origin),
        evidence_status=EvidenceStatus(evidence_status),
        citations=cites,
        created_at=created_at,
        status=SuggestionStatus(status),
        influencing_rule_set_id=influencing_rule_set_id,
        influencing_rule_set_name=influencing_rule_set_name,
        influencing_rule_id=influencing_rule_id,
        influencing_rule_title=influencing_rule_title,
        trigger_excerpt=trigger_excerpt,
    )
