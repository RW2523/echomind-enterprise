"""Pydantic models for Silent Assistant correction findings (display-only; never TTS)."""
from __future__ import annotations

import json
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class SilentAssistantMode(str, Enum):
    SILENT_ASSISTANT = "SILENT_ASSISTANT"


class FindingCategory(str, Enum):
    rules_violation = "rules_violation"
    factual_inconsistency = "factual_inconsistency"
    contradiction_with_indexed_knowledge = "contradiction_with_indexed_knowledge"
    unsupported_claim = "unsupported_claim"
    possible_misinterpretation = "possible_misinterpretation"
    useful_suggestion = "useful_suggestion"
    needs_verification = "needs_verification"


class StatusLabel(str, Enum):
    likely_correct = "likely_correct"
    possibly_wrong = "possibly_wrong"
    unsupported = "unsupported"
    contradicted = "contradicted"
    needs_verification = "needs_verification"
    suggestion_available = "suggestion_available"


class EvidenceStatus(str, Enum):
    grounded = "grounded"
    partial = "partial"
    weak = "weak"
    none = "none"


class UserAction(str, Enum):
    pending = "pending"
    accepted = "accepted"
    dismissed = "dismissed"
    marked_unhelpful = "marked_unhelpful"
    saved = "saved"
    pinned = "pinned"


class SourceOrigin(str, Enum):
    transcript = "transcript"
    rag = "rag"
    rules = "rules"
    rules_plus_rag = "rules_plus_rag"
    transcript_plus_rag = "transcript_plus_rag"
    none = "none"


class SilentAnalyzeIn(BaseModel):
    """Analyze a stable transcript segment or merged turn (caller must not send every partial token)."""

    text: str = Field(..., min_length=8, max_length=8000)
    transcript_segment_id: Optional[str] = Field(default=None, max_length=128)
    turn_id: Optional[str] = Field(default=None, max_length=128)
    use_knowledge_base: bool = False
    active_mode: str = Field(default="SILENT_ASSISTANT", max_length=64)
    active_rule_hints: List[str] = Field(
        default_factory=list,
        description="Optional substrings; if any match the segment, a rules-oriented finding may be emitted.",
    )
    context_window: str = Field(default="all", max_length=32)


class CorrectionFindingOut(BaseModel):
    id: str
    session_id: str
    transcript_segment_id: Optional[str] = None
    turn_id: Optional[str] = None
    original_text: str
    highlighted_span_start: int = 0
    highlighted_span_end: int = 0
    category: FindingCategory
    status_label: StatusLabel
    suggested_correction: str = ""
    reason: str
    evidence_status: EvidenceStatus
    confidence: float = Field(ge=0.0, le=1.0)
    source_origin: SourceOrigin
    citations: List[dict] = Field(default_factory=list)
    created_at: str
    user_action: UserAction
    influencing_rule_set_id: Optional[str] = None
    influencing_rule_set_name: Optional[str] = None
    influencing_rule_id: Optional[str] = None
    influencing_rule_title: Optional[str] = None

    model_config = {"from_attributes": True}


class SilentAnalyzeOut(BaseModel):
    findings: List[CorrectionFindingOut]
    skipped_reason: Optional[str] = None


def row_to_finding_out(row: tuple) -> CorrectionFindingOut:
    influencing_rule_set_id: Optional[str] = None
    influencing_rule_set_name: Optional[str] = None
    influencing_rule_id: Optional[str] = None
    influencing_rule_title: Optional[str] = None
    if len(row) >= 21:
        (
            fid,
            session_id,
            seg_id,
            turn_id,
            original_text,
            h0,
            h1,
            category,
            status_label,
            suggested,
            reason,
            evidence_status,
            confidence,
            source_origin,
            citations_json,
            created_at,
            user_action,
            influencing_rule_set_id,
            influencing_rule_set_name,
            influencing_rule_id,
            influencing_rule_title,
        ) = row
    else:
        (
            fid,
            session_id,
            seg_id,
            turn_id,
            original_text,
            h0,
            h1,
            category,
            status_label,
            suggested,
            reason,
            evidence_status,
            confidence,
            source_origin,
            citations_json,
            created_at,
            user_action,
        ) = row
    cites: List[Any] = []
    if citations_json:
        try:
            cites = json.loads(citations_json)
            if not isinstance(cites, list):
                cites = []
        except Exception:
            cites = []
    return CorrectionFindingOut(
        id=fid,
        session_id=session_id,
        transcript_segment_id=seg_id,
        turn_id=turn_id,
        original_text=original_text or "",
        highlighted_span_start=int(h0 or 0),
        highlighted_span_end=int(h1 or 0),
        category=FindingCategory(category),
        status_label=StatusLabel(status_label),
        suggested_correction=suggested or "",
        reason=reason or "",
        evidence_status=EvidenceStatus(evidence_status),
        confidence=float(confidence or 0.0),
        source_origin=SourceOrigin(source_origin),
        citations=cites,
        created_at=created_at,
        user_action=UserAction(user_action),
        influencing_rule_set_id=influencing_rule_set_id,
        influencing_rule_set_name=influencing_rule_set_name,
        influencing_rule_id=influencing_rule_id,
        influencing_rule_title=influencing_rule_title,
    )
