from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...assistant import insights_store
from ...assistant.live_analysis import analyze_window

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assistant", tags=["assistant"])

ClassificationLiteral = Literal["supported", "contradicted", "related", "missing_context", "warning"]
PriorityLiteral = Literal["low", "medium", "high"]
SourceTypeLiteral = Literal["document", "transcript", "book", "faq", "unknown"]
ModeLiteral = Literal["silent_assistant", "personal_assistant"]
ActionStatusLiteral = Literal["ignored", "saved_for_later", "viewed", "asked_follow_up", "spoke_now"]


class AnalysisScopeIn(BaseModel):
    documents: bool = True
    transcripts: bool = False
    books: bool = True
    faqs: bool = True


class AnalyzeWindowRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=256)
    mode: ModeLiteral = "silent_assistant"
    transcript_window: str = ""
    rolling_context: str = ""
    analysis_scope: AnalysisScopeIn = Field(default_factory=AnalysisScopeIn)


class AssistantEvidenceOut(BaseModel):
    source_name: str
    source_type: SourceTypeLiteral = "unknown"
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    matched_text: str = ""

    @classmethod
    def from_hit_dict(cls, d: dict) -> "AssistantEvidenceOut":
        p = d.get("page")
        if p is not None and not isinstance(p, int):
            try:
                p = int(str(p).split(".")[0]) if str(p).strip() else None
            except (ValueError, TypeError):
                p = None
        st = d.get("source_type") or "unknown"
        if st not in ("document", "transcript", "book", "faq", "unknown"):
            st = "unknown"
        return cls(
            source_name=str(d.get("source_name") or "source"),
            source_type=st,  # type: ignore[arg-type]
            doc_id=d.get("doc_id"),
            chunk_id=d.get("chunk_id"),
            page=p,
            section=d.get("section"),
            matched_text=str(d.get("matched_text") or ""),
        )


class AssistantInsightOut(BaseModel):
    id: str
    transcript_text: str
    classification: ClassificationLiteral
    confidence: float = Field(..., ge=0.0, le=1.0)
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    paragraph_id: Optional[str] = None
    show_highlight: bool = True
    show_hand_raise: bool = False
    priority: PriorityLiteral = "medium"
    evidence: List[AssistantEvidenceOut] = Field(default_factory=list)
    assistant_interpretation: str = ""
    suggested_action: str = ""
    suggested_response: Optional[str] = None


class AssistantInsightPersistedOut(AssistantInsightOut):
    action_status: Optional[str] = None
    created_at: Optional[str] = None


class BulkSaveInsightsRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=256)
    mode: ModeLiteral
    transcript_id: Optional[str] = Field(None, max_length=256)
    insights: List[AssistantInsightOut] = Field(default_factory=list)


class BulkSaveInsightsResponse(BaseModel):
    session_id: str
    inserted: int
    skipped: int
    duplicate_merged: int
    id_map: Dict[str, str] = Field(default_factory=dict)


class InsightActionPatch(BaseModel):
    action_status: ActionStatusLiteral


class ListSessionInsightsResponse(BaseModel):
    session_id: str
    insights: List[AssistantInsightPersistedOut]


class AnalyzeWindowResponse(BaseModel):
    session_id: str
    mode: str
    insights: List[AssistantInsightOut]
def _coerce_classification(c: str) -> ClassificationLiteral:
    v = (c or "related").strip().lower()
    if v in ("supported", "contradicted", "related", "missing_context", "warning"):
        return v  # type: ignore[return-value]
    return "related"


def _coerce_priority(p: str) -> PriorityLiteral:
    v = (p or "medium").strip().lower()
    if v in ("low", "medium", "high"):
        return v  # type: ignore[return-value]
    return "medium"


def _insight_from_raw(raw: dict) -> AssistantInsightOut:
    evs = [AssistantEvidenceOut.from_hit_dict(e) if isinstance(e, dict) else AssistantEvidenceOut(source_name="?", matched_text="") for e in (raw.get("evidence") or [])]
    return AssistantInsightOut(
        id=str(raw.get("id") or ""),
        transcript_text=str(raw.get("transcript_text") or ""),
        classification=_coerce_classification(str(raw.get("classification") or "")),
        confidence=float(raw.get("confidence") or 0),
        start_char=raw.get("start_char"),
        end_char=raw.get("end_char"),
        paragraph_id=raw.get("paragraph_id"),
        show_highlight=bool(raw.get("show_highlight", True)),
        show_hand_raise=bool(raw.get("show_hand_raise", False)),
        priority=_coerce_priority(str(raw.get("priority") or "medium")),
        evidence=evs,
        assistant_interpretation=str(raw.get("assistant_interpretation") or ""),
        suggested_action=str(raw.get("suggested_action") or ""),
        suggested_response=raw.get("suggested_response"),
    )


def _persisted_from_store_dict(d: Dict[str, Any]) -> AssistantInsightPersistedOut:
    raw = {k: v for k, v in d.items() if k not in ("created_at", "action_status", "mode")}
    base = _insight_from_raw(raw)
    return AssistantInsightPersistedOut(
        **base.model_dump(),
        action_status=d.get("action_status"),
        created_at=d.get("created_at"),
    )


@router.get("/sessions/{session_id}/insights", response_model=ListSessionInsightsResponse)
async def get_session_insights(session_id: str) -> ListSessionInsightsResponse:
    sid = (session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id required")
    rows = insights_store.list_assistant_insights_by_session(sid)
    out: List[AssistantInsightPersistedOut] = []
    for r in rows:
        try:
            out.append(_persisted_from_store_dict(r))
        except Exception as e:
            logger.warning("assistant: skip malformed persisted insight: %s", e)
    return ListSessionInsightsResponse(session_id=sid, insights=out)


@router.post("/insights/bulk-save", response_model=BulkSaveInsightsResponse)
async def post_insights_bulk_save(body: BulkSaveInsightsRequest) -> BulkSaveInsightsResponse:
    if body.mode not in ("silent_assistant", "personal_assistant"):
        raise HTTPException(status_code=400, detail="Unsupported mode")
    rows = [i.model_dump(mode="json") for i in body.insights]
    result = insights_store.bulk_save_assistant_insights(
        body.session_id.strip(),
        body.mode,
        (body.transcript_id or "").strip() or None,
        rows,
    )
    return BulkSaveInsightsResponse(**result)


@router.patch("/insights/{insight_id}/action")
async def patch_insight_action(insight_id: str, body: InsightActionPatch) -> dict:
    iid = (insight_id or "").strip()
    if not iid:
        raise HTTPException(status_code=400, detail="insight_id required")
    try:
        ok = insights_store.update_assistant_insight_action_status(iid, body.action_status)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not ok:
        raise HTTPException(status_code=404, detail="Insight not found")
    return {"ok": True, "id": iid, "action_status": body.action_status}


@router.post("/analyze-window", response_model=AnalyzeWindowResponse)
async def post_analyze_window(body: AnalyzeWindowRequest) -> AnalyzeWindowResponse:
    if body.mode not in ("silent_assistant", "personal_assistant"):
        raise HTTPException(status_code=400, detail="Unsupported mode")
    scope = body.analysis_scope.model_dump()
    result = await analyze_window(
        session_id=body.session_id.strip(),
        mode=body.mode,
        transcript_window=body.transcript_window or "",
        rolling_context=body.rolling_context or "",
        analysis_scope=scope,
    )
    insights_out: List[AssistantInsightOut] = []
    for raw in result.get("insights") or []:
        if not isinstance(raw, dict):
            continue
        try:
            insights_out.append(_insight_from_raw(raw))
        except Exception as e:
            logger.warning("assistant: skip malformed insight: %s", e)
            continue
    return AnalyzeWindowResponse(
        session_id=result.get("session_id") or body.session_id,
        mode=result.get("mode") or body.mode,
        insights=insights_out,
    )
