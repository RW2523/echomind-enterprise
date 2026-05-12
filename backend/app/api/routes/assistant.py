"""REST API for Assistant Mode suggestions (local SQLite)."""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from ...assistant.suggestion_generator import generate_suggestions
from ...assistant.kb_transcript_analyzer import analyze_transcript
from ...assistant import suggestion_store
from ...schemas.assistant_suggestion import GenerateSuggestionsIn, GenerateSuggestionsOut, SuggestionOut
from ...schemas.transcript_analyze import AnalyzeTranscriptIn, AnalyzeTranscriptOut

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assistant", tags=["assistant"])


@router.post("/analyze-transcript", response_model=AnalyzeTranscriptOut)
async def analyze_transcript_endpoint(body: AnalyzeTranscriptIn) -> AnalyzeTranscriptOut:
    """
    Unified KB-only transcript analysis for Assistant and Silent Assistant modes.
    Uses hybrid local RAG retrieval; does not use general LLM knowledge for factual labels.
    """
    return await analyze_transcript(body)


@router.get("/sessions/{session_id}/suggestions", response_model=List[SuggestionOut])
def list_session_suggestions(
    session_id: str,
    status: Optional[str] = Query(
        "pending",
        description="Filter by status, or use 'all' for every suggestion in the session.",
    ),
) -> List[SuggestionOut]:
    st: Optional[str] = None if status in ("all", "any", "*") else status
    return suggestion_store.list_suggestions(session_id, st)


@router.post("/sessions/{session_id}/suggestions/generate", response_model=GenerateSuggestionsOut)
async def generate_session_suggestions(session_id: str, body: GenerateSuggestionsIn) -> GenerateSuggestionsOut:
    return await generate_suggestions(
        session_id,
        body.recent_transcript,
        body.use_knowledge_base,
        body.context_window or "all",
    )


@router.post("/suggestions/{suggestion_id}/approve", response_model=SuggestionOut)
def approve_suggestion(suggestion_id: str) -> SuggestionOut:
    row = suggestion_store.update_status(suggestion_id, "approved", ("pending",))
    if not row:
        raise HTTPException(status_code=404, detail="Suggestion not found or not pending")
    return row


@router.post("/suggestions/{suggestion_id}/dismiss", response_model=SuggestionOut)
def dismiss_suggestion(suggestion_id: str) -> SuggestionOut:
    row = suggestion_store.update_status(suggestion_id, "dismissed", ("pending", "approved"))
    if not row:
        raise HTTPException(status_code=404, detail="Suggestion not found or cannot be dismissed")
    return row


@router.post("/suggestions/{suggestion_id}/ignore", response_model=SuggestionOut)
def ignore_suggestion(suggestion_id: str) -> SuggestionOut:
    row = suggestion_store.update_status(suggestion_id, "ignored", ("pending",))
    if not row:
        raise HTTPException(status_code=404, detail="Suggestion not found or not pending")
    return row


@router.post("/suggestions/{suggestion_id}/spoken", response_model=SuggestionOut)
def mark_suggestion_spoken(suggestion_id: str) -> SuggestionOut:
    row = suggestion_store.update_status(suggestion_id, "spoken", ("approved",))
    if not row:
        raise HTTPException(status_code=404, detail="Suggestion not found or not approved")
    return row
