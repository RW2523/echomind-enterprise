"""REST API for Silent Assistant correction findings (never TTS / no audio)."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from ...schemas.silent_finding import CorrectionFindingOut, SilentAnalyzeIn, SilentAnalyzeOut, UserAction
from ...silent_assistant import finding_store
from ...silent_assistant.silent_analyzer import analyze_segment

router = APIRouter(prefix="/silent-assistant", tags=["silent-assistant"])


@router.get("/sessions/{session_id}/findings", response_model=List[CorrectionFindingOut])
def list_findings(
    session_id: str,
    user_action: Optional[str] = Query(
        "pending",
        description="Filter by user_action, or 'all' for every finding in the session.",
    ),
) -> List[CorrectionFindingOut]:
    ua: Optional[str] = None if user_action in ("all", "any", "*") else user_action
    return finding_store.list_findings(session_id, ua)


@router.post("/sessions/{session_id}/findings/analyze", response_model=SilentAnalyzeOut)
async def analyze_finding(session_id: str, body: SilentAnalyzeIn) -> SilentAnalyzeOut:
    return await analyze_segment(session_id, body)


@router.post("/findings/{finding_id}/dismiss", response_model=CorrectionFindingOut)
def dismiss_finding(finding_id: str) -> CorrectionFindingOut:
    row = finding_store.update_user_action(finding_id, UserAction.dismissed.value, ("pending",))
    if not row:
        raise HTTPException(status_code=404, detail="Finding not found or not pending")
    return row


@router.post("/findings/{finding_id}/accept", response_model=CorrectionFindingOut)
def accept_finding(finding_id: str) -> CorrectionFindingOut:
    row = finding_store.update_user_action(finding_id, UserAction.accepted.value, ("pending",))
    if not row:
        raise HTTPException(status_code=404, detail="Finding not found or not pending")
    return row


@router.post("/findings/{finding_id}/mark_unhelpful", response_model=CorrectionFindingOut)
def mark_unhelpful(finding_id: str) -> CorrectionFindingOut:
    row = finding_store.update_user_action(finding_id, UserAction.marked_unhelpful.value, ("pending",))
    if not row:
        raise HTTPException(status_code=404, detail="Finding not found or not pending")
    return row
