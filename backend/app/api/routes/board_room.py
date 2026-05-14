from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ...board_room.export import export_report_pdf, export_report_pptx
from ...board_room.report_pipeline import generate_board_room_report, get_report
from ...transcribe.stt_streaming import board_room_stt_status

router = APIRouter(prefix="/board-room", tags=["board-room"])

ExportFormat = Literal["pdf", "pptx"]


class AnalysisScopeIn(BaseModel):
    documents: bool = True
    transcripts: bool = False
    books: bool = True
    faqs: bool = True


class GenerateReportRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=256)
    title: str = Field(..., min_length=1, max_length=256)
    transcript: str = Field(..., min_length=1)
    session_name: str = ""
    session_location: str = ""
    include_rag_validation: bool = True
    analysis_scope: AnalysisScopeIn = Field(default_factory=AnalysisScopeIn)


class KnowledgeCheckEvidenceOut(BaseModel):
    source_name: str
    source_type: str = "unknown"
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    matched_text: str = ""


class KnowledgeCheckOut(BaseModel):
    claim: str
    classification: str
    confidence: float
    interpretation: str = ""
    suggested_action: str = ""
    evidence: List[KnowledgeCheckEvidenceOut] = Field(default_factory=list)


class GenerateReportResponse(BaseModel):
    report_id: str
    session_id: str
    title: str
    session_name: str = ""
    session_location: str = ""
    polished_transcript: str
    executive_summary: str
    knowledge_checks: List[KnowledgeCheckOut]
    markdown: str


@router.get("/stt-status")
def stt_status() -> Dict[str, Any]:
    return board_room_stt_status()


@router.post("/reports/generate", response_model=GenerateReportResponse)
async def generate_report(inp: GenerateReportRequest) -> GenerateReportResponse:
    text = (inp.transcript or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Transcript is empty.")
    payload = await generate_board_room_report(
        session_id=inp.session_id,
        title=inp.title.strip(),
        transcript=text,
        session_name=(inp.session_name or "").strip(),
        session_location=(inp.session_location or "").strip(),
        include_rag_validation=inp.include_rag_validation,
        analysis_scope=inp.analysis_scope.model_dump(),
    )
    return GenerateReportResponse(**payload)


@router.get("/reports/{report_id}", response_model=GenerateReportResponse)
def fetch_report(report_id: str) -> GenerateReportResponse:
    report = get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")
    return GenerateReportResponse(**report)


@router.get("/reports/{report_id}/export")
def export_report(report_id: str, format: ExportFormat = "pdf") -> Response:
    report = get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")
    title = (report.get("title") or "board_room_report").replace(" ", "_")[:80]
    if format == "pptx":
        data = export_report_pptx(report)
        return Response(
            content=data,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={"Content-Disposition": f'attachment; filename="{title}.pptx"'},
        )
    data = export_report_pdf(report)
    return Response(
        content=data,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{title}.pdf"'},
    )
