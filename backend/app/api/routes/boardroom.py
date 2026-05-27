"""
Boardroom Mode API routes.
"""
from __future__ import annotations

import asyncio
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional

from ...boardroom import service as br

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/boardroom", tags=["boardroom"])


class CreateSessionIn(BaseModel):
    transcript_id: Optional[str] = None


@router.post("/sessions")
def create_session(inp: CreateSessionIn = CreateSessionIn()):
    """Create a new boardroom session. Returns session_id for subsequent chunk uploads."""
    try:
        sid = br.create_session(transcript_id=inp.transcript_id)
        return {"session_id": sid, "status": "recording"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
def list_sessions(limit: int = 20):
    """List recent boardroom sessions."""
    return {"sessions": br.list_sessions(limit=limit)}


@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Get boardroom session status, diarized transcript, and report."""
    s = br.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Boardroom session not found")
    return s


@router.post("/sessions/{session_id}/chunks")
async def upload_chunk(
    session_id: str,
    chunk_index: int = 0,
    audio_format: str = "webm",
    file: UploadFile = File(...),
):
    """
    Upload one audio chunk. The frontend sends sequential chunks (MediaRecorder blobs).
    chunk_index: 0-based index of this chunk.
    audio_format: 'webm', 'ogg', 'wav', or 'pcm16'.
    """
    s = br.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Boardroom session not found")
    if s["status"] not in ("recording",):
        raise HTTPException(status_code=400, detail=f"Session is in state '{s['status']}', cannot upload chunks")
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty chunk")
        count = br.store_chunk(session_id, data, chunk_index, audio_format)
        return {"ok": True, "session_id": session_id, "chunk_index": chunk_index, "chunk_count": count}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/finalize")
async def finalize_session(session_id: str, background_tasks: BackgroundTasks):
    """
    Finalize recording: concatenate audio chunks and run VibeVoice-ASR transcription.
    This is async — poll GET /sessions/{id} for status changes (processing → transcribed).
    """
    s = br.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Boardroom session not found")

    async def _run():
        try:
            await br.finalize_and_transcribe(session_id)
        except Exception as e:
            logger.error("Boardroom finalize error for %s: %s", session_id, e)
            br._update_status(session_id, "error")

    background_tasks.add_task(_run)
    return {"ok": True, "session_id": session_id, "status": "processing"}


@router.post("/sessions/{session_id}/analyse")
async def analyse_session(session_id: str, background_tasks: BackgroundTasks):
    """
    Run RAG + LLM analysis on the diarized transcript to produce a meeting report.
    Poll GET /sessions/{id} for status changes (transcribed → analysed).
    """
    s = br.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Boardroom session not found")
    if s["status"] not in ("transcribed", "analysed"):
        raise HTTPException(
            status_code=400,
            detail=f"Session must be in 'transcribed' state. Current: '{s['status']}'"
        )

    async def _run():
        try:
            await br.analyse_meeting(session_id)
        except Exception as e:
            logger.error("Boardroom analyse error for %s: %s", session_id, e)

    background_tasks.add_task(_run)
    return {"ok": True, "session_id": session_id, "status": "analysing"}


@router.get("/sessions/{session_id}/report")
def get_report(session_id: str):
    """Return the meeting report JSON (available after /analyse completes)."""
    s = br.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Boardroom session not found")
    if not s.get("report"):
        raise HTTPException(status_code=404, detail="Report not yet generated. Run /analyse first.")
    return {"session_id": session_id, "report": s["report"]}


@router.get("/sessions/{session_id}/export")
def export_session(session_id: str, format: str = "pdf"):
    """
    Export the meeting report as PDF or PPTX.
    format: 'pdf' | 'pptx'
    """
    s = br.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Boardroom session not found")
    if not s.get("report"):
        raise HTTPException(status_code=400, detail="No report to export. Run /analyse first.")

    fmt = (format or "pdf").lower().strip()
    try:
        if fmt == "pptx":
            data = br.export_pptx(session_id)
            return Response(
                content=data,
                media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                headers={"Content-Disposition": f'attachment; filename="boardroom_{session_id[:8]}.pptx"'},
            )
        else:
            data = br.export_pdf(session_id)
            return Response(
                content=data,
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="boardroom_{session_id[:8]}.pdf"'},
            )
    except ImportError as e:
        raise HTTPException(status_code=501, detail=f"Export library not available: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
