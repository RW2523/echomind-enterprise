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
from ...core.config import settings

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


class SessionMetaIn(BaseModel):
    scenario: Optional[str] = None
    namespace: Optional[str] = None
    speaker_map: Optional[dict] = None   # {"Speaker 1": "banker", "Speaker 2": "client"}


@router.patch("/sessions/{session_id}/speakers")
def set_session_meta(session_id: str, inp: SessionMetaIn):
    """Set the scenario, knowledge namespace and speaker->role map used by the statement
    checks in /analyse (Silent Assistant v2)."""
    from ...silent_assistant.profiles import SCENARIOS
    if inp.scenario is not None and inp.scenario not in SCENARIOS:
        raise HTTPException(status_code=400, detail=f"unknown scenario {inp.scenario!r}")
    ok = br.set_session_meta(session_id, scenario=inp.scenario, namespace=inp.namespace, speaker_map=inp.speaker_map)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found or nothing to update")
    return {"ok": True}


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


class LinkTranscriptIn(BaseModel):
    transcript_id: str


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Delete a boardroom session and all associated audio chunks."""
    found = br.delete_session(session_id)
    if not found:
        raise HTTPException(status_code=404, detail="Boardroom session not found")
    return {"ok": True, "deleted": session_id}


@router.post("/sessions/{session_id}/link-transcript")
def link_transcript(session_id: str, inp: LinkTranscriptIn):
    """Link a live-transcript row to this boardroom session (called when first auto-store fires)."""
    s = br.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Boardroom session not found")
    br.link_transcript(session_id, inp.transcript_id)
    return {"ok": True, "session_id": session_id, "transcript_id": inp.transcript_id}


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
    # Reject unsupported formats early (also enforced in store_chunk for defense in depth).
    if audio_format.strip().lower() not in br.ALLOWED_AUDIO_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported audio_format '{audio_format}'")
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty chunk")
        if len(data) > settings.BOARDROOM_MAX_CHUNK_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Chunk too large ({len(data)} bytes > {settings.BOARDROOM_MAX_CHUNK_BYTES})",
            )
        count = br.store_chunk(session_id, data, chunk_index, audio_format)
        return {"ok": True, "session_id": session_id, "chunk_index": chunk_index, "chunk_count": count}
    except HTTPException:
        raise
    except ValueError as e:
        # Validation failures (bad format / out-of-range index / bad path) → client error.
        raise HTTPException(status_code=400, detail=str(e))
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

    # Atomically claim the session for processing. Only one concurrent /finalize wins;
    # others (or a session already processing/transcribed) get 409 instead of spawning
    # another ~18 GB GPU subprocess.
    if not br.try_transition(session_id, "processing", allowed_from=("recording", "error")):
        cur = br.get_session(session_id)
        raise HTTPException(
            status_code=409,
            detail=f"Session cannot be finalized from state '{(cur or {}).get('status')}'",
        )

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

    # Atomically claim the session for analysis (prevents re-entrant LLM/RAG fan-out).
    prev_status = s.get("status")  # 'transcribed' or 'analysed' (re-analyse)
    if not br.try_transition(session_id, "analysing", allowed_from=("transcribed", "analysed")):
        cur = br.get_session(session_id)
        raise HTTPException(
            status_code=409,
            detail=f"Session must be 'transcribed' to analyse. Current: '{(cur or {}).get('status')}'",
        )

    async def _run():
        try:
            await br.analyse_meeting(session_id)
        except Exception as e:
            # Restore the PRIOR status (not always 'transcribed') so a failed re-analysis of an
            # already-'analysed' session keeps its existing report instead of desyncing. (audit M)
            logger.error("Boardroom analyse error for %s: %s", session_id, e)
            br._update_status(session_id, prev_status or "transcribed")

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
