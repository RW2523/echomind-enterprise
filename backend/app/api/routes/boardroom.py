"""
Board Room Mode API routes.
Provides REST endpoints for session management, report access, and PDF/PPTX export.
WebSocket for real-time multi-speaker audio capture.

Routes:
  GET    /api/boardroom/sessions          — list sessions
  POST   /api/boardroom/sessions          — create session (meta only; use WS for audio)
  GET    /api/boardroom/sessions/{id}     — get session details
  DELETE /api/boardroom/sessions/{id}     — delete session + report
  GET    /api/boardroom/sessions/{id}/report       — get report (status + data)
  POST   /api/boardroom/sessions/{id}/report       — trigger report regeneration
  GET    /api/boardroom/sessions/{id}/export       — export report (?format=pdf|pptx)
  GET    /api/boardroom/sessions/{id}/debug        — diagnostic info (speakers, diar, source)
  WS     /api/boardroom/ws               — multi-speaker streaming audio
"""
from __future__ import annotations

import json
import logging
import wave
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, WebSocket
from fastapi.responses import Response
from pydantic import BaseModel

from ...core.db import get_conn
from ...utils.ids import now_iso
from ...boardroom.ws import handler as ws_handler
from ...boardroom.report import (
    generate_report_async,
    get_report,
    get_report_by_session,
)
from ...boardroom.export import generate_pdf, generate_pptx

logger = logging.getLogger(__name__)
router = APIRouter(tags=["boardroom"])


# ── Pydantic models ───────────────────────────────────────────────────────────

class SessionCreateRequest(BaseModel):
    title: str = "Board Room Session"
    location: str = "default"


class SessionResponse(BaseModel):
    id: str
    title: str
    location: str
    status: str
    started_at: Optional[str]
    ended_at: Optional[str]
    duration_sec: Optional[float]
    speaker_count: int
    segment_count: int
    created_at: str


# ── Helpers ──────────────────────────────────────────────────────────────────

def _row_to_session(row) -> dict:
    speaker_map = {}
    segments = []
    try:
        speaker_map = json.loads(row[8]) if row[8] else {}
    except Exception:
        pass
    try:
        segments = json.loads(row[9]) if row[9] else []
    except Exception:
        pass
    return {
        "id": row[0],
        "title": row[1],
        "location": row[2],
        "status": row[3],
        "started_at": row[4],
        "ended_at": row[5],
        "duration_sec": row[6],
        "raw_transcript": row[7],
        "speaker_map": speaker_map,
        "segments": segments,
        "speaker_count": len(speaker_map),
        "segment_count": len(segments),
        "created_at": row[10],
    }


# ── WebSocket ────────────────────────────────────────────────────────────────

@router.websocket("/boardroom/ws")
async def boardroom_ws(ws: WebSocket):
    """Multi-speaker audio streaming WebSocket for Board Room Mode."""
    await ws_handler(ws)


# ── Session CRUD ─────────────────────────────────────────────────────────────

@router.get("/boardroom/sessions")
def list_sessions(limit: int = Query(50, ge=1, le=200)):
    """List Board Room sessions, most recent first."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, title, location, status, started_at, ended_at, duration_sec,
                      raw_transcript, speaker_map_json, segments_json, created_at
               FROM boardroom_sessions ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    sessions = []
    for row in rows:
        s = _row_to_session(row)
        # Omit full transcript in list view
        s.pop("raw_transcript", None)
        s.pop("segments", None)
        s.pop("speaker_map", None)
        sessions.append(s)
    return {"sessions": sessions}


@router.post("/boardroom/sessions", status_code=201)
def create_session(req: SessionCreateRequest):
    """Create a Board Room session record (without audio — audio goes through WS)."""
    import uuid
    session_id = str(uuid.uuid4())
    created_at = now_iso()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO boardroom_sessions
               (id, title, location, status, started_at, created_at, updated_at,
                raw_transcript, speaker_map_json, segments_json)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (session_id, req.title, req.location, "created", created_at, created_at, created_at, "", "{}", "[]"),
        )
        conn.commit()
    return {"session_id": session_id, "title": req.title, "location": req.location, "created_at": created_at}


@router.get("/boardroom/sessions/{session_id}")
def get_session(session_id: str):
    """Get full session details including transcript and segments."""
    with get_conn() as conn:
        row = conn.execute(
            """SELECT id, title, location, status, started_at, ended_at, duration_sec,
                      raw_transcript, speaker_map_json, segments_json, created_at
               FROM boardroom_sessions WHERE id=?""",
            (session_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return _row_to_session(row)


@router.delete("/boardroom/sessions/{session_id}")
def delete_session(session_id: str):
    """Delete a Board Room session and its associated reports."""
    with get_conn() as conn:
        exists = conn.execute(
            "SELECT id FROM boardroom_sessions WHERE id=?", (session_id,)
        ).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="Session not found")
        conn.execute("DELETE FROM boardroom_reports WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM boardroom_sessions WHERE id=?", (session_id,))
        conn.commit()
    return {"ok": True, "deleted": session_id}


# ── Report endpoints ──────────────────────────────────────────────────────────

@router.get("/boardroom/sessions/{session_id}/report")
def get_session_report(session_id: str):
    """Get the report for a session (may be pending/generating/ready/failed)."""
    report = get_report_by_session(session_id)
    if not report:
        return {"status": "not_generated", "session_id": session_id, "report": None}
    return report


@router.post("/boardroom/sessions/{session_id}/report")
async def trigger_report(session_id: str, background_tasks: BackgroundTasks):
    """Trigger (re)generation of the Board Room report for a session."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, raw_transcript, speaker_map_json, segments_json FROM boardroom_sessions WHERE id=?",
            (session_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    raw_transcript = row[1] or ""
    try:
        speaker_map = json.loads(row[2]) if row[2] else {}
    except Exception:
        speaker_map = {}
    try:
        segments = json.loads(row[3]) if row[3] else []
    except Exception:
        segments = []

    if not raw_transcript.strip():
        raise HTTPException(status_code=400, detail="Session has no transcript to analyse")

    # Generate in background so response returns immediately
    background_tasks.add_task(
        _bg_generate_report, session_id, raw_transcript, segments, speaker_map
    )
    return {"ok": True, "session_id": session_id, "message": "Report generation started"}


async def _bg_generate_report(session_id, transcript, segments, speaker_map):
    try:
        await generate_report_async(
            session_id=session_id,
            transcript=transcript,
            segments=segments,
            speaker_map=speaker_map,
        )
    except Exception as e:
        logger.error("Board Room: background report failed for %s: %s", session_id, e)


@router.get("/boardroom/sessions/{session_id}/debug")
def debug_session(session_id: str):
    """
    Return diagnostic information about a Board Room session.

    Useful for verifying diarization and multitalker results without reading DB directly.
    """
    import os, wave

    with get_conn() as conn:
        row = conn.execute(
            """SELECT id, audio_file_path, speaker_count, transcription_source,
                      diarization_model_name, segments_json, duration_sec
               FROM boardroom_sessions WHERE id=?""",
            (session_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    session_id_db = row[0]
    audio_file_path = row[1]
    speaker_count = row[2] or 0
    transcription_source = row[3]
    diarization_model_name = row[4]
    segments_json = row[5]
    duration_sec_db = row[6]

    segments = []
    try:
        segments = json.loads(segments_json) if segments_json else []
    except Exception:
        segments = []

    transcript_speakers = list({
        seg.get("speaker_name", seg.get("speaker", "?")) for seg in segments
    })

    # Compute audio duration from WAV file if available
    audio_duration: Optional[float] = duration_sec_db
    if audio_file_path and os.path.isfile(audio_file_path):
        try:
            with wave.open(audio_file_path, "rb") as wf:
                audio_duration = wf.getnframes() / wf.getframerate()
        except Exception:
            pass

    return {
        "session_id": session_id_db,
        "audio_file_path": audio_file_path,
        "audio_duration": round(audio_duration, 2) if audio_duration else None,
        "diarization_model": diarization_model_name,
        "transcript_speaker_count": speaker_count,
        "transcript_speakers": transcript_speakers,
        "transcript_segments_count": len(segments),
        "transcription_source": transcription_source,
    }


@router.get("/boardroom/reports/{report_id}")
def get_report_by_id(report_id: str):
    """Get a specific report by report ID."""
    report = get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


# ── Export endpoints ──────────────────────────────────────────────────────────

@router.get("/boardroom/sessions/{session_id}/export")
def export_report(
    session_id: str,
    format: str = Query("pdf", pattern="^(pdf|pptx)$"),
):
    """
    Export the Board Room report as PDF or PPTX.
    Query param: ?format=pdf (default) or ?format=pptx
    """
    # Fetch report
    report_row = get_report_by_session(session_id)
    if not report_row or report_row.get("status") != "ready":
        raise HTTPException(
            status_code=404,
            detail="Report not ready. Generate the report first (POST /report) and wait for status=ready.",
        )

    report = report_row.get("report") or {}

    # Fetch session metadata
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, title, location, status, started_at, ended_at, duration_sec, created_at FROM boardroom_sessions WHERE id=?",
            (session_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    session_meta = {
        "id": row[0],
        "title": row[1],
        "location": row[2],
        "status": row[3],
        "started_at": row[4],
        "ended_at": row[5],
        "duration_sec": row[6],
        "created_at": row[7],
    }

    safe_title = (row[1] or "boardroom").replace(" ", "_")[:40]

    if format == "pdf":
        try:
            pdf_bytes = generate_pdf(report, session_meta)
        except Exception as e:
            logger.error("Board Room PDF export failed: %s", e)
            raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{safe_title}_report.pdf"'},
        )
    else:  # pptx
        try:
            pptx_bytes = generate_pptx(report, session_meta)
        except Exception as e:
            logger.error("Board Room PPTX export failed: %s", e)
            raise HTTPException(status_code=500, detail=f"PPTX generation failed: {e}")
        return Response(
            content=pptx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={"Content-Disposition": f'attachment; filename="{safe_title}_report.pptx"'},
        )
