"""A meeting being recorded, as a card in the conversation (docs/echomind-legal/03 `live_transcript`; owner's request
2026-11-02). Only for people who may read the matter's meetings (`meetings.read`).

Shows the meeting being recorded now (or, when none is, the latest one with a transcript): whether it is recording,
paused or not, how many participants have consented, and its latest transcript segments as corrected. The card stores
only the meeting's reference and is rebuilt from the records each time it refreshes, so a deleted transcript disappears
from it too. Pausing or stopping stays with the person capturing, on the meeting page: the chat never holds the
microphone of a meeting and never starts recording.
"""
from __future__ import annotations

from typing import Optional
from urllib.parse import quote as urlquote

from ...core.db import get_conn
from .. import policy

SHOWN = 12
CAUTION = ("A live transcript is machine speech-to-text and may mishear; it is corrected on the meeting page. Recording runs only "
           "with every participant's consent, and only the person capturing can pause or stop it.")


def _state(conn, meeting_id: str) -> str:
    from ..meetings import capture
    with capture._registry_lock:
        live = bool(capture._live.get(meeting_id))
    if not live:
        return "not_recording"
    r = conn.execute("SELECT e.kind FROM legal_capture_events e JOIN legal_captures c ON c.id = e.capture_id WHERE c.meeting_id=? "
                     "ORDER BY e.at DESC, e.rowid DESC LIMIT 1", (meeting_id,)).fetchone()
    return "paused" if r and r[0] == "paused" else "recording"


def pick(matter_id: str) -> Optional[str]:
    """The meeting being recorded now, else the latest one with a transcript."""
    from ..meetings import capture
    with get_conn() as conn:
        rows = [r[0] for r in conn.execute("SELECT id FROM legal_meetings WHERE matter_id=? AND transcript_deleted_at IS NULL "
                                           "ORDER BY created_at DESC LIMIT 50", (matter_id,))]
        with capture._registry_lock:
            live = [m for m in rows if capture._live.get(m)]
        if live:
            return live[0]
        r = conn.execute("SELECT s.meeting_id FROM legal_transcript_segments s JOIN legal_meetings m ON m.id = s.meeting_id "
                         "WHERE m.matter_id=? AND m.transcript_deleted_at IS NULL ORDER BY s.created_at DESC LIMIT 1", (matter_id,)).fetchone()
    return r[0] if r else None


def build(principal: dict, matter_id: str, meeting_id: str) -> Optional[dict]:
    """The live_transcript props, or None when the meeting or its transcript is gone, or the person may not read meetings."""
    from ..meetings import consent, transcript
    if not policy.authorize(principal, "meetings.read", matter_id).allowed:
        return None
    with get_conn() as conn:
        m = conn.execute("SELECT m.title, m.status, mt.number FROM legal_meetings m JOIN matters mt ON mt.id = m.matter_id "
                         "WHERE m.id=? AND m.matter_id=? AND m.transcript_deleted_at IS NULL", (meeting_id, matter_id)).fetchone()
        if not m:
            return None
        title, _status, number = m
        people = consent.participants(conn, meeting_id)
        state = _state(conn, meeting_id)
    view = transcript.view(matter_id, meeting_id)
    names = {p["id"]: p["name"] for p in view.get("participants") or []}
    segs = view.get("segments") or []
    shown = segs[-SHOWN:]
    granted = sum(1 for p in people if p["consent"]["state"] == "granted")
    consented = (f"Consent recorded for {granted} of {len(people)} participant{'s' if len(people) != 1 else ''}." if people
                 else "No participants are listed.")
    return {"meeting_id": meeting_id, "title": (title or "Meeting")[:200], "state": state, "consent": consented[:200],
            "segments": [{"speaker": (names.get(s.get("speaker_participant_id")) or None), "text": (s.get("text") or "")[:1000],
                          "at": s.get("wall_start") or s.get("created_at")} for s in shown if (s.get("text") or "").strip()],
            "earlier": max(0, len(segs) - len(shown)), "path": f"/m/{urlquote(number, safe='')}/meetings/{meeting_id}",
            "caution": CAUTION}
