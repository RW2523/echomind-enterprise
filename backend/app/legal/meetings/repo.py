"""Meetings, participants, consent events, captures and transcript segments (S1.8).
Every read takes the matter id: a meeting id is never resolved outside its matter."""
from __future__ import annotations

import json
import uuid
from typing import Optional

from ...core.db import get_conn
from ..policy import now_iso
from . import consent


def _id() -> str:
    return str(uuid.uuid4())


_M_COLS = ("id", "org_id", "matter_id", "title", "kind", "scheduled_at", "status", "consent_policy", "created_by",
           "created_at", "ended_at", "version")


def create_meeting(org_id: str, matter_id: str, title: str, kind: str, scheduled_at: Optional[str], by: str) -> str:
    mid = _id()
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_meetings (id, org_id, matter_id, title, kind, scheduled_at, consent_policy, created_by, created_at) "
                     "VALUES (?,?,?,?,?,?,?,?,?)", (mid, org_id, matter_id, title, kind, scheduled_at,
                                                    json.dumps(consent.policy()), by, now_iso()))
        conn.commit()
    return mid


def get_meeting_row(matter_id: str, meeting_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(f"SELECT {', '.join(_M_COLS)} FROM legal_meetings WHERE id=? AND matter_id=?",
                           (meeting_id, matter_id)).fetchone()
    return dict(zip(_M_COLS, row)) if row else None


def _summary(conn, row) -> dict:
    m = dict(zip(_M_COLS, row))
    people = consent.participants(conn, m["id"])
    return {"id": m["id"], "title": m["title"], "kind": m["kind"], "scheduled_at": m["scheduled_at"], "status": m["status"],
            "created_at": m["created_at"], "ended_at": m["ended_at"],
            "consent": {"granted": sum(1 for p in people if p["consent"]["state"] == "granted"), "needed": len(people)}}


def list_meetings(matter_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(f"SELECT {', '.join(_M_COLS)} FROM legal_meetings WHERE matter_id=? ORDER BY created_at DESC",
                            (matter_id,)).fetchall()
        return [_summary(conn, r) for r in rows]


def meeting_detail(matter_id: str, meeting_id: str, matter_number: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(f"SELECT {', '.join(_M_COLS)} FROM legal_meetings WHERE id=? AND matter_id=?",
                           (meeting_id, matter_id)).fetchone()
        if not row:
            return None
        out = _summary(conn, row)
        pol = json.loads(dict(zip(_M_COLS, row))["consent_policy"])
        out["policy"] = {**pol, "statement": pol["statement"].format(number=matter_number)}
        out["participants"] = [{k: v for k, v in p.items() if k != "added_at"} | {"consent": {k: v for k, v in p["consent"].items() if k != "event_id"}}
                               for p in consent.participants(conn, meeting_id)]
        caps = conn.execute("SELECT c.id, c.started_at, (SELECT username FROM users WHERE id=c.started_by), c.consent_snapshot, "
                            "c.transcription, (SELECT COUNT(*) FROM legal_capture_custody k WHERE k.capture_id=c.id), "
                            "(SELECT COALESCE(SUM(bytes),0) FROM legal_capture_custody k WHERE k.capture_id=c.id), "
                            "(SELECT COUNT(*) FROM legal_transcript_segments s JOIN legal_meetings mt ON mt.id = s.meeting_id "
                            "WHERE s.capture_id=c.id AND mt.transcript_deleted_at IS NULL) "
                            "FROM legal_captures c WHERE c.meeting_id=? ORDER BY c.started_at", (meeting_id,)).fetchall()
        out["captures"] = []
        for cid, started, by, snap, tr, custody_n, audio_bytes, segs in caps:
            events = conn.execute("SELECT kind, reason, at, (SELECT username FROM users WHERE id=e.by) FROM legal_capture_events e "
                                  "WHERE capture_id=? ORDER BY at, rowid", (cid,)).fetchall()
            out["captures"].append({"id": cid, "started_at": started, "started_by": by, "consent_snapshot": snap,
                                    "transcription": tr, "custody_records": custody_n, "audio_bytes": audio_bytes,
                                    "segments": segs, "events": [{"kind": k, "reason": r, "at": a, "by": b} for k, r, a, b in events]})
    return out


def end_meeting(meeting_id: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE legal_meetings SET status='ended', ended_at=?, version=version+1 WHERE id=? AND status='planned'",
                     (now_iso(), meeting_id))
        conn.commit()


def add_participant(meeting_id: str, matter_id: str, name: str, role: str, user_id: Optional[str], by: str) -> str:
    pid = _id()
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_meeting_participants (id, meeting_id, matter_id, name, role, user_id, added_by, added_at) "
                     "VALUES (?,?,?,?,?,?,?,?)", (pid, meeting_id, matter_id, name, role, user_id, by, now_iso()))
        conn.commit()
    return pid


def participant(meeting_id: str, participant_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT id, name, role, removed_at FROM legal_meeting_participants WHERE id=? AND meeting_id=?",
                           (participant_id, meeting_id)).fetchone()
    return {"id": row[0], "name": row[1], "role": row[2], "removed_at": row[3]} if row else None


def remove_participant(meeting_id: str, participant_id: str, by: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE legal_meeting_participants SET removed_at=?, removed_by=? WHERE id=? AND meeting_id=? AND removed_at IS NULL",
                     (now_iso(), by, participant_id, meeting_id))
        conn.commit()


def add_consent_event(meeting_id: str, participant_id: str, kind: str, method: Optional[str], statement_sha256: Optional[str],
                      by: str) -> str:
    eid = _id()
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_consent_events (id, meeting_id, participant_id, kind, method, statement_sha256, recorded_by, "
                     "recorded_at) VALUES (?,?,?,?,?,?,?,?)", (eid, meeting_id, participant_id, kind, method, statement_sha256, by, now_iso()))
        conn.commit()
    return eid


def create_capture(meeting_id: str, matter_id: str, by: str, snapshot: str, transcription: str) -> str:
    cid = _id()
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_captures (id, meeting_id, matter_id, started_by, started_at, consent_snapshot, transcription) "
                     "VALUES (?,?,?,?,?,?,?)", (cid, meeting_id, matter_id, by, now_iso(), snapshot, transcription))
        conn.commit()
    return cid


def add_segment(capture_id: str, meeting_id: str, matter_id: str, seq: int, t0, t1, text: str) -> dict:
    sid, at = _id(), now_iso()
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_transcript_segments (id, capture_id, meeting_id, matter_id, seq, t_start_ms, t_end_ms, text, created_at) "
                     "VALUES (?,?,?,?,?,?,?,?,?)", (sid, capture_id, meeting_id, matter_id, seq, t0, t1, text, at))
        conn.commit()
    return {"id": sid, "seq": seq, "t_start_ms": t0, "t_end_ms": t1, "text": text}


def segments(matter_id: str, meeting_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT s.id, s.capture_id, s.seq, s.t_start_ms, s.t_end_ms, s.text, s.created_at FROM legal_transcript_segments s "
                            "JOIN legal_captures c ON c.id = s.capture_id JOIN legal_meetings mt ON mt.id = s.meeting_id "
                            "WHERE s.meeting_id=? AND s.matter_id=? AND mt.transcript_deleted_at IS NULL "
                            "ORDER BY c.started_at, s.seq", (meeting_id, matter_id)).fetchall()
    return [{"id": r[0], "capture_id": r[1], "seq": r[2], "t_start_ms": r[3], "t_end_ms": r[4], "text": r[5], "created_at": r[6]}
            for r in rows]
