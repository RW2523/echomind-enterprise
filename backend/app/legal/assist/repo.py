"""Silent Assist alerts in the database (S2.5; D-041): raised, decided once, measured."""
from __future__ import annotations

import json
import statistics
import uuid
from datetime import datetime
from typing import Optional

from ...core.config import settings
from ...core.db import get_conn
from ..policy import now_iso
from .engine import Alert

_COLS = ("id", "meeting_id", "segment_id", "pack", "rule", "verdict", "severity", "said_json", "source_json", "explanation", "confidence",
         "threshold", "method", "status", "decided_by", "decided_at", "decision_reason", "heard_at", "raised_at", "capture_id")


class Conflict(Exception):
    pass


def save(matter_id: str, meeting_id: str, capture_id: Optional[str], segment: dict, pack: str, a: Alert, heard_at: str,
         raised_at: Optional[str] = None) -> dict:
    aid, raised = str(uuid.uuid4()), raised_at or now_iso()
    said = {**a.said, "segment_id": segment["id"]}
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_assist_alerts (id, meeting_id, matter_id, capture_id, segment_id, pack, rule, verdict, severity, "
                     "said_json, source_json, explanation, confidence, threshold, method, heard_at, raised_at, document_ids_json, title) "
                     "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (aid, meeting_id, matter_id, capture_id, segment["id"], pack, a.rule, a.verdict, a.severity, json.dumps(said),
                      json.dumps(a.source) if a.source else None, a.explanation, a.confidence, settings.LEGAL_ASSIST_MIN_CONFIDENCE,
                      a.method, heard_at, raised, json.dumps(sorted(set(a.document_ids))), a.title))
        conn.commit()
    return get(matter_id, meeting_id, aid)


def _out(r) -> dict:
    d = dict(zip(_COLS + ("title", "decided_by_name"), r))
    d["said"], d["source"] = json.loads(d.pop("said_json")), json.loads(d.pop("source_json")) if d["source_json"] else None
    d["decided_by"] = d.pop("decided_by_name")             # the person's name, never their internal id
    return d


_SELECT = (f"SELECT {', '.join('a.' + c for c in _COLS)}, a.title, (SELECT username FROM users u WHERE u.id = a.decided_by) "
           f"FROM legal_assist_alerts a")


def get(matter_id: str, meeting_id: str, alert_id: str) -> Optional[dict]:
    with get_conn() as conn:
        r = conn.execute(f"{_SELECT} WHERE a.id=? AND a.meeting_id=? AND a.matter_id=?", (alert_id, meeting_id, matter_id)).fetchone()
    return _out(r) if r else None


def list_for(matter_id: str, meeting_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(f"{_SELECT} WHERE a.meeting_id=? AND a.matter_id=? AND NOT EXISTS "
                            f"(SELECT 1 FROM legal_meetings m WHERE m.id = a.meeting_id AND m.transcript_deleted_at IS NOT NULL) "
                            f"ORDER BY a.raised_at, a.rowid", (meeting_id, matter_id)).fetchall()
    return [_out(r) for r in rows]


def decide(matter_id: str, meeting_id: str, alert_id: str, status: str, by: str, reason: Optional[str]) -> dict:
    """Confirm or dismiss, once. Raises Conflict if already decided; None if there is no such alert."""
    with get_conn() as conn:
        cur = conn.execute("UPDATE legal_assist_alerts SET status=?, decided_by=?, decided_at=?, decision_reason=? "
                           "WHERE id=? AND meeting_id=? AND matter_id=? AND status='open'", (status, by, now_iso(), reason, alert_id,
                                                                                             meeting_id, matter_id))
        conn.commit()
        if not cur.rowcount:
            exists = conn.execute("SELECT status FROM legal_assist_alerts WHERE id=? AND meeting_id=? AND matter_id=?",
                                  (alert_id, meeting_id, matter_id)).fetchone()
            if exists:
                raise Conflict(f"This alert was already {exists[0]}.")
            return None
    return get(matter_id, meeting_id, alert_id)


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def telemetry(matter_id: str, meeting_id: str) -> dict:
    """Alerts per hour of capture, how many were dismissed, and how long after the words they came."""
    alerts = list_for(matter_id, meeting_id)
    with get_conn() as conn:
        spans = conn.execute("SELECT MIN(s.wall_start), MAX(s.wall_end) FROM legal_transcript_segments s WHERE s.meeting_id=? AND s.matter_id=? "
                             "GROUP BY s.capture_id", (meeting_id, matter_id)).fetchall()
    hours = sum(max(0.0, (_parse(b) - _parse(a)).total_seconds()) for a, b in spans if a and b) / 3600.0
    decided = [a for a in alerts if a["status"] != "open"]
    latency = [max(0.0, (_parse(a["raised_at"]) - _parse(a["heard_at"])).total_seconds()) for a in alerts]
    return {"alerts": len(alerts), "critical": sum(1 for a in alerts if a["severity"] == "critical"),
            "per_hour": round(len(alerts) / hours, 1) if hours >= 5 / 60 else None,      # not extrapolated from a few minutes
            "confirmed": sum(1 for a in decided if a["status"] == "confirmed"),
            "dismissed": sum(1 for a in decided if a["status"] == "dismissed"),
            "dismiss_rate": round(sum(1 for a in decided if a["status"] == "dismissed") / len(decided), 3) if decided else None,
            "latency_p50_s": round(statistics.median(latency), 2) if latency else None,
            "latency_max_s": round(max(latency), 2) if latency else None,
            "threshold": settings.LEGAL_ASSIST_MIN_CONFIDENCE, "thresholds_approved": settings.LEGAL_ASSIST_THRESHOLDS_APPROVED}
