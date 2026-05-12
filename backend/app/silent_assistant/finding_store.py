"""SQLite persistence for Silent Assistant correction findings."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence

from ..core.db import get_conn
from ..schemas.silent_finding import CorrectionFindingOut, UserAction, row_to_finding_out
from ..utils.ids import new_id, now_iso

logger = logging.getLogger(__name__)

SILENT_FINDING_COOLDOWN_SEC = 42
SILENT_MAX_PENDING_FINDINGS = 24


def is_within_cooldown(session_id: str, cooldown_sec: int = SILENT_FINDING_COOLDOWN_SEC) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT created_at FROM silent_findings WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    if not row or not row[0]:
        return False
    ts = str(row[0])
    try:
        if ts.endswith("Z"):
            t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            t = datetime.fromisoformat(ts)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - t).total_seconds() < float(cooldown_sec)
    except Exception:
        return False


def count_pending(session_id: str) -> int:
    with get_conn() as conn:
        r = conn.execute(
            "SELECT COUNT(*) FROM silent_findings WHERE session_id = ? AND user_action = 'pending'",
            (session_id,),
        ).fetchone()
        return int(r[0]) if r else 0


def list_findings(session_id: str, user_action: Optional[str] = None) -> List[CorrectionFindingOut]:
    with get_conn() as conn:
        cols = (
            "id, session_id, transcript_segment_id, turn_id, original_text, highlighted_span_start, "
            "highlighted_span_end, category, status_label, suggested_correction, reason, evidence_status, "
            "confidence, source_origin, citations_json, created_at, user_action, "
            "influencing_rule_set_id, influencing_rule_set_name, influencing_rule_id, influencing_rule_title "
        )
        if user_action and user_action != "all":
            rows = conn.execute(
                f"SELECT {cols} FROM silent_findings WHERE session_id = ? AND user_action = ? "
                "ORDER BY created_at DESC",
                (session_id, user_action),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {cols} FROM silent_findings WHERE session_id = ? ORDER BY created_at DESC",
                (session_id,),
            ).fetchall()
    return [row_to_finding_out(tuple(r)) for r in rows]


def get_finding(finding_id: str) -> Optional[CorrectionFindingOut]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, session_id, transcript_segment_id, turn_id, original_text, highlighted_span_start, "
            "highlighted_span_end, category, status_label, suggested_correction, reason, evidence_status, "
            "confidence, source_origin, citations_json, created_at, user_action, "
            "influencing_rule_set_id, influencing_rule_set_name, influencing_rule_id, influencing_rule_title "
            "FROM silent_findings WHERE id = ?",
            (finding_id,),
        ).fetchone()
    if not row:
        return None
    return row_to_finding_out(tuple(row))


def recent_fingerprints(session_id: str, limit: int = 16) -> List[str]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT category, original_text FROM silent_findings WHERE session_id = ? AND user_action = 'pending' "
            "ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    out: List[str] = []
    for cat, txt in rows:
        norm = f"{cat}|{' '.join((txt or '').lower().split())[:100]}"
        out.append(norm)
    return out


def insert_finding(
    session_id: str,
    transcript_segment_id: Optional[str],
    turn_id: Optional[str],
    original_text: str,
    span_start: int,
    span_end: int,
    category: str,
    status_label: str,
    suggested_correction: str,
    reason: str,
    evidence_status: str,
    confidence: float,
    source_origin: str,
    citations: List[dict],
    *,
    influencing_rule_set_id: Optional[str] = None,
    influencing_rule_set_name: Optional[str] = None,
    influencing_rule_id: Optional[str] = None,
    influencing_rule_title: Optional[str] = None,
) -> CorrectionFindingOut:
    fid = new_id("sf")
    created = now_iso()
    cites_json = json.dumps(citations, ensure_ascii=False)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO silent_findings (id, session_id, transcript_segment_id, turn_id, original_text, "
            "highlighted_span_start, highlighted_span_end, category, status_label, suggested_correction, reason, "
            "evidence_status, confidence, source_origin, citations_json, created_at, user_action, updated_at, "
            "influencing_rule_set_id, influencing_rule_set_name, influencing_rule_id, influencing_rule_title) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                fid,
                session_id,
                transcript_segment_id,
                turn_id,
                original_text,
                span_start,
                span_end,
                category,
                status_label,
                suggested_correction,
                reason,
                evidence_status,
                confidence,
                source_origin,
                cites_json,
                created,
                "pending",
                created,
                influencing_rule_set_id,
                influencing_rule_set_name,
                influencing_rule_id,
                influencing_rule_title,
            ),
        )
        conn.commit()
    row = get_finding(fid)
    assert row is not None
    return row


def update_user_action(finding_id: str, new_action: str, allowed_from: Sequence[str]) -> Optional[CorrectionFindingOut]:
    with get_conn() as conn:
        row = conn.execute("SELECT user_action FROM silent_findings WHERE id = ?", (finding_id,)).fetchone()
        if not row:
            return None
        cur = row[0]
        if cur not in allowed_from:
            logger.info("silent finding action skip: id=%s cur=%s wanted=%s", finding_id, cur, new_action)
            return None
        now = now_iso()
        conn.execute(
            "UPDATE silent_findings SET user_action = ?, updated_at = ? WHERE id = ?",
            (new_action, now, finding_id),
        )
        conn.commit()
    return get_finding(finding_id)
