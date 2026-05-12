"""SQLite persistence for Assistant Mode suggestions."""
from __future__ import annotations

import json
import logging
from typing import List, Optional, Sequence

from ..core.db import get_conn
from ..schemas.assistant_suggestion import SuggestionOut, row_to_suggestion_out
from ..utils.ids import new_id, now_iso

logger = logging.getLogger(__name__)

ASSISTANT_SUGGESTION_COOLDOWN_SEC = 50
ASSISTANT_MAX_PENDING_PER_SESSION = 8


def is_within_cooldown(session_id: str, cooldown_sec: int = ASSISTANT_SUGGESTION_COOLDOWN_SEC) -> bool:
    """True if the last suggestion for this session was created within cooldown_sec."""
    from datetime import datetime, timedelta, timezone

    with get_conn() as conn:
        row = conn.execute(
            "SELECT created_at FROM assistant_suggestions WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
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
        delta = datetime.now(timezone.utc) - t
        return delta.total_seconds() < float(cooldown_sec)
    except Exception:
        return False


def count_pending(session_id: str) -> int:
    with get_conn() as conn:
        r = conn.execute(
            "SELECT COUNT(*) FROM assistant_suggestions WHERE session_id = ? AND status = 'pending'",
            (session_id,),
        ).fetchone()
        return int(r[0]) if r else 0


def _suggestion_select_cols() -> str:
    return (
        "id, session_id, mode, title, short_text, speak_text, reason, category, confidence, "
        "source_origin, evidence_status, citations_json, created_at, status, "
        "influencing_rule_set_id, influencing_rule_set_name, influencing_rule_id, influencing_rule_title, "
        "trigger_excerpt "
    )


def list_suggestions(session_id: str, status: Optional[str] = None) -> List[SuggestionOut]:
    cols = _suggestion_select_cols()
    with get_conn() as conn:
        if status:
            rows = conn.execute(
                f"SELECT {cols} FROM assistant_suggestions WHERE session_id = ? AND status = ? "
                "ORDER BY created_at DESC",
                (session_id, status),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {cols} FROM assistant_suggestions WHERE session_id = ? ORDER BY created_at DESC",
                (session_id,),
            ).fetchall()
    return [row_to_suggestion_out(tuple(r)) for r in rows]


def get_suggestion(suggestion_id: str) -> Optional[SuggestionOut]:
    with get_conn() as conn:
        row = conn.execute(
            f"SELECT {_suggestion_select_cols()} FROM assistant_suggestions WHERE id = ?",
            (suggestion_id,),
        ).fetchone()
    if not row:
        return None
    return row_to_suggestion_out(tuple(row))


def insert_suggestion(
    session_id: str,
    mode: str,
    title: str,
    short_text: str,
    speak_text: str,
    reason: str,
    category: str,
    confidence: float,
    source_origin: str,
    evidence_status: str,
    citations: List[dict],
    status: str = "pending",
    *,
    influencing_rule_set_id: Optional[str] = None,
    influencing_rule_set_name: Optional[str] = None,
    influencing_rule_id: Optional[str] = None,
    influencing_rule_title: Optional[str] = None,
    trigger_excerpt: Optional[str] = None,
) -> SuggestionOut:
    sid = new_id("sug")
    created = now_iso()
    cites_json = json.dumps(citations, ensure_ascii=False)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO assistant_suggestions (id, session_id, mode, title, short_text, speak_text, reason, "
            "category, confidence, source_origin, evidence_status, citations_json, created_at, status, updated_at, "
            "influencing_rule_set_id, influencing_rule_set_name, influencing_rule_id, influencing_rule_title, "
            "trigger_excerpt) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                sid,
                session_id,
                mode,
                title,
                short_text,
                speak_text,
                reason,
                category,
                confidence,
                source_origin,
                evidence_status,
                cites_json,
                created,
                status,
                created,
                influencing_rule_set_id,
                influencing_rule_set_name,
                influencing_rule_id,
                influencing_rule_title,
                trigger_excerpt,
            ),
        )
        conn.commit()
    row = get_suggestion(sid)
    assert row is not None
    return row


def recent_pending_fingerprints(session_id: str, limit: int = 12) -> List[str]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT short_text FROM assistant_suggestions WHERE session_id = ? AND status = 'pending' "
            "ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    out: List[str] = []
    for (t,) in rows:
        if not t:
            continue
        norm = " ".join((t or "").lower().split())[:120]
        out.append(norm)
    return out


def update_status(
    suggestion_id: str,
    new_status: str,
    allowed_from: Sequence[str],
) -> Optional[SuggestionOut]:
    with get_conn() as conn:
        row = conn.execute("SELECT status FROM assistant_suggestions WHERE id = ?", (suggestion_id,)).fetchone()
        if not row:
            return None
        cur = row[0]
        if cur not in allowed_from:
            logger.info("assistant suggestion status skip: id=%s cur=%s wanted=%s", suggestion_id, cur, new_status)
            return None
        now = now_iso()
        conn.execute(
            "UPDATE assistant_suggestions SET status = ?, updated_at = ? WHERE id = ?",
            (new_status, now, suggestion_id),
        )
        conn.commit()
    return get_suggestion(suggestion_id)
