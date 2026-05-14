"""Persist assistant insights to SQLite (not RAG, not transcript ingestion)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from app.core.db import get_conn

ACTION_STATUSES = frozenset({"ignored", "saved_for_later", "viewed", "asked_follow_up", "spoke_now"})

SaveResult = Literal["inserted", "duplicate", "skipped"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_transcript_for_dedupe(text: str) -> str:
    return " ".join((text or "").split()).lower()


def primary_source_key(evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return ""
    ev0 = evidence[0] if isinstance(evidence[0], dict) else {}
    cid = str(ev0.get("chunk_id") or "").strip()
    did = str(ev0.get("doc_id") or "").strip()
    if cid:
        return f"chunk:{cid}"
    if did:
        return f"doc:{did}"
    sn = str(ev0.get("source_name") or "")[:160].strip()
    mt = str(ev0.get("matched_text") or "")[:200]
    mt_h = hashlib.sha256(mt.encode()).hexdigest()[:16] if mt else ""
    return f"src:{sn}|{mt_h}"


def compute_dedupe_key(session_id: str, transcript_text: str, classification: str, evidence: List[Any]) -> str:
    ev_ser: List[Dict[str, Any]] = []
    for e in evidence or []:
        ev_ser.append(e if isinstance(e, dict) else {})
    base = (
        f"{session_id}|{normalize_transcript_for_dedupe(transcript_text)}|"
        f"{(classification or '').lower()}|{primary_source_key(ev_ser)}"
    )
    return hashlib.sha256(base.encode()).hexdigest()


def should_persist_insight(ins: Dict[str, Any]) -> bool:
    if not bool(ins.get("show_highlight")):
        return False
    try:
        conf = float(ins.get("confidence") or 0)
    except (TypeError, ValueError):
        return False
    if conf < 0.70:
        return False
    ev = ins.get("evidence") or []
    if len(ev) < 1:
        return False
    return True


def _eligible_hand_raise_for_listing(insight: Dict[str, Any], mode: str) -> bool:
    """Mirror live_analysis._eligible_hand_raise without importing the RAG stack."""
    if mode != "personal_assistant":
        return False
    conf = float(insight.get("confidence") or 0)
    if conf < 0.75:
        return False
    ev = insight.get("evidence") or []
    if not ev:
        return False
    cls = str(insight.get("classification") or "")
    if cls == "missing_context":
        return False
    if cls not in ("warning", "contradicted", "related", "supported"):
        return False
    if cls == "supported" and conf < 0.88:
        return False
    return True


def _row_to_api_insight(row: sqlite3.Row) -> Dict[str, Any]:
    mode = str(row["mode"])
    evidence = json.loads(row["evidence_json"] or "[]")
    chk = {
        "confidence": float(row["confidence"] or 0),
        "classification": str(row["classification"] or ""),
        "evidence": evidence,
    }
    return {
        "id": row["id"],
        "transcript_text": row["transcript_text"],
        "classification": row["classification"],
        "confidence": float(row["confidence"] or 0),
        "start_char": row["start_char"],
        "end_char": row["end_char"],
        "paragraph_id": row["paragraph_id"],
        "show_highlight": True,
        "show_hand_raise": _eligible_hand_raise_for_listing(chk, mode),
        "priority": row["priority"] or "medium",
        "evidence": evidence,
        "assistant_interpretation": row["assistant_interpretation"] or "",
        "suggested_action": row["suggested_action"] or "",
        "suggested_response": row["suggested_response"],
        "action_status": row["action_status"],
        "created_at": row["created_at"],
        "mode": mode,
    }


def save_assistant_insight(
    session_id: str,
    mode: str,
    transcript_id: Optional[str],
    insight: Dict[str, Any],
) -> Tuple[SaveResult, Optional[str]]:
    """Insert one persistable insight. Returns (result, row_id)."""
    if not should_persist_insight(insight):
        return "skipped", None
    with get_conn() as conn:
        res, pid = _insert_one(conn, session_id.strip(), mode, transcript_id, insight)
        conn.commit()
        return res, pid


def _opt_int(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _insert_one(
    conn: sqlite3.Connection,
    session_id: str,
    mode: str,
    transcript_id: Optional[str],
    ins: Dict[str, Any],
) -> Tuple[SaveResult, Optional[str]]:
    if not should_persist_insight(ins):
        return "skipped", None
    dedupe_key = compute_dedupe_key(
        session_id,
        str(ins.get("transcript_text") or ""),
        str(ins.get("classification") or ""),
        ins.get("evidence") or [],
    )
    new_id = f"ins_{uuid.uuid4().hex[:16]}"
    now = _now_iso()
    evidence_json = json.dumps(ins.get("evidence") or [], ensure_ascii=False)
    sc = _opt_int(ins.get("start_char"))
    ec = _opt_int(ins.get("end_char"))
    pid = ins.get("paragraph_id")
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO assistant_insights(
            id, session_id, transcript_id, mode, dedupe_key,
            transcript_text, classification, confidence, priority, evidence_json,
            assistant_interpretation, suggested_action, suggested_response,
            start_char, end_char, paragraph_id, action_status, created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            new_id,
            session_id,
            transcript_id,
            mode,
            dedupe_key,
            str(ins.get("transcript_text") or ""),
            str(ins.get("classification") or "related"),
            float(ins.get("confidence") or 0),
            str(ins.get("priority") or "medium"),
            evidence_json,
            str(ins.get("assistant_interpretation") or ""),
            str(ins.get("suggested_action") or ""),
            ins.get("suggested_response"),
            sc,
            ec,
            str(pid) if pid else None,
            None,
            now,
            now,
        ),
    )
    if cur.rowcount == 1:
        return "inserted", new_id
    row = conn.execute(
        "SELECT id, transcript_id FROM assistant_insights WHERE session_id = ? AND dedupe_key = ?",
        (session_id, dedupe_key),
    ).fetchone()
    if row:
        existing_id = row[0]
        if transcript_id and not (row[1] or "").strip():
            conn.execute(
                "UPDATE assistant_insights SET transcript_id = ?, updated_at = ? WHERE id = ?",
                (transcript_id, now, existing_id),
            )
        return "duplicate", existing_id
    return "skipped", None


def bulk_save_assistant_insights(
    session_id: str,
    mode: str,
    transcript_id: Optional[str],
    insights: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Save many insights in one transaction; id_map maps request insight id -> stored row id."""
    sid = session_id.strip()
    id_map: Dict[str, str] = {}
    inserted = 0
    duplicates = 0
    skipped = 0
    with get_conn() as conn:
        try:
            conn.execute("BEGIN")
            for ins in insights:
                if not isinstance(ins, dict):
                    skipped += 1
                    continue
                client_key = str(ins.get("id") or "")
                if not should_persist_insight(ins):
                    skipped += 1
                    continue
                res, pid = _insert_one(conn, sid, mode, transcript_id, ins)
                if res == "inserted" and pid:
                    inserted += 1
                    if client_key:
                        id_map[client_key] = pid
                elif res == "duplicate" and pid:
                    duplicates += 1
                    if client_key:
                        id_map[client_key] = pid
                else:
                    skipped += 1
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return {
        "session_id": sid,
        "inserted": inserted,
        "skipped": skipped,
        "duplicate_merged": duplicates,
        "id_map": id_map,
    }


def list_assistant_insights_by_session(session_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM assistant_insights WHERE session_id = ? ORDER BY created_at ASC",
            (session_id.strip(),),
        ).fetchall()
    return [_row_to_api_insight(r) for r in rows]


def update_assistant_insight_action_status(insight_id: str, action_status: str) -> bool:
    if action_status not in ACTION_STATUSES:
        raise ValueError(f"Invalid action_status: {action_status}")
    now = _now_iso()
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE assistant_insights SET action_status = ?, updated_at = ? WHERE id = ?",
            (action_status, now, insight_id.strip()),
        )
        conn.commit()
        return cur.rowcount > 0
