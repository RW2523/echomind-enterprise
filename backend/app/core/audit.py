"""Phase 0b: activity log -> audit view + usage metering. Best-effort; never breaks a request."""
from __future__ import annotations

import logging
from typing import Optional

from .db import get_conn
from ..utils.ids import new_id, now_iso

logger = logging.getLogger(__name__)


def record_activity(username: Optional[str], role: Optional[str], method: str, path: str, status: int, ip: str = "") -> None:
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO activity_log (id, ts, username, role, method, path, status, ip) VALUES (?,?,?,?,?,?,?,?)",
                (new_id("act"), now_iso(), username or "anonymous", role or "", method, path, int(status), ip or ""),
            )
            conn.commit()
    except Exception as e:
        logger.debug("activity log insert failed: %s", e)


def recent_activity(limit: int = 150) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT ts, username, role, method, path, status, ip FROM activity_log ORDER BY ts DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [
        {"ts": r[0], "username": r[1], "role": r[2], "method": r[3], "path": r[4], "status": r[5], "ip": r[6]}
        for r in rows
    ]


def usage_summary(limit_users: int = 50) -> dict:
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM activity_log").fetchone()[0]
        rows = conn.execute(
            "SELECT username, COUNT(*) c, MAX(ts) last FROM activity_log GROUP BY username ORDER BY c DESC LIMIT ?",
            (int(limit_users),),
        ).fetchall()
    return {"total_events": total, "by_user": [{"username": r[0], "events": r[1], "last": r[2]} for r in rows]}
