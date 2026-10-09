"""Server-side, revocable sessions (decision D-006).

The browser holds an opaque random token; the database holds only its SHA-256. A session has a
sliding idle timeout and an absolute lifetime. A session created after the password step but
before the MFA code is "partial" (mfa_verified=0) and lives only LEGAL_MFA_PENDING_MIN minutes.
A session dies when it is revoked, expires, or its user is disabled.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from datetime import datetime, timezone
from typing import Optional

from ...core.config import settings
from ...core.db import get_conn

# Writing last_seen on every request would serialise all reads on SQLite; slide at most once a minute.
_SLIDE_GRANULARITY_S = 60


def _now() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _lifetimes(now: float, mfa_verified: bool) -> tuple[float, float]:
    if not mfa_verified:
        end = now + settings.LEGAL_MFA_PENDING_MIN * 60
        return end, end
    abs_end = now + settings.LEGAL_SESSION_ABSOLUTE_HOURS * 3600
    return min(now + settings.LEGAL_SESSION_IDLE_MIN * 60, abs_end), abs_end


def create(user_id: str, mfa_verified: bool, ip: Optional[str] = None, user_agent: Optional[str] = None) -> tuple[str, str]:
    """Create a session; returns (session_token, csrf_token). Neither is stored in clear."""
    token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    now = _now()
    idle, absolute = _lifetimes(now, mfa_verified)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, created_at, last_seen_at, idle_expires_at, abs_expires_at, "
            "mfa_verified, csrf_hash, ip, user_agent) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (hash_token(token), user_id, _iso(now), _iso(now), idle, absolute, 1 if mfa_verified else 0,
             hash_token(csrf), (ip or "")[:64], (user_agent or "")[:200]),
        )
        conn.commit()
    return token, csrf


_SELECT = (
    "SELECT s.token_hash, s.user_id, s.created_at, s.last_seen_at, s.idle_expires_at, s.abs_expires_at, "
    "s.mfa_verified, s.csrf_hash, s.ip, s.user_agent, u.username, u.role, COALESCE(u.tenant,''), "
    "COALESCE(u.status,'active'), COALESCE(u.org_id,''), u.must_change_password FROM sessions s JOIN users u ON u.id = s.user_id "
    "WHERE s.token_hash = ? AND s.revoked_at IS NULL"
)


def lookup_hash(token_hash: str, slide: bool = True) -> Optional[dict]:
    """The live session with this token hash, or None (revoked, expired, unknown, user disabled)."""
    if not token_hash:
        return None
    now = _now()
    with get_conn() as conn:
        row = conn.execute(_SELECT, (token_hash,)).fetchone()
        if not row:
            return None
        (th, uid, created, seen, idle, absolute, mfa, csrf_hash, ip, ua, username, role, tenant, status, org_id, must_change) = row
        if status != "active" or now >= absolute or now >= idle:
            return None
        if slide and mfa:
            new_idle = min(now + settings.LEGAL_SESSION_IDLE_MIN * 60, absolute)
            if new_idle - idle >= _SLIDE_GRANULARITY_S:
                conn.execute("UPDATE sessions SET idle_expires_at=?, last_seen_at=? WHERE token_hash=?",
                             (new_idle, _iso(now), th))
                conn.commit()
                idle, seen = new_idle, _iso(now)
    return {"token_hash": th, "user_id": uid, "username": username, "role": role, "tenant": tenant, "org_id": org_id,
            "mfa_verified": bool(mfa), "csrf_hash": csrf_hash, "created_at": created, "last_seen_at": seen,
            "idle_expires_at": idle, "abs_expires_at": absolute, "ip": ip, "user_agent": ua,
            "must_change_password": bool(must_change)}


def lookup(token: str, slide: bool = True) -> Optional[dict]:
    return lookup_hash(hash_token(token), slide=slide) if token else None


def csrf_ok(session: dict, csrf_token: Optional[str]) -> bool:
    return bool(csrf_token) and hmac.compare_digest(hash_token(csrf_token), session.get("csrf_hash") or "")


def revoke_hash(token_hash: str, reason: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute("UPDATE sessions SET revoked_at=?, revoke_reason=? WHERE token_hash=? AND revoked_at IS NULL",
                           (_iso(_now()), reason, token_hash))
        conn.commit()
        return cur.rowcount > 0


def revoke(token: str, reason: str) -> bool:
    return revoke_hash(hash_token(token), reason) if token else False


def revoke_all_for_user(user_id: str, reason: str, keep_token: Optional[str] = None) -> int:
    keep = hash_token(keep_token) if keep_token else ""
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE sessions SET revoked_at=?, revoke_reason=? WHERE user_id=? AND revoked_at IS NULL AND token_hash<>?",
            (_iso(_now()), reason, user_id, keep))
        conn.commit()
        return cur.rowcount


def rotate(old_token: str, mfa_verified: bool) -> tuple[str, str]:
    """Replace a session with a fresh token (after MFA, to defeat session fixation)."""
    old = lookup(old_token, slide=False)
    if not old:
        raise ValueError("session not active")
    revoke(old_token, reason="rotated")
    return create(old["user_id"], mfa_verified=mfa_verified, ip=old.get("ip"), user_agent=old.get("user_agent"))


def list_for_user(user_id: str) -> list[dict]:
    now = _now()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT token_hash, created_at, last_seen_at, ip, user_agent, mfa_verified, idle_expires_at, abs_expires_at "
            "FROM sessions WHERE user_id=? AND revoked_at IS NULL ORDER BY created_at DESC", (user_id,)).fetchall()
    return [{"id": r[0][:16], "created_at": r[1], "last_seen_at": r[2], "ip": r[3], "user_agent": r[4],
             "mfa_verified": bool(r[5])} for r in rows if now < r[6] and now < r[7]]


def revoke_by_public_id(user_id: str, public_id: str, reason: str) -> bool:
    """Revoke one of the user's own sessions by the 16-char id shown in list_for_user()."""
    if len(public_id or "") != 16:
        return False
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE sessions SET revoked_at=?, revoke_reason=? WHERE user_id=? AND substr(token_hash,1,16)=? AND revoked_at IS NULL",
            (_iso(_now()), reason, user_id, public_id))
        conn.commit()
        return cur.rowcount > 0
