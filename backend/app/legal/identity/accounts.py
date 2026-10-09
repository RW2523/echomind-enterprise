"""Account state for the Law edition: status (active/disabled), lockout after repeated failures,
firm roles, password changes. Disabling an account or changing its password revokes sessions."""
from __future__ import annotations

import time
from typing import Optional

from ...core import auth as core_auth
from ...core.config import settings
from ...core.db import get_conn
from . import sessions
from .sessions import _iso

FIRM_ROLES = ("firm_admin", "managing_partner", "attorney", "paralegal", "reviewer", "records_admin", "read_only")
_COLS = "id, username, role, COALESCE(tenant,''), COALESCE(status,'active'), mfa_enabled, failed_logins, locked_until, password_hash"


def _row(row) -> Optional[dict]:
    if not row:
        return None
    return {"id": row[0], "username": row[1], "role": row[2], "tenant": row[3], "status": row[4],
            "mfa_enabled": bool(row[5]), "failed_logins": row[6] or 0, "locked_until": row[7], "password_hash": row[8]}


def get(user_id: str) -> Optional[dict]:
    with get_conn() as conn:
        return _row(conn.execute(f"SELECT {_COLS} FROM users WHERE id=?", (user_id,)).fetchone())


def get_by_username(username: str) -> Optional[dict]:
    with get_conn() as conn:
        return _row(conn.execute(f"SELECT {_COLS} FROM users WHERE username=?", ((username or "").strip(),)).fetchone())


def is_locked(account: Optional[dict]) -> bool:
    return bool(account and account.get("locked_until") and account["locked_until"] > time.time())


def register_failure(user_id: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE users SET failed_logins = COALESCE(failed_logins,0) + 1 WHERE id=?", (user_id,))
        n = conn.execute("SELECT failed_logins FROM users WHERE id=?", (user_id,)).fetchone()
        if n and n[0] >= settings.LEGAL_LOCKOUT_THRESHOLD:
            conn.execute("UPDATE users SET locked_until=?, failed_logins=0 WHERE id=?",
                         (time.time() + settings.LEGAL_LOCKOUT_MINUTES * 60, user_id))
        conn.commit()


def register_success(user_id: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE users SET failed_logins=0, locked_until=NULL WHERE id=?", (user_id,))
        conn.commit()


def set_status(username: str, status: str) -> bool:
    if status not in ("active", "disabled"):
        raise ValueError("status must be active or disabled")
    acct = get_by_username(username)
    if not acct:
        return False
    with get_conn() as conn:
        conn.execute("UPDATE users SET status=? WHERE id=?", (status, acct["id"]))
        conn.commit()
    if status == "disabled":
        sessions.revoke_all_for_user(acct["id"], reason="account_disabled")
    return True


def change_password(user_id: str, new_password: str, keep_token: Optional[str] = None) -> int:
    """Set a new password and revoke every other session of the user. Returns sessions revoked."""
    acct = get(user_id)
    if not acct:
        raise ValueError("unknown user")
    core_auth.set_password(acct["username"], new_password)
    with get_conn() as conn:
        conn.execute("UPDATE users SET password_changed_at=?, must_change_password=0 WHERE id=?", (_iso(time.time()), user_id))
        conn.commit()
    return sessions.revoke_all_for_user(user_id, reason="password_changed", keep_token=keep_token)


def roles_of(user_id: str) -> list[str]:
    with get_conn() as conn:
        return [r[0] for r in conn.execute("SELECT role FROM user_roles WHERE user_id=? ORDER BY role", (user_id,))]


def grant_role(user_id: str, role: str, granted_by: Optional[str] = None) -> None:
    if role not in FIRM_ROLES:
        raise ValueError(f"unknown firm role {role!r}")
    with get_conn() as conn:
        conn.execute("INSERT OR IGNORE INTO user_roles (user_id, role, granted_by, granted_at) VALUES (?,?,?,?)",
                     (user_id, role, granted_by, _iso(time.time())))
        conn.commit()


def require_password_change(user_id: str) -> None:
    """After an administrator sets someone's password, they must choose their own at next sign-in."""
    with get_conn() as conn:
        conn.execute("UPDATE users SET must_change_password=1 WHERE id=?", (user_id,))
        conn.commit()
