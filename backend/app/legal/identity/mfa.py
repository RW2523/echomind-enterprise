"""TOTP multi-factor authentication (RFC 6238; decision D-007) and single-use recovery codes.

Standard library only, works fully offline with any authenticator app (SHA-1, 30 s, 6 digits).
A used time-step is remembered per user so a code cannot be replayed.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import struct
import time
import urllib.parse
from typing import Optional

from ...core.config import settings
from ...core.db import get_conn
from .sessions import _iso, hash_token

_STEP = 30
_DIGITS = 6
_WINDOW = 1          # accept the previous and next 30 s step for clock drift
_RECOVERY_CODES = 10
ISSUER = "EchoMind Law"


def new_secret() -> str:
    return base64.b32encode(secrets.token_bytes(20)).decode("ascii")


def _hotp(secret_b32: str, counter: int) -> str:
    key = base64.b32decode(secret_b32.upper() + "=" * (-len(secret_b32) % 8))
    digest = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code = (struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF) % (10 ** _DIGITS)
    return str(code).zfill(_DIGITS)


def totp(secret_b32: str, at: float) -> str:
    return _hotp(secret_b32, int(at) // _STEP)


def verify_totp(secret_b32: str, code: str, now: Optional[float] = None, last_used_step: Optional[int] = None) -> Optional[int]:
    """Return the matched time-step, or None. Steps at or before last_used_step are refused (replay)."""
    code = (code or "").strip().replace(" ", "")
    if len(code) != _DIGITS or not code.isdigit() or not secret_b32:
        return None
    current = int(time.time() if now is None else now) // _STEP
    for step in range(current - _WINDOW, current + _WINDOW + 1):
        if last_used_step is not None and step <= last_used_step:
            continue
        if hmac.compare_digest(_hotp(secret_b32, step), code):
            return step
    return None


def otpauth_uri(secret_b32: str, username: str) -> str:
    # Key-URI format: literal ':' between issuer and account; %20 (not '+') for spaces, which some
    # authenticator apps would otherwise display literally.
    label = urllib.parse.quote(f"{ISSUER}:{username}", safe=":")
    query = urllib.parse.urlencode({"secret": secret_b32, "issuer": ISSUER, "algorithm": "SHA1",
                                    "digits": _DIGITS, "period": _STEP}, quote_via=urllib.parse.quote)
    return f"otpauth://totp/{label}?{query}"


# ── Enrolment and verification against the users table ─────────────────────────
def _state(user_id: str) -> tuple[Optional[str], bool, Optional[int]]:
    with get_conn() as conn:
        row = conn.execute("SELECT mfa_secret, mfa_enabled, mfa_last_step FROM users WHERE id=?", (user_id,)).fetchone()
    return (row[0], bool(row[1]), row[2]) if row else (None, False, None)


def is_enrolled(user_id: str) -> bool:
    return _state(user_id)[1]


def begin_enrollment(user_id: str) -> str:
    """Store a new pending secret (MFA stays off until confirmed) and return it."""
    secret = new_secret()
    with get_conn() as conn:
        conn.execute("UPDATE users SET mfa_secret=?, mfa_enabled=0, mfa_last_step=NULL WHERE id=?", (secret, user_id))
        conn.commit()
    return secret


def _accept_step(user_id: str, step: int) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE users SET mfa_last_step=? WHERE id=?", (step, user_id))
        conn.commit()


def confirm_enrollment(user_id: str, code: str) -> bool:
    secret, enabled, _ = _state(user_id)
    if enabled or not secret:
        return False
    step = verify_totp(secret, code)
    if step is None:
        return False
    with get_conn() as conn:
        conn.execute("UPDATE users SET mfa_enabled=1, mfa_last_step=? WHERE id=?", (step, user_id))
        conn.commit()
    return True


def verify_login_code(user_id: str, code: str) -> bool:
    """A current TOTP code, or an unused recovery code."""
    secret, enabled, last = _state(user_id)
    if not enabled:
        return False
    step = verify_totp(secret, code, last_used_step=last)
    if step is not None:
        _accept_step(user_id, step)
        return True
    return use_recovery_code(user_id, code)


# ── Recovery codes ──────────────────────────────────────────────────────────────
def _norm(code: str) -> str:
    return (code or "").strip().lower().replace("-", "").replace(" ", "")


def _code_hash(user_id: str, code: str) -> str:
    return hashlib.sha256(f"{user_id}:{_norm(code)}".encode("utf-8")).hexdigest()


def issue_recovery_codes(user_id: str) -> list[str]:
    """Replace the user's recovery codes; the plain codes are returned once and never stored."""
    codes = []
    for _ in range(_RECOVERY_CODES):
        raw = base64.b32encode(secrets.token_bytes(7)).decode("ascii").lower()[:10]
        codes.append(f"{raw[:5]}-{raw[5:]}")
    now = _iso(time.time())
    with get_conn() as conn:
        conn.execute("DELETE FROM mfa_recovery_codes WHERE user_id=?", (user_id,))
        conn.executemany("INSERT INTO mfa_recovery_codes (user_id, code_hash, created_at) VALUES (?,?,?)",
                         [(user_id, _code_hash(user_id, c), now) for c in codes])
        conn.commit()
    return codes


def use_recovery_code(user_id: str, code: str) -> bool:
    if len(_norm(code)) != 10:
        return False
    with get_conn() as conn:
        cur = conn.execute("UPDATE mfa_recovery_codes SET used_at=? WHERE user_id=? AND code_hash=? AND used_at IS NULL",
                           (_iso(time.time()), user_id, _code_hash(user_id, code)))
        conn.commit()
        return cur.rowcount == 1


# ── Enrolment links (D-021) ─────────────────────────────────────────────────────
def issue_enrollment_token(user_id: str, created_by: str) -> tuple[str, float]:
    """A one-time link token for binding an authenticator; older unused tokens stop working."""
    token = secrets.token_urlsafe(24)
    now = time.time()
    expires = now + settings.LEGAL_ENROLLMENT_LINK_HOURS * 3600
    with get_conn() as conn:
        conn.execute("UPDATE mfa_enrollment_tokens SET revoked_at=? WHERE user_id=? AND used_at IS NULL AND revoked_at IS NULL",
                     (_iso(now), user_id))
        conn.execute("INSERT INTO mfa_enrollment_tokens (token_hash, user_id, created_by, created_at, expires_at) VALUES (?,?,?,?,?)",
                     (hash_token(token), user_id, created_by, _iso(now), expires))
        conn.commit()
    return token, expires


_LIVE_TOKEN = "token_hash=? AND user_id=? AND used_at IS NULL AND revoked_at IS NULL AND expires_at > ?"


def enrollment_token_valid(user_id: str, token: str) -> bool:
    if not token:
        return False
    with get_conn() as conn:
        return conn.execute(f"SELECT 1 FROM mfa_enrollment_tokens WHERE {_LIVE_TOKEN}",
                            (hash_token(token), user_id, time.time())).fetchone() is not None


def consume_enrollment_token(user_id: str, token: str) -> bool:
    if not token:
        return False
    with get_conn() as conn:
        cur = conn.execute(f"UPDATE mfa_enrollment_tokens SET used_at=? WHERE {_LIVE_TOKEN}",
                           (_iso(time.time()), hash_token(token), user_id, time.time()))
        conn.commit()
        return cur.rowcount == 1


def reset(user_id: str) -> None:
    """Forget the user's authenticator and recovery codes (administrator action)."""
    with get_conn() as conn:
        conn.execute("UPDATE users SET mfa_secret=NULL, mfa_enabled=0, mfa_last_step=NULL WHERE id=?", (user_id,))
        conn.execute("DELETE FROM mfa_recovery_codes WHERE user_id=?", (user_id,))
        conn.commit()
