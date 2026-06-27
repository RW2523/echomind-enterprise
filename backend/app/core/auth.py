"""Phase 0b auth core: local accounts + HMAC-signed JWT (HS256), stdlib only (offline-safe).

Opt-in via settings.AUTH_ENABLED. When disabled the app behaves exactly as before — none of
this is enforced. No third-party dependencies (pbkdf2 + hmac from the standard library).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from typing import Optional

from .config import settings
from .db import get_conn
from ..utils.ids import new_id, now_iso

logger = logging.getLogger(__name__)

_PBKDF2_ITERATIONS = 200_000


# ── Password hashing (pbkdf2-hmac-sha256) ────────────────────────────────────────
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_hex, hash_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


# ── HS256 JWT (minimal, dependency-free) ─────────────────────────────────────────
def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _b64u_dec(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _secret() -> bytes:
    if settings.AUTH_SECRET:
        return settings.AUTH_SECRET.encode("utf-8")
    # Persist a generated secret so issued tokens survive restarts.
    path = os.path.join(settings.DATA_DIR, "auth_secret.key")
    try:
        if os.path.exists(path):
            return open(path, "rb").read().strip()
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        key = secrets.token_hex(32).encode("ascii")
        with open(path, "wb") as fh:
            fh.write(key)
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass
        return key
    except Exception:
        return b"echomind-ephemeral-" + secrets.token_hex(8).encode("ascii")


def make_token(sub: str, role: str, username: str, ttl_min: Optional[int] = None) -> str:
    now = int(time.time())
    ttl = (ttl_min if ttl_min is not None else settings.AUTH_TOKEN_TTL_MIN) * 60
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"sub": sub, "role": role, "username": username, "iat": now, "exp": now + ttl}
    h = _b64u(json.dumps(header, separators=(",", ":")).encode())
    p = _b64u(json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64u(hmac.new(_secret(), f"{h}.{p}".encode(), hashlib.sha256).digest())
    return f"{h}.{p}.{sig}"


def decode_token(token: str) -> Optional[dict]:
    try:
        h, p, sig = token.split(".")
        expected = _b64u(hmac.new(_secret(), f"{h}.{p}".encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(_b64u_dec(p))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


# ── User store ───────────────────────────────────────────────────────────────────
def get_user_by_username(username: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, role, created_at FROM users WHERE username=?", (username,)
        ).fetchone()
    if not row:
        return None
    return {"id": row[0], "username": row[1], "password_hash": row[2], "role": row[3], "created_at": row[4]}


def create_user(username: str, password: str, role: str = "user") -> dict:
    username = (username or "").strip()
    if not username or not password:
        raise ValueError("username and password required")
    if get_user_by_username(username):
        raise ValueError("user already exists")
    uid = new_id("usr")
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO users (id, username, password_hash, role, created_at) VALUES (?,?,?,?,?)",
            (uid, username, hash_password(password), role, now_iso()),
        )
        conn.commit()
    return {"id": uid, "username": username, "role": role}


def set_password(username: str, password: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE users SET password_hash=? WHERE username=?", (hash_password(password), username))
        conn.commit()


def list_users() -> list:
    with get_conn() as conn:
        rows = conn.execute("SELECT id, username, role, created_at FROM users ORDER BY created_at ASC").fetchall()
    return [{"id": r[0], "username": r[1], "role": r[2], "created_at": r[3]} for r in rows]


def delete_user(username: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM users WHERE username=?", (username,))
        conn.commit()


def count_users() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def authenticate(username: str, password: str) -> Optional[dict]:
    u = get_user_by_username(username)
    if not u or not verify_password(password, u["password_hash"]):
        return None
    return {"id": u["id"], "username": u["username"], "role": u["role"]}


def seed_admin() -> None:
    """Ensure an admin login exists once auth is in play.
    - AUTH_ADMIN_PASSWORD set -> (re)set the admin password every boot.
    - else, on first run with no users -> create admin with a random password and log it once.
    Skipped entirely when auth is off and no admin password is configured (keeps the demo DB clean)."""
    if not settings.AUTH_ENABLED and not settings.AUTH_ADMIN_PASSWORD:
        return
    try:
        admin = settings.AUTH_ADMIN_USER or "admin"
        if settings.AUTH_ADMIN_PASSWORD:
            if get_user_by_username(admin):
                set_password(admin, settings.AUTH_ADMIN_PASSWORD)
            else:
                create_user(admin, settings.AUTH_ADMIN_PASSWORD, role="admin")
            logger.info("Auth: admin user '%s' ready (password from env).", admin)
        elif count_users() == 0:
            pw = secrets.token_urlsafe(12)
            create_user(admin, pw, role="admin")
            logger.warning(
                "Auth: created initial admin '%s' with generated password: %s  "
                "(set AUTH_ADMIN_PASSWORD to control this)", admin, pw,
            )
    except Exception as e:
        logger.warning("Auth: seed_admin failed: %s", e)


# ── Request helper ────────────────────────────────────────────────────────────────
def user_from_request(request) -> Optional[dict]:
    """Validate the bearer token from the Authorization header or the echomind_token cookie.
    Returns the decoded token payload (sub/role/username/exp) or None."""
    token = ""
    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    if not token:
        try:
            token = request.cookies.get("echomind_token", "")
        except Exception:
            token = ""
    if not token:
        return None
    return decode_token(token)
