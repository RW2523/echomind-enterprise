"""Minimal, dependency-free HS256 token check for the voice WebSocket (Phase 0b).

Opt-in: gates the voice /ws handshake only when VOICE_AUTH_ENABLED is set. Validates the
echomind_token cookie issued by the backend, so VOICE_AUTH_SECRET must equal the backend's
AUTH_SECRET. Default off => voice behaves exactly as before.
"""
import base64
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request

_ENABLED = os.getenv("VOICE_AUTH_ENABLED", "0").lower() in ("1", "true", "yes")
_SECRET = os.getenv("VOICE_AUTH_SECRET", "")
_BACKEND = (os.getenv("BACKEND_CHAT_URL", "http://backend:8000") or "").rstrip("/")


def auth_enabled() -> bool:
    return _ENABLED


def _b64u_dec(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def valid_token(token: str) -> bool:
    if not _SECRET or not token:
        return False
    try:
        h, p, sig = token.split(".")
        expected = base64.urlsafe_b64encode(
            hmac.new(_SECRET.encode("utf-8"), f"{h}.{p}".encode(), hashlib.sha256).digest()
        ).rstrip(b"=").decode("ascii")
        if not hmac.compare_digest(sig, expected):
            return False
        payload = json.loads(_b64u_dec(p))
        return int(payload.get("exp", 0)) >= int(time.time())
    except Exception:
        return False


def session_active(token: str, timeout_s: float = 3.0) -> bool:
    """Ask the backend whether the session is still active (it rejects logged-out tokens,
    NC-2026-012). Fails closed: if the backend cannot confirm, the voice session is refused."""
    if not valid_token(token):
        return False
    req = urllib.request.Request(f"{_BACKEND}/api/auth/me", headers={"Cookie": f"echomind_token={token}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False
