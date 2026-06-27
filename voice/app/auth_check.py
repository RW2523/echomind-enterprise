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

_ENABLED = os.getenv("VOICE_AUTH_ENABLED", "0").lower() in ("1", "true", "yes")
_SECRET = os.getenv("VOICE_AUTH_SECRET", "")


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
