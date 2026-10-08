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


# ═════════════════════════════════════════════════════════════════════════════════
# EchoMind Law edition (S1.1, SEC-01; decision D-013)
# The browser's backend session is the only credential. The voice service never verifies it
# locally: it asks the backend, exchanges it for short-lived on-behalf-of tokens for RAG calls,
# and re-checks it while the socket is open.
# ═════════════════════════════════════════════════════════════════════════════════
_SESSION_COOKIE = "__Host-echomind_s"
_CSRF_COOKIE = "__Host-echomind_csrf"
_OBO_REFRESH_MARGIN_S = 60


def _cookie_value(cookie_header: str, name: str) -> str:
    for part in (cookie_header or "").split(";"):
        k, _, v = part.strip().partition("=")
        if k == name:
            return v
    return ""


def allowed_origins() -> set:
    return {o.strip() for o in (os.getenv("VOICE_ALLOWED_ORIGINS") or "").split(",") if o.strip()}


def law_origin_ok(origin: str) -> bool:
    """Browsers always send Origin on a WebSocket handshake; it must be one of ours."""
    return bool(origin) and origin in allowed_origins()


def law_session_user(cookie_header: str, timeout_s: float = 3.0):
    """The backend's view of this browser session ({id, username, ...}) or None. Fails closed."""
    if not _cookie_value(cookie_header, _SESSION_COOKIE):
        return None
    req = urllib.request.Request(f"{_BACKEND}/api/auth/me", headers={"Cookie": cookie_header})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return json.loads(r.read() or b"{}").get("user") if r.status == 200 else None
    except (urllib.error.URLError, OSError, ValueError):
        return None


class BackendAuth:
    """Credential holder for one voice WebSocket: keeps the browser's session cookie in memory and
    trades it for on-behalf-of tokens (5 min) that authorise this user's RAG calls."""

    def __init__(self, cookie_header: str, timeout_s: float = 3.0):
        import threading
        self._cookie = cookie_header
        self._timeout = timeout_s
        self._token = ""
        self._exp = 0.0
        self._lock = threading.Lock()

    def refresh(self) -> bool:
        body = json.dumps({"audience": "voice"}).encode()
        req = urllib.request.Request(
            f"{_BACKEND}/api/auth/obo-token", data=body, method="POST",
            headers={"Cookie": self._cookie, "Content-Type": "application/json",
                     "X-CSRF-Token": _cookie_value(self._cookie, _CSRF_COOKIE)})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                data = json.loads(r.read() or b"{}")
            self._token, self._exp = data["token"], time.time() + float(data.get("expires_in", 0))
            return True
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            self._token, self._exp = "", 0.0
            return False

    def headers(self) -> dict:
        """Authorization header for a backend call, refreshed near expiry; {} if it cannot be had
        (the backend then refuses the call: fail closed)."""
        with self._lock:
            if not self._token or self._exp - time.time() < _OBO_REFRESH_MARGIN_S:
                self.refresh()
            return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def still_valid(self) -> bool:
        return law_session_user(self._cookie, self._timeout) is not None
