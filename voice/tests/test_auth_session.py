"""Voice WebSocket gate must refuse logged-out sessions (REG-04 NC-2026-012)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import importlib
import json
import time
import urllib.error

SECRET = "test-secret-for-voice"


def _token(exp_delta=3600):
    b = lambda d: base64.urlsafe_b64encode(json.dumps(d, separators=(",", ":")).encode()).rstrip(b"=").decode()
    h, p = b({"alg": "HS256", "typ": "JWT"}), b({"sub": "u", "exp": int(time.time()) + exp_delta})
    sig = base64.urlsafe_b64encode(hmac.new(SECRET.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()).rstrip(b"=").decode()
    return f"{h}.{p}.{sig}"


def _module(monkeypatch):
    monkeypatch.setenv("VOICE_AUTH_SECRET", SECRET)
    import app.auth_check as m
    return importlib.reload(m)


class _Resp:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_backend_confirms_active_session(monkeypatch):
    m = _module(monkeypatch)
    monkeypatch.setattr(m.urllib.request, "urlopen", lambda req, timeout=0: _Resp(200))
    assert m.session_active(_token()) is True


def test_revoked_session_is_refused(monkeypatch):
    m = _module(monkeypatch)
    def deny(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 401, "Not authenticated", {}, None)
    monkeypatch.setattr(m.urllib.request, "urlopen", deny)
    assert m.session_active(_token()) is False


def test_backend_unreachable_fails_closed(monkeypatch):
    m = _module(monkeypatch)
    def down(req, timeout=0):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(m.urllib.request, "urlopen", down)
    assert m.session_active(_token()) is False


def test_bad_signature_never_reaches_backend(monkeypatch):
    m = _module(monkeypatch)
    called = []
    monkeypatch.setattr(m.urllib.request, "urlopen", lambda req, timeout=0: called.append(1) or _Resp(200))
    assert m.session_active(_token()[:-2] + "xx") is False
    assert m.session_active(_token(exp_delta=-10)) is False
    assert called == []
