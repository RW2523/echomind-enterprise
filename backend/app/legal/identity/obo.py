"""Short-lived on-behalf-of tokens for service-to-service calls (decision D-013).

The voice service exchanges the user's session for a token bound to an audience ("voice"), the
user and that session, valid LEGAL_OBO_TTL_SEC seconds. Every use re-checks the underlying
session, so logging out, revocation or disabling the user stops the service immediately.
Tokens are `obo.<payload>.<sig>`, signed with a key derived from AUTH_SECRET for this purpose
only, so a legacy JWT signed with AUTH_SECRET can never pass as one.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Optional

from ...core.config import settings
from . import sessions

_PREFIX = "obo"


def _now() -> float:
    return time.time()


def _key() -> bytes:
    return hmac.new((settings.AUTH_SECRET or "").encode("utf-8"), b"echomind-law/obo/v1", hashlib.sha256).digest()


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def issue(session: dict, audience: str) -> str:
    now = int(_now())
    payload = {"typ": "obo", "aud": audience, "sub": session["user_id"], "sid": session["token_hash"],
               "iat": now, "exp": now + settings.LEGAL_OBO_TTL_SEC}
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64(hmac.new(_key(), f"{_PREFIX}.{body}".encode(), hashlib.sha256).digest())
    return f"{_PREFIX}.{body}.{sig}"


def verify(token: str, audience: str) -> Optional[dict]:
    """{"user_id", "session"} for a valid token whose session is still live; otherwise None."""
    try:
        prefix, body, sig = (token or "").split(".")
        if prefix != _PREFIX:
            return None
        expected = _b64(hmac.new(_key(), f"{prefix}.{body}".encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        p = json.loads(_unb64(body))
        if p.get("typ") != "obo" or p.get("aud") != audience or int(p.get("exp", 0)) <= _now():
            return None
        s = sessions.lookup_hash(p.get("sid", ""), slide=True)
        if not s or not s["mfa_verified"] or s["user_id"] != p.get("sub"):
            return None
        return {"user_id": s["user_id"], "session": s, "exp": int(p["exp"])}
    except Exception:
        return None
