"""Signed-in HTTP client for scripts that talk to an EchoMind Law DEVELOPMENT backend.

The Law edition (S1.1) uses an HttpOnly, Secure session cookie, a CSRF header on writes and a
TOTP second factor. `open_session()` signs in and returns a urllib opener that carries all three:

- cookies are kept in a jar that may send Secure cookies to loopback addresses only (a dev backend
  on http://127.0.0.1 is a "secure context" to browsers too);
- X-CSRF-Token is added to POST/PUT/PATCH/DELETE from the CSRF cookie;
- on the dev admin's first sign-in the authenticator is enrolled and its key and recovery codes are
  stored (mode 0600) in .law-data/, which is gitignored and must only hold fictional development data.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import http.cookiejar
import json
import os
import pathlib
import struct
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
_CSRF_COOKIE = "__Host-echomind_csrf"
_MUTATING = {"POST", "PUT", "PATCH", "DELETE"}


def totp(secret_b32: str, at: float | None = None) -> str:
    key = base64.b32decode(secret_b32.upper() + "=" * (-len(secret_b32) % 8))
    digest = hmac.new(key, struct.pack(">Q", int(time.time() if at is None else at) // 30), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    return str((struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF) % 1_000_000).zfill(6)


class _LoopbackSecurePolicy(http.cookiejar.DefaultCookiePolicy):
    def return_ok_secure(self, cookie, request):
        if urllib.parse.urlparse(request.full_url).hostname in _LOOPBACK:
            return True
        return super().return_ok_secure(cookie, request)


class _CsrfHandler(urllib.request.BaseHandler):
    handler_order = 900  # after the cookie processor

    def __init__(self, jar: http.cookiejar.CookieJar):
        self.jar = jar

    def http_request(self, req):
        if req.get_method() in _MUTATING:
            for c in self.jar:
                if c.name == _CSRF_COOKIE:
                    req.add_unredirected_header("X-CSRF-Token", c.value)
        return req

    https_request = http_request


def _write_private(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)


def open_session(base: str, username: str, password: str, data_dir: pathlib.Path | None = None):
    """Sign in (password + TOTP, enrolling on first use) and return an authenticated opener."""
    host = urllib.parse.urlparse(base).hostname
    if host not in _LOOPBACK:
        raise SystemExit(f"law_client is for local development backends only (got {host!r})")
    data_dir = pathlib.Path(data_dir or os.getenv("LAW_DATA_DIR") or ROOT / ".law-data")
    secret_file = data_dir / f"dev_{username}_totp_secret"
    jar = http.cookiejar.CookieJar(policy=_LoopbackSecurePolicy())
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar), _CsrfHandler(jar))

    def post(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(f"{base}{path}", data=data if data is not None else b"", method="POST",
                                     headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=60) as r:
            return json.loads(r.read() or b"{}")

    status = post("/auth/login", {"username": username, "password": password}).get("status", "ok")
    if status == "mfa_enrollment_required":
        secret = post("/auth/mfa/enroll")["secret"]
        _write_private(secret_file, secret + "\n")
        codes = post("/auth/mfa/confirm", {"code": totp(secret)})["recovery_codes"]
        _write_private(data_dir / f"dev_{username}_recovery_codes", "\n".join(codes) + "\n")
    elif status == "mfa_required":
        secret = (os.getenv("LAW_TOTP_SECRET") or (secret_file.read_text().strip() if secret_file.exists() else ""))
        if not secret:
            raise SystemExit(f"{username} has an authenticator but no key is stored in {secret_file}")
        for attempt in range(2):
            try:
                post("/auth/mfa/verify", {"code": totp(secret)})
                break
            except urllib.error.HTTPError as e:
                # A code can be used once: if this 30 s step was already spent, wait for the next one.
                if e.code != 401 or attempt:
                    raise
                time.sleep(31 - (time.time() % 30))
    return opener
