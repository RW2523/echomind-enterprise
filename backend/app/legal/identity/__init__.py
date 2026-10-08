"""EchoMind Law identity (S1.1, SEC-01): server-side sessions, TOTP MFA, lockout, CSRF and
on-behalf-of tokens. Only used when ECHOMIND_EDITION=law; legacy editions keep core/auth.py's JWTs."""

SESSION_COOKIE = "__Host-echomind_s"   # opaque token; HttpOnly, Secure, SameSite=Strict
CSRF_COOKIE = "__Host-echomind_csrf"   # readable by the SPA, echoed back in CSRF_HEADER
CSRF_HEADER = "x-csrf-token"
