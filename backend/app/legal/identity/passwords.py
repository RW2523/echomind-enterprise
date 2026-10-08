"""Password policy for the Law edition (S1.1). Length first (NIST SP 800-63B style): no forced
character classes, but no short, repetitive, username-based or well-known passwords."""
from __future__ import annotations

from ...core.config import settings

# A short built-in list is enough to stop the worst choices offline; length does the rest.
_COMMON = {
    "password", "passw0rd", "password1", "password123", "123456789012", "qwertyuiop", "letmein",
    "welcome", "admin", "administrator", "changeme", "iloveyou", "monkey", "dragon", "football",
    "lawfirm", "echomind", "trustno1", "abc123", "qwerty123",
}


def problems(password: str, username: str = "") -> list[str]:
    """Policy violations as short codes: length, username, common, repetitive. Empty list = ok."""
    pw = password or ""
    low = pw.lower()
    out = []
    if len(pw) < settings.LEGAL_PASSWORD_MIN_LENGTH:
        out.append("length")
    user = (username or "").strip().lower()
    if user and (user in low or low in user):
        out.append("username")
    stripped = "".join(ch for ch in low if ch.isalnum())
    if low in _COMMON or any(low == c * (len(low) // len(c)) for c in _COMMON if len(low) >= len(c) and len(low) % len(c) == 0):
        out.append("common")
    elif stripped in _COMMON:
        out.append("common")
    if pw and len(set(pw)) <= 2:
        out.append("repetitive")
    return out


MESSAGES = {
    "length": "use at least {n} characters",
    "username": "must not contain the username",
    "common": "is a commonly used password",
    "repetitive": "is too repetitive",
}


def describe(codes: list[str]) -> str:
    return "Password " + "; ".join(MESSAGES[c].format(n=settings.LEGAL_PASSWORD_MIN_LENGTH) for c in codes) + "."
