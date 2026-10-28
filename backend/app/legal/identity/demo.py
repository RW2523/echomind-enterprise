"""One-click demo sign-in (D-077). Off unless LEGAL_DEMO_SIGNIN.

The owner asked to enter the demo without passwords or codes. Only the fictional people the demo seed made (the names in
$ECHOMIND_DATA_DIR/legal/demo_credentials.json) can be entered this way; the administrator and any account added later
keep their password and two-step code. By default only from this server or a private address (the client's real address,
D-075); LEGAL_DEMO_SIGNIN_FROM=anywhere is the owner's choice. Every entry is audited (auth.demo_signin).

Commercial-readiness review, 2026-10-28 (Phase 1, F1/F10):
- the server refuses to start with demo sign-in unless it is a demo instance (LEGAL_PROFILE=demo and the seed's marker;
  core/edition.py), so it cannot be left on where real data lives;
- a demo person who holds a firm-wide power (administration, records, the managing partners who make up the two-person
  rules) is never entered in one click: they keep their password and code, even in a demo;
- an entry must come from the app's own pages (Origin), so another site cannot sign a visitor in;
- when demo sign-in is off, sessions the demo people still hold are ended at start-up.
"""
from __future__ import annotations

import ipaddress
import json
import os

from ...core.config import settings
from ...core.edition import is_law
from ..netaddr import client_address


# Firm-wide powers: never one click away, even in a demo.
PRIVILEGED = frozenset({"firm_admin", "managing_partner", "records_admin"})
MARKER = "DEMO_INSTANCE"


def seeded() -> list[str]:
    """Every fictional person the demo seed made, whatever their role."""
    try:
        with open(os.path.join(settings.DATA_DIR, "legal", "demo_credentials.json")) as f:
            return sorted(json.load(f))
    except (OSError, ValueError):
        return []


def people() -> list[str]:
    """The demo people who may be entered in one click: the seeded ones without a firm-wide power, checked on every call
    (a role granted later takes them off the list)."""
    from . import accounts
    out = []
    for name in seeded():
        acct = accounts.get_by_username(name)
        if acct is None or not (set(accounts.roles_of(acct["id"])) & PRIVILEGED):
            out.append(name)
    return out


def from_own_pages(request) -> bool:
    """A one-click entry is a POST without a CSRF token, so it must come from the app's own pages: a browser always sends
    Origin on a cross-site POST, and another site's Origin is refused."""
    origin = (request.headers.get("origin") or "").rstrip("/")
    if not origin:
        return True                                         # not a browser's cross-site request (scripts, tests)
    allowed = {o.strip().rstrip("/") for o in (settings.CORS_ALLOW_ORIGINS or "").split(",") if o.strip()}
    own = f"{request.url.scheme}://{request.headers.get('host', '')}"
    return origin in allowed or origin == own


def end_leftover_sessions() -> int:
    """Demo sign-in off: end any session a demo person still holds (one may have been made with one click)."""
    if not is_law() or settings.LEGAL_DEMO_SIGNIN:
        return 0
    from . import accounts, sessions
    ended = 0
    for name in seeded():
        acct = accounts.get_by_username(name)
        if acct:
            ended += sessions.revoke_all_for_user(acct["id"], "demo_signin_off")
    return ended


# Office and this-server addresses only (Python's is_private would also take documentation and reserved ranges).
_PRIVATE = [ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8", "::1/128", "fc00::/7")]


def _from_allowed(request) -> bool:
    if (settings.LEGAL_DEMO_SIGNIN_FROM or "private").strip().lower() == "anywhere":
        return True
    try:
        ip = ipaddress.ip_address(client_address(request) or "")
    except ValueError:
        return False
    return any(ip in n for n in _PRIVATE if n.version == ip.version)


def available(request) -> bool:
    return bool(is_law() and settings.LEGAL_DEMO_SIGNIN and _from_allowed(request) and people())


def auto() -> str | None:
    """The demo person the sign-in page enters as by itself (LEGAL_DEMO_AUTO), if that is one of the demo people."""
    name = settings.LEGAL_DEMO_AUTO
    return name if name and name in people() else None
