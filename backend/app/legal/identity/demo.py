"""One-click demo sign-in (D-077). Off unless LEGAL_DEMO_SIGNIN.

The owner asked to enter the demo without passwords or codes. Only the fictional people the demo seed made (the names in
$ECHOMIND_DATA_DIR/legal/demo_credentials.json) can be entered this way; the administrator and any account added later
keep their password and two-step code. By default only from this server or a private address (the client's real address,
D-075); LEGAL_DEMO_SIGNIN_FROM=anywhere is the owner's choice. Every entry is audited (auth.demo_signin).
"""
from __future__ import annotations

import ipaddress
import json
import os

from ...core.config import settings
from ...core.edition import is_law
from ..netaddr import client_address


def people() -> list[str]:
    try:
        with open(os.path.join(settings.DATA_DIR, "legal", "demo_credentials.json")) as f:
            return sorted(json.load(f))
    except (OSError, ValueError):
        return []


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
