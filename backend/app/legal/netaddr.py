"""The client's address (D-075).

Behind the Law frontend every request reaches the backend from nginx, so the socket's address is nginx's own: the audit
trail once recorded it for every event. nginx passes the client's address in X-Real-IP. It is believed only when the
request comes from an address in LEGAL_TRUSTED_PROXIES (IP addresses or networks, comma-separated; names for tests),
so a client talking to the backend directly cannot choose its own. Empty (the default) believes no one.
"""
from __future__ import annotations

import ipaddress
import logging
from typing import Optional

from ..core.config import settings

logger = logging.getLogger(__name__)
_warned = False


def _trusted(host: str) -> bool:
    for entry in (e.strip() for e in (settings.LEGAL_TRUSTED_PROXIES or "").split(",")):
        if not entry:
            continue
        if host == entry:
            return True
        try:
            if ipaddress.ip_address(host) in ipaddress.ip_network(entry, strict=False):
                return True
        except ValueError:
            continue
    return False


def client_address(request) -> Optional[str]:
    host = request.client.host if request is not None and request.client else None
    if host and _trusted(host):
        try:
            return str(ipaddress.ip_address((request.headers.get("x-real-ip") or "").strip()))
        except ValueError:
            pass
    return host


def behind_untrusted_proxy(request) -> bool:
    """A proxy is in front (it sent X-Real-IP) but is not in LEGAL_TRUSTED_PROXIES: every client then shares its address,
    so nothing may be counted per address (one client's failures would lock out everyone). Warns once."""
    global _warned
    host = request.client.host if request is not None and request.client else None
    if not request.headers.get("x-real-ip") or not host or _trusted(host):
        return False
    if not _warned:
        _warned = True
        logger.warning("Client addresses: X-Real-IP arrives from an address not in LEGAL_TRUSTED_PROXIES; per-address limits are off")
    return True

