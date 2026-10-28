"""Where the answer model and the search embeddings run, from their configured addresses (review P1, 2026-10-28).

The interface says where answers come from; it must not claim more than the configuration supports. "this_server": every
model address is this machine (loopback); "firm_network": private addresses or a service name inside the deployment;
"external": anything else, and then the interface says so plainly.
"""
from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

from ..core.config import settings


def _where(url: str) -> str:
    host = (urlparse(url or "").hostname or "").strip().lower()
    if not host:
        return "external"
    if host in ("localhost", "host.docker.internal") or host.endswith(".localhost"):
        return "this_server"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return "firm_network" if "." not in host else "external"     # a service name inside the deployment, else a domain
    if ip.is_loopback:
        return "this_server"
    return "firm_network" if ip.is_private and not ip.is_link_local else "external"


def location() -> str:
    places = {_where(settings.LLM_BASE_URL), _where(settings.OLLAMA_EMBED_URL)}
    for place in ("external", "firm_network", "this_server"):
        if place in places:
            return place
    return "external"
