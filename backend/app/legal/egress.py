"""Outbound-connection recorder for the network-denied test (S1.12, spec test T11).

With LAW_EGRESS_LOG=<file>, a Python audit hook appends one JSON line for every name lookup and every
socket connect or send the backend attempts — from any thread, any library, at import time too —
whether or not it succeeds. scripts/law/egress_test.sh runs the browser suite with it on, alongside a
watcher that sees the backend's open sockets from outside (catching native code), and fails on any
destination that is not this machine. Development and test only; standard library only, so it can
be installed before anything else is imported.
"""
from __future__ import annotations

import ipaddress
import json
import sys
import threading
import time

_EVENTS = {"socket.connect", "socket.sendto", "socket.sendmsg", "socket.getaddrinfo", "socket.gethostbyname",
           "socket.gethostbyname_ex", "socket.gethostbyaddr"}
_LOCAL_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback", ""}
_lock = threading.Lock()
_installed: dict = {}


def is_local(host) -> bool:
    """True for this machine: loopback addresses and names, the unspecified address, and Unix sockets."""
    if host is None:
        return True
    if isinstance(host, (bytes, bytearray)):
        host = host.decode(errors="replace")
    host = str(host).strip().lower().strip("[]")
    if host in _LOCAL_NAMES or host.startswith("/") or host.startswith("\x00"):
        return True
    try:
        ip = ipaddress.ip_address(host.split("%")[0])
    except ValueError:
        return False
    return ip.is_loopback or ip.is_unspecified


def _destination(event: str, args: tuple):
    if event in ("socket.connect", "socket.sendto", "socket.sendmsg"):
        addr = args[1] if len(args) > 1 else None
        if isinstance(addr, tuple) and addr:
            return addr[0], addr[1] if len(addr) > 1 else None
        return addr, None                                    # a Unix socket path, or nothing
    host = args[0] if args else None
    port = args[1] if event == "socket.getaddrinfo" and len(args) > 1 else None
    return host, port


def install(path: str) -> None:
    if _installed.get("path"):
        return
    _installed["path"] = path
    out = open(path, "a", buffering=1, encoding="utf-8")

    def hook(event: str, args: tuple) -> None:
        if event not in _EVENTS:
            return
        try:
            host, port = _destination(event, args)
            if isinstance(host, (bytes, bytearray)):
                host = host.decode(errors="replace")
            line = json.dumps({"ts": time.time(), "event": event, "host": host if isinstance(host, (str, type(None))) else repr(host),
                               "port": port if isinstance(port, (int, str, type(None))) else repr(port),
                               "local": is_local(host), "thread": threading.current_thread().name})
            with _lock:
                out.write(line + "\n")
        except Exception:                                    # a recorder must never break the app
            pass

    sys.addaudithook(hook)
