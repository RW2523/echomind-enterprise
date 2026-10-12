"""Forwarding the audit trail to the firm's SIEM (S1.5, SEC-04; decision D-069).

Off unless LEGAL_AUDIT_SYSLOG names a destination:

    udp://siem.firm.internal:514     one datagram per event
    tcp://siem.firm.internal:514     RFC 6587 octet counting: "<length> <message>"

Each event goes, in chain order, as one RFC 5424 syslog message (facility local4) carrying a CEF record. The record
carries what the trail already holds — identifiers, action, outcome, reason, counts — and the event's seq and hash,
so the SIEM's copy can be checked against the chain. Matter content is never in the trail, so it cannot leave
this way either.

The last event sent is kept in legal_audit_forward_state and moved on only after a send succeeds: a restart or an
unreachable SIEM skips nothing and repeats at most what was in flight. A destination seen for the first time starts
at the current end of the trail rather than receiving its whole history, unless LEGAL_AUDIT_SYSLOG_FROM_START is set.

    python -m app.legal.audit forward --once     # send what is pending now
"""
from __future__ import annotations

import itertools
import logging
import os
import socket
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlsplit

from ...core.config import settings
from ...core.db import get_conn
from ..policy import now_iso
from . import chain

logger = logging.getLogger(__name__)

BATCH = 500
CEF_SEVERITY = {"allow": 3, "deny": 6, "error": 8}            # CEF scale, 0–10
BREAK_GLASS_SEVERITY = 7                                      # at least this for break-glass actions and accesses
SYSLOG_SEVERITY = {"allow": 6, "deny": 4, "error": 3}         # info, warning, err
FACILITY = 20                                                 # local4
APP_NAME = "echomind-law"
TIMEOUT_S = 3.0
MAX_BACKOFF_S = 60.0
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_BAD_DESTINATION = "LEGAL_AUDIT_SYSLOG must be udp://host:port or tcp://host:port"

_lock = threading.Lock()
_stop = threading.Event()
_thread: Optional[threading.Thread] = None


# ── The message ──────────────────────────────────────────────────────────────────
def _header(value) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|")


def _value(value) -> str:
    return str(value).replace("\\", "\\\\").replace("=", "\\=").replace("\r", "\\r").replace("\n", "\\n")


def _epoch_ms(ts: Optional[str]) -> Optional[int]:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt - _EPOCH) // timedelta(milliseconds=1)


def cef(event: dict, version: str) -> str:
    """One audit event as a CEF record. Empty values are left out."""
    action, outcome = event.get("action") or "", event.get("outcome") or ""
    severity = CEF_SEVERITY.get(outcome, 8)
    if action.startswith("breakglass.") or event.get("reason") == "break_glass":      # the grant, and every access under one
        severity = max(severity, BREAK_GLASS_SEVERITY)
    obj = ":".join(str(v) for v in (event.get("object_type"), event.get("object_id")) if v)
    fields = [("rt", _epoch_ms(event.get("ts"))), ("act", action), ("outcome", outcome),
              ("suid", event.get("actor_user_id")), ("suser", event.get("actor_username")), ("src", event.get("ip"))]
    for key, label, value in (("cs1", "matter", event.get("matter_id")), ("cs2", "object", obj),
                              ("cs3", "reason", event.get("reason")), ("cs4", "details", event.get("details_json")),
                              ("cs5", "hash", event.get("hash")), ("cs6", "org", event.get("org_id")),
                              ("cn1", "seq", event.get("seq"))):
        if value not in (None, ""):
            fields += [(f"{key}Label", label), (key, value)]
    head = "|".join(_header(h) for h in ("EchoMind", "EchoMind Law", version, action, action, severity))
    return f"CEF:0|{head}|" + " ".join(f"{k}={_value(v)}" for k, v in fields if v not in (None, ""))


def syslog_line(event: dict, version: str) -> str:
    """RFC 5424: <PRI>1 TIMESTAMP HOSTNAME APP-NAME PROCID MSGID STRUCTURED-DATA MSG, with the CEF record as MSG."""
    pri = FACILITY * 8 + SYSLOG_SEVERITY.get(event.get("outcome"), 3)
    return f"<{pri}>1 {event.get('ts') or '-'} {socket.gethostname() or '-'} {APP_NAME} - - - {cef(event, version)}"


# ── Sending ──────────────────────────────────────────────────────────────────────
def destination(url: Optional[str] = None) -> tuple[str, str, int]:
    """(scheme, host, port) of LEGAL_AUDIT_SYSLOG. ValueError for anything else, including empty."""
    try:
        u = urlsplit((settings.LEGAL_AUDIT_SYSLOG if url is None else url).strip())
        if u.scheme in ("udp", "tcp") and u.hostname and u.port and u.path in ("", "/"):
            return u.scheme, u.hostname, u.port
    except ValueError:
        pass
    raise ValueError(_BAD_DESTINATION)


def _connect(scheme: str, host: str, port: int):
    """An open socket and a function that sends one message on it."""
    if scheme == "tcp":
        sock = socket.create_connection((host, port), timeout=TIMEOUT_S)
        return sock, lambda msg: sock.sendall(b"%d %b" % (len(msg), msg))
    family, kind, proto, _, addr = socket.getaddrinfo(host, port, type=socket.SOCK_DGRAM)[0]
    sock = socket.socket(family, kind, proto)
    sock.settimeout(TIMEOUT_S)
    return sock, lambda msg: sock.sendto(msg, addr)


def _cursor(url: str) -> int:
    """The last seq sent to this destination. A new destination starts at the current end of the trail, or before
    its first event with LEGAL_AUDIT_SYSLOG_FROM_START."""
    with get_conn() as conn:
        row = conn.execute("SELECT destination, last_seq FROM legal_audit_forward_state WHERE id = 1").fetchone()
        if row and row[0] == url:
            return row[1]
        start = 0 if settings.LEGAL_AUDIT_SYSLOG_FROM_START else \
            conn.execute("SELECT COALESCE(MAX(seq), 0) FROM audit_events").fetchone()[0]
        conn.execute("INSERT INTO legal_audit_forward_state(id, destination, last_seq, updated_at) VALUES (1, ?, ?, ?) "
                     "ON CONFLICT(id) DO UPDATE SET destination=excluded.destination, last_seq=excluded.last_seq, "
                     "updated_at=excluded.updated_at", (url, start, now_iso()))
        conn.commit()
    return start


def _advance(url: str, seq: int) -> None:
    with get_conn() as conn:                  # never backwards, e.g. when the CLI ran alongside the backend
        conn.execute("UPDATE legal_audit_forward_state SET last_seq=?, updated_at=? WHERE id = 1 AND destination=? AND last_seq < ?",
                     (seq, now_iso(), url, seq))
        conn.commit()


def forward_pending(limit: int = BATCH) -> int:
    """Send up to `limit` events the destination has not received, oldest first; returns how many were sent.
    Raises if the destination cannot be reached; whatever was sent before that stays counted."""
    url = settings.LEGAL_AUDIT_SYSLOG.strip()
    if not url:
        return 0
    scheme, host, port = destination(url)
    version = os.getenv("BUILD_VERSION", "0.0.0-dev")
    with _lock:
        events = list(itertools.islice(chain.rows(_cursor(url), batch=limit), limit))
        if not events:
            return 0
        sock, send = _connect(scheme, host, port)
        sent = 0
        try:
            for e in events:
                send(syslog_line(e, version).encode("utf-8"))
                sent += 1
        finally:
            sock.close()
            if sent:
                _advance(url, events[sent - 1]["seq"])
    return sent


# ── The background thread ────────────────────────────────────────────────────────
def _backoff(failures: int) -> float:
    return min(settings.LEGAL_AUDIT_SYSLOG_INTERVAL_SEC * 2 ** failures, MAX_BACKOFF_S)


def _loop(stop: threading.Event) -> None:
    failures = 0
    while not stop.is_set():
        sent = 0
        try:
            sent = forward_pending()
            failures = 0
        except Exception as e:
            failures += 1
            logger.warning("Audit forwarding: send failed (%s); next attempt in %.0f s", type(e).__name__, _backoff(failures))
        stop.wait(0 if sent == BATCH else _backoff(failures))


def start() -> bool:
    """Forward in a daemon thread (Law edition start-up). False when LEGAL_AUDIT_SYSLOG is empty or unusable."""
    global _stop, _thread
    if not settings.LEGAL_AUDIT_SYSLOG.strip():
        return False
    try:
        scheme = destination()[0]
    except ValueError:
        logger.error("Audit forwarding not started: " + _BAD_DESTINATION)     # not an argument: the Law log filter would mask it
        return False
    if _thread is None or not _thread.is_alive():
        _stop = threading.Event()
        _thread = threading.Thread(target=_loop, args=(_stop,), name="law-audit-forward", daemon=True)
        _thread.start()
        logger.info("Audit forwarding: on, over %s", scheme)
    return True


def stop() -> None:
    global _thread
    _stop.set()
    if _thread is not None:
        _thread.join(2 * TIMEOUT_S)
        _thread = None
