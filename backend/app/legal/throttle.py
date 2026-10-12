"""A per-address sign-in throttle (D-075).

Accounts lock after repeated failures (S1.1), but one address could still try password after password across many
accounts. Failed sign-in steps (a wrong password or authenticator code) are counted per client address
(netaddr.client_address). After LEGAL_SIGNIN_IP_MAX_FAILURES within LEGAL_SIGNIN_IP_WINDOW_MIN minutes, that address
is refused with 429 and Retry-After until the oldest failure leaves the window; other addresses are not affected and a
successful sign-in never counts. The first refusal in a window is audited (auth.signin.throttled, the count only).
Counts live in this process: a restart forgets them, which errs on the side of letting people in. 0 turns it off.
Behind a proxy that is not trusted (netaddr.behind_untrusted_proxy) nothing is counted: every client would share the
proxy's address, and one person's failures would lock everyone out.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

from ..core.config import settings

SIGNIN_PATHS = frozenset({"/api/auth/login", "/api/auth/mfa/verify"})
_FAILED = frozenset({400, 401, 403, 423})
_MAX_ADDRESSES = 20000                      # oldest forgotten beyond this (an address flood cannot exhaust memory)
_lock = threading.Lock()
_failures: dict[str, deque] = {}
_reported: dict[str, float] = {}


def _now() -> float:
    return time.monotonic()


def _window() -> float:
    return settings.LEGAL_SIGNIN_IP_WINDOW_MIN * 60.0


def reset() -> None:
    with _lock:
        _failures.clear()
        _reported.clear()


def _recent(address: str, now: float) -> deque:
    q = _failures.get(address, deque())
    while q and now - q[0] > _window():
        q.popleft()
    if not q:
        _failures.pop(address, None)
    return q


def retry_after(address: Optional[str]) -> int:
    """Seconds before `address` may try to sign in again; 0 when it may now."""
    if not address or settings.LEGAL_SIGNIN_IP_MAX_FAILURES <= 0:
        return 0
    now = _now()
    with _lock:
        q = _recent(address, now)
        if len(q) < settings.LEGAL_SIGNIN_IP_MAX_FAILURES:
            return 0
        return max(1, int(q[0] + _window() - now) + 1)


def failed(address: Optional[str], status: int) -> None:
    if not address or status not in _FAILED or settings.LEGAL_SIGNIN_IP_MAX_FAILURES <= 0:
        return
    now = _now()
    with _lock:
        q = _recent(address, now)
        q.append(now)
        _failures[address] = q
        while len(_failures) > _MAX_ADDRESSES:
            _failures.pop(next(iter(_failures)))


def first_refusal(address: str) -> Optional[int]:
    """The failure count when this is the first refusal of `address` in its window (to audit once), else None."""
    now = _now()
    with _lock:
        q = _recent(address, now)
        if not q or _reported.get(address, -1.0) >= q[0]:
            return None
        _reported[address] = now
        while len(_reported) > _MAX_ADDRESSES:
            _reported.pop(next(iter(_reported)))
        return len(q)
