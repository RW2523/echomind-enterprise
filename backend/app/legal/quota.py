"""Per-person limits on requests that start model work (commercial-readiness review F4, 2026-10-28).

The answer model is one GPU shared by the whole firm (and, on a shared host, by other services): one person's script or a
stuck browser tab must not starve everyone else. Each signed-in person has a token bucket of LEGAL_MODEL_REQUESTS_PER_MIN
requests (refilled continuously, the same number as burst); a request beyond it is refused with 429 and Retry-After.
In memory, per backend process: a restart forgets it, which only ever loosens it.
"""
from __future__ import annotations

import math
import re
import threading
import time

from ..core.config import settings

# POST paths that start model work: answers, comparisons, panels, summaries, translations, extractions, drafts.
MODEL_WORK = re.compile(
    r"^/api/legal/(?:"
    r"matters/[^/]+/(?:research/query|compare(?:/scan|/[^/]+/explain)?|panel(?:/[^/]+/proceed)?"
    r"|documents/[^/]+/(?:summaries|translate|obligations/extract|playbook-check)|summaries/[^/]+/translate"
    r"|meetings/[^/]+/minutes/draft|drafts|playbook-checks/[^/]+/redline|diligence/runs)"
    r"|counsel/conversations/[^/]+/turns|intake/[^/]+/replies|time/suggest)$")

_lock = threading.Lock()
_buckets: dict[str, tuple[float, float]] = {}          # who -> (tokens, last refill, monotonic)


def take(who: str, per_minute: int | None = None, now: float | None = None) -> float:
    """Spend one request for this person: 0 when allowed, else the seconds to wait."""
    rate = settings.LEGAL_MODEL_REQUESTS_PER_MIN if per_minute is None else per_minute
    if rate <= 0:
        return 0.0
    now = time.monotonic() if now is None else now
    with _lock:
        tokens, last = _buckets.get(who, (float(rate), now))
        tokens = min(float(rate), tokens + (now - last) * rate / 60.0)
        if tokens < 1.0:
            _buckets[who] = (tokens, now)
            return (1.0 - tokens) * 60.0 / rate
        _buckets[who] = (tokens - 1.0, now)
        if len(_buckets) > 10_000:                      # forget the idle ones (full again by now anyway)
            for k in [k for k, (_, t) in _buckets.items() if now - t > 120]:
                _buckets.pop(k, None)
        return 0.0


def retry_after(seconds: float) -> str:
    return str(max(1, math.ceil(seconds)))
