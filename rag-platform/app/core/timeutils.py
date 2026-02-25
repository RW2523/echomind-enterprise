"""Epoch and time helpers for transcript/document filtering."""
from __future__ import annotations
import time
import re
from typing import Optional, Tuple

def now_epoch() -> int:
    return int(time.time())


def parse_relative_time(query: str) -> Optional[Tuple[int, int]]:
    """
    Parse 'last N minutes/hours/days' from query. Returns (start_ts, end_ts) or None.
    end_ts = now; start_ts = now - delta.
    """
    q = (query or "").strip().lower()
    now = now_epoch()
    # last N minutes
    m = re.search(r"last\s+(\d+)\s*min(?:ute)?s?", q)
    if m:
        sec = int(m.group(1)) * 60
        return (now - sec, now)
    m = re.search(r"last\s+(\d+)\s*hours?", q)
    if m:
        sec = int(m.group(1)) * 3600
        return (now - sec, now)
    m = re.search(r"last\s+(\d+)\s*days?", q)
    if m:
        sec = int(m.group(1)) * 86400
        return (now - sec, now)
    if "today" in q or "yesterday" in q:
        # Approximate: today = last 24h, yesterday = 24–48h
        if "yesterday" in q:
            return (now - 86400 * 2, now - 86400)
        return (now - 86400, now)
    return None


def parse_between_time(query: str) -> Optional[Tuple[int, int]]:
    """
    Parse 'between A and B' (epoch or date-like). Returns (start_ts, end_ts) or None.
    Simplified: look for "between" and two numbers or leave to caller.
    """
    q = (query or "").strip().lower()
    if "between" not in q:
        return None
    # Could integrate date parsing here; for now return None and use relative only
    return None
