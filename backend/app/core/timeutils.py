"""
Time utilities for transcript timestamps, query time windows, and timezone handling.
Used by ingestion (ingested_at), router (time filters), and catalog.
"""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from typing import Optional


def now_utc_ts() -> int:
    """Current time as Unix timestamp (seconds) in UTC."""
    return int(datetime.now(timezone.utc).timestamp())


def iso_to_utc_ts(iso_str: Optional[str]) -> Optional[int]:
    """Parse ISO datetime string to Unix timestamp in UTC. Returns None if invalid."""
    if not iso_str or not iso_str.strip():
        return None
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return None


def ts_to_iso(ts: int) -> str:
    """Convert Unix timestamp to ISO string in UTC."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def parse_context_window_hours(context_window: Optional[str]) -> Optional[float]:
    """Map context_window string to hours: '24h' -> 24, '48h' -> 48, '1w' -> 168. 'all' -> None."""
    if not context_window or context_window == "all":
        return None
    if context_window == "24h":
        return 24.0
    if context_window == "48h":
        return 48.0
    if context_window == "1w":
        return 24.0 * 7
    return None


def cutoff_from_hours(hours: float, now: Optional[datetime] = None) -> datetime:
    """Return cutoff datetime (now - hours) in UTC."""
    ref = now or datetime.now(timezone.utc)
    return ref - timedelta(hours=hours)
