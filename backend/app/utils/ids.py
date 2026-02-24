import time
import uuid
from datetime import datetime, timezone


def now_iso() -> str:
    """Current time in UTC as ISO 8601 with seconds (e.g. 2025-02-24T12:00:00Z). Used for transcript echodate and RAG time filtering."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_iso_with_ms() -> str:
    """Current time in UTC as ISO 8601 with milliseconds (e.g. 2025-02-24T12:00:00.123Z)."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + "%03dZ" % (now.microsecond // 1000)


def normalize_echodate_to_utc_iso(value: str) -> str:
    """
    Normalize a client-provided echodate to UTC ISO with seconds (YYYY-MM-DDTHH:MM:SSZ).
    Accepts ISO-ish strings with or without Z, with or without fractional seconds.
    If the string has no timezone, it is treated as UTC.
    """
    if not (value or "").strip():
        return now_iso()
    s = value.strip()
    try:
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        elif "+" in s or (len(s) >= 6 and s[-6] in "+-" and s[-3] in "+-"):
            dt = datetime.fromisoformat(s)
        else:
            if "T" in s:
                dt = datetime.fromisoformat(s + "+00:00")
            else:
                dt = datetime.fromisoformat(s.strip() + "T00:00:00+00:00")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return now_iso()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
