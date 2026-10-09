"""The job worker (S1.6).

Jobs live in `legal_jobs`, so a restart loses nothing: a job left "running" by a crash is queued
again at start-up (up to LEGAL_JOB_MAX_ATTEMPTS). One daemon thread in the backend process claims
jobs one at a time; `wake()` lets an upload start its job without waiting for the next poll.
Tests call `run_pending()` directly.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

from ...core.config import settings
from ...core.db import get_conn
from ..policy import now_iso

logger = logging.getLogger(__name__)

_JOB_COLS = ("id", "kind", "org_id", "matter_id", "object_id", "requested_by", "attempts")
_wake = threading.Event()
_thread: Optional[threading.Thread] = None
_stop = threading.Event()


def _handlers():
    from ..documents import ingest
    return {"document.index": ingest.run}


def claim() -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "UPDATE legal_jobs SET status='running', started_at=?, attempts=attempts+1 WHERE id = "
            "(SELECT id FROM legal_jobs WHERE status='queued' ORDER BY created_at LIMIT 1) AND status='queued' "
            f"RETURNING {', '.join(_JOB_COLS)}", (now_iso(),)).fetchone()
        conn.commit()
    return dict(zip(_JOB_COLS, row)) if row else None


def _finish(job_id: str, status: str, error: Optional[str] = None) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE legal_jobs SET status=?, error=?, finished_at=? WHERE id=?", (status, error, now_iso(), job_id))
        conn.commit()


def run_one(job: dict) -> None:
    handler = _handlers().get(job["kind"])
    if handler is None:
        _finish(job["id"], "failed", "unknown_kind")
        return
    try:
        error = asyncio.run(handler(job))
        _finish(job["id"], "failed" if error else "done", error)
    except Exception as e:
        # The handler already recorded its own outcome; the job keeps only the error class.
        _finish(job["id"], "failed", type(e).__name__)


def run_pending(limit: Optional[int] = None) -> int:
    """Run queued jobs until none is left (or `limit`); returns how many ran."""
    n = 0
    while limit is None or n < limit:
        job = claim()
        if job is None:
            break
        run_one(job)
        n += 1
    return n


def recover() -> int:
    """At start-up: jobs a crash left running go back to the queue, or fail after too many attempts."""
    with get_conn() as conn:
        conn.execute("UPDATE legal_jobs SET status='failed', error='too_many_attempts', finished_at=? "
                     "WHERE status='running' AND attempts >= ?", (now_iso(), settings.LEGAL_JOB_MAX_ATTEMPTS))
        n = conn.execute("UPDATE legal_jobs SET status='queued' WHERE status='running'").rowcount
        conn.commit()
    return n


def wake() -> None:
    _wake.set()


def _loop() -> None:
    while not _stop.is_set():
        try:
            run_pending()
        except Exception as e:
            logger.error("Jobs: worker loop error (%s)", type(e).__name__)
        _wake.wait(settings.LEGAL_JOB_POLL_SEC)
        _wake.clear()


def start() -> None:
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    requeued = recover()
    if requeued:
        logger.info("Jobs: %d interrupted job(s) queued again", requeued)
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="law-jobs", daemon=True)
    _thread.start()


def stop() -> None:
    _stop.set()
    _wake.set()
