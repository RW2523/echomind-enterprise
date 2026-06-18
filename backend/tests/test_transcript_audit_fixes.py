"""Unit tests for the Live Transcript / Silent Assistant audit fixes."""
from __future__ import annotations

import os
import re
import tempfile

os.environ.setdefault("ECHOMIND_DATA_DIR", tempfile.mkdtemp(prefix="echomind_test_"))


# ── Reconnect collision fix: paragraph ids are unique per session ────────────
def test_paragraph_ids_are_session_scoped():
    from app.transcribe.session_state import SessionState
    a = SessionState("aaaaaaaa-1111-2222-3333-444444444444")
    b = SessionState("bbbbbbbb-5555-6666-7777-888888888888")
    ids_a = [a._next_paragraph_id() for _ in range(3)]
    ids_b = [b._next_paragraph_id() for _ in range(3)]
    # Prefixed with the session id -> no cross-session collisions on reconnect.
    assert ids_a == ["aaaaaaaa-p1", "aaaaaaaa-p2", "aaaaaaaa-p3"]
    assert ids_b == ["bbbbbbbb-p1", "bbbbbbbb-p2", "bbbbbbbb-p3"]
    assert set(ids_a).isdisjoint(set(ids_b))


def test_paragraph_id_falls_back_without_session():
    from app.transcribe.session_state import SessionState
    s = SessionState("")
    assert s._next_paragraph_id() == "p1"


# ── Silent Assistant confidence coercion (LLM may return "85%", "high", null) ─
def _coerce_confidence(value):
    # Mirror analyzer.py's defensive parse so we can unit-test the logic in isolation.
    try:
        c = float(re.sub(r"[^0-9.]", "", str(value)) or 0)
    except (ValueError, TypeError):
        c = 0.0
    return max(0.0, min(100.0, c))


def test_confidence_coercion_handles_dirty_values():
    assert _coerce_confidence(80) == 80.0
    assert _coerce_confidence("85%") == 85.0
    assert _coerce_confidence("high") == 0.0
    assert _coerce_confidence(None) == 0.0
    assert _coerce_confidence("120") == 100.0      # clamped
    assert _coerce_confidence("-5") == 5.0         # non-numeric strip drops '-'
    assert _coerce_confidence("conf: 73.5") == 73.5
