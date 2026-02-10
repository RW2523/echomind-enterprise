"""Unit tests for RAG advanced: general conversation detection, time decay, context window, insufficient-context response."""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone, timedelta

import pytest

# Use a temp data dir so index._load() does not touch /data (avoids PermissionError in tests).
os.environ.setdefault("ECHOMIND_DATA_DIR", tempfile.mkdtemp(prefix="echomind_test_"))

try:
    from app.rag.advanced import (
        INSUFFICIENT_CONTEXT_MSG,
        _is_general_conversation,
        _apply_time_decay,
        _filter_hits_by_context_window,
    )
except (ModuleNotFoundError, PermissionError) as e:
    pytest.skip("RAG deps or data dir not available: " + str(e), allow_module_level=True)


# --- _is_general_conversation: greetings vs short real queries ---
def test_is_general_conversation_empty_or_greeting():
    assert _is_general_conversation("") is True
    assert _is_general_conversation("   ") is True
    assert _is_general_conversation("hi") is True
    assert _is_general_conversation("Hello") is True
    assert _is_general_conversation("thanks!") is True
    assert _is_general_conversation("goodbye") is True
    assert _is_general_conversation("how are you?") is True


def test_is_general_conversation_short_real_queries_not_general():
    assert _is_general_conversation("pricing") is False
    assert _is_general_conversation("errors") is False
    assert _is_general_conversation("setup") is False
    assert _is_general_conversation("status") is False
    assert _is_general_conversation("help") is False
    assert _is_general_conversation("api") is False
    assert _is_general_conversation("summary") is False
    assert _is_general_conversation("cost") is False
    assert _is_general_conversation("guide") is False


def test_is_general_conversation_two_word_queries():
    assert _is_general_conversation("pricing info") is False
    assert _is_general_conversation("setup guide") is False
    assert _is_general_conversation("what is") is False  # "what" is a question word -> not general
    assert _is_general_conversation("ok thanks") is True  # 2 words, not query-like, no ? -> general


# --- Time decay: true half-life math ---
def test_apply_time_decay_halflife():
    """At age_days = halflife_days, score should be multiplied by 0.5."""
    now = datetime.now(timezone.utc)
    halflife_days = 7.0
    created_at = (now - timedelta(days=7)).isoformat()
    hits = [
        {"chunk_id": "c1", "score": 1.0, "text": "x", "source": {"doc_id": "d1"}},
    ]
    doc_created = {"d1": created_at}
    out = _apply_time_decay(hits, doc_created, halflife_days)
    assert len(out) == 1
    assert abs(out[0]["score"] - 0.5) < 0.01


def test_apply_time_decay_zero_age():
    now = datetime.now(timezone.utc)
    created_at = now.isoformat()
    hits = [{"chunk_id": "c1", "score": 0.8, "text": "x", "source": {"doc_id": "d1"}}]
    doc_created = {"d1": created_at}
    out = _apply_time_decay(hits, doc_created, 7.0)
    assert abs(out[0]["score"] - 0.8) < 0.01


def test_apply_time_decay_no_halflife():
    hits = [{"chunk_id": "c1", "score": 0.8, "text": "x", "source": {}}]
    out = _apply_time_decay(hits, {}, 0)
    assert out[0]["score"] == 0.8


# --- Context window: "all" passes through; policy is batch fetch and exclude missing doc_id/created_at ---
def test_filter_hits_by_context_window_all_passes_through():
    hits = [
        {"chunk_id": "c1", "score": 0.9, "text": "a", "source": {"doc_id": "d1"}},
    ]
    out = _filter_hits_by_context_window(hits, "all")
    assert len(out) == 1
    assert out == hits


def test_filter_hits_by_context_window_empty():
    assert _filter_hits_by_context_window([], "24h") == []
    assert _filter_hits_by_context_window([], "all") == []


# --- Document/transcript intent + insufficient retrieval returns deterministic message ---
def test_insufficient_context_message_constant():
    assert "do not contain" in INSUFFICIENT_CONTEXT_MSG or "does not contain" in INSUFFICIENT_CONTEXT_MSG
    assert len(INSUFFICIENT_CONTEXT_MSG) > 10
