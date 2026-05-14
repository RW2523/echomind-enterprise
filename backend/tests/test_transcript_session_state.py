"""Tests for live transcript session state (punctuation + streaming hypothesis sync)."""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("ECHOMIND_DATA_DIR", tempfile.mkdtemp(prefix="echomind_transcript_test_"))

from app.transcribe.session_state import SessionState


def test_append_piece_skips_redundant_leading_period():
    s = SessionState("sess")
    s.append_piece("Hello.", 100)
    s.append_piece(".", 200)
    assert s.get_display_text() == "Hello."


def test_collapse_duplicate_sentence_punctuation():
    s = SessionState("sess")
    s.append_piece("Hello..", 100)
    assert s.get_display_text() == "Hello."


def test_sync_hypothesis_revision_avoids_duplicate_period():
    s = SessionState("sess")
    s.sync_hypothesis("Hello world.", 100)
    s.sync_hypothesis("Hello world. Today", 200)
    assert s.get_display_text() == "Hello world. Today"


def test_sync_hypothesis_rewrites_unstable_tail():
    s = SessionState("sess")
    s.sync_hypothesis("Hello. World", 100)
    s.sync_hypothesis("Hello world", 200)
    assert s.get_display_text() == "Hello world"


def test_apply_stream_asr_disjoint_window_keeps_prior_text():
    s = SessionState("sess")
    s.apply_stream_asr("Hello world", 100)
    s.apply_stream_asr("Hello world today", 200)
    assert s.get_display_text() == "Hello world today"


def test_apply_stream_asr_disjoint_without_overlap_appends():
    s = SessionState("sess")
    s.apply_stream_asr("Hello world", 100)
    s.apply_stream_asr("today", 200)
    assert s.get_display_text() == "Hello world today"


def test_merge_streaming_hypothesis_appends_disjoint_windows():
    from app.transcribe.stt_streaming import _merge_streaming_hypothesis

    merged = _merge_streaming_hypothesis("Hello world", "Hello", "world today")
    assert merged == "Hello world today"


def test_apply_stream_asr_finalize_reconciles_tail():
    s = SessionState("sess")
    s.apply_stream_asr("Hello", 100)
    s.apply_stream_asr("Hello world", 200, finalize=True)
    assert s.get_display_text() == "Hello world"
