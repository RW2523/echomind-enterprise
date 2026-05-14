"""Tests for Silent Assistant live analysis and analyze-window API."""
from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("ECHOMIND_DATA_DIR", tempfile.mkdtemp(prefix="echomind_assistant_test_"))

pytest.importorskip("httpx")

from app.assistant import live_analysis as la
from app.assistant.live_analysis import (
    _apply_mode_post_process,
    _eligible_hand_raise,
    _filter_insight_confidence,
    _should_dedupe,
)


@pytest.fixture(autouse=True)
def clear_dedupe():
    la._dedupe_entries.clear()
    yield
    la._dedupe_entries.clear()


def test_filter_insight_confidence_hides_below_70():
    ins = {
        "confidence": 0.65,
        "classification": "related",
        "transcript_text": "x",
        "evidence": [{"source_name": "a"}],
    }
    assert _filter_insight_confidence(ins) is None


def test_filter_insight_confidence_keeps_72():
    ins = {
        "confidence": 0.72,
        "classification": "related",
        "transcript_text": "x",
        "evidence": [{"source_name": "a"}],
    }
    out = _filter_insight_confidence(ins)
    assert out is not None
    assert out["show_highlight"] is True
    assert out["priority"] == "medium"


def test_filter_insight_confidence_high():
    ins = {
        "confidence": 0.9,
        "classification": "supported",
        "transcript_text": "x",
        "evidence": [{"source_name": "a"}],
    }
    out = _filter_insight_confidence(ins)
    assert out is not None
    assert out["priority"] == "high"


def test_eligible_hand_raise_only_for_personal_mode():
    base = {
        "confidence": 0.8,
        "classification": "warning",
        "evidence": [{"matched_text": "x"}],
    }
    assert _eligible_hand_raise(base, "personal_assistant") is True
    assert _eligible_hand_raise(base, "silent_assistant") is False


def test_eligible_hand_raise_supported_requires_high_confidence():
    low = {"confidence": 0.76, "classification": "supported", "evidence": [{"a": 1}]}
    high = {"confidence": 0.9, "classification": "supported", "evidence": [{"a": 1}]}
    assert _eligible_hand_raise(low, "personal_assistant") is False
    assert _eligible_hand_raise(high, "personal_assistant") is True


def test_apply_mode_post_process_silent_clears_hand_raise_fields():
    ins = {
        "show_hand_raise": True,
        "suggested_response": "Please confirm with the client.",
        "classification": "warning",
    }
    out = _apply_mode_post_process({**ins}, "silent_assistant")
    assert out["show_hand_raise"] is False
    assert out["suggested_response"] is None


def test_apply_mode_post_process_personal_sets_hand_raise_from_rules():
    ins = {
        "confidence": 0.8,
        "classification": "related",
        "evidence": [{"matched_text": "policy text"}],
        "suggested_response": "  Ask about exceptions.  ",
    }
    out = _apply_mode_post_process({**ins}, "personal_assistant")
    assert out["show_hand_raise"] is True
    assert out["suggested_response"] == "Ask about exceptions."


@pytest.mark.asyncio
async def test_analyze_window_no_retrieval_returns_empty():
    async def empty_doc(q, k):
        return []

    with patch.object(la.index, "search_document_only", side_effect=empty_doc), patch.object(
        la.index, "search", new_callable=AsyncMock
    ) as fs:
        fs.return_value = []
        out = await la.analyze_window(
            "sess-a",
            "silent_assistant",
            "some spoken claim about the budget",
            "earlier context",
            {"documents": True, "transcripts": False, "books": True, "faqs": True},
        )
    assert out["insights"] == []


@pytest.mark.asyncio
async def test_analyze_window_dedupes_duplicate_insights():
    hits = [
        {
            "chunk_id": "c1",
            "score": 0.55,
            "text": "The policy requires approval for all travel over 500 miles.",
            "source": {"doc_id": "d1", "filename": "policy.pdf", "doc_id": "d1"},
        }
    ]
    # fix duplicate doc_id in source
    hits[0]["source"] = {"doc_id": "d1", "filename": "policy.pdf"}

    async def mock_doc(q, k):
        return list(hits)

    raw_llm = [
        {
            "transcript_text": "travel more than 500 miles",
            "classification": "related",
            "confidence": 0.8,
            "evidence_ids": ["E0"],
            "assistant_interpretation": "Policy mentions travel limits.",
            "suggested_action": "Verify travel approval rules.",
        },
        {
            "transcript_text": "travel more than 500 miles",
            "classification": "related",
            "confidence": 0.82,
            "evidence_ids": ["E0"],
            "assistant_interpretation": "dup",
            "suggested_action": "dup",
        },
    ]

    with patch.object(la.index, "search_document_only", side_effect=mock_doc), patch.object(
        la, "_llm_classify_insights", new_callable=AsyncMock
    ) as llm:
        llm.return_value = raw_llm
        out = await la.analyze_window(
            "sess-dedupe",
            "silent_assistant",
            "travel more than 500 miles without approval",
            "",
            {"documents": True, "transcripts": False, "books": True, "faqs": True},
        )
    assert len(out["insights"]) == 1


@pytest.mark.asyncio
async def test_analyze_window_skips_llm_when_no_evidence(monkeypatch):
    async def low_score(q, k):
        return [{"chunk_id": "c1", "score": 0.1, "text": "x", "source": {"doc_id": "d1", "filename": "a.pdf"}}]

    llm = AsyncMock()
    with patch.object(la.index, "search_document_only", side_effect=low_score), patch.object(
        la, "_llm_classify_insights", llm
    ):
        out = await la.analyze_window(
            "sess-x",
            "silent_assistant",
            "hello",
            "",
            {"documents": True, "transcripts": False, "books": True, "faqs": True},
        )
    assert out["insights"] == []
    llm.assert_not_called()
