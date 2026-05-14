"""Tests for assistant_insights SQLite persistence."""
from __future__ import annotations

import os
import tempfile

import pytest

os.environ.setdefault("ECHOMIND_DATA_DIR", tempfile.mkdtemp(prefix="echomind_insights_test_"))

from app.assistant import insights_store as store
from app.core.db import get_conn, init_db


@pytest.fixture(autouse=True)
def _init():
    init_db()
    with get_conn() as conn:
        conn.execute("DELETE FROM assistant_insights")
        conn.commit()
    yield


def _sample_insight(iid: str, text: str = "claim about travel", cls: str = "related", conf: float = 0.8):
    return {
        "id": iid,
        "transcript_text": text,
        "classification": cls,
        "confidence": conf,
        "show_highlight": True,
        "show_hand_raise": False,
        "priority": "medium",
        "evidence": [{"source_name": "Policy", "source_type": "document", "chunk_id": "c1", "matched_text": "Travel rules"}],
        "assistant_interpretation": "interp",
        "suggested_action": "verify",
        "suggested_response": None,
    }


def test_should_persist_requires_highlight_confidence_evidence():
    assert store.should_persist_insight({"show_highlight": False, "confidence": 0.9, "evidence": [{}]}) is False
    assert store.should_persist_insight({"show_highlight": True, "confidence": 0.65, "evidence": [{}]}) is False
    assert store.should_persist_insight({"show_highlight": True, "confidence": 0.72, "evidence": []}) is False
    assert store.should_persist_insight(_sample_insight("a")) is True


def test_bulk_save_and_list_round_trip():
    ins = [_sample_insight("client_eph_1")]
    r = store.bulk_save_assistant_insights("sess-1", "silent_assistant", None, ins)
    assert r["inserted"] == 1
    assert r["id_map"]["client_eph_1"]

    rows = store.list_assistant_insights_by_session("sess-1")
    assert len(rows) == 1
    assert rows[0]["transcript_text"] == "claim about travel"
    assert rows[0]["classification"] == "related"


def test_bulk_save_dedupe_same_session():
    a = _sample_insight("id-a", text="same claim text")
    b = _sample_insight("id-b", text="same claim text")
    r1 = store.bulk_save_assistant_insights("sess-d", "silent_assistant", None, [a])
    r2 = store.bulk_save_assistant_insights("sess-d", "silent_assistant", None, [b])
    assert r1["inserted"] == 1
    assert r2["duplicate_merged"] == 1
    rows = store.list_assistant_insights_by_session("sess-d")
    assert len(rows) == 1


def test_update_action_status():
    store.bulk_save_assistant_insights("sess-p", "personal_assistant", None, [_sample_insight("x")])
    rows = store.list_assistant_insights_by_session("sess-p")
    iid = rows[0]["id"]
    assert store.update_assistant_insight_action_status(iid, "viewed") is True
    rows2 = store.list_assistant_insights_by_session("sess-p")
    assert rows2[0]["action_status"] == "viewed"
