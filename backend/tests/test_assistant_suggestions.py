"""Assistant Mode suggestions: heuristics, SQLite store, cooldown (no auto-speak)."""
from __future__ import annotations

import asyncio
import importlib
import os
from typing import Generator

import pytest


@pytest.fixture(autouse=True)
def _isolate_assistant_db(tmp_path, monkeypatch) -> Generator[None, None, None]:
    """Fresh DATA_DIR + DB init per test; reload assistant modules so SQLite path is correct."""
    monkeypatch.setenv("ECHOMIND_DATA_DIR", str(tmp_path))
    import app.core.config as cfg
    import app.core.db as db

    importlib.reload(cfg)
    importlib.reload(db)
    import app.assistant.suggestion_store as ss
    import app.assistant.suggestion_generator as sg

    importlib.reload(ss)
    importlib.reload(sg)
    db.init_db()
    yield


def test_detect_trigger_too_short():
    from app.assistant.suggestion_generator import detect_trigger

    assert detect_trigger("") is None
    assert detect_trigger("short text no signals here ok") is None


def test_detect_trigger_fact_check():
    from app.assistant.suggestion_generator import detect_trigger

    t = (
        "During this session we are reviewing travel reimbursement under the DoD FMR. "
        "The traveler asked about per diem overseas versus CONUS. " * 2
    )
    t += " What about paragraph 030201 — is that true according to the uploaded document?"
    tr = detect_trigger(t)
    assert tr is not None
    assert tr.category == "fact_check"


def test_suggestion_store_crud():
    from app.assistant import suggestion_store
    from app.schemas.assistant_suggestion import SuggestionStatus

    session_id = "sess_crud_1"
    row = suggestion_store.insert_suggestion(
        session_id=session_id,
        mode="ASSISTANT",
        title="Test title",
        short_text="Short body for the card.",
        speak_text="Spoken line when user approves.",
        reason="unit test",
        category="clarification",
        confidence=0.5,
        source_origin="transcript",
        evidence_status="none",
        citations=[],
    )
    assert row.session_id == session_id
    assert row.status == SuggestionStatus.pending

    listed = suggestion_store.list_suggestions(session_id, "pending")
    assert len(listed) >= 1

    approved = suggestion_store.update_status(row.id, "approved", ("pending",))
    assert approved is not None
    assert approved.status == SuggestionStatus.approved

    spoken = suggestion_store.update_status(row.id, "spoken", ("approved",))
    assert spoken is not None
    assert spoken.status == SuggestionStatus.spoken


def test_generate_cooldown():
    from app.assistant.suggestion_generator import generate_suggestions

    session_id = "sess_cooldown"
    transcript = (
        "We are walking through DoD financial management regulation excerpts for this audit. "
        "The analyst needs to confirm policy alignment before sign-off. " * 2
    )
    transcript += " Can you verify whether that matches the FMR section we uploaded?"

    out1 = asyncio.run(generate_suggestions(session_id, transcript, use_knowledge_base=False))
    if out1.suggestions:
        out2 = asyncio.run(generate_suggestions(session_id, transcript + " ", use_knowledge_base=False))
        assert out2.skipped_reason == "cooldown" or len(out2.suggestions) == 0


def test_generate_sync_helper():
    from app.assistant.suggestion_generator import generate_suggestions_sync

    os.environ["ECHOMIND_ASSISTANT_SUGGESTION_LLM"] = "0"
    session_id = "sess_sync"
    transcript = (
        "Long enough transcript body for the assistant to consider whether the prior statement "
        "about entitlements was accurate. " * 3
    )
    transcript += " Is that true?"
    out = generate_suggestions_sync(session_id, transcript, use_knowledge_base=False)
    assert isinstance(out.skipped_reason, (str, type(None)))
