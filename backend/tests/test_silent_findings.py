"""Silent Assistant findings: store, analyzer heuristics, cooldown (never TTS)."""
from __future__ import annotations

import importlib
from typing import Generator

import pytest


@pytest.fixture(autouse=True)
def _isolate_silent_db(tmp_path, monkeypatch) -> Generator[None, None, None]:
    monkeypatch.setenv("ECHOMIND_DATA_DIR", str(tmp_path))
    import app.core.config as cfg
    import app.core.db as db
    import app.silent_assistant.finding_store as fs
    import app.silent_assistant.silent_analyzer as sa

    importlib.reload(cfg)
    importlib.reload(db)
    importlib.reload(fs)
    importlib.reload(sa)
    db.init_db()
    yield


def test_finding_store_insert():
    from app.silent_assistant import finding_store

    sid = "sess_sf_1"
    row = finding_store.insert_finding(
        session_id=sid,
        transcript_segment_id="p1",
        turn_id=None,
        original_text="Test claim about section 030201.",
        span_start=0,
        span_end=40,
        category="needs_verification",
        status_label="needs_verification",
        suggested_correction="Verify externally.",
        reason="No supporting evidence was found in the local knowledge base.",
        evidence_status="none",
        confidence=0.5,
        source_origin="transcript",
        citations=[],
    )
    assert row.session_id == sid
    assert row.user_action.value == "pending"
    listed = finding_store.list_findings(sid, "pending")
    assert len(listed) >= 1


def test_analyze_too_short():
    from app.schemas.silent_finding import SilentAnalyzeIn
    from app.silent_assistant.silent_analyzer import analyze_segment_sync

    out = analyze_segment_sync("s1", SilentAnalyzeIn(text="x" * 20))
    assert out.skipped_reason == "too_short"


def test_analyze_kb_disabled_returns_empty():
    from app.schemas.silent_finding import SilentAnalyzeIn
    from app.silent_assistant.silent_analyzer import analyze_segment_sync

    body = SilentAnalyzeIn(
        text="We must never store classified data in the personal cloud bucket per policy. " * 2,
        use_knowledge_base=False,
    )
    out = analyze_segment_sync("sess_kb_off", body)
    assert out.findings == []
    assert out.skipped_reason == "kb_disabled"


def test_no_voice_import_in_silent_package():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "app" / "silent_assistant"
    for p in root.glob("*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "voice" in node.module:
                raise AssertionError(f"silent_assistant must not import voice: {p} {node.module}")
            if isinstance(node, ast.Import):
                for n in node.names:
                    if n.name and "voice" in n.name:
                        raise AssertionError(f"silent_assistant must not import voice: {p} {n.name}")
