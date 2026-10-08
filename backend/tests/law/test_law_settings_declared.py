"""S1.0 / audit R-13 — every knob the code reads from `settings` must be a declared Settings field.

pydantic-settings ignores environment variables for undeclared fields, so a `getattr(settings,
"X", default)` on an undeclared X silently pins the default and an operator's override does
nothing. The declared default must equal the in-code fallback so declaring it changes nothing.
"""
from __future__ import annotations

import ast
import pathlib
import re

APP_DIR = pathlib.Path(__file__).resolve().parents[2] / "app"
_GETATTR_RE = re.compile(r"getattr\(\s*settings\s*,\s*\"([A-Z0-9_]+)\"\s*,\s*([^)]+?)\s*\)")

# Knobs declared by S1.0; their declared defaults must match the code's fallbacks exactly.
NEWLY_DECLARED = {
    "RAG_CE_DENSE_OVERRIDE", "RAG_CE_HARD_FLOOR", "RAG_CE_RELEVANCE_FLOOR", "RAG_CITATION_CE_FLOOR",
    "RAG_CITATION_CE_WINDOW", "RAG_DOC_TIME_DECAY_HALFLIFE_DAYS", "RAG_GLOBAL_FETCH_FLOOR",
    "RAG_INTENT_MARGIN", "RAG_INTENT_SEMANTIC", "RAG_INTENT_SIM_FLOOR", "RAG_INTENT_SMALLTALK_MAX_WORDS",
    "RAG_MAX_CITATIONS", "RAG_NS_FETCH_FRACTION", "RAG_SECTION_MIN_DOC_COVERAGE", "TRANSCRIPT_ANALYSIS_CONCURRENCY",
}


def _getattr_reads():
    for py in APP_DIR.rglob("*.py"):
        for name, default in _GETATTR_RE.findall(py.read_text(encoding="utf-8")):
            yield py.relative_to(APP_DIR), name, default


def test_every_settings_knob_read_by_the_code_is_declared():
    from app.core.config import Settings
    undeclared = sorted({f"{name} ({where})" for where, name, _ in _getattr_reads() if name not in Settings.model_fields})
    assert not undeclared, "declare these in core/config.py so env overrides work:\n" + "\n".join(undeclared)


def test_newly_declared_defaults_match_the_code_fallbacks(monkeypatch):
    from app.core.config import Settings
    for name in NEWLY_DECLARED:
        monkeypatch.delenv(name, raising=False)
    declared = Settings()
    mismatches = []
    for where, name, default in _getattr_reads():
        if name in NEWLY_DECLARED:
            fallback = ast.literal_eval(default)
            if getattr(declared, name) != fallback:
                mismatches.append(f"{name}: declared {getattr(declared, name)!r} != fallback {fallback!r} ({where})")
    assert not mismatches, "\n".join(mismatches)


def test_newly_declared_knob_honours_the_environment(monkeypatch):
    from app.core.config import Settings
    monkeypatch.setenv("RAG_MAX_CITATIONS", "9")
    monkeypatch.setenv("RAG_INTENT_SEMANTIC", "0")
    s = Settings()
    assert s.RAG_MAX_CITATIONS == 9 and s.RAG_INTENT_SEMANTIC is False
