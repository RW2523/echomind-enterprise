"""Regression tests for prompt-audit fixes: injection guards present + Lawyer disclaimer consistent."""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("ECHOMIND_DATA_DIR", tempfile.mkdtemp(prefix="echomind_test_"))


def test_analyzer_system_prompt_has_injection_guard():
    from app.transcribe import analyzer
    assert "SECURITY" in analyzer._SYSTEM_PROMPT
    assert "untrusted" in analyzer._SYSTEM_PROMPT.lower()
    assert "never follow" in analyzer._SYSTEM_PROMPT.lower()


def test_compress_system_has_guard():
    from app.rag import advanced
    assert "SECURITY" in advanced.COMPRESS_SYSTEM
    assert "untrusted" in advanced.COMPRESS_SYSTEM.lower()


def test_contextualizer_prompts_have_guard():
    from app.rag import contextualizer
    for p in (contextualizer.SECTION_SUMMARY_SYSTEM, contextualizer.CHUNK_ROLE_SYSTEM):
        assert "SECURITY" in p and "untrusted" in p.lower()


def test_lawyer_disclaimer_is_consistent_across_variants():
    from app.rag import advanced
    canon = "informational legal analysis, not formal legal advice"
    consult = "Consult a licensed attorney"
    for d in (advanced._PERSONA_RAG_PROMPTS, advanced._PERSONA_GENERAL_PROMPTS, advanced._PERSONA_STRICT_CITATION_PROMPTS):
        lawyer = d.get("Lawyer", "")
        assert canon in lawyer, f"missing canonical disclaimer in {lawyer[:60]!r}"
        assert consult in lawyer, f"missing consult-attorney line in {lawyer[:60]!r}"


def test_injection_guard_applied_to_rag_and_strict_prompts():
    from app.rag import advanced
    assert advanced._INJECTION_GUARD in advanced._rag_system_prompt(None)
    assert advanced._INJECTION_GUARD in advanced._strict_citation_system_prompt(None)
