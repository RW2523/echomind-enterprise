"""Unit tests for the Critical/High fixes (C2, H1, H4, H10).

These cover the parts testable without the full GPU/LLM stack. Live HTTP/WS behaviour
(H5/H6/H7/H8/H9/H11) is exercised by the smoke tests against the running stack.
"""
from __future__ import annotations

import os
import tempfile

import pytest

# Temp data dir so index._load() / disk writes don't touch /data.
os.environ.setdefault("ECHOMIND_DATA_DIR", tempfile.mkdtemp(prefix="echomind_test_"))


# ── C2: audio_format path-traversal allowlist ────────────────────────────────
def test_c2_validate_audio_format_allows_known():
    from app.boardroom.service import _validate_audio_format
    for fmt in ("webm", "WEBM", "ogg", "wav", "pcm16", "mp4", "m4a"):
        assert _validate_audio_format(fmt) == fmt.strip().lower()


def test_c2_validate_audio_format_rejects_traversal():
    from app.boardroom.service import _validate_audio_format
    for bad in ("wav/../../../../tmp/evil", "../../foo.py", "webm;rm -rf", "exe", "wav/x", "wav\x00.py"):
        with pytest.raises(ValueError):
            _validate_audio_format(bad)


def test_c2_validate_audio_format_empty_defaults_to_webm():
    from app.boardroom.service import _validate_audio_format
    assert _validate_audio_format("") == "webm"
    assert _validate_audio_format(None) == "webm"


def test_c2_store_chunk_rejects_bad_format_and_index(tmp_path, monkeypatch):
    from app.boardroom import service as br
    # Point the boardroom dir at a temp location.
    monkeypatch.setattr(br, "_BOARDROOM_DIR", str(tmp_path / "boardroom"))
    # Bad format must raise before any write.
    with pytest.raises(ValueError):
        br.store_chunk("sess1", b"abc", 0, "wav/../../evil")
    # Negative / out-of-range index must raise.
    with pytest.raises(ValueError):
        br.store_chunk("sess1", b"abc", -1, "webm")
    with pytest.raises(ValueError):
        br.store_chunk("sess1", b"abc", br.settings.BOARDROOM_MAX_CHUNKS + 1, "webm")


# ── H10: prompt-injection guard + fencing ────────────────────────────────────
def test_h10_fence_untrusted_wraps_content():
    from app.rag.advanced import _fence_untrusted, _UNTRUSTED_BEGIN, _UNTRUSTED_END
    fenced = _fence_untrusted("some doc text")
    assert _UNTRUSTED_BEGIN in fenced and _UNTRUSTED_END in fenced
    assert "some doc text" in fenced


def test_h10_system_prompts_contain_injection_guard():
    from app.rag.advanced import _rag_system_prompt, _strict_citation_system_prompt, _INJECTION_GUARD
    assert _INJECTION_GUARD in _rag_system_prompt(None)
    assert _INJECTION_GUARD in _strict_citation_system_prompt(None)
    # Guard tells the model not to obey instructions inside untrusted content.
    assert "never" in _INJECTION_GUARD.lower() and "instruction" in _INJECTION_GUARD.lower()


# ── H1: section-retrieval flag off must not crash (3-tuple contract) ──────────
def test_h1_section_restricted_paths_returns_triple_when_disabled(monkeypatch):
    import asyncio
    from app.rag import advanced
    monkeypatch.setattr(advanced.settings, "RAG_USE_SECTION_RETRIEVAL", False, raising=False)
    result = asyncio.run(advanced._get_section_restricted_paths("anything", None))
    assert isinstance(result, tuple) and len(result) == 3
    paths, no_global, debug = result
    assert paths == [] and no_global is False and isinstance(debug, dict)


# ── H4: clear_doc rebuilds FAISS keeping row<->meta alignment ────────────────
class _FakeEmb:
    """Deterministic fake embedder: same text -> same vector (within a process)."""
    def __init__(self, dim: int = 8):
        self.dim = dim

    async def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            h = abs(hash(t))
            v[h % self.dim] = 1.0
            v[(h // 7) % self.dim] += 0.5
            out.append(v)
        return out


def _sections(doc_id, titles):
    return [
        {
            "section_id": f"{doc_id}_{i}",
            "doc_id": doc_id,
            "section_title": t,
            "section_path": f"{doc_id} > {t}",
            "full_section_text": f"text for {doc_id} {t}",
            "canonical_section_id": None,
        }
        for i, t in enumerate(titles)
    ]


def test_h4_section_index_clear_doc_keeps_alignment(tmp_path):
    import asyncio
    from app.rag.section_index import SectionIndex

    si = SectionIndex(_FakeEmb(), str(tmp_path / "sec.index"), str(tmp_path / "sec_meta.json"))

    async def run():
        await si.add_sections(_sections("docA", ["A1", "A2"]))
        await si.add_sections(_sections("docB", ["B1", "B2"]))
        assert si._index.ntotal == 4
        # Delete docA -> index and meta must both shrink to docB's 2 rows.
        si.clear_doc("docA")
        assert si._index.ntotal == 2
        assert len(si._meta["section_ids"]) == 2
        assert all(v["doc_id"] == "docB" for v in si._meta["section_by_id"].values())
        # Alignment check: searching B2's exact text returns B2 (not a stale/wrong row).
        hits = await si.search("text for docB B2", k=1)
        assert hits and hits[0]["doc_id"] == "docB"
        assert hits[0]["section_id"] in {"docB_0", "docB_1"}

    asyncio.run(run())


def test_h4_glossary_index_clear_doc_keeps_alignment(tmp_path):
    import asyncio
    from app.rag.glossary_index import GlossaryIndex

    gi = GlossaryIndex(_FakeEmb(), str(tmp_path / "gls.index"), str(tmp_path / "gls_meta.json"))

    async def run():
        await gi.add_entries([
            {"doc_id": "docA", "section_path": "A>g", "text": "alpha term definition"},
            {"doc_id": "docB", "section_path": "B>g", "text": "bravo term definition"},
        ])
        assert gi._index.ntotal == 2
        gi.clear_doc("docA")
        assert gi._index.ntotal == 1
        assert len(gi._meta["ids"]) == 1
        hits = await gi.search("bravo term definition", k=1)
        assert hits and hits[0]["doc_id"] == "docB"

    asyncio.run(run())
