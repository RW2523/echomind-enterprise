"""Unit tests for the Medium/Low + prompt audit fixes (pure-function parts)."""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("ECHOMIND_DATA_DIR", tempfile.mkdtemp(prefix="echomind_test_"))


# ── M3: rerank score normalization (CE logit -> sigmoid; cosine passthrough) ──
def test_m3_normalize_rerank_score():
    from app.rag.evidence_gate import _normalize_rerank_score
    assert _normalize_rerank_score(0.0) == 0.0          # in-range cosine passthrough
    assert _normalize_rerank_score(0.7) == 0.7
    assert 0.49 < _normalize_rerank_score(0.0001) < 0.51 or _normalize_rerank_score(0.0001) == 0.0001
    # Large positive logit -> ~1; large negative -> ~0
    assert _normalize_rerank_score(11.0) > 0.99
    assert _normalize_rerank_score(-11.0) < 0.01
    assert _normalize_rerank_score("nan-ish") == 0.0    # bad input -> 0


# ── M6: section-id extraction excludes calendar years ─────────────────────────
def test_m6_extract_codes_excludes_years():
    from app.rag.book.section_id import extract_all_codes
    codes = extract_all_codes("In fiscal year 2024 the budget was 150000 per 030201 and section 0301.")
    assert "2024" not in codes              # year excluded
    assert any(c.startswith("0301") or c.startswith("030201") for c in codes)  # real codes kept


# ── M7: TOC parser accepts non-zero-prefixed section codes ────────────────────
def test_m7_toc_section_regex_accepts_nonzero():
    from app.rag.book.toc_parser import _SECTION_RE
    assert _SECTION_RE.match("7001 Funding Authority")   # was dropped before
    assert _SECTION_RE.match("0301 Purpose")             # still matches zero-prefixed
    assert _SECTION_RE.match("5050 Reimbursements")


# ── L20: tagging keeps unigrams that are substrings (not words) of a bigram ───
def test_l20_tagging_keeps_substring_unigrams():
    from app.tagging import get_tags
    # "category" should not be dropped just because the bigram "cat report" exists.
    text = ("category category category budget budget report report cat cat report "
            "category report category report")
    tags = get_tags(text, max_tags=10)
    joined = " ".join(tags)
    assert "category" in joined  # substring-of-bigram unigram retained
