"""
BookRAG evaluation harness for regulatory books (e.g. DoD FMR).

Runs 30–50 test queries of types: citation, procedural, definition, comparison, situational.
Metrics: section-hit accuracy (explicit ref queries), inferred-rate, citation coverage.
Uses full answer pipeline when run_full=True for inferred-rate and citation coverage.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

# Run from project root: python -m app.rag.eval.book_eval
# Or from backend/: PYTHONPATH=. python -m app.rag.eval.book_eval

from app.rag.advanced import answer, retrieve_semantic_first
from app.rag.book.section_id import extract_all_codes
from app.core.config import settings

logger = logging.getLogger(__name__)

# ~30 test queries for DoD FMR eval
SAMPLE_QUERIES = [
    # Citation (explicit ref) — 8
    {"q": "What does paragraph 030201 say about purpose?", "type": "citation", "expected_codes": ["030201"]},
    {"q": "Section 0402 requirements", "type": "citation", "expected_codes": ["0402"]},
    {"q": "Section 0705 guidance", "type": "citation", "expected_codes": ["0705"]},
    {"q": "Paragraph 010101 scope", "type": "citation", "expected_codes": ["010101"]},
    {"q": "What does 080302 specify?", "type": "citation", "expected_codes": ["080302"]},
    {"q": "030201 vs 030202", "type": "comparison", "expected_codes": ["030201", "030202"]},
    {"q": "Compare section 0301 and section 0402", "type": "comparison", "expected_codes": ["0301", "0402"]},
    {"q": "Difference between 0501 and 0502", "type": "comparison", "expected_codes": ["0501", "0502"]},
    # Procedural — 6
    {"q": "How do I submit a payment request?", "type": "procedural", "expected_codes": []},
    {"q": "What are the steps for certification?", "type": "procedural", "expected_codes": []},
    {"q": "How to obtain obligational authority?", "type": "procedural", "expected_codes": []},
    {"q": "What is required for travel voucher approval?", "type": "procedural", "expected_codes": []},
    {"q": "Procedure for fund allocation", "type": "procedural", "expected_codes": []},
    {"q": "Steps to file a reimbursement claim", "type": "procedural", "expected_codes": []},
    # Definition — 6
    {"q": "What is audit readiness?", "type": "definition", "expected_codes": []},
    {"q": "Define obligational authority", "type": "definition", "expected_codes": []},
    {"q": "What is a continuing resolution?", "type": "definition", "expected_codes": []},
    {"q": "Define reprogramming", "type": "definition", "expected_codes": []},
    {"q": "What does cost ceiling mean?", "type": "definition", "expected_codes": []},
    {"q": "Explain fund control", "type": "definition", "expected_codes": []},
    # Situational — 6
    {"q": "When must funds be obligated?", "type": "situational", "expected_codes": []},
    {"q": "Who approves travel vouchers?", "type": "situational", "expected_codes": []},
    {"q": "When is a supplemental request needed?", "type": "situational", "expected_codes": []},
    {"q": "Who is responsible for budget execution?", "type": "situational", "expected_codes": []},
    {"q": "When can funds be deobligated?", "type": "situational", "expected_codes": []},
    {"q": "Who certifies financial reports?", "type": "situational", "expected_codes": []},
    # Volume/Chapter — 4
    {"q": "What does Volume 2B say about advances?", "type": "citation", "expected_codes": []},
    {"q": "Chapter 8 requirements", "type": "citation", "expected_codes": []},
    {"q": "Volume 1 Chapter 3 summary", "type": "citation", "expected_codes": []},
    {"q": "Section 0705 in Volume 2B", "type": "citation", "expected_codes": ["0705"]},
]


def _section_hit_accuracy(hits: List[Dict], expected_codes: List[str]) -> float:
    """1.0 if all expected codes appear in top-5 chunk section_ids/paths; else fraction per code."""
    if not expected_codes:
        return 1.0
    top5 = hits[:5]
    found = set()
    for h in top5:
        src = h.get("source") or {}
        sp = (src.get("section_path") or "").strip()
        sid = (src.get("section_id") or "").strip()
        for code in expected_codes:
            if code in sp or code in sid or (sp and code in sp.split(">")):
                found.add(code)
    return len(found) / len(expected_codes) if expected_codes else 1.0


def _citation_coverage(answer_text: str, citations: List) -> float:
    """1.0 if answer has inline citation (section/page) and citations list; else 0.5 if citations only; else 0.0."""
    has_cite_list = bool(citations)
    a = (answer_text or "").lower()
    has_inline = bool(
        re.search(r"\([^)]{5,100},\s*(?:p(?:age)?\.?\s*)?\d+\)", answer_text or "")
        or re.search(r"\(p(?:age)?\.?\s*\d+\)", answer_text or "")
        or re.search(r"\[Volume\s+\d+|Chapter\s+\d+|Section\s+\d", a)
    )
    if has_cite_list and has_inline:
        return 1.0
    if has_cite_list:
        return 0.5
    return 0.0


def _is_inferred_or_not_found(answer_text: str) -> bool:
    """True if answer indicates inference or not-found (AnswerGating, citation postprocess)."""
    a = (answer_text or "").lower()
    return bool(
        "inferred" in a
        or "not found" in a
        or "couldn't find" in a
        or "closest sections:" in a
        or "insufficient context" in a
    )


def _is_refusal(answer_text: str) -> bool:
    """True if answer is a refusal (AnswerGating: could not find, shows closest sections)."""
    a = (answer_text or "").lower()
    return bool(
        "could not find" in a
        or "not found in retrieved" in a
        or "closest sections:" in a
        or "i could not find this" in a
    )


async def run_eval(
    queries: Optional[List[Dict]] = None,
    run_full: bool = True,
) -> Dict:
    """Run evaluation and return metrics.

    run_full=True: runs full answer pipeline for inferred-rate and citation coverage.
    run_full=False: retrieval-only (section-hit from hits; citation/inferred approximated).
    """
    queries = queries or SAMPLE_QUERIES
    results = []
    section_hits = []
    citation_scores = []
    inferred_flags = []
    refusal_flags = []
    explicit_ref_queries = [i for i, item in enumerate(queries) if item.get("expected_codes")]

    for i, item in enumerate(queries):
        q = item.get("q", "")
        qtype = item.get("type", "conceptual")
        expected = item.get("expected_codes", [])
        try:
            if run_full:
                out = await answer(
                    q,
                    history=[],
                    use_knowledge_base=True,
                    advanced_rag=False,
                )
                answer_text = out.get("answer", "")
                citations = out.get("citations", [])
                source_type = "document" if citations or "general" not in (answer_text or "").lower() else "general"
                hits = []  # We don't have hits from answer(); use citations for coverage
            else:
                source_type, hits = await retrieve_semantic_first(
                    q, k=15, source_options={"transcript": False, "document": True, "general": True}
                )
                answer_text = ""
                citations = [h.get("source", {}) for h in hits if h.get("source")] if source_type == "document" and hits else []

            sh = 1.0
            if expected:
                if run_full:
                    # Section-hit from citations (answer used these sections)
                    found = set()
                    for c in citations:
                        sp = (c.get("section_path") or "").strip()
                        for code in expected:
                            if code in sp:
                                found.add(code)
                    sh = len(found) / len(expected) if expected else 1.0
                else:
                    sh = _section_hit_accuracy(hits, expected)
            section_hits.append(sh)

            cite_score = _citation_coverage(answer_text, citations) if run_full else (0.5 if citations and any(c.get("section_path") or c.get("page_number") for c in citations) else 0.0)
            citation_scores.append(cite_score)

            inf = 1.0 if _is_inferred_or_not_found(answer_text) else 0.0
            inferred_flags.append(inf)
            refusal = 1.0 if _is_refusal(answer_text) else 0.0
            refusal_flags.append(refusal)

            results.append({
                "query": q[:60],
                "type": qtype,
                "source": source_type,
                "hits": len(hits) if not run_full else len(citations),
                "section_hit": sh,
                "citation_score": cite_score,
                "inferred": inf,
                "refusal": refusal,
                "expected_codes": expected,
            })
        except Exception as e:
            logger.warning("Eval query failed: %s", e)
            results.append({"query": q[:60], "type": qtype, "error": str(e), "section_hit": 0.0, "citation_score": 0.0, "inferred": 0.0, "refusal": 0.0})
            section_hits.append(0.0)
            citation_scores.append(0.0)
            inferred_flags.append(0.0)
            refusal_flags.append(0.0)

    n = len(results)
    section_accuracy = sum(section_hits) / n if n else 0.0
    section_accuracy_explicit = (
        sum(section_hits[i] for i in explicit_ref_queries) / len(explicit_ref_queries)
        if explicit_ref_queries
        else 1.0
    )
    citation_pct = sum(citation_scores) / n * 100 if n else 0.0
    inferred_pct = sum(inferred_flags) / n * 100 if n else 0.0
    refusal_pct = sum(refusal_flags) / n * 100 if n else 0.0

    return {
        "n_queries": n,
        "section_hit_accuracy": round(section_accuracy, 3),
        "section_hit_accuracy_explicit_refs": round(section_accuracy_explicit, 3),
        "citation_coverage_pct": round(citation_pct, 1),
        "refusal_rate_pct": round(refusal_pct, 1),
        "inferred_rate_pct": round(inferred_pct, 1),
        "results": results,
    }


def print_report(report: Dict) -> None:
    """Print a small report to stdout."""
    print("\n" + "=" * 60)
    print("BookRAG Evaluation Report")
    print("=" * 60)
    print(f"Queries: {report['n_queries']}")
    print(f"Section-hit accuracy (all): {report['section_hit_accuracy']:.1%}")
    print(f"Section-hit accuracy (explicit refs): {report.get('section_hit_accuracy_explicit_refs', report['section_hit_accuracy']):.1%}")
    print(f"Citation coverage: {report['citation_coverage_pct']:.1f}%")
    print(f"Refusal rate: {report.get('refusal_rate_pct', 0):.1f}%")
    print(f"Inferred rate: {report['inferred_rate_pct']:.1f}%")
    print("=" * 60)
    for r in report.get("results", [])[:10]:
        q = r.get("query", "")
        sh = r.get("section_hit", 0)
        print(f"  [{sh:.0%}] {q}...")
    if len(report.get("results", [])) > 10:
        print("  ...")
    print()


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="BookRAG eval: section-hit, inferred-rate, citation coverage")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip full answer pipeline (faster)")
    args = parser.parse_args()
    report = asyncio.run(run_eval(run_full=not args.retrieval_only))
    print_report(report)
    out_path = Path(settings.DATA_DIR) / "book_eval_report.json"
    Path(settings.DATA_DIR).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {out_path}")
