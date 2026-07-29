#!/usr/bin/env python3
"""Golden-question evaluation harness for the EchoMind RAG + conversation stack.

Runs a fixed set of golden questions against the live backend and scores:
  - routing        : small talk / refusals / off-corpus questions must NOT cite sources
  - retrieval      : expected documents appear in citations (doc hit-rate + precision)
  - answer facts   : expected fact strings appear in the answer (grouped alternatives)
  - hallucination  : forbidden strings must not appear
  - latency        : wall time per question

Usage:
  python3 eval/run_eval.py                       # all sets in eval/golden/
  python3 eval/run_eval.py --set health          # one set
  python3 eval/run_eval.py --base http://localhost:3000/api
  python3 eval/run_eval.py --judge               # add LLM-as-judge 0-10 grading (slower)

Exit code: 0 when every question passes, 1 otherwise (CI-friendly).
Reports:  eval/reports/eval_<timestamp>.json

Golden item schema (JSONL, one object per line; '#' lines are comments):
  id               unique kebab-case id
  type             retrieval | smalltalk | refusal | offcorpus
  namespace        KB namespace ("" = whole KB)
  persona          persona string sent to the API
  question         the user message
  setup            optional list of prior user messages sent first (same chat)
  expect_docs      [substr, ...] every entry must match some cited filename (retrieval)
  expect_facts     [[variant, ...], ...] ALL groups required, ANY variant per group
  forbid_facts     [substr, ...] none may appear in the answer
  expect_citations "0" | ">=1"  (defaults: 0 for conversational types, >=1 for retrieval)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GOLDEN_DIR = ROOT / "golden"
REPORTS_DIR = ROOT / "reports"

DEFAULT_BASE = "http://localhost:3000/api"
CONVERSATIONAL_TYPES = {"smalltalk", "refusal", "offcorpus"}


def _post(base: str, path: str, payload: dict, timeout: int = 240) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base}{path}", body, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _ask(base: str, chat_id: str, message: str, persona: str, namespace: str) -> dict:
    return _post(base, "/chat/ask", {
        "chat_id": chat_id,
        "message": message,
        "persona": persona or "General Assistant",
        "namespace": namespace or "",
        "use_knowledge_base": True,
    })


def _ci(hay: str, needle: str) -> bool:
    return needle.lower() in (hay or "").lower()


def load_items(only_set: str | None) -> list[dict]:
    items: list[dict] = []
    for f in sorted(GOLDEN_DIR.glob("*.jsonl")):
        if only_set and f.stem != only_set:
            continue
        for ln, line in enumerate(f.read_text().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                it = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"{f.name}:{ln}: bad JSON — {e}")
            it["_set"] = f.stem
            items.append(it)
    if not items:
        sys.exit(f"no golden items found in {GOLDEN_DIR}" + (f" for set '{only_set}'" if only_set else ""))
    dup = [i for i in {x["id"] for x in items} if sum(1 for x in items if x["id"] == i) > 1]
    if dup:
        sys.exit(f"duplicate ids: {dup}")
    return items


def evaluate_item(base: str, run_id: str, it: dict) -> dict:
    chat_id = f"eval-{run_id}-{it['id']}"
    persona = it.get("persona") or "General Assistant"
    ns = it.get("namespace") or ""
    for msg in it.get("setup") or []:
        _ask(base, chat_id, msg, persona, ns)

    t0 = time.monotonic()
    try:
        resp = _ask(base, chat_id, it["question"], persona, ns)
    except Exception as e:
        return {"id": it["id"], "set": it["_set"], "type": it.get("type", "retrieval"),
                "passed": False, "error": str(e)[:200], "checks": {}, "latency_s": round(time.monotonic() - t0, 2)}
    latency = time.monotonic() - t0

    answer = resp.get("answer") or ""
    citations = resp.get("citations") or []
    cited_names = [
        (c.get("filename") or c.get("doc_title") or "") for c in citations
    ]

    qtype = it.get("type", "retrieval")
    checks: dict[str, bool] = {}
    notes: list[str] = []

    # Citations expectation
    expect_cit = it.get("expect_citations") or ("0" if qtype in CONVERSATIONAL_TYPES else ">=1")
    if expect_cit == "0":
        checks["citations"] = len(citations) == 0
        if not checks["citations"]:
            notes.append(f"expected 0 citations, got {len(citations)}: {cited_names[:3]}")
    else:
        checks["citations"] = len(citations) >= 1
        if not checks["citations"]:
            notes.append("expected >=1 citation, got 0")

    # Expected docs among citations (retrieval only)
    doc_precision = None
    if it.get("expect_docs"):
        missing = [d for d in it["expect_docs"] if not any(_ci(n, d) for n in cited_names)]
        checks["docs"] = not missing
        if missing:
            notes.append(f"expected docs not cited: {missing}; cited={cited_names[:5]}")
        if cited_names:
            matched = sum(1 for n in cited_names if any(_ci(n, d) for d in it["expect_docs"]))
            doc_precision = round(matched / len(cited_names), 2)

    # Answer facts: ALL groups, ANY variant per group
    if it.get("expect_facts"):
        failed_groups = [
            grp for grp in it["expect_facts"] if not any(_ci(answer, v) for v in grp)
        ]
        checks["facts"] = not failed_groups
        if failed_groups:
            notes.append(f"missing fact groups: {failed_groups}")

    # Forbidden strings (hallucination / dropped-topic canaries)
    if it.get("forbid_facts"):
        present = [f for f in it["forbid_facts"] if _ci(answer, f)]
        checks["forbid"] = not present
        if present:
            notes.append(f"forbidden strings present: {present}")

    return {
        "id": it["id"], "set": it["_set"], "type": qtype,
        "passed": all(checks.values()),
        "checks": checks, "notes": notes,
        "doc_precision": doc_precision,
        "latency_s": round(latency, 2),
        "n_citations": len(citations),
        "cited": cited_names[:6],
        "answer_head": answer[:220],
    }


def judge_item(base_llm: str, it: dict, result: dict) -> None:
    """Optional LLM-as-judge: grade answer 0-10 against the golden expectations."""
    rubric = (
        "You grade a RAG assistant's answer. Question: {q}\n"
        "Required facts (groups of acceptable variants; all groups must be covered): {f}\n"
        "Answer to grade: {a}\n"
        "Reply with ONLY a JSON object: {{\"score\": <0-10 integer>, \"reason\": \"<short>\"}}. "
        "10 = complete and correct; 0 = wrong or empty."
    ).format(q=it["question"], f=json.dumps(it.get("expect_facts") or []), a=result.get("answer_head", ""))
    try:
        r = _post(base_llm, "/chat/completions", {
            "model": "qwen2.5:7b-instruct-q4_K_M",
            "messages": [{"role": "user", "content": rubric}],
            "max_tokens": 80, "temperature": 0.0,
        }, timeout=120)
        txt = r["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            j = json.loads(m.group(0))
            result["judge_score"] = int(j.get("score", -1))
            result["judge_reason"] = str(j.get("reason", ""))[:120]
    except Exception as e:
        result["judge_error"] = str(e)[:120]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE, help=f"backend API base (default {DEFAULT_BASE})")
    ap.add_argument("--set", dest="only_set", default=None, help="run one golden set (file stem)")
    ap.add_argument("--judge", action="store_true", help="add LLM-as-judge grading via ollama")
    ap.add_argument("--judge-base", default="http://localhost:11434/v1", help="OpenAI-compatible base for --judge")
    args = ap.parse_args()

    items = load_items(args.only_set)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    print(f"golden-eval run {run_id}: {len(items)} questions against {args.base}\n")

    results = []
    for it in items:
        r = evaluate_item(args.base, run_id, it)
        if args.judge and r.get("checks") and it.get("type", "retrieval") == "retrieval":
            judge_item(args.judge_base, it, r)
        results.append(r)
        mark = "PASS" if r["passed"] else "FAIL"
        extra = f" judge={r['judge_score']}/10" if "judge_score" in r else ""
        print(f"[{mark}] {r['set']:>14s}/{r['id']:<34s} {r['latency_s']:6.1f}s cites={r['n_citations']}{extra}")
        for n in r.get("notes") or []:
            print(f"        - {n}")
        if r.get("error"):
            print(f"        - ERROR: {r['error']}")

    # Summary
    by_type: dict[str, list] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)
    total_pass = sum(1 for r in results if r["passed"])
    print(f"\n{'='*62}\nSUMMARY  {total_pass}/{len(results)} passed")
    for t, rs in sorted(by_type.items()):
        p = sum(1 for r in rs if r["passed"])
        lat = sorted(r["latency_s"] for r in rs)
        med = lat[len(lat) // 2]
        precs = [r["doc_precision"] for r in rs if r.get("doc_precision") is not None]
        prec = f"  doc-precision(avg)={sum(precs)/len(precs):.2f}" if precs else ""
        print(f"  {t:<10s} {p}/{len(rs)} passed   median latency {med:.1f}s{prec}")

    REPORTS_DIR.mkdir(exist_ok=True)
    report = REPORTS_DIR / f"eval_{run_id}.json"
    report.write_text(json.dumps({"run_id": run_id, "base": args.base,
                                  "passed": total_pass, "total": len(results),
                                  "results": results}, indent=2))
    print(f"\nreport: {report}")
    return 0 if total_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
