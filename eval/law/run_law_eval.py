#!/usr/bin/env python3
"""EchoMind Law research evaluation (S2.2; spec tests T05 and T06; IMPLEMENTATION_PLAN §8).

Builds a throwaway Law data folder and a fictional firm:
- alice leads the evaluation matter;
- bob leads a second matter that alice cannot see.

It then imports the fictional corpus in eval/law/corpus with each document's status (the MSA as a
signed version 1 and a draft version 2), reads it with the real embedding model, and asks every
question as alice through the real research endpoint and the real answer model. Nothing leaves
this host. The data folder is deleted at the end.

    cd backend && python3 ../eval/law/run_law_eval.py --set golden          # tuning set
    cd backend && python3 ../eval/law/run_law_eval.py --set heldout         # never used for tuning

Scores, per item and overall:
- **answer accuracy:** every expected fact group present, and not abstained;
- **abstention accuracy:** questions the matter cannot answer get the explicit abstention;
- **T06 leaks:** an answer to the forbidden matter's questions — must be 0;
- **citation precision:** the share of verified evidence quotes taken from the expected documents;
- **quote verification rate:** verified quotes over all quotes given;
- **T05 conflicts:** where signed and draft texts differ, the conflict is reported;
- **R-7 multi-source:** every expected document appears in the evidence;
- **latency:** p50 and p95, to the first checked paragraph and to the end of the answer.

The report is written to eval/law/reports/<time>-<set>.json and .md, with the commit and models.
"""
from __future__ import annotations

import argparse
import hashlib
import base64
import json
import os
import re
import secrets
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BACKEND = ROOT / "backend"


def setup_env(args) -> Path:
    data = Path(tempfile.mkdtemp(prefix="emlaw-eval-"))
    key = data.parent / f"{data.name}.key"
    fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(base64.b64encode(secrets.token_bytes(32)).decode() + "\n")
    os.environ.update({
        "ECHOMIND_DATA_DIR": str(data), "ECHOMIND_EDITION": "law", "LEGAL_MASTER_KEY_FILE": str(key),
        "AUTH_ENABLED": "1", "AUTH_SECRET": secrets.token_hex(32), "AUTH_ADMIN_PASSWORD": "",
        "CORS_ALLOW_ORIGINS": "https://testserver", "LLM_BASE_URL": args.llm, "LLM_MODEL": args.model,
        "OLLAMA_EMBED_URL": args.embed, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": "",
        "DOCGEN_IMAGE_BACKEND": "none", "ECHOMIND_EXIT_ON_FATAL_CUDA": "0",
    })
    sys.path[:0] = [str(BACKEND), str(BACKEND / "tests" / "law")]
    return data


def norm(s: str) -> str:
    s = (s or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-").replace(" ", " ")
    return re.sub(r"\s+", " ", s).lower()


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, int(round(q * (len(v) - 1))))], 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["golden", "heldout", "heldout2", "heldout3", "heldout4"], default="golden")
    ap.add_argument("--limit", type=int, default=0, help="first N items only (smoke runs)")
    ap.add_argument("--only", default="", help="comma-separated item ids")
    ap.add_argument("--llm", default=os.getenv("LAW_LLM_BASE_URL", "http://localhost:8355/v1"))
    ap.add_argument("--model", default=os.getenv("LAW_LLM_MODEL", "nvidia/Qwen3-30B-A3B-FP4"))
    ap.add_argument("--embed", default=os.getenv("LAW_EMBED_URL", "http://localhost:11435/api/embeddings"))
    ap.add_argument("--keep", action="store_true", help="keep the throwaway data folder")
    ap.add_argument("--search-only", action="store_true", help="retrieval diagnostics: top passages and scores, no answers")
    ap.add_argument("--floor", type=float, default=None, help="override LEGAL_RERANK_MIN_SCORE (diagnostics)")
    ap.add_argument("--gate", action="store_true", help="release gate: refuse a changed tree or held-out set, apply "
                                                         "eval/law/thresholds.json, exit 2 when a bar is missed")
    ap.add_argument("--allow-dirty", action="store_true", help="with --gate: run on uncommitted changes anyway (never for a release)")
    args = ap.parse_args()
    if args.gate and (problems := gate_preconditions(args)):
        print("GATE REFUSED: " + "; ".join(problems), file=sys.stderr)
        return 3
    data = setup_env(args)
    try:
        return run(args, data)
    finally:
        if not args.keep:
            shutil.rmtree(data, ignore_errors=True)
            (data.parent / f"{data.name}.key").unlink(missing_ok=True)


def run(args, data: Path) -> int:
    import logging
    from fastapi.testclient import TestClient
    logging.disable(logging.INFO)                       # keep the report readable; warnings and errors still show

    from app.core.config import settings
    from app.core.db import init_db
    if args.floor is not None:
        settings.LEGAL_RERANK_MIN_SCORE = args.floor
    init_db()
    import app.main as main_mod
    from app.legal.jobs import worker
    from app.legal.matters import repo
    from app.legal.orgs import default_org_id
    from conftest import csrf_headers, make_user, sign_in

    org = default_org_id()
    alice = make_user("alice", mfa=True, firm_roles=["attorney"])
    bob = make_user("bob", mfa=True, firm_roles=["attorney"])
    voss = repo.create_client(org, "Voss-Lindqvist AB", "VOSS", alice["id"])
    beta = repo.create_client(org, "Beta Corp", "BETA", bob["id"])
    m_eval = repo.create_matter(org, voss["id"], "EVAL-1", "Harrowgate renewal and related advice (evaluation)", alice["id"])
    m_beta = repo.create_matter(org, beta["id"], "EVAL-2", "Beta investigation (forbidden to alice)", bob["id"],
                                sensitivity="highly_restricted")
    clients = {}
    for name, user in (("alice", alice), ("bob", bob)):
        c = TestClient(main_mod.app, base_url="https://testserver")
        sign_in(c, user)
        clients[name] = c
    manifest = json.loads((HERE / "corpus" / "manifest.json").read_text())
    doc_ids: dict[str, str] = {}
    t0 = time.monotonic()
    for d in manifest["documents"]:
        who, mid = ("bob", m_beta["id"]) if d["matter"] == "forbidden" else ("alice", m_eval["id"])
        c = clients[who]
        form = {"kind": d["kind"], "status": d["status"], "title": d["title"]}
        if d.get("version_of"):
            form["version_of"] = doc_ids[d["version_of"]]
        r = c.post(f"/api/legal/matters/{mid}/documents/import", headers=csrf_headers(c),
                   files={"file": (Path(d["file"]).name, (HERE / d["file"]).read_bytes(), "text/markdown")}, data=form)
        assert r.status_code == 202, r.text
        doc_ids.setdefault(d["key"], r.json()["document"]["id"])
        worker.run_pending()                            # real embeddings; one document at a time, in order
    print(f"corpus read in {time.monotonic() - t0:.1f}s: {len(manifest['documents'])} files", flush=True)

    items = [json.loads(line) for line in (HERE / f"{args.set}.jsonl").read_text().splitlines() if line.strip()]
    if args.only:
        keep = set(args.only.split(","))
        items = [i for i in items if i["id"] in keep]
    if args.limit:
        items = items[:args.limit]
    alice_c = clients["alice"]
    if args.search_only:
        from app.legal.retrieval import search as searchmod
        from app.legal.retrieval import scope as scopemod
        import asyncio
        principal = {"sub": alice["id"], "org_id": org, "auth": "session"}
        found = 0
        for it in items:
            sc = scopemod.authorize(principal, m_eval["id"])
            res = asyncio.run(searchmod.search(sc, it["question"], k=6, versions=it["versions"]))
            ranks = [k for k, p in enumerate(res["passages"])
                     if it["expect"]["facts"] and all(any(norm(a) in norm(p["text"]) for a in g) for g in it["expect"]["facts"])]
            hit = bool(ranks) if it["expect"]["facts"] else None
            found += bool(hit)
            top = res["passages"][0]["score"] if res["passages"] else None
            good = res["passages"][ranks[0]]["score"] if ranks else None
            print(f"DIAG {it['id']:28} expect={'abstain' if it['expect']['abstain'] else 'answer'} top={top} "
                  f"fact_rank={ranks[0] if ranks else None} fact_score={good}")
            for p in res["passages"][:3]:
                print(f"    {p.get('score'):>7}  {p['citation']['document_title'][:24]:24} {norm(p['text'])[:110]}")
        print(f"facts present in the passages for {found} of {sum(1 for i in items if i['expect']['facts'])} answerable items")
        return 0
    results = []
    for it in items:
        start = time.monotonic()
        first = None
        events = []
        with alice_c.stream("POST", f"/api/legal/matters/{m_eval['id']}/research/query", headers=csrf_headers(alice_c),
                            json={"question": it["question"], "versions": it["versions"]}) as r:
            for line in r.iter_lines():
                if not line.strip():
                    continue
                ev = json.loads(line)
                if ev["type"] == "paragraph" and first is None:
                    first = time.monotonic() - start
                events.append(ev)
        total = time.monotonic() - start
        done = next((e for e in events if e["type"] == "done"), {"status": "error", "answer": ""})
        results.append(score(it, done, events, first, total, doc_ids))
        r_ = results[-1]
        print(f"{'PASS' if r_['correct'] else 'FAIL'} {it['id']:28} {done.get('status'):10} {total:5.1f}s  {norm(done.get('answer'))[:110]}",
              flush=True)
    summary = summarise(results)
    misses = check_bars(args.set, summary) if args.gate else []
    if args.gate:
        summary["gate"] = {"passed": not misses, "missed": misses, "thresholds": json.loads(THRESHOLDS.read_text())["status"]}
    report(args, summary, results)
    print(json.dumps(summary, indent=1))
    if misses:
        print("GATE FAILED: " + "; ".join(misses), file=sys.stderr, flush=True)
        return 2
    return 0


THRESHOLDS = HERE / "thresholds.json"


def _ratio(v) -> float | None:
    """"5/7" → 0.714; a number stays a number; None stays None."""
    if isinstance(v, str) and "/" in v:
        a, b = v.split("/")
        return int(a) / int(b) if int(b) else None
    return v


def check_bars(set_: str, summary: dict) -> list[str]:
    """Every bar this set misses, in words (empty when it passes). A measure that could not be computed misses its bar."""
    bars = json.loads(THRESHOLDS.read_text())["bars"].get(set_, {})
    misses = []
    for key, bar in bars.items():
        name, kind = key.rsplit("_", 1)
        value = _ratio(summary.get(name))
        if value is None:
            misses.append(f"{name} not measured (bar {kind} {bar})")
        elif (kind == "min" and value < bar) or (kind == "max" and value > bar):
            misses.append(f"{name} {round(value, 3)} ({'below' if kind == 'min' else 'above'} {bar})")
    return misses


def gate_preconditions(args) -> list[str]:
    """A release gate runs on committed code and on the frozen held-out set."""
    problems = []
    dirty = subprocess.run(["git", "status", "--porcelain", "--", ".", ":!eval/law/reports"], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    if dirty and not args.allow_dirty:
        problems.append("the tree has uncommitted changes (commit them, or --allow-dirty for a trial run that is not a release gate)")
    frozen = json.loads(THRESHOLDS.read_text())
    for name in ("heldout", "heldout2", "heldout3", "heldout4"):
        want = frozen.get(f"{name}_sha256")
        have = hashlib.sha256((HERE / f"{name}.jsonl").read_bytes()).hexdigest()
        if want and have != want:
            problems.append(f"{name}.jsonl changed (sha256 {have[:12]}…, frozen {want[:12]}…): update thresholds.json deliberately, with a note")
    return problems


def score(it: dict, done: dict, events: list, first, total: float, doc_ids: dict) -> dict:
    exp = it["expect"]
    answer = done.get("answer") or ""
    status = done.get("status")
    abstained = status == "abstained"
    blob = norm(json.dumps(events))
    leaked = [f for f in exp["forbidden"] if norm(f) in blob]
    facts = all(any(norm(a) in norm(answer) for a in group) for group in exp["facts"])
    correct = (abstained and not leaked) if exp["abstain"] else (not abstained and facts)
    evidence = done.get("evidence") or []
    verified = [e for e in evidence if e["verification"] in ("verbatim", "normalized")]
    quoted = [e for e in evidence if e["verification"] != "not_quoted"]
    want = {doc_ids[k] for k in exp["docs"] if k in doc_ids}
    precision = (sum(1 for e in verified if e["document_id"] in want) / len(verified)) if verified and want else None
    have = {e["document_id"] for e in verified}
    multi = all(doc_ids[k] in have for k in exp["docs_all"]) if exp["docs_all"] else None
    return {"id": it["id"], "tags": it.get("tags", []), "question": it["question"], "versions": it["versions"],
            "expect_abstain": exp["abstain"], "status": status, "abstention_reason": done.get("abstention_reason"),
            "correct": correct, "facts_found": facts, "leaked": leaked, "answer": answer,
            "evidence": [{k: e.get(k) for k in ("n", "quote", "verification", "document_title", "version_number", "version_status",
                                                 "clause", "page_start", "line")} for e in evidence],
            "citation_precision": precision, "verified_quotes": len(verified), "quotes": len(quoted),
            "conflict_expected": exp["conflict"], "conflict_found": bool(done.get("conflicts")), "multi_source": multi,
            "first_paragraph_s": round(first, 2) if first is not None else None, "total_s": round(total, 2)}


def summarise(rs: list[dict]) -> dict:
    answerable = [r for r in rs if not r["expect_abstain"]]
    abstain = [r for r in rs if r["expect_abstain"]]
    precisions = [r["citation_precision"] for r in rs if r["citation_precision"] is not None]
    quotes = sum(r["quotes"] for r in rs)
    conflicts = [r for r in rs if r["conflict_expected"]]
    multi = [r for r in rs if r["multi_source"] is not None]
    statuses: dict[str, int] = {}
    for r in rs:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    return {
        "items": len(rs),
        "answer_accuracy": f"{sum(r['correct'] for r in answerable)}/{len(answerable)}",
        "abstention_accuracy": f"{sum(r['correct'] for r in abstain)}/{len(abstain)}",
        "false_abstentions": sum(1 for r in answerable if r["status"] == "abstained"),
        "t06_leaks": sum(1 for r in rs if r["leaked"]),
        "citation_precision": round(statistics.mean(precisions), 3) if precisions else None,
        "quote_verification_rate": round(sum(r["verified_quotes"] for r in rs) / quotes, 3) if quotes else None,
        "t05_conflicts_found": f"{sum(r['conflict_found'] for r in conflicts)}/{len(conflicts)}",
        "r7_multi_source": f"{sum(bool(r['multi_source']) for r in multi)}/{len(multi)}",
        "statuses": statuses,
        "latency_total_p50_s": pct([r["total_s"] for r in rs], 0.5), "latency_total_p95_s": pct([r["total_s"] for r in rs], 0.95),
        "latency_first_paragraph_p50_s": pct([r["first_paragraph_s"] for r in rs if r["first_paragraph_s"]], 0.5),
        "latency_first_paragraph_p95_s": pct([r["first_paragraph_s"] for r in rs if r["first_paragraph_s"]], 0.95),
    }


def report(args, summary: dict, rs: list[dict]) -> None:
    out = HERE / "reports"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", ".", ":!eval/law/reports"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    meta = {"set": args.set, "commit": commit + ("+changes" if dirty else ""), "model": args.model, "embeddings": "nomic-embed-text",
            "items_sha256": hashlib.sha256((HERE / f"{args.set}.jsonl").read_bytes()).hexdigest(), "gate": bool(getattr(args, "gate", False)),
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2", "time": stamp, "limit": args.limit or None, "only": args.only or None}
    (out / f"{stamp}-{args.set}.json").write_text(json.dumps({"meta": meta, "summary": summary, "items": rs}, indent=1))
    lines = [f"# EchoMind Law research evaluation — {args.set}", "", f"- Commit: `{meta['commit']}`", f"- Answer model: `{args.model}`",
             "- Embeddings: nomic-embed-text; reranker: ms-marco-MiniLM-L-6-v2", f"- Run: {stamp}", "", "| Measure | Result |", "|---|---|"]
    lines += [f"| {k.replace('_', ' ')} | {v} |" for k, v in summary.items()]
    lines += ["", "| Item | Expected | Status | Correct | Precision | Quotes verified | Seconds |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['id']} | {'abstain' if r['expect_abstain'] else 'answer'} | {r['status']} | {'yes' if r['correct'] else '**no**'} | "
              f"{r['citation_precision'] if r['citation_precision'] is not None else '—'} | {r['verified_quotes']}/{r['quotes']} | {r['total_s']} |"
              for r in rs]
    (out / f"{stamp}-{args.set}.md").write_text("\n".join(lines) + "\n")
    print(f"report: eval/law/reports/{stamp}-{args.set}.md", flush=True)


if __name__ == "__main__":
    sys.exit(main())
