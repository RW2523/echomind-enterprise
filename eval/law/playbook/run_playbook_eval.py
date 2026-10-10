#!/usr/bin/env python3
"""EchoMind Law playbook evaluation (S3.2, DOC-02a; decision D-046).

A fictional playbook (golden.json tunes; heldout.json is never used to adjust the check) is approved,
and fictional contracts are checked against it through the real endpoint with the real models on this
host. Each topic's expected class is known.

Reported:
- **accuracy:** lines classed as expected (standard, fallback, outside, not found);
- **deviations caught:** expected "outside" lines classed outside (or unclear, for a lawyer);
- **passed in error:** an expected "outside" or "not found" classed standard or fallback. This must be 0;
- **quotes verified**, and the time per check.

    cd backend && python3 ../eval/law/playbook/run_playbook_eval.py --set golden
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAW = HERE.parent
ROOT = LAW.parent.parent
sys.path.insert(0, str(LAW))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["golden", "heldout"], default="golden")
    ap.add_argument("--llm", default="http://localhost:8355/v1")
    ap.add_argument("--model", default="nvidia/Qwen3-30B-A3B-FP4")
    ap.add_argument("--embed", default="http://localhost:11435/api/embeddings")
    args = ap.parse_args()
    from run_law_eval import setup_env
    data = setup_env(args)
    try:
        return run(args)
    finally:
        shutil.rmtree(data, ignore_errors=True)
        (data.parent / f"{data.name}.key").unlink(missing_ok=True)


def run(args) -> int:
    import logging
    logging.disable(logging.INFO)
    from fastapi.testclient import TestClient
    from app.core.db import init_db
    init_db()
    import app.main as main_mod
    from app.legal.jobs import worker
    from app.legal.matters import repo
    from app.legal.orgs import default_org_id
    from app.legal.playbooks import check as checkmod
    from app.legal.playbooks import library
    from conftest import csrf_headers, make_user, sign_in

    sc = json.loads((HERE / f"{args.set}.json").read_text())
    org = default_org_id()
    alice = make_user("alice", mfa=True, firm_roles=["attorney"])
    mp = make_user("mp", mfa=True, firm_roles=["managing_partner"])
    pb = library.create(org, sc["playbook"]["name"], None, None, alice["id"])
    for pos in sc["playbook"]["positions"]:
        pb = library.add_position(org, pb["id"], pos, alice["id"])
    for pos in pb["positions"]:
        library.approve(org, pb["id"], pos["id"], mp["id"], pos["content_sha256"])
    client = repo.create_client(org, "Evaluation client", "EVAL", alice["id"])
    m = repo.create_matter(org, client["id"], "EVAL-P", "Playbook evaluation", alice["id"])
    c = TestClient(main_mod.app, base_url="https://testserver")
    sign_in(c, alice)
    base = f"/api/legal/matters/{m['id']}"
    results = []
    for k in sc["contracts"]:
        r = c.post(f"{base}/documents/import", headers=csrf_headers(c), files={"file": (f"{k['id']}.txt", k["text"].encode(), "text/plain")},
                   data={"kind": "contract", "status": "executed"})
        assert r.status_code == 202, r.text
        doc = r.json()["document"]["id"]
        worker.run_pending()
        t0 = time.monotonic()
        r = c.post(f"{base}/documents/{doc}/playbook-check", headers=csrf_headers(c), json={"playbook_id": pb["id"]})
        seconds = round(time.monotonic() - t0, 1)
        assert r.status_code == 200, r.text
        got = {x["topic"]: x for x in r.json()["check"]["results"]}
        lines = []
        for topic, want in k["expect"].items():
            x = got[topic]
            lines.append({"topic": topic, "expected": want, "got": x["classification"], "ok": x["classification"] == want,
                          "method": x["method"], "quote": (x.get("quote") or {}).get("text"), "reason": x.get("reason")})
        results.append({"id": k["id"], "seconds": seconds, "lines": lines})
    allx = [x for r in results for x in r["lines"]]
    outside = [x for x in allx if x["expected"] == "outside"]
    summary = {"lines": len(allx), "accuracy": f"{sum(x['ok'] for x in allx)}/{len(allx)}",
               "deviations_caught": f"{sum(x['got'] in ('outside', 'unclear') for x in outside)}/{len(outside)}",
               "passed_in_error": sum(x["expected"] in ("outside", "not_found") and x["got"] in ("standard", "fallback") for x in allx),
               "by_rule": sum(x["method"] == "rule" for x in allx),
               "quotes": sum(1 for x in allx if x["quote"]),
               "seconds_per_check": round(sum(r["seconds"] for r in results) / len(results), 1)}
    out = LAW / "reports"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", ".", ":!eval/law/reports"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    meta = {"set": args.set, "commit": commit + ("+changes" if dirty else ""), "model": args.model, "embeddings": "nomic-embed-text",
            "time": stamp, "prompt_version": checkmod.VERSION}
    (out / f"{stamp}-playbook-{args.set}.json").write_text(json.dumps({"meta": meta, "summary": summary, "contracts": results}, indent=1, ensure_ascii=False))
    md = [f"# EchoMind Law playbook evaluation — {args.set}", "", f"- Commit: `{meta['commit']}`", f"- Model: `{args.model}`", f"- Run: {stamp}", "",
          "| Measure | Result |", "|---|---|"] + [f"| {k.replace('_', ' ')} | {v} |" for k, v in summary.items()]
    for r in results:
        md += ["", f"## {r['id']} ({r['seconds']} s)", ""]
        md += [f"- {x['topic']}: expected {x['expected']}, got {x['got']} ({x['method']}){'' if x['ok'] else ' **✗**'} — {x['reason'] or ''}" for x in r["lines"]]
    (out / f"{stamp}-playbook-{args.set}.md").write_text("\n".join(md) + "\n")
    print(json.dumps(summary))
    print(f"report: eval/law/reports/{stamp}-playbook-{args.set}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
