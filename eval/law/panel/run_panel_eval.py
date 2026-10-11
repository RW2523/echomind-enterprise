#!/usr/bin/env python3
"""EchoMind Law review-panel benchmark (S4.5, OPT-05; decision D-057).

Fictional matters, each with documents and a question, and whether a careful panel must stop for facts the documents
do not give (golden.json tunes; heldout.json is written before it is run and never used to change the panel). Through
the real endpoint and job, with the real models on this host (five model calls a panel). Reported:
- **stop right:** the panel stopped exactly when the question needs facts the documents lack;
- **quotes verified:** each perspective's quotes found in the documents;
- **uncited sentences:** sentences the perspectives wrote without a source (marked for the partner);
- **perspectives answered:** perspectives that answered rather than abstained;
- **matrix issues / disagreements:** what the moderator laid out; **matrix figures unsupported:** figures no answer gave;
- **seconds per panel.**

    cd backend && python3 ../eval/law/panel/run_panel_eval.py --set golden

The report goes to eval/law/reports/<time>-panel-<set>.json and .md. Nothing leaves this host.
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
    from app.core.config import settings
    from app.core.db import init_db
    init_db()
    settings.LEGAL_PANEL = True
    import app.main as main_mod
    from app.legal.jobs import worker
    from app.legal.matters import repo
    from app.legal.orgs import default_org_id
    from app.legal.panel import run as panel
    from conftest import csrf_headers, make_user, sign_in

    sc = json.loads((HERE / f"{args.set}.json").read_text())
    org = default_org_id()
    alice = make_user("alice", mfa=True, firm_roles=["attorney"])
    client = repo.create_client(org, "Evaluation client", "EVAL", alice["id"])
    c = TestClient(main_mod.app, base_url="https://testserver")
    sign_in(c, alice)
    results = []
    for n, s in enumerate(sc["scenarios"]):
        m = repo.create_matter(org, client["id"], f"EVAL-P{n}", s["id"], alice["id"])
        base = f"/api/legal/matters/{m['id']}"
        for d in s["documents"]:
            r = c.post(f"{base}/documents/import", headers=csrf_headers(c), files={"file": (d["name"], d["text"].encode(), "text/plain")},
                       data={"kind": "contract", "status": "executed"})
            assert r.status_code == 202, r.text
        worker.run_pending()
        t0 = time.monotonic()
        r = c.post(f"{base}/panel", headers=csrf_headers(c), json={"question": s["question"], "specialist": s["specialist"]})
        assert r.status_code == 200, r.text
        rid = r.json()["review"]["id"]
        worker.run_pending()
        seconds = round(time.monotonic() - t0, 1)
        v = c.get(f"{base}/panel/{rid}").json()
        a = v["artifact"]
        if not a:
            results.append({"id": s["id"], "seconds": seconds, "status": v["review"]["status"], "failed": True})
            continue
        ct = a["content"]
        ev = [e for p in ct["perspectives"] for e in p["evidence"]]
        results.append({"id": s["id"], "seconds": seconds, "status": v["review"]["status"], "stop_expected": s["stop"], "stopped": ct["stopped"],
                        "stop_right": ct["stopped"] == s["stop"], "missing_facts": ct["missing_facts"],
                        "quotes": len(ev), "verified": sum(e["verification"] in ("verbatim", "normalized") for e in ev),
                        "uncited": sum(len(p["unsupported"]) for p in ct["perspectives"]),
                        "answered": sum(not p["abstained"] for p in ct["perspectives"]),
                        "issues": len(ct["matrix"]), "disagreements": sum(m["disagreement"] for m in ct["matrix"]),
                        "matrix_figures_unsupported": sum(len(m["figures_unsupported"]) for m in ct["matrix"]),
                        "perspectives": [{"role": p["role"], "text": p["text"], "unsupported": p["unsupported"], "missing": p["missing_facts"]} for p in ct["perspectives"]],
                        "matrix": ct["matrix"]})
    ok = [r for r in results if not r.get("failed")]
    summary = {"panels": len(results), "failed": len(results) - len(ok), "stop_right": f"{sum(r['stop_right'] for r in ok)}/{len(ok)}",
               "quotes_verified": f"{sum(r['verified'] for r in ok)}/{sum(r['quotes'] for r in ok)}",
               "uncited_sentences": sum(r["uncited"] for r in ok), "perspectives_answered": f"{sum(r['answered'] for r in ok)}/{4 * len(ok)}",
               "matrix_issues": sum(r["issues"] for r in ok), "disagreements": sum(r["disagreements"] for r in ok),
               "matrix_figures_unsupported": sum(r["matrix_figures_unsupported"] for r in ok),
               "seconds_per_panel": round(sum(r["seconds"] for r in results) / max(len(results), 1), 1)}
    out = LAW / "reports"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", ".", ":!eval/law/reports", ":!docs/echomind-law/design/screens"],
                                cwd=ROOT, capture_output=True, text=True).stdout.strip())
    meta = {"set": args.set, "commit": commit + ("+changes" if dirty else ""), "model": args.model, "time": stamp, "prompt_version": panel.VERSION}
    (out / f"{stamp}-panel-{args.set}.json").write_text(json.dumps({"meta": meta, "summary": summary, "scenarios": results}, indent=1, ensure_ascii=False))
    md = [f"# EchoMind Law review-panel benchmark — {args.set}", "", f"- Commit: `{meta['commit']}`", f"- Model: `{args.model}`",
          f"- Prompt: `{panel.VERSION}`", f"- Run: {stamp}", "", "| Measure | Result |", "|---|---|"] + [f"| {k.replace('_', ' ')} | {v} |" for k, v in summary.items()]
    for r in results:
        md += ["", f"## {r['id']} ({r['seconds']} s, {r['status']})", ""]
        if r.get("failed"):
            md.append("- **no result**")
            continue
        md += [f"- stop: expected {r['stop_expected']}, got {r['stopped']} {'✓' if r['stop_right'] else '✗'}" + (f" (missing: {'; '.join(r['missing_facts'])})" if r["missing_facts"] else ""),
               f"- quotes verified {r['verified']}/{r['quotes']}; uncited sentences {r['uncited']}; perspectives answered {r['answered']}/4",
               f"- matrix: {r['issues']} issues, {r['disagreements']} disagreements, {r['matrix_figures_unsupported']} unsupported figures"]
    (out / f"{stamp}-panel-{args.set}.md").write_text("\n".join(md) + "\n")
    print(json.dumps(summary))
    print(f"report: eval/law/reports/{stamp}-panel-{args.set}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
