#!/usr/bin/env python3
"""EchoMind Law governance checks evaluation (S3.8, DOC-07; decision D-052).

Fictional companies (golden.json tunes the rules; heldout.json never does and is run once per version), and a
consistent company as a control, are imported into a matter each and run through the real endpoints and job
against the firm's "Corporate governance review" preset. No model is involved.

Reported: expected findings found (kind and the start of the title), extra findings (the control must have
none), every finding preliminary, and both quotes found at their offsets.

    cd backend && python3 ../eval/law/governance/run_governance_eval.py --set golden
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
KINDS = ("share_capital", "authority", "constitution", "ip_chain")


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
    from app.core.db import get_conn, init_db
    init_db()
    import app.main as main_mod
    from app.legal.diligence import checklists, run as ddrun
    from app.legal.jobs import worker
    from app.legal.matters import repo
    from app.legal.orgs import default_org_id
    from conftest import csrf_headers, make_user, sign_in

    sc = json.loads((HERE / f"{args.set}.json").read_text())
    org = default_org_id()
    alice = make_user("alice", mfa=True, firm_roles=["attorney"])
    mp = make_user("mp", mfa=True, firm_roles=["managing_partner"])
    preset = next(p for p in checklists.PRESETS if p.get("governance"))
    cl = checklists.create(org, preset["name"], preset["items"], preset["expected_law"], alice["id"], governance=preset["governance"])
    cl = checklists.approve(org, cl["id"], mp["id"], cl["content_sha256"])
    client = repo.create_client(org, "Evaluation client", "EVAL", alice["id"])
    c = TestClient(main_mod.app, base_url="https://testserver")
    sign_in(c, alice)
    results = []
    for n, co in enumerate(sc["companies"]):
        m = repo.create_matter(org, client["id"], f"EVAL-G{n}", f"Governance evaluation {co['id']}", alice["id"])
        base = f"/api/legal/matters/{m['id']}"
        for d in co["documents"]:
            r = c.post(f"{base}/documents/import", headers=csrf_headers(c), files={"file": (d["name"], d["text"].encode(), "text/plain")},
                       data={"kind": "contract", "status": "executed"})
            assert r.status_code == 202, r.text
        worker.run_pending()
        t0 = time.monotonic()
        r = c.post(f"{base}/diligence/runs", headers=csrf_headers(c), json={"checklist_id": cl["id"]})
        worker.run_pending()
        ms = round((time.monotonic() - t0) * 1000)
        v = c.get(f"{base}/diligence/runs/{r.json()['run']['id']}").json()
        got = [i for i in v["issues"] if i["kind"] in KINDS]
        used, lines = set(), []
        for kind, title in co["expect"]:
            i = next((i for i in got if i["id"] not in used and i["kind"] == kind and i["title"].startswith(title)), None)
            if i:
                used.add(i["id"])
            lines.append({"kind": kind, "title": title, "found": i is not None})
        extra = [{"kind": i["kind"], "title": i["title"], "document": i["document_title"]} for i in got if i["id"] not in used]
        with get_conn() as conn:
            def at(vid, s, e, q):
                return conn.execute("SELECT substr(text, ?, ?) FROM legal_version_text WHERE version_id=?", (s + 1, e - s, vid)).fetchone()[0] == q
            quotes_ok = all(at(i["version_id"], i["evidence"]["start"], i["evidence"]["end"], i["evidence"]["quote"])
                            and all(at(x["version_id"], x["start"], x["end"], x["quote"]) for x in i.get("related", [])) for i in got)
        outputs = [{"kind": i["kind"], "title": i["title"], "detail": i["detail"], "document": i["document_title"], "quote": i["evidence"]["quote"],
                    "related": [{"document": x["document_title"], "quote": x["quote"]} for x in i.get("related", [])]} for i in got]   # for the lawyer packet
        results.append({"id": co["id"], "ms": ms, "control": not co["expect"], "lines": lines, "extra": extra, "outputs": outputs,
                        "all_preliminary": all(i["status"] == "preliminary" for i in got), "quotes_ok": quotes_ok,
                        "two_sources": sum(1 for i in got if i.get("related"))})
    allx = [x for r in results for x in r["lines"]]
    summary = {"companies": len(results), "found": f"{sum(x['found'] for x in allx)}/{len(allx)}", "extra": sum(len(r["extra"]) for r in results),
               "controls_clean": f"{sum(not r['extra'] for r in results if r['control'])}/{sum(r['control'] for r in results)}",
               "by_kind": {k: f"{sum(x['found'] for x in allx if x['kind'] == k)}/{sum(1 for x in allx if x['kind'] == k)}" for k in KINDS if any(x["kind"] == k for x in allx)},
               "all_preliminary": all(r["all_preliminary"] for r in results), "quotes_at_offsets": all(r["quotes_ok"] for r in results),
               "ms_per_company": round(sum(r["ms"] for r in results) / len(results))}
    out = LAW / "reports"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", ".", ":!eval/law/reports"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    meta = {"set": args.set, "commit": commit + ("+changes" if dirty else ""), "time": stamp, "prompt_version": ddrun.VERSION, "model": "none (rules only)"}
    (out / f"{stamp}-governance-{args.set}.json").write_text(json.dumps({"meta": meta, "summary": summary, "companies": results}, indent=1, ensure_ascii=False))
    md = [f"# EchoMind Law governance checks evaluation — {args.set}", "", f"- Commit: `{meta['commit']}`", f"- Rules: `{ddrun.VERSION}`", f"- Run: {stamp}", "",
          "| Measure | Result |", "|---|---|"] + [f"| {k.replace('_', ' ')} | {v} |" for k, v in summary.items()]
    for r in results:
        md += ["", f"## {r['id']} ({r['ms']} ms{', control' if r['control'] else ''})", ""]
        md += [f"- {x['kind']}: `{x['title']}` {'found' if x['found'] else '**missed** ✗'}" for x in r["lines"]] or ["- (control: nothing expected)"]
        md += [f"- not expected: {e['kind']}: {e['title']} ({e['document']})" for e in r["extra"]]
    (out / f"{stamp}-governance-{args.set}.md").write_text("\n".join(md) + "\n")
    print(json.dumps(summary))
    print(f"report: eval/law/reports/{stamp}-governance-{args.set}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
