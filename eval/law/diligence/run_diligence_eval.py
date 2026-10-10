#!/usr/bin/env python3
"""EchoMind Law due-diligence evaluation (S3.7, DOC-06; decision D-051).

Synthetic data rooms (golden.json tunes the rules; heldout.json never does and is run once per
version) are imported into a matter each and run, through the real endpoints and job, against the
firm's preset checklist. No model is involved.

Reported:
- **missing:** the items reported missing against those expected (both ways);
- **issues found:** each expected (document, kind) issue; **extra issues** for a person to judge (the
  control room must have none);
- every issue preliminary, with a quote found at its offsets.

    cd backend && python3 ../eval/law/diligence/run_diligence_eval.py --set golden
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
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
    preset = checklists.PRESETS[0]
    cl = checklists.create(org, preset["name"], preset["items"], preset["expected_law"], alice["id"])
    cl = checklists.approve(org, cl["id"], mp["id"], cl["content_sha256"])
    client = repo.create_client(org, "Evaluation client", "EVAL", alice["id"])
    c = TestClient(main_mod.app, base_url="https://testserver")
    sign_in(c, alice)
    results = []
    for n, room in enumerate(sc["rooms"]):
        m = repo.create_matter(org, client["id"], f"EVAL-DD{n}", f"Diligence evaluation {room['id']}", alice["id"])
        base = f"/api/legal/matters/{m['id']}"
        for d in room["documents"]:
            mime = "text/markdown" if d["name"].endswith(".md") else "text/plain"
            r = c.post(f"{base}/documents/import", headers=csrf_headers(c), files={"file": (d["name"], d["text"].encode(), mime)},
                       data={"kind": "contract", "status": "executed"})
            assert r.status_code == 202, r.text
        worker.run_pending()
        t0 = time.monotonic()
        r = c.post(f"{base}/diligence/runs", headers=csrf_headers(c), json={"checklist_id": cl["id"]})
        assert r.status_code == 200, r.text
        worker.run_pending()
        ms = round((time.monotonic() - t0) * 1000)
        v = c.get(f"{base}/diligence/runs/{r.json()['run']['id']}").json()
        got_missing = sorted(i["item_id"] for i in v["missing"])
        want_missing = sorted(room["missing"])
        got = Counter((i["document_title"], i["kind"]) for i in v["issues"])
        want = Counter(tuple(x) for x in room["issues"])
        found = sum(min(got[k], want[k]) for k in want)
        extra = [{"document": k[0], "kind": k[1], "count": got[k] - want.get(k, 0),
                  "titles": [i["title"] for i in v["issues"] if (i["document_title"], i["kind"]) == k]} for k in got if got[k] > want.get(k, 0)]
        missed = [{"document": k[0], "kind": k[1], "count": want[k] - got.get(k, 0)} for k in want if want[k] > got.get(k, 0)]
        with get_conn() as conn:
            quotes_ok = all(conn.execute("SELECT substr(text, ?, ?) FROM legal_version_text WHERE version_id=?",
                                         (i["evidence"]["start"] + 1, i["evidence"]["end"] - i["evidence"]["start"], i["version_id"])).fetchone()[0]
                            == i["evidence"]["quote"] for i in v["issues"])
        results.append({"id": room["id"], "ms": ms, "control": not room["issues"], "missing_got": got_missing, "missing_want": want_missing,
                        "missing_ok": got_missing == want_missing, "issues_expected": sum(want.values()), "issues_found": found,
                        "missed": missed, "extra": extra, "all_preliminary": all(i["status"] == "preliminary" for i in v["issues"]),
                        "quotes_ok": quotes_ok, "status": v["run"]["status"]})
    summary = {"rooms": len(results), "missing_lists_right": f"{sum(r['missing_ok'] for r in results)}/{len(results)}",
               "issues_found": f"{sum(r['issues_found'] for r in results)}/{sum(r['issues_expected'] for r in results)}",
               "extra_issues": sum(x["count"] for r in results for x in r["extra"]),
               "controls_clean": f"{sum(not r['extra'] for r in results if r['control'])}/{sum(r['control'] for r in results)}",
               "all_preliminary": all(r["all_preliminary"] for r in results), "quotes_at_offsets": all(r["quotes_ok"] for r in results),
               "ms_per_room": round(sum(r["ms"] for r in results) / len(results))}
    out = LAW / "reports"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", ".", ":!eval/law/reports"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    meta = {"set": args.set, "commit": commit + ("+changes" if dirty else ""), "time": stamp, "prompt_version": ddrun.VERSION, "model": "none (rules only)"}
    (out / f"{stamp}-diligence-{args.set}.json").write_text(json.dumps({"meta": meta, "summary": summary, "rooms": results}, indent=1, ensure_ascii=False))
    md = [f"# EchoMind Law due-diligence evaluation — {args.set}", "", f"- Commit: `{meta['commit']}`", f"- Rules: `{ddrun.VERSION}`", f"- Run: {stamp}", "",
          "| Measure | Result |", "|---|---|"] + [f"| {k.replace('_', ' ')} | {v} |" for k, v in summary.items()]
    for r in results:
        md += ["", f"## {r['id']} ({r['ms']} ms{', control' if r['control'] else ''})", "",
               f"- Missing: {'right' if r['missing_ok'] else '**wrong** ✗'} (got {', '.join(r['missing_got']) or 'none'}; expected {', '.join(r['missing_want']) or 'none'})",
               f"- Issues: {r['issues_found']}/{r['issues_expected']} found"]
        md += [f"- **missed** ✗ {x['document']}: {x['kind']} ×{x['count']}" for x in r["missed"]]
        md += [f"- not expected: {x['document']}: {x['kind']} ×{x['count']} ({'; '.join(x['titles'])})" for x in r["extra"]]
    (out / f"{stamp}-diligence-{args.set}.md").write_text("\n".join(md) + "\n")
    print(json.dumps(summary))
    print(f"report: eval/law/reports/{stamp}-diligence-{args.set}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
