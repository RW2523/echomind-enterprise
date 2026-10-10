#!/usr/bin/env python3
"""EchoMind Law defined terms and cross-references evaluation (S3.5, DOC-04; decision D-049).

Fictional documents with planted drafting errors (golden.json tunes the checker; heldout.json never
does and is run once per checker version), and clean documents as controls, are imported through the
real endpoint and checked. No model is involved.

Reported:
- **planted caught:** each expected finding found (same type, and the same words or message);
- **fixes right:** for a finding that should be fixable, the first fix offered is the expected wording;
- **false findings:** findings no one expected, listed for a person to judge (controls must have none);
- time per check.

    cd backend && python3 ../eval/law/definitions/run_definitions_eval.py --set golden
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


def _hit(e: dict, f: dict) -> bool:
    if f["type"] != e["type"]:
        return False
    if "message" in e:
        return f["message"].startswith(e["message"])
    return f["text"] == e["text"] and ("conflicting" not in e or f.get("conflicting") == e["conflicting"])


def run(args) -> int:
    import logging
    logging.disable(logging.INFO)
    from fastapi.testclient import TestClient
    from app.core.db import init_db
    init_db()
    import app.main as main_mod
    from app.legal.definitions import check as checkmod
    from app.legal.jobs import worker
    from app.legal.matters import repo
    from app.legal.orgs import default_org_id
    from conftest import csrf_headers, make_user, sign_in

    sc = json.loads((HERE / f"{args.set}.json").read_text())
    org = default_org_id()
    alice = make_user("alice", mfa=True, firm_roles=["attorney"])
    client = repo.create_client(org, "Evaluation client", "EVAL", alice["id"])
    m = repo.create_matter(org, client["id"], "EVAL-D", "Definitions evaluation", alice["id"])
    c = TestClient(main_mod.app, base_url="https://testserver")
    sign_in(c, alice)
    base = f"/api/legal/matters/{m['id']}"
    results = []
    for d in sc["documents"]:
        mime = "text/markdown" if d["name"].endswith(".md") else "text/plain"
        r = c.post(f"{base}/documents/import", headers=csrf_headers(c), files={"file": (d["name"], d["text"].encode(), mime)},
                   data={"kind": "contract", "status": "draft"})
        assert r.status_code == 202, r.text
        doc = r.json()["document"]["id"]
        worker.run_pending()
        t0 = time.monotonic()
        r = c.post(f"{base}/documents/{doc}/definitions-check", headers=csrf_headers(c))
        ms = round((time.monotonic() - t0) * 1000)
        assert r.status_code == 200, r.text
        findings = r.json()["check"]["findings"]
        used, lines = set(), []
        for e in d["expect"]:
            f = next((f for f in findings if f["id"] not in used and _hit(e, f)), None)
            if f:
                used.add(f["id"])
            fix_ok = None
            if "fix" in e:
                fix_ok = bool(f and f["fixes"] and f["fixes"][0]["revised"] == e["fix"])
            lines.append({**e, "found": f is not None, "fix_ok": fix_ok, "got": f["message"] if f else None})
        extra = [{"type": f["type"], "text": f["text"], "message": f["message"]} for f in findings if f["id"] not in used]
        results.append({"id": d["id"], "ms": ms, "control": not d["expect"], "lines": lines, "extra": extra})
    allx = [x for r in results for x in r["lines"]]
    fixable = [x for x in allx if x["fix_ok"] is not None]
    summary = {"documents": len(results), "planted": len(allx), "caught": f"{sum(x['found'] for x in allx)}/{len(allx)}",
               "fixes_right": f"{sum(bool(x['fix_ok']) for x in fixable)}/{len(fixable)}",
               "false_findings": sum(len(r["extra"]) for r in results),
               "controls_clean": f"{sum(not r['extra'] for r in results if r['control'])}/{sum(r['control'] for r in results)}",
               "by_type": {t: f"{sum(x['found'] for x in allx if x['type'] == t)}/{sum(1 for x in allx if x['type'] == t)}"
                           for t in sorted({x["type"] for x in allx})},
               "ms_per_check": round(sum(r["ms"] for r in results) / len(results))}
    out = LAW / "reports"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", ".", ":!eval/law/reports"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    meta = {"set": args.set, "commit": commit + ("+changes" if dirty else ""), "time": stamp, "prompt_version": checkmod.VERSION, "model": "none (rules only)"}
    (out / f"{stamp}-definitions-{args.set}.json").write_text(json.dumps({"meta": meta, "summary": summary, "documents": results}, indent=1, ensure_ascii=False))
    md = [f"# EchoMind Law definitions and cross-references evaluation — {args.set}", "", f"- Commit: `{meta['commit']}`", f"- Checker: `{checkmod.VERSION}`",
          f"- Run: {stamp}", "", "| Measure | Result |", "|---|---|"] + [f"| {k.replace('_', ' ')} | {v} |" for k, v in summary.items()]
    for r in results:
        md += ["", f"## {r['id']} ({r['ms']} ms{', control' if r['control'] else ''})", ""]
        md += [f"- {x['type']} `{x.get('text') or x.get('message')}`: {'found' if x['found'] else '**missed** ✗'}"
               f"{'' if x['fix_ok'] is None else (', fix right' if x['fix_ok'] else ', **fix wrong** ✗')}" for x in r["lines"]] or ["- (control: nothing expected)"]
        if r["extra"]:
            md += ["", "Not expected:"] + [f"- {e['type']} `{e['text']}`: {e['message']}" for e in r["extra"]]
    (out / f"{stamp}-definitions-{args.set}.md").write_text("\n".join(md) + "\n")
    print(json.dumps(summary))
    print(f"report: eval/law/reports/{stamp}-definitions-{args.set}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
