#!/usr/bin/env python3
"""EchoMind Law obligations and deadlines evaluation (S3.6, DOC-05; decision D-050).

Fictional documents (golden.json tunes the rules; heldout.json never does and is run once per version),
and a document with no obligations as a control, are imported through the real endpoint and read.
No model is involved.

Reported:
- **found:** each expected obligation (clause and kind), and its timing in the document's words;
- **dates right:** an expected date computed exactly; **dates guessed:** a date where the document gives
  none, or a wrong one. This must be 0;
- **extra:** obligations no one expected, listed for a person to judge (the control must have none);
- every obligation proposed, none confirmed by the system.

    cd backend && python3 ../eval/law/obligations/run_obligations_eval.py --set golden
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
    from app.legal.obligations import extract as ex
    from app.legal.orgs import default_org_id
    from conftest import csrf_headers, make_user, sign_in

    sc = json.loads((HERE / f"{args.set}.json").read_text())
    org = default_org_id()
    alice = make_user("alice", mfa=True, firm_roles=["attorney"])
    client = repo.create_client(org, "Evaluation client", "EVAL", alice["id"])
    m = repo.create_matter(org, client["id"], "EVAL-O", "Obligations evaluation", alice["id"])
    c = TestClient(main_mod.app, base_url="https://testserver")
    sign_in(c, alice)
    base = f"/api/legal/matters/{m['id']}"
    results = []
    for d in sc["documents"]:
        mime = "text/markdown" if d["name"].endswith(".md") else "text/plain"
        r = c.post(f"{base}/documents/import", headers=csrf_headers(c), files={"file": (d["name"], d["text"].encode(), mime)},
                   data={"kind": "contract", "status": "executed"})
        assert r.status_code == 202, r.text
        doc = r.json()["document"]["id"]
        worker.run_pending()
        t0 = time.monotonic()
        r = c.post(f"{base}/documents/{doc}/obligations/extract", headers=csrf_headers(c))
        ms = round((time.monotonic() - t0) * 1000)
        assert r.status_code == 200, r.text
        obs = r.json()["obligations"]
        used, lines = set(), []
        for e in d["expect"]:
            o = next((o for o in obs if o["id"] not in used and o["clause"] == e["clause"] and o["kind"] == e["kind"]), None)
            if o:
                used.add(o["id"])
            when_ok = None if "when" not in e else bool(o and e["when"].casefold() in (o["when"] or "").casefold())
            date_ok = bool(o) and o["due_date"] == e.get("due")
            lines.append({**e, "found": o is not None, "when_ok": when_ok, "date_ok": date_ok, "got_due": o["due_date"] if o else None,
                          "got_when": o["when"] if o else None, "status": o["status"] if o else None})
        extra = [{"clause": o["clause"], "kind": o["kind"], "action": o["action"], "due": o["due_date"]} for o in obs if o["id"] not in used]
        outputs = [{"clause": o["clause"], "kind": o["kind"], "party": o["party"], "action": o["action"], "when": o["when"], "due_date": o["due_date"],
                    "due_note": o["due_note"], "source": o["source"], "quote": (o.get("evidence") or {}).get("quote")} for o in obs]   # for the lawyer packet
        results.append({"id": d["id"], "ms": ms, "control": not d["expect"], "lines": lines, "extra": extra,
                        "all_proposed": all(o["status"] == "proposed" for o in obs), "outputs": outputs})
    allx = [x for r in results for x in r["lines"]]
    dated = [x for x in allx if x.get("due")]
    # a date given where the document fixes none, or a different date from the one it fixes (an empty date is a miss, not a guess)
    guessed = sum(1 for x in allx if x["found"] and x["got_due"] and x["got_due"] != x.get("due")) + sum(1 for r in results for e in r["extra"] if e["due"])
    timed = [x for x in allx if x["when_ok"] is not None]
    summary = {"documents": len(results), "expected": len(allx), "found": f"{sum(x['found'] for x in allx)}/{len(allx)}",
               "timing_in_its_words": f"{sum(bool(x['when_ok']) for x in timed)}/{len(timed)}",
               "dates_right": f"{sum(x['date_ok'] for x in dated)}/{len(dated)}", "dates_guessed": guessed,
               "extra": sum(len(r["extra"]) for r in results), "controls_clean": f"{sum(not r['extra'] for r in results if r['control'])}/{sum(r['control'] for r in results)}",
               "all_proposed": all(r["all_proposed"] for r in results),
               "by_kind": {k: f"{sum(x['found'] for x in allx if x['kind'] == k)}/{sum(1 for x in allx if x['kind'] == k)}" for k in sorted({x["kind"] for x in allx})},
               "ms_per_document": round(sum(r["ms"] for r in results) / len(results))}
    out = LAW / "reports"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", ".", ":!eval/law/reports"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    from app.core.config import settings
    from app.legal.obligations import model as om
    # the checked model pass runs after the rules when LEGAL_OBLIGATIONS_MODEL is on (the default; S3.6)
    meta = {"set": args.set, "commit": commit + ("+changes" if dirty else ""), "time": stamp, "prompt_version": ex.VERSION,
            "model": args.model if settings.LEGAL_OBLIGATIONS_MODEL else "none (rules only)",
            "model_version": om.VERSION if settings.LEGAL_OBLIGATIONS_MODEL else None}
    (out / f"{stamp}-obligations-{args.set}.json").write_text(json.dumps({"meta": meta, "summary": summary, "documents": results}, indent=1, ensure_ascii=False))
    md = [f"# EchoMind Law obligations evaluation — {args.set}", "", f"- Commit: `{meta['commit']}`", f"- Rules: `{ex.VERSION}`", f"- Model pass: `{meta['model']}` ({meta['model_version'] or 'off'})", f"- Run: {stamp}", "",
          "| Measure | Result |", "|---|---|"] + [f"| {k.replace('_', ' ')} | {v} |" for k, v in summary.items()]
    for r in results:
        md += ["", f"## {r['id']} ({r['ms']} ms{', control' if r['control'] else ''})", ""]
        md += [f"- {x['clause']} {x['kind']}: {'found' if x['found'] else '**missed** ✗'}"
               f"{'' if x['when_ok'] is None else (', timing in its words' if x['when_ok'] else f', **timing missing** ✗ (got {x['got_when']!r})')}"
               f"{'' if not x.get('due') else (', date right' if x['date_ok'] else f', **date wrong** ✗ (got {x['got_due']})')}"
               f"{', **dated without a date in the document** ✗' if x['found'] and not x.get('due') and x['got_due'] else ''}" for x in r["lines"]] or ["- (control: nothing expected)"]
        if r["extra"]:
            md += ["", "Not expected:"] + [f"- {e['clause']} {e['kind']}: {e['action']}{' (dated ' + e['due'] + ')' if e['due'] else ''}" for e in r["extra"]]
    (out / f"{stamp}-obligations-{args.set}.md").write_text("\n".join(md) + "\n")
    print(json.dumps(summary))
    print(f"report: eval/law/reports/{stamp}-obligations-{args.set}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
