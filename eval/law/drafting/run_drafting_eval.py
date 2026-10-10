#!/usr/bin/env python3
"""EchoMind Law drafting evaluation (S3.1, DOC-01; decision D-045).

For each fictional scenario (golden.json is the tuning set; heldout.json is never used to adjust the
drafting), a throwaway matter receives the scenario's documents. The firm's standard presets are
installed and approved, and a draft is made from the named preset through the real endpoint, with the
real answer and embedding models on this host.

Reported:
- **field accuracy:** expected facts found, with a checked quote;
- **absent facts left open:** a fact the documents do not state stays [[NEEDS INPUT]] (never invented);
- **wrong values:** a field filled with something other than the expected fact. This must be 0;
- **invented figures in written passages:** amounts, dates and numbers in the draft's written text
  that no document states. This must be 0;
- **unsupported sentences** in written passages, and the time per draft.

    cd backend && python3 ../eval/law/drafting/run_drafting_eval.py --set golden

The report goes to eval/law/reports/<time>-drafting-<set>.json and .md. Nothing leaves this host.
"""
from __future__ import annotations

import argparse
import json
import re
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
    from app.legal.drafting import engine, templates
    from app.legal.jobs import worker
    from app.legal.matters import repo
    from app.legal.orgs import default_org_id
    from conftest import csrf_headers, make_user, sign_in

    sc = json.loads((HERE / f"{args.set}.json").read_text())
    org = default_org_id()
    alice = make_user("alice", mfa=True, firm_roles=["attorney"])
    mp = make_user("mp", mfa=True, firm_roles=["managing_partner"])
    templates.install_presets(org, "system")
    by_key = {}
    for t in templates.listing(org):
        v = t["versions"][0]
        templates.approve(org, t["id"], v["id"], mp["id"], v["content_sha256"])
        by_key[t["preset"]] = t["id"]
    client = repo.create_client(org, "Evaluation client", "EVAL", alice["id"])
    c = TestClient(main_mod.app, base_url="https://testserver")
    sign_in(c, alice)
    results = []
    for n, s in enumerate(sc["scenarios"], start=1):
        m = repo.create_matter(org, client["id"], f"EVAL-D{n}", s["title"], alice["id"])
        base = f"/api/legal/matters/{m['id']}"
        for d in s["documents"]:
            r = c.post(f"{base}/documents/import", headers=csrf_headers(c), files={"file": (d["name"], d["text"].encode(), "text/plain")},
                       data={"kind": "correspondence", "status": d["status"]})
            assert r.status_code == 202, r.text
        worker.run_pending()
        t0 = time.monotonic()
        r = c.post(f"{base}/drafts", headers=csrf_headers(c), json={"template_id": by_key[s["template"]], "title": s["title"]})
        seconds = round(time.monotonic() - t0, 1)
        assert r.status_code == 200, r.text
        draft = r.json()["draft"]["content"]
        fields = {f["key"]: f for f in draft["fields"]}
        found, wrong = [], []
        for key, want in s["expect"].items():
            f = fields[key]
            ok = f["status"] == "sourced" and want.casefold() in (f["value"] or "").casefold()
            found.append({"field": key, "expected": want, "value": f["value"], "status": f["status"], "ok": ok,
                          "quote": f["sources"][0]["quote"] if f["sources"] else None, "suggestion": f.get("suggestion")})
            if f["status"] == "sourced" and not ok:
                wrong.append(key)
        absent = [{"field": k, "status": fields[k]["status"], "value": fields[k]["value"], "ok": fields[k]["status"] == "needs_input"}
                  for k in s["absent"]]
        wrong += [a["field"] for a in absent if not a["ok"]]
        docs_text = " ".join(d["text"] for d in s["documents"])
        written = [b for b in draft["blocks"] if b.get("origin") == "written"]
        invented = []
        for b in written:
            outside = engine.GAP.sub("", b["text"])
            invented += engine._figures_unsupported(outside, docs_text, " ".join(str(f["value"] or "") for f in draft["fields"]))
        results.append({"id": s["id"], "template": s["template"], "seconds": seconds, "fields": found, "absent": absent, "wrong": wrong,
                        "written": [{"text": b["text"], "sources": len(b["sources"]), "unsupported": b["unsupported"]} for b in written],
                        "invented_figures": invented, "gaps": draft["gaps"]})
    expected = [f for r in results for f in r["fields"]]
    absent = [a for r in results for a in r["absent"]]
    summary = {"scenarios": len(results), "field_accuracy": f"{sum(f['ok'] for f in expected)}/{len(expected)}",
               "absent_left_open": f"{sum(a['ok'] for a in absent)}/{len(absent)}", "wrong_values": sum(len(r["wrong"]) for r in results),
               "invented_figures": sum(len(r["invented_figures"]) for r in results),
               "written_passages": sum(len(r["written"]) for r in results),
               "unsupported_sentences": sum(len(w["unsupported"]) for r in results for w in r["written"]),
               "seconds_per_draft": round(sum(r["seconds"] for r in results) / max(len(results), 1), 1)}
    out = LAW / "reports"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", ".", ":!eval/law/reports"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    meta = {"set": args.set, "commit": commit + ("+changes" if dirty else ""), "model": args.model, "embeddings": "nomic-embed-text",
            "time": stamp, "prompt_version": engine.VERSION}
    (out / f"{stamp}-drafting-{args.set}.json").write_text(json.dumps({"meta": meta, "summary": summary, "scenarios": results}, indent=1, ensure_ascii=False))
    lines = [f"# EchoMind Law drafting evaluation — {args.set}", "", f"- Commit: `{meta['commit']}`", f"- Model: `{args.model}`", f"- Run: {stamp}", "",
             "| Measure | Result |", "|---|---|"] + [f"| {k.replace('_', ' ')} | {v} |" for k, v in summary.items()]
    for r in results:
        lines += ["", f"## {r['id']} ({r['template']}, {r['seconds']} s, {r['gaps']} gaps)", ""]
        lines += [f"- {f['field']}: {'ok' if f['ok'] else '**missed**'} — {f['status']}: {f['value']!r} (expected {f['expected']!r})"
                  + (f" — suggested {f['suggestion']['value']!r}, not used: {f['suggestion']['reason']}" if f.get("suggestion") else "") for f in r["fields"]]
        lines += [f"- {a['field']} (not in the documents): {'left open' if a['ok'] else '**filled: ' + repr(a['value']) + '**'}" for a in r["absent"]]
        lines += [f"- written ({w['sources']} sources, {len(w['unsupported'])} unsupported): {w['text']}" for w in r["written"]]
        if r["invented_figures"]:
            lines.append(f"- **invented figures:** {r['invented_figures']}")
    (out / f"{stamp}-drafting-{args.set}.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary))
    print(f"report: eval/law/reports/{stamp}-drafting-{args.set}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
