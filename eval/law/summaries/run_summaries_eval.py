#!/usr/bin/env python3
"""EchoMind Law summaries evaluation (S4.2, OPT-02; decision D-055).

Fictional documents (golden.json is the tuning set; heldout.json is written before it is run and never used to
change the summaries), each with the key facts a lawyer's summary must carry and where each is. Through the real
endpoint and job, with the real model on this host. Reported:
- **facts covered:** expected facts found in a point (its words or its quote);
- **citations right:** for a covered fact with an expected place, the point's computed citation (clause, page and
  line, or timecode) is that place;
- **quotes verified:** points whose quote was found in the document;
- **figures unsupported:** points stating a figure their quote does not contain (for a lawyer to check; should be 0);
- points per document, seconds per document.

    cd backend && python3 ../eval/law/summaries/run_summaries_eval.py --set golden

The report goes to eval/law/reports/<time>-summaries-<set>.json and .md. Nothing leaves this host.
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


def _covers(p: dict, anchor: str) -> bool:
    a = anchor.casefold()
    return a in (p.get("quote") or "").casefold() or a in (p.get("text") or "").casefold()


def run(args) -> int:
    import logging
    logging.disable(logging.INFO)
    from fastapi.testclient import TestClient
    from app.core.config import settings
    from app.core.db import init_db
    init_db()
    settings.LEGAL_SUMMARIES = True
    import app.main as main_mod
    from app.legal.jobs import worker
    from app.legal.matters import repo
    from app.legal.orgs import default_org_id
    from app.legal.summaries import summarize
    from conftest import csrf_headers, make_user, sign_in

    sc = json.loads((HERE / f"{args.set}.json").read_text())
    org = default_org_id()
    alice = make_user("alice", mfa=True, firm_roles=["attorney"])
    client = repo.create_client(org, "Evaluation client", "EVAL", alice["id"])
    m = repo.create_matter(org, client["id"], "EVAL-S", "Summaries evaluation", alice["id"])
    c = TestClient(main_mod.app, base_url="https://testserver")
    sign_in(c, alice)
    base = f"/api/legal/matters/{m['id']}"
    results = []
    for d in sc["documents"]:
        if d.get("pdf_pages"):
            import fitz
            pdf = fitz.open()
            for page in d["pdf_pages"]:
                pdf.new_page().insert_text((72, 72), page, fontsize=10)
            body, mime = pdf.tobytes(), "application/pdf"
        else:
            body, mime = d["text"].encode(), "text/markdown" if d["name"].endswith(".md") else "text/plain"
        r = c.post(f"{base}/documents/import", headers=csrf_headers(c), files={"file": (d["name"], body, mime)},
                   data={"kind": "contract", "status": "final"})
        assert r.status_code == 202, r.text
        doc = r.json()["document"]["id"]
        worker.run_pending()
        t0 = time.monotonic()
        r = c.post(f"{base}/documents/{doc}/summaries", headers=csrf_headers(c))
        assert r.status_code == 200, r.text
        worker.run_pending()
        seconds = round(time.monotonic() - t0, 1)
        got = c.get(f"{base}/documents/{doc}/summaries").json()
        if not got["summaries"]:
            results.append({"id": d["id"], "seconds": seconds, "failed": got.get("job"), "facts": [], "points": []})
            continue
        content = got["summaries"][0]["content"]
        pts = content["points"]
        facts = []
        for f in d["facts"]:
            hit = next((p for p in pts if _covers(p, f["anchor"])), None)
            where = f.get("where")
            cite_ok = None if not where or not hit else all((hit.get("cite") or {}).get(k) == v for k, v in where.items())
            facts.append({**f, "covered": hit is not None, "cite_ok": cite_ok, "point": hit["text"] if hit else None,
                          "got_cite": hit.get("cite") if hit else None})
        results.append({"id": d["id"], "seconds": seconds, "transcript": content["transcript"], "facts": facts,
                        "points": [{"text": p["text"], "quote": p["quote"], "verification": p["verification"], "cite": p["cite"],
                                    "figures_unsupported": p["figures_unsupported"]} for p in pts]})
    facts = [f for r in results for f in r["facts"]]
    placed = [f for f in facts if f["cite_ok"] is not None]
    points = [p for r in results for p in r["points"]]
    summary = {"documents": len(results), "facts": len(facts), "facts_covered": f"{sum(f['covered'] for f in facts)}/{len(facts)}",
               "citations_right": f"{sum(bool(f['cite_ok']) for f in placed)}/{len(placed)}",
               "quotes_verified": f"{sum(p['verification'] != 'failed' for p in points)}/{len(points)}",
               "figures_unsupported": sum(1 for p in points if p["figures_unsupported"]),
               "points_per_document": round(len(points) / max(len(results), 1), 1),
               "seconds_per_document": round(sum(r["seconds"] for r in results) / max(len(results), 1), 1)}
    out = LAW / "reports"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", ".", ":!eval/law/reports", ":!docs/echomind-law/design/screens"],
                                cwd=ROOT, capture_output=True, text=True).stdout.strip())
    meta = {"set": args.set, "commit": commit + ("+changes" if dirty else ""), "model": args.model, "time": stamp, "prompt_version": summarize.VERSION}
    (out / f"{stamp}-summaries-{args.set}.json").write_text(json.dumps({"meta": meta, "summary": summary, "documents": results}, indent=1, ensure_ascii=False))
    md = [f"# EchoMind Law summaries evaluation — {args.set}", "", f"- Commit: `{meta['commit']}`", f"- Model: `{args.model}`",
          f"- Prompt: `{summarize.VERSION}`", f"- Run: {stamp}", "", "| Measure | Result |", "|---|---|"] + [f"| {k.replace('_', ' ')} | {v} |" for k, v in summary.items()]
    for r in results:
        md += ["", f"## {r['id']} ({r['seconds']} s)", ""]
        md += [f"- {'✓' if f['covered'] else '**missed** ✗'} {f['fact']}" + ("" if f["cite_ok"] is None else (" · place ✓" if f["cite_ok"] else f" · **place ✗** (got {f['got_cite']})"))
               for f in r["facts"]]
        md += [f"- point not verified: {p['text']}" for p in r["points"] if p["verification"] == "failed"]
        md += [f"- figure to check: {p['text']} ({', '.join(p['figures_unsupported'])})" for p in r["points"] if p["figures_unsupported"]]
    (out / f"{stamp}-summaries-{args.set}.md").write_text("\n".join(md) + "\n")
    print(json.dumps(summary))
    print(f"report: eval/law/reports/{stamp}-summaries-{args.set}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
