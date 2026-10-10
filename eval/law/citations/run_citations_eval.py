#!/usr/bin/env python3
"""EchoMind Law citation and quote checker evaluation (S3.4, DOC-03; spec test T07; decision D-048).

Fictional documents with planted citations (golden.json tunes the checker; heldout.json never does and
is run once per checker version) are imported through the real endpoint, and the citing document is
checked. No model is involved; the documents are read (and their passages indexed) as any import is.

Reported:
- **references:** resolved to the right clause of the right document, or flagged missing;
  **false missing** (a clause that exists reported missing) must be 0;
- **quotations:** accurate, altered or not found as a careful reader would say; **altered caught**,
  and **false alarms** (an accurate quotation reported altered or not found);
- **authorities:** found, **malformed caught**, **wrongly flagged** (a well-formed citation called
  impossible) must be 0; every authority unverified and nothing looked up (P-009);
- **extra items:** found but not expected, listed for a person to judge.

    cd backend && python3 ../eval/law/citations/run_citations_eval.py --set golden
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


def _match(items: list[dict], e: dict) -> dict | None:
    rx = re.compile(re.escape(e["text"]) + (r"(?![\d.])" if e["kind"] == "reference" else ""))
    hits = [i for i in items if i["kind"] == e["kind"] and rx.search(i["text"])]
    n = e.get("occurrence", 1)
    return hits[n - 1] if len(hits) >= n else None


def _judge(e: dict, i: dict | None, citing_id: str, titles: dict) -> tuple[bool, str]:
    if i is None:
        return False, "not found by the parser"
    if e["kind"] == "authority":
        ok = i["status"] == "unverified" and bool(i["malformed"]) == e["malformed"]
        return ok, f"unverified, {'malformed: ' + '; '.join(i['malformed']) if i['malformed'] else 'well-formed'}"
    t = i.get("target") or {}
    got = f"{i['status']}" + (f", clause {t.get('clause')} of {'this document' if t.get('document_id') == citing_id else titles.get(t.get('document_id'))}"
                              if t and t.get("clause") else "")
    ok = i["status"] == e["status"]
    if ok and e.get("clause"):
        ok = t.get("clause") == e["clause"] and (t.get("document_id") == citing_id if e["in"] == "self" else titles.get(t.get("document_id")) == e["in"])
    if ok and e.get("note"):
        ok = e["note"] in (i.get("note") or "")
    return ok, got + (f" — {i['note']}" if i.get("note") else "")


def run(args) -> int:
    import logging
    logging.disable(logging.INFO)
    from fastapi.testclient import TestClient
    from app.core.db import init_db
    init_db()
    import app.main as main_mod
    from app.legal.citations import check as checkmod
    from app.legal.jobs import worker
    from app.legal.matters import repo
    from app.legal.orgs import default_org_id
    from conftest import csrf_headers, make_user, sign_in

    sc = json.loads((HERE / f"{args.set}.json").read_text())
    org = default_org_id()
    alice = make_user("alice", mfa=True, firm_roles=["attorney"])
    client = repo.create_client(org, "Evaluation client", "EVAL", alice["id"])
    c = TestClient(main_mod.app, base_url="https://testserver")
    sign_in(c, alice)
    results = []
    for n, s in enumerate(sc["scenarios"]):
        m = repo.create_matter(org, client["id"], f"EVAL-C{n}", f"Citation evaluation {s['id']}", alice["id"])
        base = f"/api/legal/matters/{m['id']}"
        titles = {}

        def imp(title: str, text: str, status: str) -> str:
            r = c.post(f"{base}/documents/import", headers=csrf_headers(c), files={"file": (f"{title}.txt", text.encode(), "text/plain")},
                       data={"kind": "contract", "status": status, "title": title})
            assert r.status_code == 202, r.text
            titles[r.json()["document"]["id"]] = title
            return r.json()["document"]["id"]
        for src in s["sources"]:
            imp(src["title"], src["text"], src["status"])
        citing = imp(s["citing"]["title"], s["citing"]["text"], "draft")
        worker.run_pending()
        t0 = time.monotonic()
        r = c.post(f"{base}/documents/{citing}/citation-check", headers=csrf_headers(c))
        ms = round((time.monotonic() - t0) * 1000)
        assert r.status_code == 200, r.text
        ch = r.json()["check"]
        used, lines = set(), []
        for e in s["expect"]:
            i = _match(ch["items"], e)
            if i:
                used.add(i["id"])
            ok, got = _judge(e, i, citing, titles)
            lines.append({**e, "ok": ok, "got": got, "got_status": i["status"] if i else None, "got_malformed": bool(i and i["malformed"])})
        extra = [{"kind": i["kind"], "text": i["text"], "status": i["status"]} for i in ch["items"] if i["id"] not in used]
        results.append({"id": s["id"], "ms": ms, "lines": lines, "extra": extra, "external_lookup": ch["external_lookup"]})
    allx = [x for r in results for x in r["lines"]]
    of = lambda k: [x for x in allx if x["kind"] == k]  # noqa: E731
    refs, quotes, auths = of("reference"), of("quote"), of("authority")
    summary = {
        "items": len(allx), "correct": f"{sum(x['ok'] for x in allx)}/{len(allx)}",
        "references": f"{sum(x['ok'] for x in refs)}/{len(refs)}",
        "missing_caught": f"{sum(x['got_status'] == 'missing' for x in refs if x['status'] == 'missing')}/{sum(x['status'] == 'missing' for x in refs)}",
        "false_missing": sum(x["status"] == "resolved" and x["got_status"] == "missing" for x in refs),
        "quotations": f"{sum(x['ok'] for x in quotes)}/{len(quotes)}",
        "altered_caught": f"{sum(x['got_status'] == 'altered' for x in quotes if x['status'] == 'altered')}/{sum(x['status'] == 'altered' for x in quotes)}",
        "false_alarms": sum(x["status"] == "accurate" and x["got_status"] in ("altered", "not_found") for x in quotes),
        "authorities_found": f"{sum(x['got_status'] is not None for x in auths)}/{len(auths)}",
        "malformed_caught": f"{sum(x['got_malformed'] for x in auths if x['malformed'])}/{sum(x['malformed'] for x in auths)}",
        "wrongly_flagged": sum(x["got_malformed"] and not x["malformed"] for x in auths),
        "all_unverified": all(x["got_status"] in (None, "unverified") for x in auths),
        "external_lookups": sum(bool(r["external_lookup"]) for r in results),
        "extra_items": sum(len(r["extra"]) for r in results),
        "ms_per_check": round(sum(r["ms"] for r in results) / len(results)),
    }
    out = LAW / "reports"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", ".", ":!eval/law/reports"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    meta = {"set": args.set, "commit": commit + ("+changes" if dirty else ""), "time": stamp, "prompt_version": checkmod.VERSION,
            "model": "none (rules only)", "embeddings": "nomic-embed-text (indexing on import only)"}
    (out / f"{stamp}-citations-{args.set}.json").write_text(json.dumps({"meta": meta, "summary": summary, "scenarios": results}, indent=1, ensure_ascii=False))
    md = [f"# EchoMind Law citation checker evaluation (T07) — {args.set}", "", f"- Commit: `{meta['commit']}`", f"- Checker: `{checkmod.VERSION}`",
          f"- Run: {stamp}", "", "| Measure | Result |", "|---|---|"] + [f"| {k.replace('_', ' ')} | {v} |" for k, v in summary.items()]
    for r in results:
        md += ["", f"## {r['id']} ({r['ms']} ms)", ""]
        md += [f"- {x['kind']} `{x['text']}`: expected {x.get('status') or ('malformed' if x.get('malformed') else 'well-formed')}"
               f"{', clause ' + x['clause'] + ' of ' + x['in'] if x.get('clause') else ''}; got {x['got']}{'' if x['ok'] else ' **✗**'}" for x in r["lines"]]
        if r["extra"]:
            md += ["", "Also found (not in the expectations):"] + [f"- {e['kind']} `{e['text']}`: {e['status']}" for e in r["extra"]]
    (out / f"{stamp}-citations-{args.set}.md").write_text("\n".join(md) + "\n")
    print(json.dumps(summary))
    print(f"report: eval/law/reports/{stamp}-citations-{args.set}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
