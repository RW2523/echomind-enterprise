#!/usr/bin/env python3
"""EchoMind Law compare and contradictions evaluation (S2.3, RAG-02; decision D-039).

Builds a throwaway Law data folder and imports the fictional corpus (eval/law/corpus) as alice in
one matter, then the set's own files (compare/<set>.json): further versions of corpus documents,
and a document with five planted contradictions.
- **golden** (tuning): the restructured MSA (msa-v3) and the order form.
- **heldout** (never used to adjust the system): a lease draft and a lease side letter. Everything is read with the real embedding model. Then, through the real endpoints:

1. **Alignment.** Each version pair in the set is scored:
   - every change between clause-numbered units: kind, moved, renumbered from, labels;
   - precision and recall over the changes found.
2. **Explanations.** Every material change of one pair is explained by the real model:
   - the quote verification rate;
   - how many changed clauses are explained with a verified quote from both versions;
   - latency.
3. **Scan.** Conflicts across the matter's signed documents, checked by the real model:
   - recall of the planted conflicts;
   - precision, where conflicts listed as acceptable count neither way;
   - unclear checks, the number of candidates, and the time taken.

    cd backend && python3 ../eval/law/run_compare_eval.py --set golden
    cd backend && python3 ../eval/law/run_compare_eval.py --set heldout

Nothing leaves this host. The report goes to eval/law/reports/<time>-compare-<set>.json and .md.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from run_law_eval import norm, pct, setup_env  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default=os.getenv("LAW_LLM_BASE_URL", "http://localhost:8355/v1"))
    ap.add_argument("--model", default=os.getenv("LAW_LLM_MODEL", "nvidia/Qwen3-30B-A3B-FP4"))
    ap.add_argument("--embed", default=os.getenv("LAW_EMBED_URL", "http://localhost:11435/api/embeddings"))
    ap.add_argument("--keep", action="store_true", help="keep the throwaway data folder")
    ap.add_argument("--set", choices=["golden", "heldout"], default="golden")
    ap.add_argument("--skip", default="", help="comma-separated parts to skip: explain,scan")
    args = ap.parse_args()
    data = setup_env(args)
    try:
        return run(args)
    finally:
        if not args.keep:
            shutil.rmtree(data, ignore_errors=True)
            (data.parent / f"{data.name}.key").unlink(missing_ok=True)


def _events(r) -> list[dict]:
    assert r.status_code == 200, r.text
    return [json.loads(line) for line in r.text.splitlines() if line.strip()]


def score_alignment(result: dict, expected: list[dict]) -> dict:
    """Scored over clause-numbered units only."""
    def key(p):
        return ("right", p["right"]["label"]) if p.get("right") and p["right"].get("label") else \
               ("left", p["left"]["label"]) if p.get("left") and p["left"].get("label") else None
    predicted = {}
    for p in result["pairs"]:
        k = key(p)
        if k and (p["kind"] != "unchanged" or p["moved"] or p["renumbered"]):
            predicted[k] = p
    rows, correct = [], 0
    for e in expected:
        k = ("right", e["right"]) if "right" in e else ("left", e["left"])
        p = predicted.get(k)
        ok = bool(p) and p["kind"] == e["kind"] and bool(p["moved"]) == bool(e.get("moved")) and \
            (e.get("renumbered_from") is None or (p.get("left") or {}).get("label") == e["renumbered_from"]) and \
            set(e.get("flags", [])) <= set(p.get("flags", []))
        correct += ok
        rows.append({"unit": f"{k[0]} {k[1]}", "expected": e, "got": None if not p else
                     {"kind": p["kind"], "moved": p["moved"], "renumbered": p["renumbered"], "flags": p["flags"],
                      "from": (p.get("left") or {}).get("label")}, "correct": ok})
    expected_keys = {("right", e["right"]) if "right" in e else ("left", e["left"]) for e in expected}
    extra = [f"{k[0]} {k[1]}: {predicted[k]['kind']}" for k in predicted if k not in expected_keys]
    unlabelled = sum(1 for p in result["pairs"] if not key(p) and p["kind"] != "unchanged")
    return {"expected": len(expected), "correct": correct, "found": len(predicted),
            "precision": round(correct / len(predicted), 3) if predicted else None,
            "recall": round(correct / len(expected), 3) if expected else None,
            "extra": extra, "unlabelled_changes": unlabelled, "rows": rows}


def run(args) -> int:
    import logging
    from fastapi.testclient import TestClient
    logging.disable(logging.INFO)
    from app.core.db import init_db
    init_db()
    import app.main as main_mod
    from app.legal.jobs import worker
    from app.legal.matters import repo
    from app.legal.orgs import default_org_id
    from conftest import csrf_headers, make_user, sign_in

    expected = json.loads((HERE / "compare" / f"{args.set}.json").read_text())
    org = default_org_id()
    alice = make_user("alice", mfa=True, firm_roles=["attorney"])
    client = repo.create_client(org, "Voss-Lindqvist AB", "VOSS", alice["id"])
    matter = repo.create_matter(org, client["id"], "EVAL-3", "Harrowgate renewal (compare evaluation)", alice["id"])
    mid = matter["id"]
    c = TestClient(main_mod.app, base_url="https://testserver")
    sign_in(c, alice)

    manifest = json.loads((HERE / "corpus" / "manifest.json").read_text())
    docs = [d for d in manifest["documents"] if d["matter"] != "forbidden"]
    titles = {d["key"]: d["title"] for d in docs}
    docs += [{"key": v["of"], "file": v["file"], "title": titles[v["of"]], "kind": "contract", "status": v["status"], "version_of": v["of"]}
             for v in expected["versions"]]
    docs += [{**d, "key": d["file"]} for d in expected["scan"]["extra_documents"]]
    doc_ids, version_of_file = {}, {}
    t0 = time.monotonic()
    for d in docs:
        form = {"kind": d["kind"], "status": d["status"], "title": d["title"]}
        if d.get("version_of"):
            form["version_of"] = doc_ids[d["version_of"]]
        r = c.post(f"/api/legal/matters/{mid}/documents/import", headers=csrf_headers(c),
                   files={"file": (Path(d["file"]).name, (HERE / d["file"]).read_bytes(), "text/markdown")}, data=form)
        assert r.status_code == 202, r.text
        did = r.json()["document"]["id"]
        doc_ids.setdefault(d["key"], did)
        worker.run_pending()
        versions = c.get(f"/api/legal/matters/{mid}/documents/{did}/versions").json()["versions"]
        version_of_file[d["file"]] = max(versions, key=lambda v: v["number"])["id"]
    print(f"corpus read in {time.monotonic() - t0:.1f}s: {len(docs)} files", flush=True)

    report: dict = {"alignment": [], "explanations": None, "scan": None}
    comparisons = {}
    for pair in expected["alignment"]:
        t = time.monotonic()
        r = c.post(f"/api/legal/matters/{mid}/compare", headers=csrf_headers(c),
                   json={"left_version_id": version_of_file[pair["left"]], "right_version_id": version_of_file[pair["right"]]})
        assert r.status_code == 200, r.text
        res = r.json()["comparison"]
        comparisons[pair["right"]] = res
        s = score_alignment(res, pair["changes"])
        s.update(pair=f"{Path(pair['left']).name} → {Path(pair['right']).name}", seconds=round(time.monotonic() - t, 2),
                 passes=res["passes"], summary=res["summary"])
        report["alignment"].append(s)
        print(f"alignment {s['pair']}: {s['correct']}/{s['expected']} correct, precision {s['precision']}, recall {s['recall']}, "
              f"extra {s['extra']}", flush=True)

    skip = set(filter(None, args.skip.split(",")))
    if "explain" not in skip:
        res = comparisons[expected["explain"]]
        t = time.monotonic()
        ev = _events(c.post(f"/api/legal/matters/{mid}/compare/{res['id']}/explain", headers=csrf_headers(c), json={}))
        total = time.monotonic() - t
        ex = [e for e in ev if e["type"] == "explanation"]
        quotes = [q for e in ex for q in e["evidence"] if q["verification"] != "not_quoted"]
        verified = [q for q in quotes if q["verification"] in ("verbatim", "normalized")]
        changed = [e for e in ex if res["pairs"][e["index"]]["kind"] == "changed"]
        both = [e for e in changed if {q["side"] for q in e["evidence"] if q["verification"] in ("verbatim", "normalized")} == {"left", "right"}]
        report["explanations"] = {
            "explained": len(ex), "errors": sum(1 for e in ev if e["type"] == "explanation_error"),
            "quotes": len(quotes), "verified": len(verified),
            "quote_verification_rate": round(len(verified) / len(quotes), 3) if quotes else None,
            "changed_with_both_sides_quoted": f"{len(both)}/{len(changed)}",
            "statuses": {s: sum(1 for e in ex if e["status"] == s) for s in {e["status"] for e in ex}},
            "seconds_total": round(total, 2), "seconds_per_change": round(total / len(ex), 2) if ex else None,
            "samples": [{"unit": (res["pairs"][e["index"]].get("right") or res["pairs"][e["index"]].get("left"))["label"],
                         "text": e["text"], "evidence": [(q["side"], q["verification"], q["quote"]) for q in e["evidence"]]}
                        for e in ex]}
        print(f"explanations: {report['explanations']['explained']} explained, verification "
              f"{report['explanations']['quote_verification_rate']}, both sides {report['explanations']['changed_with_both_sides_quoted']}, "
              f"{report['explanations']['seconds_per_change']} s each", flush=True)

    if "scan" not in skip:
        t = time.monotonic()
        ev = _events(c.post(f"/api/legal/matters/{mid}/compare/scan", headers=csrf_headers(c), json={}))
        total = time.monotonic() - t
        checked = [e["finding"] for e in ev if e["type"] == "checked"]
        conflicts = [f for f in checked if f["verdict"] == "conflict"]
        sc = expected["scan"]

        def side_of(f, title):
            return f["a"] if f["a"]["document_title"] == title else f["b"] if f["b"]["document_title"] == title else None

        def planted_match(f, p):
            mine = side_of(f, sc["subject_title"])
            if not mine or f["kind"] != p["kind"] or norm(p["value"]) not in norm(mine["value"]):
                return False
            other = f["b"] if mine is f["a"] else f["a"]
            if p.get("other_title") and other["document_title"] != p["other_title"]:
                return False
            return not p.get("other_value") or norm(p["other_value"]) in norm(other["value"])

        def acceptable(f):
            vals = {norm(f["a"]["value"]), norm(f["b"]["value"])}
            return any(f["kind"] == a["kind"] and {norm(v) for v in a["values"]} == vals for a in sc["acceptable"])
        found = [p for p in sc["planted"] if any(planted_match(f, p) for f in conflicts)]
        true_pos = [f for f in conflicts if any(planted_match(f, p) for p in sc["planted"])]
        neutral = [f for f in conflicts if f not in true_pos and acceptable(f)]
        false_pos = [f for f in conflicts if f not in true_pos and f not in neutral]
        missed_candidates = [p for p in sc["planted"] if not any(planted_match(f, p) for f in checked)]
        report["scan"] = {
            "documents": len(ev[0]["documents"]), "candidates": ev[0]["candidates"], "checked": len(checked),
            "conflicts": len(conflicts), "rejected": sum(1 for f in checked if f["verdict"] == "no_conflict"),
            "unclear": sum(1 for f in checked if f["verdict"] == "unclear"),
            "planted_recall": f"{len(found)}/{len(sc['planted'])}",
            "precision": round(len(true_pos) / (len(true_pos) + len(false_pos)), 3) if true_pos or false_pos else None,
            "acceptable_extras": len(neutral), "false_positives": [_brief(f) for f in false_pos],
            "planted_never_proposed": [p["kind"] for p in missed_candidates],
            "planted_rejected_by_model": [p["kind"] for p in sc["planted"] if p not in found and p not in missed_candidates],
            "quote_verification_rate": _rate([q for f in checked for q in f["evidence"] if q["verification"] != "not_quoted"]),
            "seconds": round(total, 2), "findings": [_brief(f) for f in checked]}
        print(f"scan: planted {report['scan']['planted_recall']}, precision {report['scan']['precision']}, "
              f"{report['scan']['candidates']} candidates, {report['scan']['unclear']} unclear, {report['scan']['seconds']} s; "
              f"false positives {report['scan']['false_positives']}", flush=True)
    args.explain_pair = expected["explain"]
    write(args, report)
    return 0


def _rate(quotes: list[dict]):
    ok = [q for q in quotes if q["verification"] in ("verbatim", "normalized")]
    return round(len(ok) / len(quotes), 3) if quotes else None


def _brief(f: dict) -> dict:
    return {"kind": f["kind"], "verdict": f["verdict"], "a": f"{f['a']['document_title']}: {f['a']['value']}",
            "b": f"{f['b']['document_title']}: {f['b']['value']}", "explanation": f.get("explanation", "")[:300],
            "quotes": [q["verification"] for q in f.get("evidence", [])]}


def write(args, report: dict) -> None:
    out = HERE / "reports"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    meta = {"set": args.set, "commit": commit + ("+changes" if dirty else ""), "model": args.model, "embeddings": "nomic-embed-text",
            "time": stamp}
    (out / f"{stamp}-compare-{args.set}.json").write_text(json.dumps({"meta": meta, **report}, indent=1, ensure_ascii=False))
    lines = [f"# EchoMind Law compare evaluation — {args.set}", "", f"- Commit: `{meta['commit']}`", f"- Model: `{args.model}`; embeddings nomic-embed-text",
             f"- Run: {stamp}", "", "## Alignment", "", "| Pair | Correct | Precision | Recall | Extra | Passes | Seconds |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {a['pair']} | {a['correct']}/{a['expected']} | {a['precision']} | {a['recall']} | {', '.join(a['extra']) or '—'} | "
              f"{', '.join(a['passes'])} | {a['seconds']} |" for a in report["alignment"]]
    for a in report["alignment"]:
        lines += ["", f"**{a['pair']}**", "", "| Unit | Expected | Got | Correct |", "|---|---|---|---|"]
        lines += [f"| {r['unit']} | {r['expected']['kind']}{' moved' if r['expected'].get('moved') else ''}"
                  f"{' from ' + r['expected']['renumbered_from'] if r['expected'].get('renumbered_from') else ''} "
                  f"{','.join(r['expected'].get('flags', []))} | "
                  f"{(r['got']['kind'] + (' moved' if r['got']['moved'] else '') + (' from ' + str(r['got']['from']) if r['got']['renumbered'] else '') + ' ' + ','.join(r['got']['flags'])) if r['got'] else '—'} | "
                  f"{'yes' if r['correct'] else '**no**'} |" for r in a["rows"]]
    if report["explanations"]:
        e = report["explanations"]
        lines += ["", f"## Explanations ({Path(args.explain_pair).name})", "", "| Measure | Result |", "|---|---|"]
        lines += [f"| {k.replace('_', ' ')} | {v} |" for k, v in e.items() if k != "samples"]
        lines += ["", "| Unit | Explanation | Quotes |", "|---|---|---|"]
        lines += [f"| {s['unit']} | {s['text'].replace('|', '/')} | {'; '.join(f'{x[0]} {x[1]}' for x in s['evidence'])} |" for s in e["samples"]]
    if report["scan"]:
        s = report["scan"]
        lines += ["", "## Conflicts across documents (signed versions)", "", "| Measure | Result |", "|---|---|"]
        lines += [f"| {k.replace('_', ' ')} | {v} |" for k, v in s.items() if k != "findings"]
        lines += ["", "| Kind | Verdict | Source 1 | Source 2 | Quotes |", "|---|---|---|---|---|"]
        lines += [f"| {f['kind']} | {f['verdict']} | {f['a']} | {f['b']} | {', '.join(f['quotes'])} |" for f in s["findings"]]
    (out / f"{stamp}-compare-{args.set}.md").write_text("\n".join(lines) + "\n")
    print(f"report: eval/law/reports/{stamp}-compare-{args.set}.md", flush=True)


if __name__ == "__main__":
    sys.exit(main())
