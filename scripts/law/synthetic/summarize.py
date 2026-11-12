#!/usr/bin/env python3
"""Summarise chat runs of the synthetic set (2026-11-12): two or more run_chat_keys.py reports, side by side.

    python3 scripts/law/synthetic/summarize.py eval/law/reports/<time>-chat-syn-r1.json eval/law/reports/<time>-chat-syn-r2.json

Per run: passed, by kind, by matter and by the format of the document that answers each item (Word, typeset PDF, scan,
plain text; from eval/fixtures/synthetic/manifest.json), the answer states, and answer times. Across runs: items that
pass in every run, in some, in none. Items marked `needs` in the keys (Word comments, tracked changes: words the product
does not read by design) are counted apart. Writes eval/law/reports/<time>-synthetic-summary.md and prints it.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "eval" / "fixtures" / "synthetic"


def main() -> int:
    runs = [json.loads(Path(p).read_text()) for p in sys.argv[1:]]
    if not runs:
        print(__doc__)
        return 2
    manifest = json.loads((BASE / "manifest.json").read_text())
    fmt = {d["title"]: d["render"] for c in manifest["cases"] for d in c["documents"]}
    keys = {}
    for k in sorted(BASE.glob("answer_key_*.jsonl")):
        for line in k.read_text().splitlines():
            if line.strip():
                it = json.loads(line)
                keys[it["id"]] = it
    labels = [r["label"] for r in runs]
    by_id = [{i["id"]: i for i in r["items"]} for r in runs]
    ids = [i["id"] for i in runs[0]["items"]]

    def group_of(iid: str, how: str) -> str:
        it = keys.get(iid, {})
        if how == "kind":
            return it.get("kind", "?")
        if how == "matter":
            return it.get("matter", "?")
        if it.get("needs"):
            return f"needs {it['needs']}"
        if it.get("kind") == "other_matter":
            return "other matter"
        return {"docx": "Word", "pdf": "typeset PDF", "scan": "scan (OCR)", "txt": "plain text", "md": "Markdown"}.get(
            fmt.get(it.get("source"), ""), "no single source")

    lines = [f"# Synthetic set: {len(ids)} items, {len(runs)} runs ({', '.join(labels)})", ""]
    passed = [sum(bool(b[i]["ok"]) for i in ids) for b in by_id]
    every = [i for i in ids if all(b[i]["ok"] for b in by_id)]
    some = [i for i in ids if any(b[i]["ok"] for b in by_id) and i not in every]
    lines += [f"Passed: {' and '.join(f'**{p}/{len(ids)}** ({l})' for p, l in zip(passed, labels))}. In every run {len(every)}, "
              f"in some {len(some)}, in none {len(ids) - len(every) - len(some)}.", ""]
    excluded = [i for i in ids if keys.get(i, {}).get("needs")]
    if excluded:
        p2 = [sum(bool(b[i]["ok"]) for i in ids if i not in excluded) for b in by_id]
        lines += [f"Without the {len(excluded)} items that need Word markup the product does not read by design "
                  f"({', '.join(excluded)}): {' and '.join(f'{p}/{len(ids) - len(excluded)}' for p in p2)}.", ""]
    for how, title in (("kind", "Kind"), ("matter", "Matter"), ("format", "Format of the answering document")):
        groups = defaultdict(list)
        for i in ids:
            groups[group_of(i, how)].append(i)
        lines += [f"| {title} | Items | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
        for g in sorted(groups):
            lines.append(f"| {g} | {len(groups[g])} | " + " | ".join(f"{sum(bool(b[i]['ok']) for i in groups[g])}" for b in by_id) + " |")
        lines.append("")
    lines += ["| Run | Answer states (last turn) | Median s | p95 s |", "|---|---|---|---|"]
    for r, b in zip(runs, by_id):
        states = Counter((b[i]["turns"][-1]["state"] if b[i].get("turns") else "error") or "card" for i in ids)
        lines.append(f"| {r['label']} | {', '.join(f'{k} {v}' for k, v in states.most_common())} | {r['summary']['p50_s']} | "
                     f"{r['summary']['p95_s']} |")
    lines += ["", "## Items that failed in at least one run", ""]
    for i in ids:
        if i in every:
            continue
        it = keys.get(i, {})
        marks = " ".join("PASS" if b[i]["ok"] else "fail" for b in by_id)
        lines.append(f"- **{i}** ({it.get('kind')}, {group_of(i, 'format')}) [{marks}]: {it.get('question') or it['turns'][-1]['question']}")
        for b, label in zip(by_id, labels):
            r = b[i]
            if r["ok"] or not r.get("turns"):
                continue
            bad = next((t for t in r["turns"] if not t["ok"]), r["turns"][-1])
            text = " ".join((bad.get("text") or "").split())[:260]
            lines.append(f"  - {label}: {bad.get('state')}; missing {bad.get('missing')}"
                         + (f"; said {bad.get('leaked')}" if bad.get("leaked") else "") + f": \"{text}\"")
    out = ROOT / "eval" / "law" / "reports" / f"{time.strftime('%Y%m%d-%H%M%S')}-synthetic-summary.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))
    print(f"-> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
