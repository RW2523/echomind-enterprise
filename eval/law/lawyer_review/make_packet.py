#!/usr/bin/env python3
"""Phase 2 lawyer review packet (S2.8; rubric: docs/echomind-law/releases/phase2-rubric.md).

Builds a packet a practising lawyer scores, from the newest HELD-OUT reports (never used for tuning):
research answers with their evidence, compare explanations and conflicts, Silent Assist alerts, and
minutes items. Items are shuffled and labelled only by what they are (answer, explanation, alert,
minutes item), not by suite, setting or model.

    python3 eval/law/lawyer_review/make_packet.py                 # writes packet.html, sheet.csv, key.json here

- packet.html: what the lawyer reads (fictional material only);
- sheet.csv: one row per item with blank score columns, to fill in;
- key.json: which report and item each row came from (for score.py; not for the reviewer).
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent / "reports"


def newest(suffix: str, research: bool = False) -> Path:
    found = sorted(p for p in REPORTS.glob(f"*{suffix}") if not research or p.name.count("-") == 2)
    if not found:
        raise SystemExit(f"no report ending {suffix}: run the gate first (eval/law/run_gate.py)")
    return found[-1]


def items() -> list[dict]:
    out = []
    r = newest("-heldout.json", research=True)
    for it in json.loads(r.read_text())["items"]:
        ev = [f"[{e['n']}] “{e['quote']}” — {e.get('document_title')}, version {e.get('version_number')} ({e.get('version_status')})"
              + (f", clause {e['clause']}" if e.get("clause") else "") + f" · {e['verification']}" for e in it.get("evidence") or []]
        out.append({"kind": "research_answer", "source": [r.name, it["id"]], "title": it["question"],
                    "body": it.get("answer") or "(no answer)", "evidence": ev, "extra": {"versions": it.get("versions"), "expect_abstain": it.get("expect_abstain")}})
    c = newest("-compare-heldout.json")
    data = json.loads(c.read_text())
    for i, s in enumerate(data.get("explanations", {}).get("samples", [])):
        out.append({"kind": "compare_explanation", "source": [c.name, f"explanation {i}"], "title": "Explained change",
                    "body": s["text"], "evidence": [f"{side}: “{q}” · {v}" for side, v, q in s.get("evidence", [])]})
    for i, f in enumerate(data.get("scan", {}).get("findings", [])):
        out.append({"kind": "conflict_finding", "source": [c.name, f"finding {i}"], "title": f"Possible conflict ({f['kind']})",
                    "body": f"{f['a']} / {f['b']} — {f.get('explanation') or ''}", "evidence": [f"verdict: {f['verdict']}"]})
    a = newest("-assist-heldout.json")
    for sc in json.loads(a.read_text())["scenarios"]:
        for j, line in enumerate(sc.get("all", [])):
            out.append({"kind": "assist_alert", "source": [a.name, f"{sc['id']} alert {j}"], "title": f"Alert during a call ({sc['pack']})",
                        "body": line, "evidence": []})
    m = newest("-minutes-heldout.json")
    for it in json.loads(m.read_text())["items"]:
        out.append({"kind": "minutes_item", "source": [m.name, it["id"]], "title": f"Minutes item ({it['section']})",
                    "body": f"{it['kind'].capitalize()}: {it['text']}" + (f" — owner {it.get('owner') or 'not stated'}, by {it.get('deadline') or 'not stated'}"
                                                                           if it["kind"] == "action" else ""),
                    "evidence": [f"{s.get('speaker')}: “{s['quote']}” · {s['verification']}" for s in it.get("sources", [])]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    rows = items()
    random.Random(args.seed).shuffle(rows)
    out = Path(args.out)
    for n, r in enumerate(rows, start=1):
        r["id"] = f"R{n:03d}"
    (out / "key.json").write_text(json.dumps([{"id": r["id"], "kind": r["kind"], "source": r["source"], **r.get("extra", {})} for r in rows], indent=1))
    with open(out / "sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "kind", "score", "abstained_correctly", "version_right", "missing_or_invented", "notes"])
        for r in rows:
            w.writerow([r["id"], r["kind"], "", "", "", "", ""])
    label = {"research_answer": "Answer", "compare_explanation": "Explained change", "conflict_finding": "Conflict",
             "assist_alert": "Alert", "minutes_item": "Minutes item"}
    parts = ["<!doctype html><meta charset=utf-8><title>EchoMind Law — review packet</title>",
             "<style>body{font:15px/1.55 Georgia,serif;max-width:820px;margin:40px auto;padding:0 16px;color:#15171C}"
             "h1{font:600 22px system-ui}section{border-top:1px solid #ddd;padding:14px 0}.k{font:600 12px system-ui;color:#4A4F59;"
             "text-transform:uppercase;letter-spacing:.04em}.e{font:13px system-ui;color:#4A4F59}</style>",
             "<h1>Review packet — fictional material</h1><p class=e>Score each item in sheet.csv using the rubric. Items are shuffled.</p>"]
    for r in rows:
        ev = "".join(f"<li>{html.escape(e)}</li>" for e in r["evidence"])
        parts.append(f"<section><div class=k>{r['id']} · {label[r['kind']]}</div><h2 style='font-size:16px'>{html.escape(r['title'])}</h2>"
                     f"<p>{html.escape(r['body']).replace(chr(10), '<br>')}</p>{f'<ul class=e>{ev}</ul>' if ev else ''}</section>")
    (out / "packet.html").write_text("\n".join(parts))
    print(f"{len(rows)} items → {out / 'packet.html'}, {out / 'sheet.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
