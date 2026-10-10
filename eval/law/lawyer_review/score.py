#!/usr/bin/env python3
"""Score a filled review sheet against the rubric's proposed bars (Phase 2: S2.8; Phase 3: S3.9).

    python3 eval/law/lawyer_review/score.py sheet.csv [second_reviewer.csv]
    python3 eval/law/lawyer_review/score.py --phase 3 phase3/sheet.csv [second_reviewer.csv]
    python3 eval/law/lawyer_review/score.py --phase 4 phase4/sheet.csv [second_reviewer.csv]

Writes eval/law/reports/<time>-lawyer-phase<N>.json and .md. Bars are proposals for the owner (and P-008
for alerts); a measure below its bar is reported, never hidden.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent / "reports"
SCALE = {"research_answer": 3, "compare_explanation": 3, "conflict_finding": 3, "minutes_item": 3, "assist_alert": 2}
# Phase 3: drafts and playbook lines on 0-3; each finding on 0-2 (2 useful, 1 right but not worth a lawyer's time, 0 wrong).
SCALE3 = {"draft": 3, "playbook_line": 3, "citation": 2, "definition": 2, "obligation": 2, "diligence_issue": 2, "dd_missing": 2,
          "governance_issue": 2}
FINDINGS = ("citation", "definition", "obligation", "diligence_issue", "dd_missing", "governance_issue")
# Phase 4: summary points 0-2 (2 accurate and worth keeping, 1 accurate but not needed, 0 wrong or misleading);
# panel perspectives 0-3 as drafts; the moderator's matrix rows 0-2; the stop decision 1 right / 0 wrong.
SCALE4 = {"summary_point": 2, "panel_perspective": 3, "panel_issue": 2, "panel_stop": 1}


def read(path: str) -> dict[str, dict]:
    with open(path, newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f) if r.get("score", "").strip() != "" or r.get("missed", "").strip() != ""}


def _yes(v: str) -> bool:
    return v.strip().lower() in ("yes", "y", "1")


def measures3(rows: dict[str, dict], key: list[dict]) -> dict:
    kinds = {k["id"]: k for k in key}
    out = {}
    for kind, top in SCALE3.items():
        scores = [int(r["score"]) for i, r in rows.items() if kinds.get(i, {}).get("kind") == kind]
        if scores:
            out[kind] = {"scored": len(scores), "zero": sum(s == 0 for s in scores), "good_share": round(sum(s >= top - 1 if top == 3 else s == 2
                                                                                                           for s in scores) / len(scores), 3)}
    found = [int(r["score"]) for i, r in rows.items() if kinds.get(i, {}).get("kind") in FINDINGS]
    if found:
        out["findings"] = {"scored": len(found), "wrong_share": round(sum(s == 0 for s in found) / len(found), 3),
                           "useful_share": round(sum(s == 2 for s in found) / len(found), 3)}
    flag = lambda kind, word: sum(1 for i, r in rows.items() if kinds.get(i, {}).get("kind") == kind
                                  and r.get("wrong_or_invented", "").strip().lower().startswith(word))
    out["drafts_invented"] = flag("draft", "invent")
    out["obligation_dates_wrong"] = flag("obligation", "date")
    out["groups_with_misses"] = sorted(i for i, r in rows.items() if kinds.get(i, {}).get("kind") == "missed" and _yes(r.get("missed", "")))
    return out


def measures4(rows: dict[str, dict], key: list[dict]) -> dict:
    kinds = {k["id"]: k for k in key}
    out = {}
    for kind, top in SCALE4.items():
        scores = [int(r["score"]) for i, r in rows.items() if kinds.get(i, {}).get("kind") == kind]
        if scores:
            good = sum(s >= top - 1 for s in scores) if top == 3 else sum(s == top for s in scores)
            out[kind] = {"scored": len(scores), "zero": sum(s == 0 for s in scores), "good_share": round(good / len(scores), 3)}
    out["groups_with_misses"] = sorted(i for i, r in rows.items() if kinds.get(i, {}).get("kind") == "missed" and _yes(r.get("missed", "")))
    return out


def measures(rows: dict[str, dict], key: list[dict]) -> dict:
    kinds = {k["id"]: k for k in key}
    out = {}
    for kind, top in SCALE.items():
        scores = [int(r["score"]) for i, r in rows.items() if kinds.get(i, {}).get("kind") == kind]
        if not scores:
            continue
        good = sum(s >= top - 1 for s in scores) if top == 3 else sum(s == 2 for s in scores)
        out[kind] = {"scored": len(scores), "zero": sum(s == 0 for s in scores), "good_share": round(good / len(scores), 3)}
    abst = [r for i, r in rows.items() if kinds.get(i, {}).get("expect_abstain")]
    if abst:
        out["abstentions"] = {"items": len(abst), "correct": sum(r.get("abstained_correctly", "").strip().lower() in ("yes", "y", "1") for r in abst)}
    invented = [r for i, r in rows.items() if kinds.get(i, {}).get("kind") == "minutes_item" and r.get("missing_or_invented", "").strip().lower().startswith("invent")]
    out["minutes_invented"] = len(invented)
    return out


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sheets", nargs="+", help="the filled sheet, and optionally the second reviewer's")
    ap.add_argument("--phase", type=int, choices=[2, 3, 4], default=2)
    ap.add_argument("--key", default=None, help="key.json (default: next to the packet for the phase)")
    ap.add_argument("--reports", default=str(REPORTS))
    args = ap.parse_args(argv)
    argv = args.sheets
    key = json.loads(Path(args.key or (HERE / f"phase{args.phase}" / "key.json" if args.phase in (3, 4) else HERE / "key.json")).read_text())
    first = read(argv[0])
    result = {"measures": {3: measures3, 4: measures4}.get(args.phase, measures)(first, key)}
    if len(argv) > 1:
        second = read(argv[1])
        both = [i for i in first if i in second and first[i].get("score", "").strip() and second[i].get("score", "").strip()]
        if both:
            result["agreement_within_1"] = round(sum(abs(int(first[i]["score"]) - int(second[i]["score"])) <= 1 for i in both) / len(both), 3)
            result["double_scored"] = len(both)
    m = result["measures"]
    if args.phase == 3:
        return _write3(args, result, m)
    if args.phase == 4:
        return _write4(args, result, m)
    bars = {"research harmful (0)": (m.get("research_answer", {}).get("zero"), "= 0"),
            "research 2 or 3": (m.get("research_answer", {}).get("good_share"), "≥ 0.85"),
            "compare 2 or 3": (m.get("compare_explanation", {}).get("good_share"), "≥ 0.85"),
            "minutes items 2 or 3": (m.get("minutes_item", {}).get("good_share"), "≥ 0.85"),
            "minutes invented": (m.get("minutes_invented"), "= 0"),
            "agreement ±1": (result.get("agreement_within_1"), "≥ 0.80")}
    result["bars"] = {k: {"value": v, "bar": b} for k, (v, b) in bars.items()}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.reports) / f"{stamp}-lawyer-phase2.json"
    out.write_text(json.dumps({"meta": {"set": "phase2", "time": stamp, "sheets": [Path(a).name for a in argv]}, "summary": result}, indent=1))
    out.with_suffix(".md").write_text("# Phase 2 lawyer review\n\n| Measure | Value | Proposed bar |\n|---|---|---|\n" +
                                      "".join(f"| {k} | {v['value']} | {v['bar']} |\n" for k, v in result["bars"].items()))
    print(json.dumps(result, indent=1))
    return 0


def _write3(args, result: dict, m: dict) -> int:
    bars = {"drafts 2 or 3": (m.get("draft", {}).get("good_share"), "≥ 0.80"),
            "drafts with an invented fact": (m.get("drafts_invented"), "= 0"),
            "playbook lines 2 or 3": (m.get("playbook_line", {}).get("good_share"), "≥ 0.85"),
            "playbook lines scored 0": (m.get("playbook_line", {}).get("zero"), "= 0"),
            "findings wrong (0)": (m.get("findings", {}).get("wrong_share"), "≤ 0.05"),
            "findings useful (2)": (m.get("findings", {}).get("useful_share"), "≥ 0.70"),
            "obligation dates wrong": (m.get("obligation_dates_wrong"), "= 0"),
            "groups with something missed": (len(m.get("groups_with_misses", [])), "reported"),
            "agreement ±1": (result.get("agreement_within_1"), "≥ 0.80")}
    result["bars"] = {k: {"value": v, "bar": b} for k, (v, b) in bars.items()}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.reports) / f"{stamp}-lawyer-phase3.json"
    out.write_text(json.dumps({"meta": {"set": "phase3", "time": stamp, "sheets": [Path(a).name for a in args.sheets]}, "summary": result}, indent=1))
    out.with_suffix(".md").write_text("# Phase 3 lawyer review\n\n| Measure | Value | Proposed bar |\n|---|---|---|\n" +
                                      "".join(f"| {k} | {v['value']} | {v['bar']} |\n" for k, v in result["bars"].items()))
    print(json.dumps(result, indent=1))
    return 0


def _write4(args, result: dict, m: dict) -> int:
    bars = {"summary points wrong (0)": (m.get("summary_point", {}).get("zero"), "= 0"),
            "summary points worth keeping (2)": (m.get("summary_point", {}).get("good_share"), "≥ 0.80"),
            "panel perspectives 2 or 3": (m.get("panel_perspective", {}).get("good_share"), "≥ 0.75"),
            "panel perspectives harmful (0)": (m.get("panel_perspective", {}).get("zero"), "= 0"),
            "matrix rows misleading (0)": (m.get("panel_issue", {}).get("zero"), "= 0"),
            "stop decisions right": (m.get("panel_stop", {}).get("good_share"), "= 1.0"),
            "groups with something missed": (len(m.get("groups_with_misses", [])), "reported"),
            "agreement ±1": (result.get("agreement_within_1"), "≥ 0.80")}
    result["bars"] = {k: {"value": v, "bar": b} for k, (v, b) in bars.items()}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.reports) / f"{stamp}-lawyer-phase4.json"
    out.write_text(json.dumps({"meta": {"set": "phase4", "time": stamp, "sheets": [Path(a).name for a in args.sheets]}, "summary": result}, indent=1))
    out.with_suffix(".md").write_text("# Phase 4 lawyer review\n\n| Measure | Value | Proposed bar |\n|---|---|---|\n" +
                                      "".join(f"| {k} | {v['value']} | {v['bar']} |\n" for k, v in result["bars"].items()))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
