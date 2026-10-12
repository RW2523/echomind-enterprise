#!/usr/bin/env python3
"""Lawyer review packets: Phase 2 (S2.8; docs/echomind-law/releases/phase2-rubric.md) and Phase 3 (S3.9;
docs/echomind-law/releases/phase3-rubric.md).

Builds a packet a practising lawyer scores, from the newest HELD-OUT reports (never used for tuning):
research answers with their evidence, compare explanations and conflicts, Silent Assist alerts, and
minutes items. Items are shuffled and labelled only by what they are (answer, explanation, alert,
minutes item), not by suite, setting or model.

    python3 eval/law/lawyer_review/make_packet.py                 # Phase 2: writes packet.html, sheet.csv, key.json here
    python3 eval/law/lawyer_review/make_packet.py --phase 3       # Phase 3: the same files in lawyer_review/phase3/
    python3 eval/law/lawyer_review/make_packet.py --phase 4       # Phase 4: summaries and the review panel, in lawyer_review/phase4/
    python3 eval/law/lawyer_review/make_packet.py --phase 5       # the pilot additions: regulatory matching and translation (D-068, D-070)

Phase 3 takes the newest held-out report of each document tool (drafting, playbook checks, citations,
definitions, obligations, due diligence, governance), grouped by the documents they were run on, with the
documents themselves so the reviewer can say what was missed. The groups are shuffled.

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
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAW = HERE.parent
ROOT = LAW.parent.parent
REPORTS = LAW / "reports"


def newest(suffix: str, research: bool = False) -> Path:
    found = sorted(p for p in REPORTS.glob(f"*{suffix}") if not research or p.name.count("-") == 2)
    if not found:
        raise SystemExit(f"no report ending {suffix}: run the gate first (eval/law/run_gate.py)")
    return found[-1]


def items2() -> list[dict]:
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


def _heldout(suite: str) -> dict:
    return json.loads((LAW / suite / "heldout.json").read_text())


def _needs(report: Path, rows: list[dict], key: str) -> None:
    if any(key not in r for r in rows):
        raise SystemExit(f"{report.name} has no {key!r}: it predates full outputs in the reports; run that suite's held-out set on the current runner")


def groups3() -> list[dict]:
    """One group per document set: its documents, and every item a tool produced on them."""
    out = []
    r = newest("-drafting-heldout.json")
    sets = {s["id"]: s for s in _heldout("drafting")["scenarios"]}
    for sc in json.loads(r.read_text())["scenarios"]:
        src = sets[sc["id"]]
        def shown(f: dict) -> str:
            if f["value"]:
                return f"{f['field']}: {f['value']}" + (f" — “{f['quote']}”" if f.get("quote") else "")
            sug = f.get("suggestion")
            return f"{f['field']}: [left for a person]" + (f" — the model suggested “{sug['value']}”, not used: {sug['reason']}" if sug else "")
        fields = [shown(f) for f in sc["fields"] + sc.get("absent", [])]
        unsupported = [f"marked as unsupported: {u}" for w in sc.get("written", []) for u in w["unsupported"]]
        body = "\n\n".join(w["text"] for w in sc.get("written", [])) or "(no written passages)"
        out.append({"group": f"drafting:{sc['id']}", "title": src["title"], "documents": [(d["name"], d["text"]) for d in src["documents"]],
                    "items": [{"kind": "draft", "source": [r.name, sc["id"]], "title": f"Draft from the {src['template'].replace('_', ' ')} template",
                               "body": body, "evidence": fields + unsupported}]})
    r = newest("-playbook-heldout.json")
    sets = {c["id"]: c for c in _heldout("playbook")["contracts"]}
    for c in json.loads(r.read_text())["contracts"]:
        out.append({"group": f"playbook:{c['id']}", "title": "Contract checked against the firm's playbook",
                    "documents": [(c["id"], sets[c["id"]]["text"])],
                    "items": [{"kind": "playbook_line", "source": [r.name, f"{c['id']} {x['topic']}"], "title": f"Playbook check — {x['topic'].replace('_', ' ')}",
                               "body": f"{x['got']}: {x.get('reason') or ''}", "evidence": [f"“{x['quote']}”"] if x.get("quote") else []}
                              for x in c["lines"]]})
    r = newest("-citations-heldout.json")
    sets = {s["id"]: s for s in _heldout("citations")["scenarios"]}
    for sc in json.loads(r.read_text())["scenarios"]:
        src = sets[sc["id"]]
        found = [x for x in sc["lines"] if x.get("got_status")]
        found += [{"kind": e.get("kind", "item"), "text": e.get("text", ""), "got": e.get("got") or e.get("status", "")} for e in sc.get("extra", [])]
        out.append({"group": f"citations:{sc['id']}", "title": src["citing"]["title"],
                    "documents": [(src["citing"]["title"] + " (the document checked)", src["citing"]["text"])] + [(d["title"], d["text"]) for d in src["sources"]],
                    "items": [{"kind": "citation", "source": [r.name, f"{sc['id']} {x['text']}"], "title": f"{x['kind'].capitalize()}: {x['text']}",
                               "body": x["got"], "evidence": []} for x in found]})
    r = newest("-definitions-heldout.json")
    sets = {d["id"]: d for d in _heldout("definitions")["documents"]}
    for d in json.loads(r.read_text())["documents"]:
        found = [{"text": x.get("text") or x.get("message", ""), "type": x["type"], "body": x["got"]} for x in d["lines"] if x["found"]]
        found += [{"text": e.get("text") or "", "type": e["type"], "body": e["message"]} for e in d.get("extra", [])]
        out.append({"group": f"definitions:{d['id']}", "title": sets[d["id"]]["name"], "documents": [(sets[d["id"]]["name"], sets[d["id"]]["text"])],
                    "items": [{"kind": "definition", "source": [r.name, f"{d['id']} {x['type']} {x['text']}"], "title": f"Drafting check ({x['type']})",
                               "body": x["body"], "evidence": []} for x in found]})
    r = newest("-obligations-heldout.json")
    rows = json.loads(r.read_text())["documents"]
    _needs(r, rows, "outputs")
    sets = {d["id"]: d for d in _heldout("obligations")["documents"]}
    for d in rows:
        out.append({"group": f"obligations:{d['id']}", "title": sets[d["id"]]["name"], "documents": [(sets[d["id"]]["name"], sets[d["id"]]["text"])],
                    "items": [{"kind": "obligation", "source": [r.name, f"{d['id']} {o['clause']} {o['kind']}"],
                               "title": f"Obligation ({o['kind']}), clause {o['clause'] or '—'}",
                               "body": f"{o.get('party') or 'A party'}: {o['action']}" + (f" — {o['when']}" if o.get("when") else "")
                                       + (f" — due {o['due_date']}" if o.get("due_date") else "") + (f" ({o['due_note']})" if o.get("due_note") else ""),
                               "evidence": ([f"“{o['quote']}”"] if o.get("quote") else []) + [f"found by: {o['source']}"]} for o in d["outputs"]]})
    for suite, key in (("diligence", "rooms"), ("governance", "companies")):
        r = newest(f"-{suite}-heldout.json")
        rows = json.loads(r.read_text())[key]
        _needs(r, rows, "outputs")
        sets = {x["id"]: x for x in _heldout(suite)[key]}
        for g in rows:
            items = [{"kind": f"{suite}_issue", "source": [r.name, f"{g['id']} {i['kind']} {i['title']}"], "title": f"{i['title']} ({i['document']})",
                      "body": i.get("detail") or "", "evidence": [f"“{i['quote']}” — {i['document']}"] + [f"“{x['quote']}” — {x['document']}" for x in i.get("related", [])]}
                     for i in g["outputs"]]
            if suite == "diligence":
                items.append({"kind": "dd_missing", "source": [r.name, f"{g['id']} missing"], "title": "Documents the checklist expects and the room lacks",
                              "body": "; ".join(g.get("missing_titles") or g["missing_got"]) or "(none)", "evidence": []})
            out.append({"group": f"{suite}:{g['id']}", "title": "Data room" if suite == "diligence" else "Company records",
                        "documents": [(d["name"], d["text"]) for d in sets[g["id"]]["documents"]], "items": items})
    return out


def groups4() -> list[dict]:
    """Phase 4: each summary's points with where they cite, and each panel's four answers, matrix and stop decision."""
    out = []
    r = newest("-summaries-heldout.json")
    sets = {d["id"]: d for d in _heldout("summaries")["documents"]}
    for d in json.loads(r.read_text())["documents"]:
        src = sets[d["id"]]
        text = src.get("text") or "\n\n[page break]\n\n".join(src.get("pdf_pages") or [])
        cite = lambda c: ("not verified" if not c else c.get("timecode") or ", ".join(x for x in (c.get("clause") and f"clause {c['clause']}",
                          c.get("page") and f"p. {c['page']}", c.get("line") and f"l. {c['line']}") if x))
        out.append({"group": f"summaries:{d['id']}", "title": src["name"], "documents": [(src["name"], text)],
                    "items": [{"kind": "summary_point", "source": [r.name, f"{d['id']} point {i}"], "title": f"Summary point ({cite(p.get('cite'))})",
                               "body": p["text"], "evidence": [f"“{p['quote']}” · {p['verification']}"] +
                               ([f"figure to check: {', '.join(p['figures_unsupported'])}"] if p.get("figures_unsupported") else [])}
                              for i, p in enumerate(d["points"], start=1)]})
    r = newest("-panel-heldout.json")
    sets = {x["id"]: x for x in _heldout("panel")["scenarios"]}
    for sc in json.loads(r.read_text())["scenarios"]:
        if sc.get("failed"):
            continue
        src = sets[sc["id"]]
        items = [{"kind": "panel_perspective", "source": [r.name, f"{sc['id']} {p['role']}"], "title": f"Perspective: {p['role']}",
                  "body": p["text"], "evidence": [f"cites nothing: {u}" for u in p["unsupported"]] + ([f"says it lacks: {'; '.join(p['missing'])}"] if p["missing"] else [])}
                 for p in sc["perspectives"]]
        items += [{"kind": "panel_issue", "source": [r.name, f"{sc['id']} issue {i}"], "title": f"Moderator: {m['issue']}",
                   "body": "; ".join(f"{k}: {v}" for k, v in m["positions"].items()) + (f" — {m['summary']}" if m.get("summary") else ""),
                   "evidence": ["marked as a disagreement" if m["disagreement"] else "not marked as a disagreement"]}
                  for i, m in enumerate(sc["matrix"], start=1)]
        items.append({"kind": "panel_stop", "source": [r.name, f"{sc['id']} stop"], "title": "Did the panel stop for missing facts?",
                      "body": ("Stopped, asking for: " + "; ".join(sc["missing_facts"])) if sc["stopped"] else "Did not stop.", "evidence": []})
        out.append({"group": f"panel:{sc['id']}", "title": f"Panel: {src['question']} (specialist: {src['specialist']})",
                    "documents": [(d["name"], d["text"]) for d in src["documents"]], "items": items})
    return out


def groups5() -> list[dict]:
    """The pilot additions. Regulatory updates (D-068): every match of a real daily bundle against each practice area,
    for a matter in England & Wales and one in New York; the group's row is for updates that should have matched.
    Translation (D-070): each paragraph of the fictional corpus beside its translation by the model."""
    sys.path.insert(0, str(ROOT / "backend"))
    from app.legal.regulatory import store, topics
    bundle = json.loads((ROOT / "eval/fixtures/law/regulatory/bundle-20261012.json").read_text())
    updates = []
    for it in bundle["items"]:
        try:
            updates.append(store.clean(it, today=__import__("datetime").date(2026, 10, 12)))
        except store.Invalid:
            pass
    listing = "\n".join(f"[{u['jurisdiction']}] {u['kind']} · {u['published']} · {u['title']}" + (f" — {u['summary'][:300]}" if u["summary"] else "")
                         for u in updates)
    out = []
    for keys, terms in topics.AREAS.items():
        area = keys[0].title()
        for where in ("England & Wales", "New York"):
            matter = {"practice_area": area, "jurisdiction": where}
            items = [{"kind": "regulatory_match", "source": ["bundle-20261012.json", u["source"], u["external_id"], area, where],
                      "title": f"{u['kind']} · {u['published']}: {u['title']}", "body": u["summary"] or "(no summary published)",
                      "evidence": [f"matched on: {', '.join(found)}", u["url"]]}
                     for u in updates for found in [topics.match(u, matter)] if found]
            out.append({"group": f"regulatory:{area}:{where}", "title": f"Regulatory updates for a {area.lower()} matter in {where} "
                        f"(terms: {', '.join(terms)})", "documents": [(f"All {len(updates)} updates of 12 October 2026 (real public law)", listing)],
                        "items": items})
    r = newest("-translation-review.json")
    rows = json.loads(r.read_text())["rows"]
    for name in sorted({x["document"] for x in rows}):
        for code in sorted({x["language"] for x in rows}):
            mine = [x for x in rows if x["document"] == name and x["language"] == code]
            out.append({"group": f"translation:{name}:{code}", "title": f"Translation into {code.upper()}: {name} (fictional)",
                        "documents": [(name, (LAW / "corpus" / name).read_text())],
                        "items": [{"kind": "translation_paragraph", "source": [r.name, name, code, x["paragraph"]],
                                   "title": f"Paragraph {x['paragraph'] + 1}", "body": f"ORIGINAL: {x['source']}\n\nTRANSLATION: {x['text']}",
                                   "evidence": ([f"figure flagged: {', '.join(x['figures_missing'])}"] if x["figures_missing"] else []) +
                                               (["text addressed to an AI was withheld"] if x["withheld"] else [])} for x in mine]})
    return out


def phase3(args) -> int:
    groups = groups5() if args.phase == 5 else groups4() if args.phase == 4 else groups3()
    random.Random(args.seed).shuffle(groups)
    out = Path(args.out or HERE / f"phase{args.phase}")
    out.mkdir(exist_ok=True)
    n, key = 0, []
    for gi, g in enumerate(groups, start=1):
        g["id"] = f"G{gi:02d}"
        for it in g["items"]:
            n += 1
            it["id"] = f"R{n:03d}"
            key.append({"id": it["id"], "kind": it["kind"], "group": g["id"], "source": it["source"]})
        key.append({"id": g["id"], "kind": "missed", "group": g["id"], "source": [g["group"]]})
    (out / "key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    with open(out / "sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "group", "kind", "score", "wrong_or_invented", "missed", "notes"])
        for g in groups:
            for it in g["items"]:
                w.writerow([it["id"], g["id"], it["kind"], "", "", "", ""])
            w.writerow([g["id"], g["id"], "missed", "", "", "", ""])
    label = {"draft": "Draft", "playbook_line": "Playbook check", "citation": "Citation check", "definition": "Drafting check",
             "obligation": "Obligation", "diligence_issue": "Diligence issue", "dd_missing": "Missing documents", "governance_issue": "Governance issue",
             "summary_point": "Summary point", "panel_perspective": "Panel perspective", "panel_issue": "Moderator's matrix", "panel_stop": "Panel stop decision",
             "regulatory_match": "Regulatory match", "translation_paragraph": "Translated paragraph"}
    parts = ["<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
             f"<title>Phase {args.phase} review packet</title>",
             "<style>body{font:15px/1.55 Georgia,serif;max-width:860px;margin:40px auto;padding:0 16px;color:#15171C;background:#fff}"
             "h1{font:600 22px system-ui}h2{font:600 18px system-ui;margin-top:36px}section{border-top:1px solid #ddd;padding:12px 0}"
             ".k{font:600 12px system-ui;color:#4A4F59;text-transform:uppercase;letter-spacing:.04em}.e{font:13px system-ui;color:#4A4F59}"
             "pre{white-space:pre-wrap;font:13px/1.5 ui-monospace,monospace;background:#f6f6f4;padding:10px;border-radius:6px}</style>",
             f"<h1>Phase {args.phase} review packet — fictional material</h1><p class=e>Each group shows the documents, then what the system produced on them. "
             f"Score each item in sheet.csv using docs/echomind-law/releases/phase{args.phase}-rubric.md; on the group's own row, list anything the system missed.</p>"]
    for g in groups:
        docs = "".join(f"<details><summary class=e>{html.escape(t)}</summary><pre>{html.escape(x)}</pre></details>" for t, x in g["documents"])
        parts.append(f"<h2>{g['id']} · {html.escape(g['title'])}</h2>{docs}")
        for it in g["items"]:
            ev = "".join(f"<li>{html.escape(e)}</li>" for e in it["evidence"])
            parts.append(f"<section><div class=k>{it['id']} · {label[it['kind']]}</div><h3 style='font-size:16px'>{html.escape(it['title'])}</h3>"
                         f"<p>{html.escape(it['body']).replace(chr(10), '<br>')}</p>{f'<ul class=e>{ev}</ul>' if ev else ''}</section>")
        if not g["items"]:
            parts.append("<section class=e>Nothing was reported on these documents.</section>")
    (out / "packet.html").write_text("\n".join(parts))
    print(f"{n} items in {len(groups)} groups → {out / 'packet.html'}, {out / 'sheet.csv'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", type=int, choices=[2, 3, 4, 5], default=2)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.phase in (3, 4, 5):
        return phase3(args)
    args.out = args.out or str(HERE)
    rows = items2()
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
