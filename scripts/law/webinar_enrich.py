#!/usr/bin/env python3
"""The work a firm would already have done on a demo dataset's matters, read from a spec file (the webinar, 2026-11-16).
Fictional data only. Generic: the matters, documents, updates and tasks all come from the spec, nothing from this code.

    python3 scripts/law/webinar_enrich.py --base http://127.0.0.1:8170/api --data-dir <demo data folder> \
        [--spec eval/fixtures/webinar/enrich.json] [--only casework,obligations,...]

Run after scripts/law/demo_seed.sh and scripts/law/synthetic/load.py --dataset <folder>, against a backend on that data
folder. Everything goes through the product's API as the person who would do it (credentials from the seed; none is
printed), so permissions, audit and quote checks apply as for a person. Steps (each reports and goes on if it fails):
casework (stage, updates, plan), obligations, scans (conflict scans), drafts (then draft_fill: the author fills them
in), research (a note sent for review), checks
(playbook, definitions, citation checks and summaries), compare (two versions of one document), diligence, panel (Council),
time, enquiries, chats.

Dates in the spec: an ISO date ("2026-09-21") is used as written, so updates match the documents; a number is days from
today (negative = ago), for the plan's open tasks.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
from datetime import date, timedelta

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "law"))
import demo_casework  # noqa: E402
from demo_enrich import Api, step  # noqa: E402


def when(v) -> str:
    return v if isinstance(v, str) else (date.today() + timedelta(days=int(v))).isoformat()


@step("casework: stages, updates and plans")
def casework(api: Api, spec: dict) -> str:
    plans = []
    for p in spec.get("casework", []):
        updates = [(u[0], u[1], u[2]) for u in p["updates"]]
        tasks = [(t[0], t[1], t[2], t[3], t[4]) for t in p["tasks"]]
        plans.append((p["matter"], p["lead"], p["track"], updates, tasks, p.get("standard", False)))
    demo_casework.PLANS = plans
    demo_casework.day = when            # ISO dates as written, numbers counted from today
    return "done" if demo_casework.run(api) == 0 else "some matters failed (see above)"


@step("obligations")
def obligations(api: Api, spec: dict) -> str:
    out = []
    for user, number, title in spec.get("obligations", {}).get("read", []):
        mid, did = api.matter(user, number), api.doc(user, number, title)
        r = api.call(user, "POST", f"/legal/matters/{mid}/documents/{did}/obligations/extract")
        out.append(f"{number} {r.get('added', '?')}")
    confirm = spec.get("obligations", {}).get("confirm", {})
    for number, n in confirm.items():
        mid = api.matter("alice", number)
        dated = [o for o in api.call("alice", "GET", f"/legal/matters/{mid}/obligations")["obligations"]
                 if o.get("status") == "proposed" and o.get("due_date")][:n]
        for o in dated:
            api.call("alice", "PATCH", f"/legal/matters/{mid}/obligations/{o['id']}", {"expected_version": o["version"], "action": "confirm"})
        out.append(f"confirmed {len(dated)} in {number}")
    return ", ".join(out)


@step("conflict scans")
def scans(api: Api, spec: dict) -> str:
    out = []
    for number in spec.get("scans", []):
        events = api.call("alice", "POST", f"/legal/matters/{api.matter('alice', number)}/compare/scan", {"versions": "signed"})
        done = [e for e in events if isinstance(e, dict) and e.get("type") == "done"] if isinstance(events, list) else []
        out.append(f"{number} {done[-1].get('found', '?') if done else '?'} found")
    return ", ".join(out)


@step("drafts")
def drafts(api: Api, spec: dict) -> str:
    made = []
    for d in spec.get("drafts", []):
        user, number = d["user"], d["matter"]
        mid = api.matter(user, number)
        ids = [api.doc(user, number, t) for t in d.get("documents", [])]
        names = {t["name"]: t["id"] for t in api.call(user, "GET", "/legal/templates")["templates"]}
        tid = next((names[n] for n in d["template"].split("|") if n in names), None)   # "Demand letter|Letter before claim"
        if not tid:
            made.append(f"{number}: no template {d['template']!r}")
            continue
        r = api.call(user, "POST", f"/legal/matters/{mid}/drafts", {"template_id": tid, "title": d["title"], "documents": ids})
        # a draft waits in Drafts for a second lawyer's review (one by carol is what alice approves on stage)
        made.append(f"{number} “{d['title']}” by {user} ({r['draft']['content'].get('gaps', '?')} gaps)")
    draft_fill(api, spec)
    return "; ".join(made)


_UK_DATE = re.compile(r"^(\d{1,2}) ([A-Z][a-z]+) (\d{4})$")


@step("drafts filled in by their author")
def draft_fill(api: Api, spec: dict) -> str:
    """The author fills in what a draft still needs ("fill" in the spec: field key -> value; "passage": the text for a written
    passage the model left with a gap; "fill_reason") before a
    second lawyer can approve it: a new version through the product's own edit, as a person would. A letter dated in the
    other date style than the firm's (made before LEGAL_FIRM_STYLE dated letters) is re-dated in the same edit. Safe to run
    again: fields already holding their value are left alone, and nothing is sent when nothing would change."""
    out = []
    for d in spec.get("drafts", []):
        if not d.get("fill"):
            continue
        user, number = d["user"], d["matter"]
        mid = api.matter(user, number)
        found = next((x for x in api.call(user, "GET", f"/legal/matters/{mid}/drafts")["drafts"] if x["title"] == d["title"]), None)
        if not found:
            out.append(f"{number}: no draft “{d['title']}”")
            continue
        a = api.call(user, "GET", f"/legal/matters/{mid}/artifacts/{found['id']}")["artifact"]
        style = (api.call(user, "GET", "/auth/me").get("features") or {}).get("firm_style", "uk")
        fields = [{"key": f["key"], "value": d["fill"][f["key"]]} for f in a["content"]["fields"]
                  if f["key"] in d["fill"] and f.get("value") != d["fill"][f["key"]]]
        for f in a["content"]["fields"]:
            m = _UK_DATE.match(str(f.get("value") or "")) if f.get("source") == "today" and style == "us" else None
            if m:
                fields.append({"key": f["key"], "value": f"{m.group(2)} {int(m.group(1))}, {m.group(3)}"})
        # a written passage the model left with a gap is rewritten by the author ("passage"), and shows as edited by them
        gapped = next((i for i, b in enumerate(a["content"]["blocks"])
                       if b.get("origin") in ("written", "edited") and "[[NEEDS INPUT" in b["text"]), None)
        blocks = [{"index": gapped, "text": d["passage"]}] if d.get("passage") and gapped is not None else []
        if not fields and not blocks:
            out.append(f"{number} “{d['title']}” already filled ({a['content']['gaps']} gaps)")
            continue
        r = api.call(user, "POST", f"/legal/matters/{mid}/drafts/{found['id']}/edit",
                     {"content_sha256": a["content_sha256"], "fields": fields, "blocks": blocks, "reason": d.get("fill_reason")})
        out.append(f"{number} “{d['title']}” version {r['draft']['version']} by {user}: {len(fields)} fields, {len(blocks)} passage, "
                   f"{r['draft']['content']['gaps']} gaps left")
    return "; ".join(out)


@step("research note sent for review")
def research(api: Api, spec: dict) -> str:
    out = []
    for r in spec.get("research", []):
        mid = api.matter("alice", r["matter"])
        events = api.call("alice", "POST", f"/legal/matters/{mid}/research/query", {"question": r["question"]})
        done = next(e for e in events if e.get("type") == "done")
        if r.get("review", True):
            api.call("alice", "POST", f"/legal/matters/{mid}/research/{done['turn_id']}/review")
        out.append(f"{r['matter']} {done.get('status')} with {len(done.get('citations') or [])} sources")
    return "; ".join(out)


@step("document checks")
def checks(api: Api, spec: dict) -> str:
    out = []
    for kind, number, title in spec.get("checks", []):
        mid, did = api.matter("alice", number), api.doc("alice", number, title)
        if kind == "playbook":
            pb = api.call("alice", "GET", "/legal/playbooks")["playbooks"][0]["id"]
            api.call("alice", "POST", f"/legal/matters/{mid}/documents/{did}/playbook-check", {"playbook_id": pb})
        elif kind == "summary":
            api.call("alice", "POST", f"/legal/matters/{mid}/documents/{did}/summaries")
        else:                                     # definitions-check, citation-check
            api.call("alice", "POST", f"/legal/matters/{mid}/documents/{did}/{kind}-check", {})
        out.append(f"{kind} {number}")
    return ", ".join(out)


@step("version comparison")
def compare(api: Api, spec: dict) -> str:
    out = []
    for number, title in spec.get("compare", []):
        mid, did = api.matter("alice", number), api.doc("alice", number, title)
        vs = sorted(api.call("alice", "GET", f"/legal/matters/{mid}/documents/{did}/versions")["versions"], key=lambda v: v["number"])
        if len(vs) < 2:
            out.append(f"{number} {title}: one version only")
            continue
        api.call("alice", "POST", f"/legal/matters/{mid}/compare", {"left_version_id": vs[0]["id"], "right_version_id": vs[-1]["id"]})
        out.append(f"{number} v{vs[0]['number']} → v{vs[-1]['number']}")
    return ", ".join(out)


@step("due diligence")
def diligence(api: Api, spec: dict) -> str:
    lists = {c["name"]: c["id"] for c in api.call("alice", "GET", "/legal/diligence/checklists")["checklists"]}
    out = []
    for number, name in spec.get("diligence", []):
        if name not in lists:
            out.append(f"{number}: no checklist {name!r} (have {', '.join(lists)})")
            continue
        r = api.call("alice", "POST", f"/legal/matters/{api.matter('alice', number)}/diligence/runs", {"checklist_id": lists[name]})
        out.append(f"{number} {name}: {(r.get('run') or r).get('status', 'started')}")
    return "; ".join(out)


@step("review panel (Council)")
def panel(api: Api, spec: dict) -> str:
    out = []
    for p in spec.get("panel", []):
        mid = api.matter("alice", p["matter"])
        r = api.call("alice", "POST", f"/legal/matters/{mid}/panel", {"question": p["question"], "specialist": p.get("specialist", "commercial")})
        rid, status = r["review"]["id"], r["review"]["status"]
        for _ in range(90):
            review = api.call("alice", "GET", f"/legal/matters/{mid}/panel/{rid}")["review"]
            status = review["status"]
            if status in ("needs_facts", "questions", "awaiting_facts"):
                api.call("alice", "POST", f"/legal/matters/{mid}/panel/{rid}/proceed",
                         {"reason": "Proceed on the documents as they stand; the client confirms nothing further for now."})
            elif status not in ("queued", "running", "working"):
                break
            time.sleep(10)
        out.append(f"{p['matter']} {status}")
    return "; ".join(out)


@step("time suggestions")
def time_entries(api: Api, spec: dict) -> str:
    r = api.call("alice", "POST", "/legal/time/suggest", {})
    return f"{len(r.get('entries') or r.get('suggestions') or [])} suggested"


@step("enquiries")
def enquiries(api: Api, spec: dict) -> str:
    out = []
    for e in spec.get("enquiries", []):
        iid = api.call("alice", "POST", "/legal/intake", {"channel": e["channel"]})["intake"]["id"]
        api.call("alice", "POST", f"/legal/intake/{iid}/disclaimer", {"version": "intake-disclaimer-1"})
        v = api.call("alice", "POST", f"/legal/intake/{iid}/submit", {"answers": e["answers"]})
        out.append(f"{e['answers']['your_name']}: {v['intake']['screening']['result']}")
    return "; ".join(out)


@step("chats")
def chats(api: Api, spec: dict) -> str:
    out = []
    for number, questions in spec.get("chats", []):
        conv = api.call("alice", "POST", "/legal/counsel/conversations", {"matter_id": api.matter("alice", number)})["conversation"]["id"]
        kinds = []
        for q in questions:
            events = api.call("alice", "POST", f"/legal/counsel/conversations/{conv}/turns", {"text": q})
            final = next((e["data"] for e in events if e.get("event") == "ui.final"), None)
            kinds.append(",".join(c["type"] for c in (final or {}).get("components", [])) or "failed")
        out.append(f"{number} [{'; '.join(kinds)}]")
    return " ".join(out)


STEPS = {"casework": casework, "obligations": obligations, "scans": scans, "drafts": drafts, "draft_fill": draft_fill, "research": research, "checks": checks,
         "compare": compare, "diligence": diligence, "panel": panel, "time": time_entries, "enquiries": enquiries, "chats": chats}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--data-dir", required=True, help="the demo data folder (its legal/demo_credentials.json)")
    ap.add_argument("--spec", default=str(ROOT / "eval" / "fixtures" / "webinar" / "enrich.json"))
    ap.add_argument("--only", default="", help="comma-separated steps (default: all, in order)")
    args = ap.parse_args()
    creds = json.loads((pathlib.Path(args.data_dir) / "legal" / "demo_credentials.json").read_text())
    spec = json.loads(pathlib.Path(args.spec).read_text())
    api = Api(args.base, creds)
    for name in [s.strip() for s in args.only.split(",") if s.strip()] or list(STEPS):
        STEPS[name](api, spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
