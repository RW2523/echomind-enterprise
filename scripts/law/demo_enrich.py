#!/usr/bin/env python3
"""The work a firm would already have done on the demo's documents, so every feature of EchoMind Legal can be seen and
used end to end on the public demo (owner's request 2026-11-02). Fictional data only.

    python3 scripts/law/demo_enrich.py --base http://127.0.0.1:8121/api --data-dir <demo data folder>

Run after scripts/law/demo_seed.sh and scripts/law/demo_load.py, against a backend on that folder. Everything goes
through the product's own API as the lawyer who would do it (signing in with the credentials the seed wrote; none is
printed), so permissions, audit and quote checks apply exactly as for a person:
- alice (the demo's one-click visitor) joins the six practice cases as an attorney; the restricted matters stay closed;
- obligations read from the main agreements, two confirmed with their dates; conflict scans on the dispute matters;
- drafts from firm templates (one by carol waiting for alice's approval) and a research note sent for review;
- a playbook check, a citation check, a definitions check, two summaries, a version comparison, two due-diligence runs,
  a review-panel question;
- time suggestions from the day's work; two enquiries (one naming an adverse party, so screening flags it);
- a few chats, so the chat history shows answers with their sources, a comparison card, a chronology and a briefing;
- each matter's stage, updates and plan (demo_casework.py).
Each step reports what it did and goes on if it fails; re-running adds more of the same work.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "law"))
from law_client import open_session  # noqa: E402

PRACTICE = {"M-301": "harper", "M-302": "sofia", "M-303": "kenji", "M-304": "sofia", "M-305": "harper", "M-306": "kenji"}


class Api:
    def __init__(self, base: str, creds: dict):
        self.base, self.creds, self.sessions, self.matters = base, creds, {}, {}

    def _s(self, user: str):
        if user not in self.sessions:
            os.environ["LAW_TOTP_SECRET"] = self.creds[user]["totp_secret"]
            self.sessions[user] = open_session(self.base, user, self.creds[user]["password"])
            os.environ.pop("LAW_TOTP_SECRET", None)
        return self.sessions[user]

    def call(self, user: str, method: str, path: str, body=None, timeout: int = 600):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data is not None else {})
        with self._s(user).open(req, timeout=timeout) as r:
            raw = r.read()
        text = raw.decode("utf-8", "replace")
        if "\n" in text.strip() and text.lstrip().startswith("{"):          # NDJSON: every event, in order
            return [json.loads(x) for x in text.splitlines() if x.strip()]
        return json.loads(text or "{}")

    def matter(self, user: str, number: str) -> str:
        key = (user, number)
        if key not in self.matters:
            self.matters[key] = {m["number"]: m["id"] for m in self.call(user, "GET", "/legal/matters")["matters"]}.get(number)
        if not self.matters[key]:
            raise LookupError(f"{user} cannot see {number}")
        return self.matters[key]

    def doc(self, user: str, number: str, title: str) -> str:
        mid = self.matter(user, number)
        for d in self.call(user, "GET", f"/legal/matters/{mid}/documents")["documents"]:
            if d["title"] == title:
                return d["id"]
        raise LookupError(f"no document {title!r} in {number}")


def step(name: str):
    def wrap(fn):
        def run(*a, **k):
            t = time.time()
            try:
                said = fn(*a, **k)
                print(f"ok    {name}: {said} ({time.time() - t:.0f} s)", flush=True)
            except urllib.error.HTTPError as e:
                print(f"FAIL  {name}: HTTP {e.code} {e.read()[:200]!r}", flush=True)
            except Exception as e:                                          # noqa: BLE001 — report and go on
                print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return run
    return wrap


@step("alice joins the practice cases")
def join(api: Api) -> str:
    done = []
    for number, lead in PRACTICE.items():
        mid = api.matter(lead, number)
        have = {m["username"] for m in api.call(lead, "GET", f"/legal/matters/{mid}/memberships")["memberships"]}
        if "alice" not in have:
            api.call(lead, "POST", f"/legal/matters/{mid}/memberships", {"username": "alice", "role": "attorney"})
            done.append(number)
    return f"added to {', '.join(done) or 'none (already a member)'}"


@step("obligations")
def obligations(api: Api) -> str:
    out = []
    for user, number, title in [("alice", "M-101", "Master Services Agreement (DEMO)"), ("alice", "MAT-1042", "Services Agreement (DEMO)"),
                                ("alice", "MAT-1042", "Amendment No. 1 (DEMO)"), ("alice", "M-301", "01 Logistics Services Agreement (DEMO)"),
                                ("alice", "M-304", "01 Lease of Unit 12 Riverside Works (DEMO)"),
                                ("alice", "M-305", "01 SaaS Agreement (Lumen draft) (DEMO)")]:
        mid, did = api.matter(user, number), api.doc(user, number, title)
        r = api.call(user, "POST", f"/legal/matters/{mid}/documents/{did}/obligations/extract")
        out.append(f"{number} {r.get('added', '?')}")
    mid = api.matter("alice", "M-101")
    dated = [o for o in api.call("alice", "GET", f"/legal/matters/{mid}/obligations")["obligations"]
             if o.get("status") == "proposed" and o.get("due_date")][:2]
    for o in dated:
        api.call("alice", "PATCH", f"/legal/matters/{mid}/obligations/{o['id']}", {"expected_version": o["version"], "action": "confirm"})
    return f"read {', '.join(out)}; confirmed {len(dated)} in M-101"


@step("conflict scans")
def scans(api: Api) -> str:
    out = []
    for number in ("MAT-1042", "M-301", "M-101", "M-114"):
        events = api.call("alice", "POST", f"/legal/matters/{api.matter('alice', number)}/compare/scan", {"versions": "signed"})
        done = [e for e in events if isinstance(e, dict) and e.get("type") == "done"] if isinstance(events, list) else []
        out.append(f"{number} {done[-1].get('found', '?') if done else '?'} found")
    return ", ".join(out)


def _template(api: Api, user: str, name: str) -> str:
    return next(t["id"] for t in api.call(user, "GET", "/legal/templates")["templates"] if t["name"] == name)


@step("drafts")
def drafts(api: Api) -> str:
    made = []
    for user, number, template, title, docs in [
        ("carol", "M-101", "Engagement letter", "Engagement letter — Harrowgate renewal", ["Engagement request (DEMO)"]),
        ("alice", "MAT-1042", "Letter before claim", "Letter before claim to Apex Industries about the late delivery",
         ["Services Agreement (DEMO)", "Amendment No. 1 (DEMO)", "Shipment log (DEMO)"]),
        ("alice", "M-101", "Internal memorandum", "Memo: liability cap options for the renewal",
         ["Master Services Agreement (DEMO)", "Advice memo on the MSA (DEMO)"]),
    ]:
        mid = api.matter(user, number)
        ids = [api.doc(user, number, d) for d in docs]
        r = api.call(user, "POST", f"/legal/matters/{mid}/drafts", {"template_id": _template(api, user, template), "title": title, "documents": ids})
        made.append(f"{number} “{title}” ({r['draft']['content']['gaps']} gaps)")
    return "; ".join(made)


@step("research note sent for review")
def research_note(api: Api) -> str:
    mid = api.matter("alice", "M-101")
    events = api.call("alice", "POST", f"/legal/matters/{mid}/research/query",
                      {"question": "What is each party's liability cap under the Master Services Agreement, and what is excluded?"})
    done = next(e for e in events if e.get("type") == "done")
    api.call("alice", "POST", f"/legal/matters/{mid}/research/{done['turn_id']}/review")
    return f"{done.get('status')} with {len(done.get('citations') or [])} sources"


@step("document checks")
def checks(api: Api) -> str:
    out = []
    mid = api.matter("alice", "M-101")
    pb = api.call("alice", "GET", "/legal/playbooks")["playbooks"][0]["id"]
    api.call("alice", "POST", f"/legal/matters/{mid}/documents/{api.doc('alice', 'M-101', 'Master Services Agreement (DEMO)')}/playbook-check",
             {"playbook_id": pb})
    out.append("playbook M-101 MSA")
    mid = api.matter("alice", "MAT-1042")
    api.call("alice", "POST", f"/legal/matters/{mid}/documents/{api.doc('alice', 'MAT-1042', 'Research memo with an unverified citation (DEMO)')}/citation-check", {})
    out.append("citations MAT-1042 memo")
    mid = api.matter("alice", "M-114")
    api.call("alice", "POST", f"/legal/matters/{mid}/documents/{api.doc('alice', 'M-114', 'Supply Agreement (DEMO)')}/definitions-check", {})
    out.append("definitions M-114 supply agreement (planted errors)")
    for number, title in [("M-101", "Consultancy Agreement (DEMO)"), ("MAT-1042", "Services Agreement (DEMO)")]:
        mid = api.matter("alice", number)
        api.call("alice", "POST", f"/legal/matters/{mid}/documents/{api.doc('alice', number, title)}/summaries")
        out.append(f"summary {number}")
    return ", ".join(out)


@step("version comparison")
def compare(api: Api) -> str:
    mid = api.matter("alice", "M-130")
    did = api.doc("alice", "M-130", "Harrowgate MSA")
    vs = sorted(api.call("alice", "GET", f"/legal/matters/{mid}/documents/{did}/versions")["versions"], key=lambda v: v["number"])
    r = api.call("alice", "POST", f"/legal/matters/{mid}/compare", {"left_version_id": vs[0]["id"], "right_version_id": vs[-1]["id"]})
    return f"Harrowgate MSA v{vs[0]['number']} → v{vs[-1]['number']}: {json.dumps(r)[:120]}"


@step("due diligence")
def diligence(api: Api) -> str:
    mid = api.matter("alice", "M-114")
    lists = {c["name"]: c["id"] for c in api.call("alice", "GET", "/legal/diligence/checklists")["checklists"]}
    out = []
    for name in ("Share acquisition — core documents", "Corporate governance review"):
        if name in lists:
            r = api.call("alice", "POST", f"/legal/matters/{mid}/diligence/runs", {"checklist_id": lists[name]})
            out.append(f"{name}: {(r.get('run') or r).get('status', 'started')}")
    return "; ".join(out)


@step("review panel")
def panel(api: Api) -> str:
    mid = api.matter("alice", "M-101")
    r = api.call("alice", "POST", f"/legal/matters/{mid}/panel", {
        "question": "Should our client accept Harrowgate's request to cap each party's liability at the fees of the previous twelve months "
                    "in the renewal, given what the current agreement and the advice memo say?", "specialist": "commercial"})
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
    return f"status {status}"


@step("time suggestions")
def time_entries(api: Api) -> str:
    r = api.call("alice", "POST", "/legal/time/suggest", {})
    return f"{len(r.get('entries') or r.get('suggestions') or [])} suggested"


@step("enquiries")
def enquiries(api: Api) -> str:
    out = []
    for channel, answers in [
        ("phone", {"your_name": "Dana Whitfield", "organisation": "Whitfield Bakeries Ltd (fictional)", "email": "dana.whitfield@example.invalid",
                   "matter_type": "commercial", "other_parties": ["Granary Mills Ltd (fictional)"],
                   "summary": "We want a supply agreement reviewed before signing next month; the supplier's draft caps their liability very low.",
                   "heard_from": "referral"}),
        ("in_person", {"your_name": "Marco Ellison", "organisation": "Ellison Data Centres (fictional)", "phone": "+44 20 7946 0000",
                       "matter_type": "dispute",
                       "other_parties": ["Harrowgate Systems Inc."],
                       "summary": "A dispute with our platform provider about missed service levels and service credits.",
                       "heard_from": "website"}),
    ]:
        iid = api.call("alice", "POST", "/legal/intake", {"channel": channel})["intake"]["id"]
        api.call("alice", "POST", f"/legal/intake/{iid}/disclaimer", {"version": "intake-disclaimer-1"})
        v = api.call("alice", "POST", f"/legal/intake/{iid}/submit", {"answers": answers})
        out.append(f"{answers['your_name']}: {v['intake']['screening']['result']}")
    return "; ".join(out)


CHATS = [
    ("M-101", ["What needs my attention today?",
               "When does the initial term of the Master Services Agreement end, and when is the last day to give notice of non-renewal?",
               "What is each party's liability cap?"]),
    ("MAT-1042", ["Find conflicting dates or amounts", "Could these refer to different events?", "Which contractual deadline applied?"]),
    ("MAT-1042", ["Build a chronology of events"]),
    ("M-301", ["What did Calder Freight say in response to the notice of claim?", "Chart the amounts in the documents"]),
    ("M-114", ["Who are the parties to the supply agreement with Northwind, and what does each have to do?"]),
]


@step("chats")
def chats(api: Api) -> str:
    out = []
    for number, questions in CHATS:
        conv = api.call("alice", "POST", "/legal/counsel/conversations", {"matter_id": api.matter("alice", number)})["conversation"]["id"]
        kinds = []
        for q in questions:
            events = api.call("alice", "POST", f"/legal/counsel/conversations/{conv}/turns", {"text": q})
            final = next((e["data"] for e in events if e.get("event") == "ui.final"), None)
            kinds.append(",".join(c["type"] for c in (final or {}).get("components", [])) or "failed")
        out.append(f"{number} [{'; '.join(kinds)}]")
    return " ".join(out)


@step("case management: stages, updates and plans")
def casework(api: Api) -> str:
    import demo_casework                                       # noqa: PLC0415 — it imports Api from here
    return "done" if demo_casework.run(api) == 0 else "some matters failed (see above)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data-dir", required=True, help="the demo data folder (its demo_credentials.json)")
    ap.add_argument("--only", default="", help="comma-separated steps (default: all)")
    args = ap.parse_args()
    creds = json.loads((pathlib.Path(args.data_dir) / "legal" / "demo_credentials.json").read_text())
    api = Api(args.base, creds)
    steps = {"join": join, "obligations": obligations, "scans": scans, "drafts": drafts, "research": research_note,
             "checks": checks, "compare": compare, "diligence": diligence, "panel": panel, "time": time_entries, "enquiries": enquiries, "chats": chats,
             "casework": casework}
    wanted = [s.strip() for s in args.only.split(",") if s.strip()] or list(steps)
    for name in wanted:
        steps[name](api)
    return 0


if __name__ == "__main__":
    sys.exit(main())
