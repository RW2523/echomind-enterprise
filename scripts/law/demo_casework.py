#!/usr/bin/env python3
"""Case management for the demo (owner's request 2026-11-07): each demo matter's stage, the updates its people would have
posted and its plan, so the pipeline, the overview and the chat's "what's the latest" have something real to show.
Fictional data only; the updates restate what the demo's own documents say and add only the firm's own steps.

    python3 scripts/law/demo_casework.py --base http://127.0.0.1:8121/api --data-dir <demo data folder>

Runs through the product's API as each matter's lead (the credentials the seed wrote; none is printed), so permissions and
audit apply as for a person. Dates are counted from the day it runs. A matter that already has updates is left alone, so
running it again changes nothing.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.error
from datetime import date, timedelta

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "law"))
from demo_enrich import Api  # noqa: E402


def day(n: int) -> str:
    return (date.today() + timedelta(days=n)).isoformat()


# (matter, lead, track, [(days ago, kind or "stage:<key>", text)], [(title, kind, due in days or None, assign to, done)], standard tasks)
PLANS = [
    ("MAT-1042", "alice", "disputes", [
        (-40, "instruction", "Morgan Company instructs us to pursue Apex Industries for the late delivery and the cost of the replacement parts."),
        (-36, "stage:assessment", "Assessment started on the shipment log, the inspection report and the May emails."),
        (-33, "call", "Call with the client's operations lead: the shipment log and the inspection report are the key records; she is sending the May email thread."),
        (-26, "note", "Reviewed the 14 May amendment that changed the delivery date. The main question is whether Apex's 24 April warning counts as notice under the agreement."),
        (-19, "filing", "Letter before claim sent to Apex Industries, with 14 days to respond."),
        (-12, "stage:pleadings", "Apex did not accept liability; moving to pleadings."),
        (-5, "meeting", "Team meeting: draft particulars of claim to be ready for partner review next week."),
    ], [
        ("Collect the inspection photos from the client", "task", -2, "alice", False),
        ("Draft the particulars of claim", "task", 5, "alice", False),
        ("Partner review of the particulars of claim", "task", 9, None, False),
        ("Diary Apex's deadline to respond to the claim", "filing", 30, "alice", False),
        ("Send the letter before claim", "task", -19, "alice", True),
    ], False),
    ("M-101", "alice", "transactions", [
        (-45, "instruction", "Voss-Lindqvist wants the Harrowgate MSA renewed for two more years, with the service credits reviewed first."),
        (-30, "note", "Reviewed the Q3 2026 service level report: August availability was 98.71%, below the 99.5% target, so a service credit is due for August."),
        (-22, "stage:negotiation", "First renewal draft sent to Harrowgate."),
        (-8, "call", "Harrowgate's counsel asked for the liability cap to stay as it is; the client will decide."),
        (-3, "instruction", "Client accepts the current cap if the August service credit is settled with the renewal."),
    ], [
        ("Confirm the August service credit figure with the client", "task", 2, "alice", False),
        ("Send the revised renewal draft to Harrowgate", "task", 6, "alice", False),
        ("Client sign-off on the renewal terms", "key_date", 14, None, False),
    ], False),
    ("M-114", "alice", "transactions", [
        (-28, "instruction", "Alder Finch instructs us on the Series B investment: a due diligence report is wanted before the term sheet is signed."),
        (-20, "stage:diligence", "Data room opened; the review has started."),
        (-10, "note", "Register of members reviewed; the share classes reconcile with the cap table."),
        (-4, "call", "Founders' counsel will answer the open diligence questions by the end of next week."),
    ], [
        ("Check the change-of-control thresholds against the issued shares", "task", 3, "alice", False),
    ], True),
    ("M-301", "harper", "disputes", [
        (-50, "instruction", "Northwind instructs us to recover service credits and losses from Calder Freight for the late deliveries."),
        (-35, "stage:assessment", "Assessment started on the delivery records and the service level schedule."),
        (-21, "stage:pleadings", "The client approved bringing the claim; drafting it now."),
        (-14, "filing", "Claim issued and served on Calder Freight."),
        (-6, "note", "Service credits recalculated with the 25% monthly cap applied to each month."),
    ], [
        ("Prepare the service credit schedule for the claim", "task", 4, "harper", False),
        ("Diary the deadline for Calder Freight's defence", "filing", 8, "harper", False),
        ("Review Calder Freight's defence with the client", "task", 15, "alice", False),
    ], False),
    ("M-302", "sofia", "disputes", [
        (-25, "instruction", "Brightwater asks for advice on Dr Rahman's grievance about on-call pay for 2025."),
        (-18, "stage:assessment", "Assessing the grievance against the On-Call Allowance Policy and the rota records."),
        (-11, "note", "At the policy rates, 83 nights and 12 weekend days come to GBP 19,340, not the GBP 18,400 claimed; HR's investigation counted 64 nights."),
        (-4, "meeting", "Grievance meeting held with HR; the outcome letter is to be drafted."),
    ], [
        ("Reconcile the rota records with HR's count", "task", -1, "sofia", False),
        ("Draft the grievance outcome letter", "task", 5, "alice", False),
        ("Grievance appeal hearing, if Dr Rahman appeals", "hearing", 21, "sofia", False),
    ], False),
    ("M-303", "kenji", "transactions", [
        (-60, "instruction", "Kestrel Robotics instructs us on its Series A financing."),
        (-48, "stage:diligence", "Investor's diligence requests received; answering them with the company."),
        (-20, "stage:negotiation", "Term sheet agreed; the long-form documents are in negotiation."),
        (-7, "call", "Lead investor's counsel wants broader consent rights; the founders want them limited to major decisions."),
    ], [
        ("Agree the investor consent rights", "task", 4, "kenji", False),
    ], True),
    ("M-304", "sofia", "transactions", [
        (-34, "instruction", "Harbourview Studios wants the Unit 12 lease renewed on similar terms."),
        (-24, "stage:negotiation", "Landlord's draft lease received."),
        (-9, "call", "The landlord's agent proposed a rent review in year three; the client will consider it."),
    ], [
        ("Send comments on the landlord's draft lease", "task", 3, "sofia", False),
        ("Client decision on the rent review", "key_date", 10, None, False),
    ], False),
    ("M-305", "harper", "transactions", [
        (-55, "instruction", "Lumen Software asks us to negotiate its SaaS agreement with Orbis Bank."),
        (-40, "stage:negotiation", "Orbis Bank's mark-up received."),
        (-18, "note", "Service credit regime agreed: a percentage of the monthly fee for each availability band, capped each month."),
        (-6, "stage:signing", "Final form agreed; collecting the signatures."),
    ], [
        ("Collect Orbis Bank's signature", "task", 2, "harper", False),
        ("First quarterly service review with Orbis Bank", "key_date", 60, None, False),
    ], False),
    ("M-306", "kenji", "advisory", [
        (-30, "instruction", "Meridian Foods asks for help with the Authority's information request about allergen labelling."),
        (-26, "stage:research", "Collecting the labelling records and the product recall timeline."),
        (-17, "filing", "Response to the Authority's information request sent, before the 22 October deadline."),
        (-16, "stage:advised", "Advice on the labelling changes given to the client."),
    ], [
        ("Send the response to the information request", "filing", -17, "kenji", True),
        ("Follow up with the Authority on next steps", "task", 6, "kenji", False),
    ], False),
]


def run(api: Api) -> int:
    failures = 0
    for number, lead, track, updates, tasks, standard in PLANS:
        try:
            mid = api.matter(lead, number)
        except (LookupError, urllib.error.HTTPError) as e:
            print(f"skip  {number}: {e}")
            continue
        if api.call(lead, "GET", f"/legal/matters/{mid}/updates")["updates"]:
            print(f"keep  {number}: it already has updates")
            continue
        try:
            people = {m["username"]: m["user_id"] for m in api.call(lead, "GET", f"/legal/matters/{mid}/memberships")["memberships"]}
            matter = api.call(lead, "GET", f"/legal/matters/{mid}")["matter"]
            if matter["track"] != track:                         # put it on its track first, from the day it came in
                matter = api.call(lead, "POST", f"/legal/matters/{mid}/stage", {
                    "track": track, "stage": "intake" if track == "disputes" else "instruction", "version": matter["version"],
                    "effective_on": day(updates[0][0]), "note": "Matter opened."})["matter"]
            for ago, kind, text in updates:
                if kind.startswith("stage:"):
                    matter = api.call(lead, "POST", f"/legal/matters/{mid}/stage", {
                        "stage": kind.split(":", 1)[1], "version": matter["version"], "effective_on": day(ago), "note": text})["matter"]
                else:
                    api.call(lead, "POST", f"/legal/matters/{mid}/updates", {"kind": kind, "body": text, "occurred_on": day(ago)})
            if standard:
                api.call(lead, "POST", f"/legal/matters/{mid}/plan/standard-tasks", {})
            for title, kind, due, who, done in tasks:
                body = {"title": title, "kind": kind, **({"due_on": day(due)} if due is not None else {}),
                        **({"assignee_id": people[who]} if who in people else {})}
                t = api.call(lead, "POST", f"/legal/matters/{mid}/tasks", body)["task"]
                if done:
                    api.call(lead, "PATCH", f"/legal/matters/{mid}/tasks/{t['id']}", {"status": "done", "version": t["version"]})
            print(f"ok    {number}: {matter['stage']} ({track}), {len(updates)} updates, {len(tasks)} tasks")
        except urllib.error.HTTPError as e:
            failures += 1
            print(f"FAIL  {number}: HTTP {e.code} {e.read()[:200]!r}")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data-dir", required=True, help="the demo data folder (its demo_credentials.json)")
    args = ap.parse_args()
    creds = json.loads((pathlib.Path(args.data_dir) / "legal" / "demo_credentials.json").read_text())
    return run(Api(args.base, creds))


if __name__ == "__main__":
    sys.exit(main())
