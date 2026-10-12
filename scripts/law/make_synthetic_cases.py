#!/usr/bin/env python3
"""Write the FICTIONAL demo cases' documents with the local model (D-064), from fixed case briefs.

    cd backend && LLM_BASE_URL=http://localhost:8355/v1 python3 ../scripts/law/make_synthetic_cases.py [--only M-301]

Each case has a brief: parties, dates, figures and what each document must say, including deliberate
disagreements between documents (for Compare, Silent Assist and the review panel to find). The model writes each
document from the brief alone; every figure and date in the brief must appear exactly, or the document is written
again. The results are committed under eval/fixtures/law/synthetic/, so the demo is the same every time, with
manifest.json giving the matters, the practice group and each document's kind and status. Every name is invented.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "eval" / "fixtures" / "law" / "synthetic"
sys.path.insert(0, str(ROOT / "backend"))

# The practice group: fictional people, firm roles and a line for the sign-in helper.
TEAM = {
    "harper": (["attorney"], "senior associate, disputes"),
    "sofia": (["attorney"], "associate, employment and real estate"),
    "kenji": (["attorney"], "associate, corporate and regulatory"),
    "lena": (["attorney"], "trainee solicitor"),
    "omar": (["paralegal"], "paralegal, disputes"),
    "theo": (["reviewer"], "knowledge lawyer (reviews drafts)"),
}

CASES = [
    {
        "number": "M-301", "title": "Northwind v Calder Freight — late delivery claim", "client": ("Northwind Logistics Ltd", "NWL"),
        "practice_area": "Commercial litigation", "jurisdiction": "England & Wales", "lead": "harper",
        "members": {"lena": "attorney", "omar": "paralegal", "nakamura": "reviewer"},
        "parties": [("Calder Freight Ltd", "adverse")],
        "brief": ("Northwind Logistics Ltd (the Customer) contracted Calder Freight Ltd (the Carrier) to deliver goods from "
                  "Felixstowe to Northwind's three warehouses. The Logistics Services Agreement was signed on 3 February 2025 "
                  "for three years. Clause 6.1: on-time delivery KPI of 98% per calendar month. Clause 6.3: service credits of 5% "
                  "of the monthly fee for each full percentage point below the KPI, capped at 25% of the monthly fee. The monthly "
                  "fee is GBP 140,000. Clause 11.2: the Carrier's total liability in any contract year is capped at GBP 400,000. "
                  "Clause 11.4: no liability for loss of profit. Clause 12.1: the Customer must notify any claim within 30 days "
                  "after the end of the month in which the failure occurred. Clause 14: force majeure covers port closure but "
                  "expressly not congestion or labour shortages at the Carrier's subcontractors. Clause 17: payment within 45 days. "
                  "Clause 20: governed by the law of England and Wales; disputes to be negotiated for 20 business days, then the "
                  "High Court. In April, May and June 2026 on-time delivery was 91.5%, 93% and 95%. Northwind sent a notice of "
                  "claim on 28 July 2026 for service credits of GBP 112,000 and lost sales of GBP 260,000. Calder replied on "
                  "12 August 2026 relying on force majeure (port congestion at Felixstowe), the loss-of-profit exclusion and late "
                  "notice for April and May."),
        "documents": [
            ("01", "Logistics Services Agreement", "contract", "executed",
             "the signed Logistics Services Agreement with clauses 1 to 21, including clauses 6, 11, 12, 14, 17 and 20 as in the "
             "brief, signature blocks for both companies dated 3 February 2025",
             ["98%", "5%", "25%", "GBP 140,000", "GBP 400,000", "30 days", "45 days", "3 February 2025"]),
            ("02", "Email chain on missed deliveries", "correspondence", "received",
             "an email chain from 2 July to 9 July 2026 between Northwind's operations manager Priya Nandakumar and Calder's "
             "account manager Gareth Lowe about on-time delivery of 91.5%, 93% and 95% in April, May and June 2026 and 14 missed "
             "delivery slots; Gareth blames congestion at Felixstowe",
             ["91.5%", "93%", "95%"]),
            ("03", "Notice of claim", "correspondence", "final",
             "Northwind's notice of claim dated 28 July 2026 under clause 12.1, claiming service credits of GBP 112,000 and lost "
             "sales of GBP 260,000",
             ["28 July 2026", "GBP 112,000", "GBP 260,000"]),
            ("04", "Calder Freight response", "correspondence", "received",
             "Calder Freight's letter of 12 August 2026 rejecting the claim: force majeure under clause 14 for port congestion, "
             "the loss-of-profit exclusion in clause 11.4, and notice for April and May given after the 30 days in clause 12.1",
             ["12 August 2026", "clause 14", "clause 11.4"]),
            ("05", "Internal note of call with operations", "note", "final",
             "an internal file note by trainee Lena Brandt of a call on 14 August 2026 with Priya Nandakumar about which "
             "deliveries failed and why, and that Calder's own subcontractor was short of drivers in May",
             ["14 August 2026"]),
        ],
    },
    {
        "number": "M-302", "title": "Brightwater Clinics — on-call pay grievance", "client": ("Brightwater Clinics Ltd", "BWC"),
        "practice_area": "Employment", "jurisdiction": "England & Wales", "lead": "sofia",
        "members": {"lena": "attorney", "theo": "reviewer"},
        "parties": [("Dr Amir Rahman", "adverse")],
        "brief": ("Brightwater Clinics Ltd employs Dr Amir Rahman as a consultant radiologist under a contract dated "
                  "1 September 2022. Clause 4: salary GBP 118,000 a year. Clause 7: on-call duties paid under the On-Call "
                  "Allowance Policy. Clause 12: three months' notice by either side. Clause 15: a six-month restriction on working "
                  "for a competing clinic within 10 miles. The On-Call Allowance Policy (version 3, 1 January 2024) pays GBP 220 "
                  "for each night on call and GBP 90 for each weekend day. On 19 June 2026 Dr Rahman raised a grievance: unpaid "
                  "on-call allowance of GBP 18,400 for 2025 and a rota change in March 2026 made without consultation. HR's "
                  "investigation notes record 64 unpaid nights (which at GBP 220 is GBP 14,080, not GBP 18,400) and that the rota "
                  "change was emailed to staff on 2 March 2026."),
        "documents": [
            ("01", "Employment contract — Dr A. Rahman", "contract", "executed",
             "the signed contract of employment with clauses 1 to 18, including clauses 4, 7, 12 and 15 as in the brief",
             ["GBP 118,000", "three months", "six-month", "10 miles", "1 September 2022"]),
            ("02", "On-Call Allowance Policy v3", "other", "final",
             "the firm's on-call allowance policy, version 3, effective 1 January 2024, with rates, how claims are made monthly "
             "and how disputes are handled",
             ["GBP 220", "GBP 90", "1 January 2024"]),
            ("03", "Grievance letter", "correspondence", "received",
             "Dr Rahman's grievance letter of 19 June 2026 claiming GBP 18,400 of unpaid on-call allowance for 2025 and "
             "objecting to the rota change in March 2026",
             ["19 June 2026", "GBP 18,400"]),
            ("04", "HR investigation notes", "note", "final",
             "HR's investigation notes recording 64 unpaid nights in 2025, the calculation at the policy rate (GBP 14,080) and "
             "the email of 2 March 2026 announcing the rota change",
             ["64", "GBP 14,080", "2 March 2026"]),
        ],
    },
    {
        "number": "M-303", "title": "Kestrel Robotics — Series A financing", "client": ("Kestrel Robotics, Inc.", "KES"),
        "practice_area": "Corporate / venture", "jurisdiction": "Delaware", "lead": "kenji",
        "members": {"harper": "attorney", "omar": "paralegal", "morgan": "reviewer"},
        "parties": [("Aldermoor Ventures LP", "adverse")],
        "brief": ("Kestrel Robotics, Inc., a Delaware corporation, is raising a Series A of USD 8,000,000 led by Aldermoor "
                  "Ventures LP at a pre-money valuation of USD 24,000,000. Term sheet of 4 September 2026: 1x non-participating "
                  "liquidation preference; board of five (two founders, one Aldermoor designee, two independents); protective "
                  "provisions for the Series A majority; the option pool to be topped up to 12% of the post-money capitalisation "
                  "before closing; pro rata rights; closing targeted for 30 October 2026. Capitalisation before the round: "
                  "founders 70%, angel investors 15%, option pool 15% (of which 9% granted). The draft Stock Purchase Agreement "
                  "of 20 September 2026 (Aldermoor's counsel draft) says the option pool top-up is to 10% (inconsistent with the "
                  "term sheet) and puts closing on 15 November 2026."),
        "documents": [
            ("01", "Series A term sheet", "contract", "final",
             "the signed non-binding term sheet of 4 September 2026 with the economic and control terms in the brief",
             ["USD 8,000,000", "USD 24,000,000", "12%", "30 October 2026", "4 September 2026"]),
            ("02", "Draft Stock Purchase Agreement (investor draft)", "contract", "draft",
             "an excerpt of Aldermoor's counsel's draft Series A Stock Purchase Agreement of 20 September 2026: purchase price, "
             "closing conditions (including the option pool top-up to 10%), closing on 15 November 2026, representations",
             ["10%", "15 November 2026", "20 September 2026", "USD 8,000,000"]),
            ("03", "Capitalisation table summary", "other", "final",
             "a capitalisation table summary before the round: founders 70%, angels 15%, option pool 15% of which 9% granted",
             ["70%", "15%", "9%"]),
            ("04", "Board consent approving the financing (draft)", "other", "draft",
             "a draft unanimous written consent of the board approving the Series A on the term sheet's terms",
             ["USD 8,000,000", "12%"]),
        ],
    },
    {
        "number": "M-304", "title": "Harbourview Studios — Unit 12 lease renewal", "client": ("Harbourview Studios Ltd", "HBV"),
        "practice_area": "Real estate", "jurisdiction": "England & Wales", "lead": "sofia",
        "members": {"lena": "attorney", "omar": "paralegal"},
        "parties": [("Pembroke Estates Ltd", "adverse")],
        "brief": ("Harbourview Studios Ltd is the tenant of Unit 12, Riverside Works, under a lease from Pembroke Estates Ltd "
                  "dated 24 June 2016 for 15 years. Clause 3.1: rent GBP 96,000 a year, payable quarterly. Clause 4: upward-only "
                  "open-market rent review on 24 June 2026. Clause 8: tenant's full repairing and insuring obligations. "
                  "Clause 9: tenant's break on 24 June 2027 on at least six months' written notice. Pembroke's rent review "
                  "proposal of 1 May 2026 asks for GBP 128,000 a year. The tenant's surveyor's note of 3 June 2026 gives "
                  "comparable evidence for GBP 108,000 to GBP 112,000. Draft heads of terms for a new 10-year lease propose "
                  "GBP 110,000 with a rent-free period of 6 months."),
        "documents": [
            ("01", "Lease of Unit 12 Riverside Works", "contract", "executed",
             "the lease dated 24 June 2016 with clauses 1 to 12, including clauses 3.1, 4, 8 and 9 as in the brief",
             ["GBP 96,000", "24 June 2026", "24 June 2027", "six months", "24 June 2016"]),
            ("02", "Landlord rent review proposal", "correspondence", "received",
             "Pembroke Estates' letter of 1 May 2026 proposing a reviewed rent of GBP 128,000 a year",
             ["1 May 2026", "GBP 128,000"]),
            ("03", "Surveyor's comparables note", "memo", "final",
             "the tenant's surveyor's note of 3 June 2026 with four comparable lettings supporting GBP 108,000 to GBP 112,000",
             ["3 June 2026", "GBP 108,000", "GBP 112,000"]),
            ("04", "Heads of terms for a new lease (draft)", "contract", "draft",
             "draft heads of terms for a new 10-year lease at GBP 110,000 a year with a 6-month rent-free period",
             ["GBP 110,000", "10-year", "6-month"]),
        ],
    },
    {
        "number": "M-305", "title": "Lumen Software — SaaS agreement with Orbis Bank", "client": ("Lumen Software Ltd", "LUM"),
        "practice_area": "Technology & data", "jurisdiction": "England & Wales", "lead": "harper",
        "members": {"kenji": "attorney", "theo": "reviewer", "nakamura": "reviewer"},
        "parties": [("Orbis Bank plc", "adverse")],
        "brief": ("Lumen Software Ltd supplies its Ledgerline platform to Orbis Bank plc. Lumen's draft SaaS Agreement of "
                  "8 September 2026: subscription fees GBP 360,000 a year; availability of 99.9% a month with service credits up "
                  "to 10% of the monthly fee; each party's liability capped at the fees paid in the previous 12 months; data "
                  "breach notification within 72 hours under the Data Processing Addendum. Orbis's mark-up email of "
                  "22 September 2026 asks for 99.95% availability, breach notification within 24 hours, and unlimited liability "
                  "for data protection breaches. The security schedule requires annual penetration tests and ISO 27001 "
                  "certification."),
        "documents": [
            ("01", "SaaS Agreement (Lumen draft)", "contract", "draft",
             "Lumen's draft SaaS Agreement of 8 September 2026 with clauses on services, fees, service levels, data, liability "
             "and term",
             ["GBP 360,000", "99.9%", "10%", "12 months", "8 September 2026"]),
            ("02", "Data Processing Addendum (Lumen draft)", "contract", "draft",
             "the draft Data Processing Addendum: processing purposes, sub-processors, breach notification within 72 hours, "
             "audits",
             ["72 hours"]),
            ("03", "Orbis mark-up email", "correspondence", "received",
             "Orbis Bank's procurement lead's email of 22 September 2026 with their requested changes",
             ["22 September 2026", "99.95%", "24 hours"]),
            ("04", "Security schedule", "contract", "draft",
             "the security schedule: ISO 27001, annual penetration tests, encryption, access control, incident response",
             ["ISO 27001"]),
        ],
    },
    {
        "number": "M-306", "title": "Meridian Foods — allergen labelling inquiry", "client": ("Meridian Foods plc", "MER"),
        "practice_area": "Regulatory", "jurisdiction": "England & Wales", "lead": "kenji",
        "members": {"sofia": "attorney", "omar": "paralegal", "morgan": "reviewer"},
        "parties": [("Food Labelling Authority", "related")],
        "brief": ("The Food Labelling Authority (a fictional regulator) wrote to Meridian Foods plc on 1 October 2026 about its "
                  "Harvest Seeded Loaf: a consumer reported a reaction to sesame, which the label does not list. The Authority "
                  "requests, within 21 days, the product specification, the supplier's allergen declaration, labelling approvals "
                  "and complaint records for the last 12 months. The product specification (version 5, 12 March 2026) lists "
                  "sesame as a 'may contain' trace from the bakery line, but the printed label omits it. An internal compliance "
                  "memo of 6 October 2026 recommends a precautionary recall of batches 4471 to 4479 and notes three earlier "
                  "complaints."),
        "documents": [
            ("01", "Authority information request", "correspondence", "received",
             "the Food Labelling Authority's letter of 1 October 2026 with its numbered requests and the 21-day deadline",
             ["1 October 2026", "21 days", "12 months"]),
            ("02", "Product specification v5 — Harvest Seeded Loaf", "other", "final",
             "the product specification, version 5 of 12 March 2026, including ingredients, the sesame 'may contain' statement "
             "and label copy that omits it",
             ["12 March 2026", "sesame"]),
            ("03", "Internal compliance memo", "memo", "final",
             "the compliance team's memo of 6 October 2026 recommending a precautionary recall of batches 4471 to 4479 and noting "
             "three earlier complaints",
             ["6 October 2026", "4471", "4479"]),
            ("04", "Draft response to the Authority", "correspondence", "draft",
             "a draft response from the client's lawyers answering each request and offering a corrective action plan",
             ["21 days"]),
        ],
    },
]

SYSTEM = ("You write realistic but entirely FICTIONAL legal documents for a law-firm software demonstration. Use only the "
          "facts in the brief; invent nothing that contradicts it; every company and person is fictional. Write the document "
          "itself in plain Markdown (a title line, numbered clauses or paragraphs where a real one would have them, dates, "
          "signature or sign-off blocks). No commentary, no notes to the reader, no placeholders in square brackets. "
          "Keep every figure and date exactly as the brief writes it.")


def _prompt(case: dict, doc: tuple) -> list[dict]:
    num, title, kind, status, what, _ = doc
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Case brief:\n{case['brief']}\n\nWrite this document: {title} (DEMO) — {what}. "
                                        f"About 450 to 800 words. Start with the title line \"{title} (DEMO)\". "
                                        f"Use each of these exact phrases at least once: {'; '.join(doc[5])}."}]


async def _write(case: dict, doc: tuple) -> str:
    from app.legal.research import llm
    must = doc[5]
    for attempt in range(3):
        text = await llm.complete(_prompt(case, doc), temperature=0.4 if attempt == 0 else 0.2, max_tokens=1800)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        missing = [m for m in must if m not in text]
        if not missing and len(text.split()) >= 250:
            return text + "\n"
        print(f"  {case['number']} {doc[1]}: retry ({'missing ' + ', '.join(missing) if missing else 'too short'})", flush=True)
    raise SystemExit(f"{case['number']} {doc[1]}: the model kept leaving out {missing}")


async def main_async(only: list[str]) -> None:
    manifest = {"note": "FICTIONAL demo cases (D-064): every party, person, figure and fact is invented.",
                "team": {u: {"roles": r, "about": a} for u, (r, a) in TEAM.items()}, "cases": []}
    for case in CASES:
        folder = OUT / case["number"]
        folder.mkdir(parents=True, exist_ok=True)
        entry = {k: case[k] for k in ("number", "title", "practice_area", "jurisdiction", "lead", "members")}
        entry.update(client={"name": case["client"][0], "code": case["client"][1]}, parties=case["parties"], documents=[])
        for doc in case["documents"]:
            name = f"{doc[0]} {doc[1]} (DEMO).md"
            path = folder / name
            if (not only or case["number"] in only) and not path.exists():
                path.write_text(await _write(case, doc), encoding="utf-8")
                print(f"{case['number']}: wrote {name}", flush=True)
            entry["documents"].append({"file": f"{case['number']}/{name}", "title": f"{doc[1]} (DEMO)", "kind": doc[2], "status": doc[3]})
        manifest["cases"].append(entry)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=[])
    asyncio.run(main_async(ap.parse_args().only))
    return 0


if __name__ == "__main__":
    sys.exit(main())
