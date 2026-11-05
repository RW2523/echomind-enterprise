> FICTIONAL DEMO DOCUMENT. All parties, people, amounts and events are invented. Not legal advice.

# Presenter guide — M-101 and M-114

Files: `M-101/`, `M-114/` (one `.md` per document; the file name is the exact title), `manifest_m101_m114.json`, `answer_key_m101_m114.jsonl` (59 items, every `evidence` string checked verbatim against its source).

The firm is Calder & Wynne LLP (Helena Marsh, partner; Alice Hart, associate).

## M-101 — Harrowgate MSA renewal (client Voss-Lindqvist AB; adverse Harrowgate Systems Inc.; Delaware)

| Document | In one line |
|---|---|
| Master Services Agreement (DEMO) | Executed March 2, 2026. 12-month Initial Term to March 1, 2027, auto-renews 12 months unless 90 days' notice (deadline December 1, 2026). USD 18,500/month, invoiced monthly in advance, net 30, 1%/month interest. CPI cap on renewal increases (4.4). Cap is the greater of USD 250,000 or 12 months' fees (8.1), with a separate USD 1,000,000 data cap (8.4). 48-hour breach notice (6.3). Only the Client may terminate for convenience (12.1). 30 days' change-of-control notice (14.2). SLA 99.5% monthly with a credit table (Schedule 2). |
| Advice memo on the MSA (DEMO) | Calder & Wynne, October 28, 2026. Renewal risks; correct on every point except one: it quotes 4.2 as net sixty. |
| Engagement request (DEMO) | Petra Voss to Helena Marsh, October 19, 2026. Scope (a)–(d); fixed fee USD 28,000 in two instalments; deadlines October 30, November 3, November 20 and December 1. |
| Reseller Agreement (DEMO) | Voss-Lindqvist (Supplier) and Fjellstad Distribution AS (Reseller), April 20, 2026, Swedish law. Net 45 monthly in arrears; 10% a year interest; EUR 400,000 cap; 72-hour breach notice; change of control: 15 days to tell, 45 days to terminate; runs to April 19, 2028. |
| Consultancy Agreement (DEMO) | Northbridge Analytics AB, October 26, 2026. SEK 14,500 / SEK 10,800 day rates; Fee Cap SEK 1,200,000; estimate SEK 914,500; deliverables assigned to the client. |
| Renewal proposal from Harrowgate (DEMO) | Thomas Reyes, October 14, 2026. USD 19,750/month; removes the USD 250,000 floor from 8.1; 400 to 500 Named Users; draft Amendment No. 1 (A1–A6). |
| Service level report Q3 2026 (DEMO) | Harrowgate, October 7, 2026. July 99.82%, August 98.71% (below target), September 99.64%; quarter 99.39% "for information only". The credit amount is not stated. |
| Board minutes approving the renewal (DEMO) | Voss-Lindqvist board, November 3, 2026. 5 of 6 present. Limits: a fee above USD 18,500 only with a 24-month term (max USD 19,750); cap at least USD 400,000; no non-renewal notice. Anders Holm did not vote on Northbridge. |

Deliberate issues and what the product should show:
- **Memo misquote (the headline).** The memo's 4.2 quote says "net sixty (60) days"; the MSA says net thirty (30). The quote check should fail the quotation, and "find inconsistencies" should pair the memo sentence with MSA 4.2. The rule-based scan proposes this pair.
- **Lower cap in the proposal.** Draft A2 deletes the USD 250,000 floor, so the cap falls to 12 months' fees (about USD 222,000, at most USD 237,000). The amendment check should link A1 to 4.1, A2 to 8.1, A3 to 3.4 and A4 to Schedule 3.
- Expected differences that are not errors (the product may list them, and should explain them): the proposal's USD 19,750 against the MSA's USD 18,500; the board's USD 400,000 floor against the cap; the report's 98.71% against the 99.5% target; net 45 and 72 hours in the Reseller Agreement against net 30 and 48 hours in the MSA (different contracts and parties).
- Playbook check on the MSA (demo playbook is supplier-side, Swedish law): expect a governing-law deviation (Delaware), a termination-for-convenience deviation (Client only), and a cap close to the standard.

Six demo questions:
1. "What are the payment terms in the Master Services Agreement (DEMO)? Include amounts, invoice deadlines and late-payment provisions." → USD 18,500/month, invoiced monthly in advance, net 30 from the invoice date, 1% a month on undisputed late amounts. Never net 60.
2. "Does the advice memo state the MSA's payment terms correctly?" → No: the memo says net sixty (60) days; MSA 4.2 says net thirty (30).
3. "What service credit is Voss-Lindqvist owed for August 2026?" → 98.71% is in the 10% band: 10% × USD 18,500 = USD 1,850, credited on the November 2, 2026 invoice. No termination right, because that needs below 97.0% in two consecutive months.
4. "What are Fjellstad Distribution's payment terms under the Reseller Agreement?" → Net 45 days from the invoice date, monthly in arrears, 10% a year interest. Not the MSA's terms.
5. "Can Harrowgate terminate the MSA for convenience?" → No. Only Voss-Lindqvist may, on 90 days' notice (with an early termination fee of up to 3 months' fees in the Initial Term).
6. "How would Harrowgate's proposal change the liability cap, and what did the board decide?" → It removes the USD 250,000 floor, leaving 12 months' fees. The board requires at least USD 400,000 in any amendment.

## M-114 — Series B due diligence (client Alder Finch Ltd; England & Wales)

| Document | In one line |
|---|---|
| Articles of Association (DEMO) | Adopted June 14, 2023. Ordinary and Series A Preferred shares (GBP 0.0001). Authority to allot capped at 8,500,000 Ordinary (3.3) and 2,000,000 Series A (3.4); option plan up to 600,000. Pre-emption 14 days, ROFR 21 days, drag 60%, tag at 50%. 2–7 directors; 7 days' notice; quorum of two including the Investor Director. |
| Board minutes January 2026 (DEMO) | January 20, 2026; all 4 directors present; quorate. Six resolutions, all passed 4–0: 120,000 option shares; founder IP deeds; Northwind (GBP 864,000, CEO to sign); Tidewater (36 months, about GBP 1,372,000, two directors); Series B preparations; policies. Names the founders. |
| Board procedures policy (DEMO) | v2.0. Agrees with the articles: 2–7 directors, 7 days' notice, quorum of two including the Investor Director, reserved matters, committees. |
| Delegation of authority (DEMO) | v3.0. Contracts: Head of Ops GBP 25,000, CFO GBP 150,000, CEO GBP 1,000,000 (Board approval first above GBP 500,000 or 24 months), above that the Board and two directors. Customer, capex, banking and litigation limits. |
| IP Assignment - A Moreau (DEMO) | Deed of February 2, 2026; consideration GBP 1; schedule: patent application GB2104417.6, FINCHSENSE trade mark application, firmware to v1.4, enclosure designs, domain. |
| Register of members (DEMO) | At October 31, 2026: 8,520,000 ordinary and 2,000,000 Series A = 10,520,000 (adds up); 480,000 options outstanding. |
| Supply Agreement - Northwind (DEMO) | Northwind Components Ltd (Supplier) and Alder Finch (Customer), January 29, 2026, 24 months. Minimum 50,000 NW-T4 and 2,400 NW-H1 = GBP 864,000; pay within 30 days of the invoice date; GBP 500,000 cap; signed by the CEO. |
| Supply Agreement (DEMO) | Alder Finch (Supplier) and Tidewater Retail Ltd (Customer), February 16, 2026, 36 months. Pay within 60 days after month-end; 4% over base rate; GBP 750,000 cap; customer may terminate on a >50% change of control. Carries the planted definitions errors. |
| Series B term sheet (DEMO) | October 9, 2026. 3,600,000 Series B at GBP 3.50 = GBP 12,600,000 (Larchmont 3,000,000; Brightwater 600,000). Pre-money GBP 38,500,000 on 11,000,000 fully diluted shares. Exclusivity to November 23, 2026. Condition 18(d): IP deed from Benedikt Lind. |
| Disclosure letter (draft) (DEMO) | Draft of November 3, 2026. Share capital and options; both supply agreements; GBP 12,012 sensor replacement; Lind has not signed an IP deed; revenue GBP 6,850,000. |

Deliberate issues and what the product should show:
- **Definitions check on "Supply Agreement (DEMO)"** (the rules find exactly these three, checked with `definitions/check.py`): (1) "Commencement Date" is used in 1.1 ("Contract Year"), 3.1 and 5.6 but never defined (medium); (2) "Territory" is defined in 1.1 but never used (low); (3) "Price" is defined twice with different meanings: 1.1 says the Schedule 2 price, 5.1 says the price in the Order (high).
- **Authority to allot exceeded (register against articles).** 8,520,000 ordinary shares are in issue against an 8,500,000 authority (Article 3.3). The 20,000 excess came from the January 20, 2026 option exercise, and the 480,000 outstanding options cannot be exercised either. No document discloses it. The rule-based governance check does not test this; the chat and the review panel should find it.
- **Governance run (rules, checked):** one finding, "No IP assignment from Benedikt Lind". This is consistent with the documents: the minutes name him as a founder, the disclosure letter (3.8) discloses the gap, and term sheet 18(d) makes the deed a condition. Share capital, signing authority and constitution (quorum, notice, number of directors) all pass.

Six demo questions:
1. "What is the quorum for board meetings?" → Two directors, one of them the Investor Director if one is in office (Article 15.2; policy 4.1 agrees).
2. "Do the shares in the register add up, and are they within the articles?" → They add up: 8,520,000 + 2,000,000 = 10,520,000. But 8,520,000 ordinary exceeds the 8,500,000 authority by 20,000.
3. "Was the CEO authorised to sign the Northwind supply agreement?" → Yes: GBP 864,000 is within her GBP 1,000,000 limit, and the Board approval needed above GBP 500,000 was given on January 20, 2026.
4. "Compare payment terms with Northwind and with Tidewater." → Alder Finch pays Northwind within 30 days of the invoice date. Tidewater pays Alder Finch within 60 days after the end of the month of receipt. Alder Finch is the Customer in one and the Supplier in the other.
5. "What consideration was paid for Amélie Moreau's IP assignment, and is the founder IP chain complete?" → GBP 1. Not complete: Benedikt Lind has not signed; his deed is due at Completion.
6. "Run a definitions check on the Supply Agreement (DEMO)." → Commencement Date is undefined; Territory is unused; Price is defined twice (Schedule 2 against the Order).

## Notes for whoever loads these

- These files replace the thin fixtures in `eval/fixtures/law/` for M-101 and M-114. Some facts changed on purpose: the MSA term is now 12 months, not 36; the Reseller Agreement's Reseller is now Fjellstad, not Harrowgate; the M-114 company is Alder Finch Ltd, not Tidewater Holdings.
- The seeded client call in `backend/app/legal/cli.py` (`_seed_meeting`) still says the redline raises "the liability cap to USD 500,000". The new documents have Harrowgate lowering the cap; the USD 19,750 fee in that call matches.
- Every file starts with the required disclaimer line, so `_first_line` and the conflict scan's document name read that line instead of a title. Roles still come from the titles, and the governance checks above were run that way.
