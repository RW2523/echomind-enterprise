# MAT-2104 — Brightline Software / Atlas Federal Credit Union — enterprise SaaS agreement — matter fact sheet

Fictional. Commercial contracts (technology), New York law. Calder & Wynne LLP acts for the VENDOR, Brightline
Software, Inc. Emails `.example`.

## 1. Parties and people

**Brightline Software, Inc.** — the client ("Brightline", "Provider"). Delaware corporation, 200 Varick Street, 6th
Floor, New York, NY 10014. Product: **Brightline Lending Cloud** (loan-origination SaaS).
- Nathan Briggs — General Counsel (nbriggs@brightline.example) — instructs us.
- Erin Castillo — VP, Sales (deal owner).

**Atlas Federal Credit Union** — the counterparty ("Atlas", "Customer"). Federally chartered credit union, 75 State
Street, Albany, NY 12207.
- Monica Sheridan — Director, Vendor Management (msheridan@atlasfcu.example); author of the markup.
- Ravi Patel — Chief Information Security Officer.

**Our firm:** Calder & Wynne LLP (New York office: 250 Park Avenue South, 12th Floor, New York, NY 10003) — Kenji
Arai (lead, Associate), Alice Moreno (Partner), Theo Lindgren (knowledge lawyer, reviews). File CW-2104.

## 2. Master Subscription Agreement — Version 1 (Brightline's form, October 1, 2026)

- 1 Definitions: "Subscription Fees", "Monthly Fees" (one twelfth of the annual Subscription Fees), "Availability",
  "Security Incident", "Customer Data".
- 3 Subscription: Brightline Lending Cloud for up to 400 named users; **Subscription Fees $480,000 per year**, invoiced
  annually in advance, net 30. Initial term **three (3) years** from the Effective Date.
- 5 Service levels: Availability per the SLA Schedule; **monthly Availability commitment 99.9%**; **service credits are
  Customer's sole and exclusive remedy** for Availability failures.
- 8 Security: SOC 2 Type II report annually; **Brightline notifies Customer of a Security Incident without undue delay
  and in any event within seventy-two (72) hours** after confirming it.
- 10 Term and termination: termination for material breach uncured 30 days; **no termination for convenience**.
- **11 Limitation of liability:** 11.1 no indirect damages; **11.2 general cap = the Subscription Fees paid or payable
  in the twelve (12) months before the event giving rise to the claim**; **11.3 data-breach super-cap: for breach of
  Section 8 (Security) or confidentiality involving Customer Data, two times (2x) the annual Subscription Fees
  ($960,000)**.
- 14 Governing law: New York; courts of New York County.

## 3. Master Subscription Agreement — Version 2 (Atlas markup, November 6, 2026)

Atlas returned Version 1 with **tracked changes by "M. Sheridan" dated 2026-11-06** and comments. A new version of the
same document. Changes:
- 5: **Availability 99.9% → 99.95%**; deletes "sole and exclusive remedy" (credits "without prejudice to Customer's other
  rights and remedies"). Comment: "Atlas's board requires 99.95% for member-facing systems."
- 8: **notice of Security Incident 72 hours → twenty-four (24) hours** after discovery (not confirmation).
- 10: **adds termination for convenience by Customer on ninety (90) days' notice at any time after the first twelve (12)
  months**, with a pro-rata refund of prepaid fees.
- 11.3: **super-cap 2x → five times (5x) the annual Subscription Fees ($2,400,000)**. Comment: "Required by NCUA vendor
  guidance per our CISO."
- 11.2 general cap **unchanged** (12 months' fees).
- adds 8.6: annual on-site security audit right on 30 days' notice.

## 4. SLA Schedule (Schedule 2 to Version 1)

Monthly Availability and service credits (percentage of the Monthly Fees for that month):

| Monthly Availability | Service credit |
|---|---|
| ≥ 99.9% | none |
| < 99.9% but ≥ 99.5% | 5% |
| < 99.5% but ≥ 99.0% | 10% |
| < 99.0% | 25% |

Credits capped at 25% of the Monthly Fees in any month; scheduled maintenance (up to 8 hours a month, notified 72 hours
ahead) excluded. Monthly Fees = $40,000 (= $480,000 / 12).

## 5. Other documents

**Data Processing Addendum — Brightline form, draft (October 1, 2026):** Brightline as service provider; notify
Customer of a Personal Information breach "without undue delay and in any event within seventy-two (72) hours";
subprocessors listed (cloud hosting: a named fictional host "Northbeam Cloud"); data deleted within 30 days after
termination.

**Atlas Information Security Addendum (Atlas form, received November 6, 2026):** SOC 2 Type II annually; encryption in
transit (TLS 1.2+) and at rest (AES-256); MFA for administrative access; **security incident notice within forty-eight
(48) hours**; penetration test annually with summary to Atlas; right to audit.

**Email — Monica Sheridan to Erin Castillo, cc Nathan Briggs (November 6, 2026, 4:52 p.m.)**, sending the markup:
"…we have accepted your liability cap as drafted and only tightened the uptime to 99.95%…" — this MISDESCRIBES the
markup (11.3 super-cap went to 5x; T4C and 24-hour notice were added).

**Email — Nathan Briggs to Alice Moreno and Kenji Arai (November 9, 2026), our instructions:** Brightline can accept
99.95% only if service credits stay capped at 15% … (our negotiating position): accept 99.95% with credits capped at
15% of Monthly Fees; **will not accept a super-cap above 3x ($1,440,000)**; T4C acceptable only after 24 months with 180
days' notice and no refund of the current year's fees; security incident notice 48 hours (matching Atlas's own
addendum) not 24; wants signature by **December 18, 2026** (quarter end).

## 6. Planted issues

1. **Three different security-incident notice periods:** 72 hours (MSA v1 and the DPA), 24 hours (MSA v2), 48 hours
   (Atlas's own Security Addendum).
2. Super-cap 2x ($960,000) → 5x ($2,400,000); client's limit 3x ($1,440,000).
3. Uptime 99.9% → 99.95%; credits no longer sole remedy.
4. The Atlas email says the liability cap was accepted as drafted — the markup changed the super-cap.
5. A service-credit calculation: at 99.3% Availability in a month under the v1 SLA, credit = 10% × $40,000 = **$4,000**.

## 7. Not stated anywhere

- Atlas's number of members; the hosting region; any price escalator; the number of named users in Version 2 (unchanged
  400 — but don't add a different figure anywhere).

## 8. Documents (paper: letter)

| # | file | title | kind | status | render / style |
|---|---|---|---|---|---|
| 01 | Master Subscription Agreement - Brightline form v1.docx | Master Subscription Agreement - Brightline / Atlas | contract | draft | docx, contract, numbering auto, header "BRIGHTLINE FORM - V1 - 10/01/2026" |
| 02 | Master Subscription Agreement - Atlas markup v2.docx | Master Subscription Agreement - Brightline / Atlas | contract | draft | docx, contract, numbering auto, author M. Sheridan, changes_date 2026-11-06, **version_of: 01-…src.md**, tracked changes `{++ ++}` `{-- --}` `{~~ ~> ~~}` and comments `{== ==}{>> <<}` |
| 03 | SLA Schedule - Schedule 2.docx | SLA Schedule (Schedule 2) - Availability and service credits | contract | draft | docx, table |
| 04 | Data Processing Addendum - Brightline form.docx | Data Processing Addendum - Brightline form (October 1, 2026) | contract | draft | docx, contract, numbering auto |
| 05 | Atlas Information Security Addendum.pdf | Atlas Information Security Addendum (received November 6, 2026) | contract | received | pdf, contract |
| 06 | Email - Atlas markup cover - 2026-11-06.pdf | Email from Atlas (Monica Sheridan) sending the markup (November 6, 2026) | correspondence | received | pdf, email |
| 07 | Email - Brightline instructions - 2026-11-09.pdf | Email from Brightline's General Counsel with negotiating instructions (November 9, 2026) | correspondence | received | pdf, email; header PRIVILEGED & CONFIDENTIAL |

Note: Version 2 must contain the FULL agreement text with the tracked changes marked inline (accepted text = the new
version), so the product can read the whole agreement from it.
