# Intake Form — the details only you can supply

| Field | Value |
|---|---|
| Purpose | Everything the QMS needs from you that cannot be read out of the repository |
| How to use | Fill in the answers. Hand it back and the whole document set can be populated from it in one pass. |
| Time | About 90 minutes. Part 1 is 15 minutes, Part 2 is the thinking. |
| Status | **Completed 2026-10-07.** Part 1 answered by the Managing Director; Parts 2–4 answered from the documents approved under `records/approvals/2026-10-07_AR-2026-001_document_approval.md`. Where a decision is still open, the answer names the owner, the due date and the register entry that tracks it. |

> Nothing here is busywork. Every question below is blocking at least one ISO 9001 clause, and the
> clause is named so you can see why it is asked.

---

## Part 1 — Facts (15 minutes)

### 1.1 The legal entity · *clause 4.3 — the scope must say who and where*

| # | Question | Your answer |
|---|---|---|
| 1 | Registered company name | AJACE Inc. |
| 2 | Registered address | 14159 Robert Paris Ct., Suite A, Chantilly, VA 20151, USA |
| 3 | Principal place of business (if different) | Same as registered address |
| 4 | Company registration number | To be added from the incorporation record |
| 5 | The location the service is delivered **from** — this goes into the scope statement | 14159 Robert Paris Ct., Suite A, Chantilly, VA 20151, USA |

### 1.2 People and roles · *clause 5.3 — roles and authorities must be assigned*

If one person holds all four, write the same name four times. The roles stay distinct in the
records even when the person does not.

| # | Role | Name | Job title |
|---|---|---|---|
| 6 | Managing Director — owns the QMS, approves documents, authorises releases | Anita Johan | Managing Director |
| 7 | Lead Engineer — design, verification, configuration, release execution | Richard Watson Stephen Amudha | Lead Engineer |
| 8 | Quality representative — registers, audit programme, corrective actions | Sheryl Nazareth | QA / Management Representative |
| 9 | Data custodian — customer property and data handling | Kishan Haravu Pradeep | Data custodian |
| 10 | Anyone else who contributes code or touches customer data | Alexander Peter | EchoMind Project Lead (reviewer) |

### 1.3 Approval · *clauses 5.1, 5.2, 7.3 — this is the single blocking item*

**Done 2026-10-07** — approvals confirmed by the role holders and recorded in `records/approvals/2026-10-07_AR-2026-001_document_approval.md`.

| # | Question | Your answer |
|---|---|---|
| 11 | Who signs the Quality Policy? | Anita Johan (Managing Director) |
| 12 | On what date will you sign it? | 2026-10-07 |
| 13 | Do you accept the policy text in `QP-01` as written, or do you want changes? | Accepted as written |

---

## Part 2 — Decisions (the part that takes thought)

### 2.1 Data retention · *clause 8.5.3 — and the question every regulated buyer asks*

There is no retention policy today, and data grows without limit — at one measurement 97.6% of the
corpus was auto-accumulated transcript. Give a number of days for each class, or "keep until the
customer deletes it".

| # | Data class | Where it lives | How long do you keep it? |
|---|---|---|---|
| 14 | Uploaded documents | `documents`, `chunks`, `/data/uploads` | Kept until the customer deletes it — current rule (`SOP-11` §10). Time-based retention is an open treatment: `REG-03` R-06 (Kishan Haravu Pradeep, 2026-10-31) |
| 15 | Live transcripts | `transcripts`, `transcript_segments` | Kept until the customer deletes it — current rule (`SOP-11` §10). Time-based retention is an open treatment: `REG-03` R-06 (Kishan Haravu Pradeep, 2026-10-31) |
| 16 | Boardroom audio recordings | `/data/boardroom/` (up to ~5.5 h per session) | Kept until the customer deletes it — current rule (`SOP-11` §10). Time-based retention is an open treatment: `REG-03` R-06 (Kishan Haravu Pradeep, 2026-10-31) |
| 17 | Chat messages | `messages` | Kept until the customer deletes it — current rule (`SOP-11` §10). Time-based retention is an open treatment: `REG-03` R-06 (Kishan Haravu Pradeep, 2026-10-31) |
| 18 | Access log (usernames + IP addresses) | `activity_log` | Kept in full; reviewed for access anomalies (`records/capa/2026-10-07_NC-2026-009_access_log_review.md`). A time limit is part of the open retention treatment (`REG-03` R-06) |
| 19 | Generated documents | `docgen_jobs` | Kept until the customer deletes it — current rule (`SOP-11` §10). Time-based retention is an open treatment: `REG-03` R-06 (Kishan Haravu Pradeep, 2026-10-31) |
| 20 | Does a customer contract override any of the above? | | Yes. A retention or backup term agreed with a customer is recorded under `SOP-04` and takes precedence (`SOP-11` §12: kept for the life of the contract + 3 years) |

### 2.2 Backup · *clause 8.5.4 — the scripts exist; these are the settings*

| # | Question | Your answer |
|---|---|---|
| 21 | Where should the off-host copy go? (`user@host:/path`, or a mounted share) | To be set as `BACKUP_REMOTE` by Kishan Haravu Pradeep by 2026-10-13 (`REG-03` R-02). Not set yet |
| 22 | What time should the nightly backup run? (default 02:30) | 02:30 (default in `scripts/install_backup_timer.sh`) |
| 23 | How many generations to keep? (default 7) | 7 (default) |
| 24 | Must backups be encrypted at rest? (the archive is not encrypted today) | Not for the reference instance, which holds no customer data. Required before backups of customer data leave the host — `REG-03` R-13, encryption at rest (Kishan Haravu Pradeep, 2026-10-31) |

### 2.3 Objectives and risk · *clauses 6.1, 6.2*

| # | Question | Your answer |
|---|---|---|
| 25 | Do you accept the 7 objectives and targets in `QO-01`, or change them? Only agree to what you will actually measure. | Accepted as written — `QO-01` approved 2026-10-07 |
| 26 | Do you accept the 14 risk ratings in `REG-03`? They are marked *proposed* — they are my reading of the evidence, not fact. | Accepted — ratings confirmed at approval 2026-10-07 |
| 27 | Who owns each risk treatment, and by when? | Named per risk in `REG-03` (owners Richard Watson Stephen Amudha and Kishan Haravu Pradeep; due dates 2026-10-13 to 2026-10-31) |

### 2.4 Running the system · *clauses 9.2, 9.3*

| # | Question | Your answer |
|---|---|---|
| 28 | Management review cadence — quarterly, or annual (the ISO minimum)? | Quarterly, and at least annually (`SOP-15` §4) |
| 29 | Internal audit route: (a) contract an external auditor, (b) wait for a second person, (c) interim self-assessment? **Only (a) and (b) satisfy 9.2.2 c.** | Through the AJACE Inc. corporate internal audit programme, by an auditor independent of the EchoMind work (`SOP-13` §4) |
| 30 | If (a), do you have an auditor in mind, and a budget? | Not applicable — corporate audit programme, coordinated by the QA / MR (Sheryl Nazareth) |
| 31 | Which clauses get audited in which period? (or "spread evenly over 12 months") | Nov 2026: clauses 4–7 · Dec 2026: clause 8 · Jan 2027: clauses 9–10 (`SOP-13` §4) |

### 2.5 Product decisions with compliance consequences

| # | Question | Your answer |
|---|---|---|
| 32 | Should `AUTH_ENABLED=1` become the default for customer deployments? | Login is enforced on the public reference instance and checked every 6 hours. The default for customer deployments is an open decision under `REG-03` R-04 (Kishan Haravu Pradeep, 2026-10-31) |
| 33 | What should the default CORS origin be, instead of `*`? | Decided with question 32 under `REG-03` R-04 (2026-10-31) |
| 34 | What counts as a "significant" change requiring a formal change request, rather than just a commit? | Criteria S-1 to S-7 in `SOP-06` §6 (approved 2026-10-07) |

---

## Part 3 — Things you need to find out (not answer from memory)

These are research tasks, not questions. They are listed here because they are yours to obtain and
nobody else can.

| # | Task | Why it matters | Done |
|---|---|---|---|
| 35 | **Licence terms for all 11 models** in `REG-02` — commercial use, attribution, redistribution | You sell commercially into regulated sectors. Their legal teams will ask. | ☐ Open — GitHub ticket #56 |
| 36 | **Nemotron gated-model terms specifically** — does acceptance permit redistribution *baked into an image you ship to a customer*? | The model is compiled into images already delivered. If not permitted, that is live legal exposure. | ☐ |
| 37 | **SDXL-Turbo commercial-use terms** | Stability AI licensing has commercial conditions | ☐ |
| 38 | A licensed copy of **ISO 9001:2015** | Copyrighted; cannot live in the repo. You and any auditor need the text. | ☐ |
| 39 | Whether the **MIT Ubuntu mirror** in the Dockerfiles is deliberate, or should be removed | An undeclared third-party dependency in the build path | ☒ Deliberate — workaround for `ports.ubuntu.com` sync failures; recorded in `SOP-09` §9 |

---

## Part 4 — Commercial context (only if you are pursuing certification)

| # | Question | Your answer |
|---|---|---|
| 40 | Is there a customer, tender or deadline driving this? | The AJACE Inc. ISO 9001:2015 surveillance audit — final audit 14 October 2026 |
| 41 | Do you need certification, or is a demonstrable QMS enough for your buyers? | AJACE Inc. is already certified (scope *Product Development and Technology Consulting Services*); EchoMind operates within that QMS |
| 42 | If certification: which accredited body? (UKAS / ANAB / national equivalent) | The body that holds AJACE Inc.'s current certificate (expires 28 October 2026) |
| 43 | Target date for the stage 2 audit | Not applicable — surveillance audit 14 October 2026 |

> On 41 — worth pausing on. Many enterprise buyers accept evidence of a functioning quality system
> without a certificate. Certification adds an accredited body, an audit cycle and cost. If no buyer
> is actually requiring the certificate, the cheaper answer is to run the system, accumulate the
> records, and certify only when someone asks.

---

## What happens once this is filled in

| Your input | Populates |
|---|---|
| 1–5 | `QM-01` §1 and the scope statement in §4 |
| 6–10 | `QM-01` §6.1; `REG-05` competence matrix |
| 11–13 | `QP-01` signature block; all 36 approval headers; `REG-01` |
| 14–20 | `SOP-11` §7 retention policy; implementation task D10 |
| 21–24 | `install_backup_timer.sh` arguments; `SOP-11` §8 |
| 25–27 | `QO-01` targets; `REG-03` revision 1.1 with owners and dates |
| 28–31 | `SOP-15` cadence; `SOP-13` audit programme |
| 32–34 | `SOP-11` §5; `SOP-06` §5 significance criteria |
| 35–39 | `REG-02` licence column; `NOTICE` file |
| 40–43 | Sequencing of the certification path |

**Answers to 11 and 12 alone unblock more than everything else combined** — they convert the
document set into a management system. Everything else can follow.
