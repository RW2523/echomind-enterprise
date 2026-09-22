# Intake Form — the details only you can supply

| Field | Value |
|---|---|
| Purpose | Everything the QMS needs from you that cannot be read out of the repository |
| How to use | Fill in the answers. Hand it back and the whole document set can be populated from it in one pass. |
| Time | About 90 minutes. Part 1 is 15 minutes, Part 2 is the thinking. |
| Status | Blank — awaiting your input |

> Nothing here is busywork. Every question below is blocking at least one ISO 9001 clause, and the
> clause is named so you can see why it is asked.

---

## Part 1 — Facts (15 minutes)

### 1.1 The legal entity · *clause 4.3 — the scope must say who and where*

| # | Question | Your answer |
|---|---|---|
| 1 | Registered company name | `____________________` |
| 2 | Registered address | `____________________` |
| 3 | Principal place of business (if different) | `____________________` |
| 4 | Company registration number | `____________________` |
| 5 | The location the service is delivered **from** — this goes into the scope statement | `____________________` |

### 1.2 People and roles · *clause 5.3 — roles and authorities must be assigned*

If one person holds all four, write the same name four times. The roles stay distinct in the
records even when the person does not.

| # | Role | Name | Job title |
|---|---|---|---|
| 6 | Managing Director — owns the QMS, approves documents, authorises releases | `__________` | `__________` |
| 7 | Lead Engineer — design, verification, configuration, release execution | `__________` | `__________` |
| 8 | Quality representative — registers, audit programme, corrective actions | `__________` | `__________` |
| 9 | Data custodian — customer property and data handling | `__________` | `__________` |
| 10 | Anyone else who contributes code or touches customer data | `__________` | `__________` |

### 1.3 Approval · *clauses 5.1, 5.2, 7.3 — this is the single blocking item*

Until this is done every document reads `DRAFT — not yet approved` and the QMS does not exist as a
system. **It must be your own act** — I will not pre-fill a name, a signature or a date, and none
should ever be back-dated.

| # | Question | Your answer |
|---|---|---|
| 11 | Who signs the Quality Policy? | `__________` |
| 12 | On what date will you sign it? | `____ / ____ / ________` |
| 13 | Do you accept the policy text in `QP-01` as written, or do you want changes? | `__________` |

---

## Part 2 — Decisions (the part that takes thought)

### 2.1 Data retention · *clause 8.5.3 — and the question every regulated buyer asks*

There is no retention policy today, and data grows without limit — at one measurement 97.6% of the
corpus was auto-accumulated transcript. Give a number of days for each class, or "keep until the
customer deletes it".

| # | Data class | Where it lives | How long do you keep it? |
|---|---|---|---|
| 14 | Uploaded documents | `documents`, `chunks`, `/data/uploads` | `__________` |
| 15 | Live transcripts | `transcripts`, `transcript_segments` | `__________` |
| 16 | Boardroom audio recordings | `/data/boardroom/` (up to ~5.5 h per session) | `__________` |
| 17 | Chat messages | `messages` | `__________` |
| 18 | Access log (usernames + IP addresses) | `activity_log` | `__________` |
| 19 | Generated documents | `docgen_jobs` | `__________` |
| 20 | Does a customer contract override any of the above? | | `__________` |

### 2.2 Backup · *clause 8.5.4 — the scripts exist; these are the settings*

| # | Question | Your answer |
|---|---|---|
| 21 | Where should the off-host copy go? (`user@host:/path`, or a mounted share) | `__________` |
| 22 | What time should the nightly backup run? (default 02:30) | `__________` |
| 23 | How many generations to keep? (default 7) | `__________` |
| 24 | Must backups be encrypted at rest? (the archive is not encrypted today) | `__________` |

### 2.3 Objectives and risk · *clauses 6.1, 6.2*

| # | Question | Your answer |
|---|---|---|
| 25 | Do you accept the 7 objectives and targets in `QO-01`, or change them? Only agree to what you will actually measure. | `__________` |
| 26 | Do you accept the 14 risk ratings in `REG-03`? They are marked *proposed* — they are my reading of the evidence, not fact. | `__________` |
| 27 | Who owns each risk treatment, and by when? | `__________` |

### 2.4 Running the system · *clauses 9.2, 9.3*

| # | Question | Your answer |
|---|---|---|
| 28 | Management review cadence — quarterly, or annual (the ISO minimum)? | `__________` |
| 29 | Internal audit route: (a) contract an external auditor, (b) wait for a second person, (c) interim self-assessment? **Only (a) and (b) satisfy 9.2.2 c.** | `__________` |
| 30 | If (a), do you have an auditor in mind, and a budget? | `__________` |
| 31 | Which clauses get audited in which period? (or "spread evenly over 12 months") | `__________` |

### 2.5 Product decisions with compliance consequences

| # | Question | Your answer |
|---|---|---|
| 32 | Should `AUTH_ENABLED=1` become the default for customer deployments? | `__________` |
| 33 | What should the default CORS origin be, instead of `*`? | `__________` |
| 34 | What counts as a "significant" change requiring a formal change request, rather than just a commit? | `__________` |

---

## Part 3 — Things you need to find out (not answer from memory)

These are research tasks, not questions. They are listed here because they are yours to obtain and
nobody else can.

| # | Task | Why it matters | Done |
|---|---|---|---|
| 35 | **Licence terms for all 11 models** in `REG-02` — commercial use, attribution, redistribution | You sell commercially into regulated sectors. Their legal teams will ask. | ☐ |
| 36 | **Nemotron gated-model terms specifically** — does acceptance permit redistribution *baked into an image you ship to a customer*? | The model is compiled into images already delivered. If not permitted, that is live legal exposure. | ☐ |
| 37 | **SDXL-Turbo commercial-use terms** | Stability AI licensing has commercial conditions | ☐ |
| 38 | A licensed copy of **ISO 9001:2015** | Copyrighted; cannot live in the repo. You and any auditor need the text. | ☐ |
| 39 | Whether the **MIT Ubuntu mirror** in the Dockerfiles is deliberate, or should be removed | An undeclared third-party dependency in the build path | ☐ |

---

## Part 4 — Commercial context (only if you are pursuing certification)

| # | Question | Your answer |
|---|---|---|
| 40 | Is there a customer, tender or deadline driving this? | `__________` |
| 41 | Do you need certification, or is a demonstrable QMS enough for your buyers? | `__________` |
| 42 | If certification: which accredited body? (UKAS / ANAB / national equivalent) | `__________` |
| 43 | Target date for the stage 2 audit | `__________` |

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
