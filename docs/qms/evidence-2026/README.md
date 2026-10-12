# EchoMind Enterprise — ISO 9001:2015 evidence pack (2026)

The 2026 counterpart of the 2025 EchoMind evidence pack (`fileiso9001.zip`), built for the rebuilt
product in this repository. Same structure and numbering as 2025, so an auditor can follow the same path.
Prepared 2026-10-06 for the pre-audit (2026-10-07) and the audit (2026-10-14); approved 2026-10-07 (AR-2026-001), revision 1.1 of the review and NFR evidence approved 2026-10-10 (AR-2026-002); Excel registers EM26-19…22 added 2026-10-11.

**Start with** `00_Overview_and_Map/EM26-00_ISO9001_Audit_Package_EchoMind_Enterprise.docx`, then the
map `00_Overview_and_Map/EM26-00_Evidence_Map_2025_to_2026.xlsx` (every 2025 file → its 2026 counterpart).

The whole set — this pack, the QMS documents and raw proof (commit history, release tags, GitHub issues,
live-system snapshot, evaluation reports) — is also available as one download:
[`docs/qms/ISO9001_EchoMind_2026.zip`](../ISO9001_EchoMind_2026.zip) (open `00_START_HERE.docx` first). Clients appear as Client A–D here; the controlled copy given to the auditor carries the names.

## Rules this pack follows

- **Nothing is invented.** Every record comes from a commit, a QMS register, or a test executed on
  2026-10-06 against the reference instance. Records summarising earlier work say so and give the original dates.
- **Approvals are recorded, not assumed.** Signatures are the typed names of the role holders, entered after each
  confirmed approval (AR-2026-001, AR-2026-002 in `docs/qms/records/approvals/`). Meeting documents are templates
  for meetings still to be held; the management-review file is an input pack.
- **Dates come from records.** 2025 dates from the 2025 minutes, review documents and audits; 2026 dates from commits,
  registers and test runs. Where a remembered month and the commit record differed, the commit date is used.
- **Failures are recorded as observed.** Every failure is logged in REG-04 with its fix commit and the re-test that verified it.

## Contents

| Folder | What | 2025 counterpart |
|---|---|---|
| `00_Overview_and_Map` | Audit package (with §0 predecessor comparison), Clause 8 records, evidence map | 0. Project Documentation, Clause 8 Records |
| `01_Privacy_Security` | Privacy, security and compliance overview — current state | 1. Privacy … overview (university pilot customer) |
| `02_Meeting_Records` | Minutes templates (design, provider, QA/release) + management-review input pack | 0. Meeting Minutes …, QA Sprint minutes |
| `11_Deliverables_Review` | Deliverables log Feb–Oct 2026 | 11. Deliverables Review |
| `12_Scheduling_Review` | Actual timeline, milestones, SDLC record workbook | 12. Scheduling Review, SDLC plan |
| `13_Acceptance_Criteria` | Acceptance-criteria mapping from executed tests | 13. Acceptance Criteria Review, Jira SCRUM-36 |
| `14_Provider_Review` | Provider inventory and performance review (scores confirmed 2026-10-07) | 14, 14b |
| `15_Testing` | 13 test cases with execution records, 5 failure records, defect linking, regression, summary | 15, 15a–c, QA-FAIL / DEF / REG |
| `16_Lessons_Learnt` | 17 lessons, logbook, workbook | 16, 16b, Lessons Learned Log |
| `17_NFR_Register` | EM26-17 NFR record and monitoring + workbook (2025 finding #6) | NFR tracker |
| `18_Review_Actions` | EM26-18 review feedback and action log + workbook (2025 finding #7) | Action tracking register |
| `19_Excel_Registers` | EM26-19 deliverable review feedback and action log 2025 → 2026 (33 reviews, two-monthly review cycle); EM26-20 functional requirements; EM26-21 non-functional requirements with measurement history; EM26-22 product roadmap 2025 → 2027 | Lessons log, SDLC plan, roadmap |
| `Screens` | 21 real screenshots: app, GitHub, test and service evidence | screencapture-*.pdf/png/jpg |
| `QMS_Word` | Word copies of every `docs/qms` markdown document (the markdown stays the controlled source) | QMS-DOC-08-x |

## Test results (2026-10-06)

| Run | Build | Outcome |
|---|---|---|
| 1 | v1.4.0 code | 7 pass · 4 fail → NC-2026-012 (logout token), -013 (binary upload), -014 (citations on small talk), -015 (API key not redacted) |
| 2 | fixes `e0ed4d0` | the four fixes verified; the logout fix had made logouts log as anonymous → corrected `3feb5db` |
| 3 | release v1.4.1 | embedding service down after redeploy while its healthcheck stayed green → NC-2026-016, embeddings moved to CPU |
| 4 | release v1.4.1 | **all 11 executable cases pass**; 1 not executed (live transcription, needs a microphone), 1 blocked (golden evaluation, NC-2026-008) |

All four runs are kept in `15_Testing/Execution_Records/run1…run4/`.

## Still open (owners and dates in REG-03 and EM26-22 › Forward plan)

1. Off-host nightly backup — Kishan Haravu Pradeep, 2026-10-13 (R-02).
2. Restore the evaluation corpus and re-run the golden suite — Richard Watson Stephen Amudha, 2026-10-13 (NC-2026-008).
3. Model licences — 2026-10-31 (R-10, ticket #56); login default, retention and encryption decisions — 2026-10-31 (R-04, R-06, R-13).
4. Run TC-EM-TRN-112 (live transcription) manually.
5. Internal audits Nov 2026 → Jan 2027 and the first management review; record real minutes in the templates.
