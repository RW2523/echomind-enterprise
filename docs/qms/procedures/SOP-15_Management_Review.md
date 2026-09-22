# SOP-15 — Management Review

| Field | Value |
|---|---|
| Document ID | SOP-15 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Managing Director |
| Approved by | `________________________` |
| Approval date | `____ / ____ / ________` |
| ISO 9001:2015 clauses | 5.1, 9.3.1, 9.3.2, 9.3.3 |

---

## 1. Purpose

To define how top management at Ajace AI reviews the quality management system (QMS) at planned
intervals against real evidence, to ensure its continuing suitability, adequacy, effectiveness and
alignment with the strategic direction of the organisation, and to record the decisions that come
out of that review.

## 2. Scope

The whole QMS and the whole of EchoMind Enterprise. Every input listed in ISO 9001:2015 9.3.2 is
considered at every review. Where an input has nothing to report, the minutes say so and say why —
"nothing to report" that was actually asked is a finding; an omitted heading is a gap.

## 3. Why this matters in a one-person organisation

Management review is sometimes treated as a meeting, and a meeting needs more than one person. It
is not a meeting. **It is the scheduled point at which the organisation stops producing and looks at
the evidence of whether what it produced met its objectives.** That is as necessary with one person
as with a hundred — arguably more so, because with one person there is no colleague whose question
forces the reflection.

What changes with one person is not the substance but the discipline required. Without a second
party in the room, the protection against a review that says what the owner already believes is the
**evidence requirement**: every input in §5 must be answered from a named source — a file, a
register, an evaluation report, a commit range — and the source must be opened and looked at during
the review, not recalled. Minutes that do not cite sources are not minutes of a review that happened.

## 4. Cadence, attendance and preparation (9.3.1)

**Cadence.** Reviews are held **quarterly**, and in any event **at least annually**. Quarterly is
chosen because the product changes fast — 202 commits between 2026-02-05 and 2026-09-21 — and an
annual-only cycle would review a system that no longer exists.

**Additional reviews** are convened by the Managing Director on any of:

| Trigger | Reason |
|---|---|
| A major nonconformity (`SOP-13` §8) | The QMS failed to prevent something significant |
| An internal audit report | Findings need a decision, not just a record |
| Entry into a new customer sector, or a first production customer | Context and interested parties change materially (`SOP-02` §4, §5) |
| A change in the number of people in the organisation | Roles, competence and the audit-impartiality position all change |
| A change of hardware platform or of a core model | Large parts of the verification evidence become historical |

**Attendance.** Top management — currently the Managing Director, who is also the Lead Engineer and
sole worker. As the organisation grows, the process owner for each procedure attends for the parts
covering their process. Attendance is recorded on `forms/FRM-07_Management_Review_Minutes.md`,
including where the attendee list is one person.

**Preparation.** Before the review, the Managing Director assembles the evidence pack. The review
does not begin until it exists:

| Pack item | Source |
|---|---|
| Previous minutes and the status of every action from them | `docs/qms/records/management-review/` |
| Current quality objectives and their measured values | `QO-01`, and the measurement sources in `SOP-12` |
| Latest golden-question evaluation report and its score | `eval/reports/`, produced by `eval/run_eval.py` |
| Latest recorded test runs | `backend/tests/` (40 test functions), `voice/tests/` (38 test functions) |
| CAPA log, open and closed since last review | `registers/REG-04_Nonconformity_and_CAPA_Log.md` |
| Risk register | `registers/REG-03_Risk_Register.md` |
| External provider register | `registers/REG-02_External_Documents_and_Providers.md` |
| Customer feedback and complaints since last review | Completed `FRM-08` records |
| Audit reports since last review | Completed `FRM-06` records |
| Commit range since the last review | `git log <previous review date>..HEAD` |

## 5. Standing agenda — the 9.3.2 inputs

Worked through in this order. Each item names where the evidence actually comes from in this
organisation. `forms/FRM-07_Management_Review_Minutes.md` mirrors these headings exactly, so the
minutes are produced by working down the agenda.

### 5.1 Status of actions from previous management reviews (9.3.2 a)

**Source:** the previous minutes in `docs/qms/records/management-review/`.

Each open action is listed with its owner, due date and current state: complete, in progress, or
overdue. An overdue action is discussed, not silently rolled forward. An action carried over twice
without progress is either re-scoped, re-owned or abandoned with the reason recorded — the one thing
it may not be is carried a third time unchanged.

### 5.2 Changes in external and internal issues relevant to the QMS (9.3.2 b)

**Source:** `SOP-02` §4 (context) and §5 (interested parties), re-read against current reality.

Specific questions at each review: has a new customer sector been entered? Has the hardware or model
platform changed? Has the legal or regulatory position on AI systems moved in a way that affects
customers in defence, government, legal, finance or healthcare? Has the dependency supply chain
changed? Is the organisation still one person? The determination tables in `SOP-02` are amended in
the same session if the answer requires it.

### 5.3 Customer satisfaction and feedback from interested parties (9.3.2 c 1)

**Source:** completed `forms/FRM-08_Customer_Feedback_and_Complaint_Record.md` records; direct
customer and prospect communication; feedback from academic and conference reviewers as an interested
party (`SOP-02` §5).

Where there is no feedback in the period, the minutes record that, and record whether the absence is
because there were no customers, because no one was asked, or because a feedback route exists and
produced nothing — these mean very different things.

### 5.4 The extent to which quality objectives have been met (9.3.2 c 2)

**Source:** `QO-01`.

Each objective is reported with its target, its measured value, the measurement date and the source
of the measurement. An objective that cannot be measured because the measurement does not exist is
reported as **not measured**, never as met. Objectives found to be unmeasurable are revised at this
review under 9.3.3 b.

### 5.5 Process performance and conformity of products and services (9.3.2 c 3)

**Source:** `SOP-12`; the 52-question golden evaluation (`eval/run_eval.py`, whose suite gate is
binary — `return 0 if total_pass == len(results) else 1` at `eval/run_eval.py:281`); the unit test
suites; container healthchecks; voice per-turn timing logs.

Reported at each review:

| Measure | What is reported |
|---|---|
| Golden-question evaluation | Score this period, previous score, direction of travel, and the identity of any item that regressed. Best recorded to date is 50/52; the most recent recorded run is 49/52 on 2026-08-06; 52/52 has never been achieved. |
| Unit tests | Whether they were run, when, and the result |
| Build and deployment | Whether a clean from-scratch rebuild has been performed in the period and whether all six containers came up healthy |
| Runtime health | Healthcheck coverage — currently 4 of 6 services — and any service observed unhealthy |
| Voice performance | Per-turn timing logs against the latency expectation |

A measure that was not taken in the period is reported as not taken. The gap between "we did not
measure" and "we measured and it was fine" is the whole point of this heading.

### 5.6 Nonconformities and corrective actions (9.3.2 c 4)

**Source:** `registers/REG-04_Nonconformity_and_CAPA_Log.md` and completed `FRM-05` records.

Reported: nonconformities raised in the period by source and severity; those closed; those open and
overdue; and — specifically — **any record closed without documented effectiveness verification**,
which is a nonconformity against `SOP-14` §8 in its own right. Trends across cause categories
(`SOP-14` §7.2) are examined: the same category recurring is a signal about a control, not about the
individual defects.

### 5.7 Standing item — the QMS's own open nonconformity on auditor impartiality

**Source:** `SOP-13` §5 and §12; `REG-04`.

Until an external or contract auditor is engaged, or a competent second person joins, clause 9.2.2 c
cannot be met internally. This is reviewed at **every** review: what progress has been made towards
option (a) or (b) in `SOP-13` §5, and whether the interim self-assessment position remains
acceptable. It is listed separately from §5.6 so that it cannot quietly age.

### 5.8 Audit results (9.3.2 c 6)

**Source:** completed `forms/FRM-06_Internal_Audit_Report.md` reports; `SOP-13`.

Reported: audits conducted in the period against the programme in `SOP-13` §4.1; findings by
classification; the state of follow-up; and coverage — which clauses in the cycle remain unaudited.
Where an audit that was scheduled did not happen, that is reported as a failure of the programme,
with the reason.

### 5.9 Monitoring and measurement results (9.3.2 c 5)

**Source:** `SOP-12`.

This heading covers measurement of the QMS itself as distinct from product conformity in §5.5:
whether the measurements defined in `SOP-12` were actually taken, whether they remain the right
measurements, and whether anything is being measured that no one uses.

### 5.10 Performance of external providers (9.3.2 c 7)

**Source:** `registers/REG-02_External_Documents_and_Providers.md`; `SOP-09`; completed
`FRM-04_Supplier_Evaluation.md` records.

Reported: any provider failure in the period; any component whose version moved when it was relied
on not to — the `nemo_toolkit[asr]` installation from the moving branch `@main` at
`backend/Dockerfile:19` is a standing item under this heading, having already caused a real outage
(commit `724fb98`); licence position of each model and component; and any provider due
re-evaluation.

### 5.11 Adequacy of resources (9.3.2 d)

**Source:** the Managing Director's assessment, supported by `SOP-03` (competence) and the risk
register.

Considered: people — capacity, competence gaps and bus factor 1; infrastructure — GPU memory
headroom, storage, and the absence of a backup for the `echomind_data` volume; tooling — the absence
of CI; and time, which in a one-person organisation is the binding constraint on every other
improvement and should be recorded as such rather than treated as unlimited.

### 5.12 Effectiveness of actions taken to address risks and opportunities (9.3.2 e)

**Source:** `registers/REG-03_Risk_Register.md`; `SOP-02` §8.

For each treated risk the question from `SOP-02` §8 is applied: has the risk recurred, and is the
evidence cited in the register still true of the code today? A treatment whose cited evidence no
longer exists in the codebase is an untreated risk. Risks scored High or Critical are reviewed
individually; residual-risk acceptances are re-confirmed or withdrawn.

### 5.13 Opportunities for improvement (9.3.2 f)

**Source:** OFIs in `REG-04`; audit OFIs; `SOP-14` §10.

Each is accepted, deferred with a reason, or rejected with a reason. Accepted opportunities become
actions under §6 with an owning role and a date.

## 6. Outputs (9.3.3)

Every review produces decisions and actions under the three required output headings. An output
recorded without an owning role and a date is a comment, not a decision.

| Output (9.3.3) | What is recorded |
|---|---|
| **a — Opportunities for improvement** | Each accepted opportunity, with the owning role, the target date and how its completion will be recognised |
| **b — Any need for changes to the quality management system** | Changes to procedures, registers, forms, the scope, the quality policy or the quality objectives, each with the document ID affected and the owning role. Changes are implemented under `SOP-01` §6. |
| **c — Resource needs** | People, competence, infrastructure, tooling, time and budget — stated specifically, with what each is needed for, and whether it is approved, deferred or declined |

**Decisions with no action.** Where the review decides to change nothing, that is recorded as a
decision with its reasoning. Silence is not a decision.

**Retention.** Minutes are retained as documented information (9.3.3) on
`forms/FRM-07_Management_Review_Minutes.md`, completed, dated and stored under
`docs/qms/records/management-review/`. The next review date is set before the review closes.

## 7. Leadership commitment (5.1)

Clause 5.1 requires top management to demonstrate leadership and commitment with respect to the QMS.
In this organisation the demonstration is this procedure being followed on time with real evidence,
plus:

| 5.1.1 requirement | How it is demonstrated, and where it is visible |
|---|---|
| Accountability for the effectiveness of the QMS | The Managing Director owns every procedure in `docs/qms/procedures/` and authorises every closure under `SOP-14` |
| Quality policy and objectives established and compatible with strategic direction | `QP-01`, `QO-01`; reviewed at §5.2 and §5.4 |
| Integration of QMS requirements into business processes | The QMS uses Git as its control mechanism rather than a parallel system (`SOP-01` §3) |
| Promoting the process approach and risk-based thinking | `SOP-02` §6–§8 |
| Ensuring resources are available | §5.11 and §6 c |
| Communicating the importance of effective quality management | Applies once there is more than one person; recorded in `registers/REG-05_Competence_and_Training_Record.md` |
| Ensuring the QMS achieves its intended results | §5.4, §5.5 and the review conclusion |
| Engaging, directing and supporting persons | As above, once there is more than one person |
| Promoting improvement | §5.13, §6 a, `SOP-14` §10 |
| Supporting other relevant management roles | As above |

**The honest statement of commitment for this organisation** is that the QMS was written to describe
what actually happens, with the gaps written down rather than papered over. That is checkable: every
procedure ends with a *Current state and gaps* section, and each one is unflattering.

## 8. Records

| Record | Location | Retention |
|---|---|---|
| Management review minutes, including inputs considered, decisions and actions | Completed copy of `forms/FRM-07_Management_Review_Minutes.md`, stored under `docs/qms/records/management-review/` | 3 years (`SOP-01` §8) |
| Evidence pack referenced by the minutes | Cited by path, commit SHA or report filename within the minutes; not duplicated | Per the retention of each source record (`SOP-01` §8) |
| Actions arising | Recorded in the minutes and, where they are corrective actions or accepted OFIs, in `registers/REG-04_Nonconformity_and_CAPA_Log.md` | 3 years after closure |
| Amendments to context, interested parties or risk arising from the review | `SOP-02` §4, §5; `registers/REG-03_Risk_Register.md` | Life of the QMS + 3 years |
| Revised quality objectives | `QO-01`; revision history in Git | Life of the QMS + 3 years |
| Review schedule and next review date | Recorded on the minutes of the preceding review | 3 years |

## 9. Current state and gaps

**Stated plainly: no management review has yet been held at Ajace AI. No minutes exist. No review
cadence has run.** This procedure is the first management review arrangement the organisation has
had, and the first review under it will have no previous actions to report under §5.1 and no audit
results to report under §5.8.

| Gap | Current state | Consequence |
|---|---|---|
| No management review has taken place | No records exist in `docs/qms/records/management-review/`; the directory is created at the first review | Clause 9.3.1 unmet; the QMS has never been reviewed for suitability, adequacy or effectiveness |
| Several 9.3.2 inputs will be empty at the first review | No previous actions (§5.1), no audits (§5.8), no customer feedback records (§5.3) | The first review is largely a baseline exercise; it must record *why* each input is empty rather than skipping the heading |
| Quality objectives not yet measured against targets | `QO-01` is drafted; no measurement cycle has run | §5.4 will report *not measured* for most objectives at the first review |
| Measurement is manual and intermittent | No CI; the golden evaluation runs when someone runs it (most recent recorded run 49/52 on 2026-08-06) | §5.5 reports on whatever happened to be measured, not on a continuous record |
| Product conformity has never reached the organisation's own gate | The golden-question suite gate is binary and has never returned 52/52; best recorded is 50/52 | The gate as currently defined has never been passed; the first review must decide whether the gate or the product is wrong |
| Top management, process owners and the workforce are the same person | `SOP-02` §4 | No challenge function inside the review; the evidence discipline in §3 is the only substitute |
| No versioning to anchor a review period to | No git tags, no CHANGELOG; `frontend/package.json` is `"0.0.0"` | Review periods must be delimited by date and commit SHA rather than by release |
| Resource gaps that will recur under §5.11 until resolved | No backup of the `echomind_data` volume; no data-retention policy; no log rotation; no LICENCE/NOTICE, model-licence inventory or SBOM; application authentication off by default with WebSocket endpoints outside the auth middleware (`backend/app/main.py:135`) | Each is a standing resource or control item for §5.11 and §6 c until closed |

Each gap above is carried into `ISO9001_Gap_Analysis.md`.
