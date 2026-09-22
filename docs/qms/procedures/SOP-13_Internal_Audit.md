# SOP-13 — Internal Audit

| Field | Value |
|---|---|
| Document ID | SOP-13 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Managing Director |
| Approved by | `________________________` |
| Approval date | `____ / ____ / ________` |
| ISO 9001:2015 clauses | 9.2.1, 9.2.2 |

---

## 1. Purpose

To define how Ajace AI plans, conducts, reports and follows up internal audits of the quality
management system (QMS), so that the organisation obtains objective evidence of whether the QMS
conforms to ISO 9001:2015 and to its own requirements, and whether it is effectively implemented
and maintained.

## 2. Scope

The whole QMS: every clause of ISO 9001:2015 that applies to Ajace AI, every procedure in
`docs/qms/procedures/`, and the engineering processes those procedures describe — requirements
capture, design and development, build and release, evaluation, deployment and support of
EchoMind Enterprise.

Audit of a **customer's** use of the deployed product is outside scope; Ajace AI has no access to
customer deployments or customer data.

## 3. Responsibilities

| Role | Responsibility |
|---|---|
| Managing Director | Owns this procedure. Approves the audit programme and any change to it. Selects and appoints the auditor. Receives the audit report. Ensures findings are actioned under `SOP-14`. |
| Auditor | Plans and conducts the audit against the criteria in §6. Gathers objective evidence. Classifies and reports findings. Does not propose the corrective action — that is the auditee's responsibility, so that the audit stays an evaluation and does not become a design activity. |
| Auditee (process owner) | Makes evidence available. Agrees the factual accuracy of each finding, or records disagreement in the report. Raises a nonconformity record under `SOP-14` for each nonconformity finding and owns the resulting action. |

Ajace AI currently has one person, who holds the Managing Director and auditee roles. The
consequences of this for auditor independence are dealt with explicitly in §5 and §11 — not
glossed over.

## 4. The audit programme (9.2.2 a)

**Cycle.** The audit programme runs on a **12-month cycle**. Over one full cycle, every clause of
ISO 9001:2015 that applies and every QMS process is audited at least once.

**Frequency and split.** The cycle is worked in two audit events rather than one, so that a finding
raised early in the year has a scheduled point at which its effectiveness is re-checked:

| Event | Nominal timing | Coverage |
|---|---|---|
| Audit A | Mid-cycle | Clauses 4, 5, 6, 7 — the management-system frame, plus documented information and competence |
| Audit B | Pre-management-review, late cycle | Clauses 8, 9, 10 — the operational core, plus performance evaluation and improvement |

**Importance-weighted frequency.** Clause 9.2.2 a requires the programme to take account of the
importance of the processes concerned, changes affecting the organisation, and the results of
previous audits. The following processes are audited at **every** audit event regardless of the
table in §4.1, because a failure in them has a direct customer or data consequence:

| Process | Why it is audited every time |
|---|---|
| Tenant / namespace isolation of retrieved content | A failure crosses a customer boundary. It has failed before: commit `4e27109` records a real isolation breach on the sparse transcript path. |
| Control of externally provided components (models, base images, packages) | The dependency chain is a moving target; `backend/Dockerfile:19` installs `nemo_toolkit[asr]` from the moving branch `@main`, which has already caused a real build and runtime outage (`724fb98`). |
| Release and change control | The organisation deploys direct-to-main with no CI; the control is entirely procedural, so its operation must be checked. |
| Nonconformity and corrective action | Verification of effectiveness is the step most often skipped anywhere; see `SOP-14` §8. |

**Programme changes.** The Managing Director may bring an audit forward, extend its scope, or add
an unscheduled audit following a significant nonconformity, a customer complaint, a change of
platform or a change in the number of people in the organisation. Any change is recorded in the
audit programme table below and the reason stated.

### 4.1 Audit schedule — clause and process coverage

Planned periods are placeholders until the programme is approved. They are completed by the
Managing Director when the programme is set, and not before.

| ISO 9001:2015 clause | Process / procedure audited | Principal evidence the auditor will seek | Planned period |
|---|---|---|---|
| 4.1, 4.2 | Context and interested parties — `SOP-02` §4, §5 | Current determination; date last reviewed; traceability into `QO-01` | `__________` |
| 4.3, 4.4 | Scope and QMS processes — `QM-01` | Stated scope matches what is actually delivered; justified exclusions | `__________` |
| 5.1, 5.2, 5.3 | Leadership, policy, roles — `QP-01` | Policy communicated and current; roles assigned in each procedure §3 | `__________` |
| 6.1 | Risk and opportunity — `SOP-02` §6–§8, `REG-03` | Register live; treatments evidenced by path; residual acceptances dated and signed | `__________` |
| 6.2 | Quality objectives — `QO-01` | Objectives measurable; measurement actually taken; source data exists | `__________` |
| 6.3 | Planning of changes — `SOP-02` §9 | Commit bodies show purpose, consequences, integrity, resources | `__________` |
| 7.1 | Resources, infrastructure, monitoring resources | Hardware and GPU capacity adequate for the stated workload; calibration not applicable — no measuring instrumentation | `__________` |
| 7.2, 7.3 | Competence and awareness — `SOP-03`, `REG-05` | `FRM-09` records; evidence behind each claimed competence | `__________` |
| 7.4 | Communication | How changes reach customers; release notes or their absence | `__________` |
| 7.5 | Documented information — `SOP-01` | Header tables complete; status values honest; nothing prohibited by `SOP-01` §9 in the repository | `__________` |
| 8.1 | Operational planning and control | Deployment procedure matches what is actually run | `__________` |
| 8.2 | Customer requirements — `SOP-04` | Requirements recorded and reviewed before commitment; `FRM-08` records | `__________` |
| 8.3 | Design and development — `SOP-05` | Design reviews (`FRM-01`), verification and validation results, design changes | `__________` |
| 8.4 | Externally provided processes, products and services — `SOP-09`, `REG-02` | `FRM-04` evaluations; licence obligations identified; pinned versions | `__________` |
| 8.5 | Production and service provision, change control, customer property — `SOP-11`, `SOP-06` | `FRM-02` change requests; handling of customer data and documents | `__________` |
| 8.6 | Release of products and services — `SOP-06` | `FRM-03` release records; evidence that verification actually passed before release | `__________` |
| 8.7 | Control of nonconforming outputs — `SOP-14` §4–§5 | Containment recorded; affected deployments identified | `__________` |
| 9.1 | Monitoring, measurement, analysis and evaluation — `SOP-12` | Golden-evaluation runs; unit-test runs; healthcheck coverage; what is done with the numbers | `__________` |
| 9.2 | Internal audit — this procedure | Programme exists and has been followed; auditor competence and independence | `__________` |
| 9.3 | Management review — `SOP-15` | Minutes exist, cover every 9.3.2 input, and produce 9.3.3 outputs | `__________` |
| 10.1, 10.2, 10.3 | Improvement, nonconformity and corrective action — `SOP-14`, `REG-04` | Effectiveness verification present and evidenced on closed records | `__________` |

Clause 8.5.1 f (validation of processes where output cannot be verified by subsequent monitoring)
and clause 7.1.5.2 (measurement traceability) are examined for applicability at each audit rather
than assumed inapplicable.

## 5. Auditor selection, competence and impartiality (9.2.2 c)

**The requirement.** ISO 9001:2015 9.2.2 c requires the organisation to select auditors and conduct
audits in a way that ensures objectivity and the impartiality of the audit process, and states that
**auditors shall not audit their own work**.

**The position at Ajace AI.** The organisation currently consists of one person, who writes the
code, operates the deployment, owns every QMS procedure and would be the only available auditor.
**It is therefore not possible to satisfy 9.2.2 c internally.** This procedure does not attempt to
disguise that by redefining independence.

**Competence criteria** for whoever conducts the audit:

| Criterion | Minimum |
|---|---|
| Knowledge of ISO 9001:2015 | Able to audit against clause text, not against a generic checklist |
| Auditing method | Trained or demonstrably experienced in evidence-based auditing and finding classification |
| Technical understanding | Sufficient to read a Git history, a Dockerfile, a test suite and an evaluation report, or willing to be walked through them and to challenge what is shown |
| Independence | Not the author or operator of the work being audited, and no interest in the outcome |

**Options for meeting the impartiality requirement**, in order of preference:

| Option | Description | Status |
|---|---|---|
| (a) External / contract auditor | Engage a competent independent auditor or consultancy to conduct the annual internal audit against this programme. This is the only option that fully satisfies 9.2.2 c while the organisation has one person. | **Preferred. Not yet engaged.** |
| (b) Competent second person | Once Ajace AI has a second competent person, allocate audits so that neither person audits work they performed. This satisfies 9.2.2 c for the parts they did not perform, and does not satisfy it for the parts they did — so the allocation must be recorded. | Not available; the organisation has one person. |
| (c) Interim structured self-assessment | The Managing Director works through the checklist in `forms/FRM-06_Internal_Audit_Report.md` against objective evidence — file paths, commit SHAs, test and evaluation output — recording the evidence examined for every clause, including where the answer is "no evidence found". | Available now. **Does not satisfy 9.2.2 c.** |

**Option (c) is explicitly not conformant.** A self-assessment is a useful management tool and it
produces real findings, but it is not an internal audit within the meaning of 9.2.2 c and must never
be reported as one. Where a self-assessment is performed:

1. The report is titled **Self-assessment**, not *Internal audit*, and the independence statement in
   `FRM-06` records plainly that the auditor audited their own work.
2. The absence of an independent internal audit is itself raised and maintained as an **open
   nonconformity of the QMS** in `registers/REG-04_Nonconformity_and_CAPA_Log.md`, against clause
   9.2.2 c, and remains open until option (a) or (b) is delivered.
3. That open nonconformity is a standing item at every management review (`SOP-15` §5.7).

## 6. Audit criteria

Each audit is conducted against:

1. ISO 9001:2015, clause by clause, for the clauses in the audit scope;
2. the Ajace AI QMS documents in `docs/qms/` at the revision in force on the audit date;
3. applicable customer and statutory requirements identified under `SOP-04`;
4. the organisation's own quality objectives in `QO-01`.

The criteria are stated in the audit plan before the audit starts, and reproduced in the report.

## 7. Conducting the audit

1. **Plan.** The auditor prepares the plan: scope, criteria, clauses and processes, dates, and who
   will be interviewed. The plan is issued to the auditee before the audit.
2. **Opening.** The auditor confirms scope, criteria, method and the finding classifications in §8.
3. **Gather objective evidence.** Evidence is examined, not asserted. For this organisation the
   evidence is overwhelmingly documentary and inspectable:

   | Evidence type | Where the auditor looks |
   |---|---|
   | Change and rationale records | `git log`, commit bodies (`SOP-01` §3 makes the commit the change record) |
   | Product conformity | `eval/run_eval.py` reports in `eval/reports/`; `eval/paper/results/SUMMARY.json`; `eval/paper/results/REGRESSION_AND_FIXES.json` |
   | Test results | `backend/tests/`, `voice/tests/` and the recorded output of a run |
   | Runtime controls | `docker-compose.yml`, `backend/app/core/config.py`, `backend/app/main.py` |
   | QMS records | `docs/qms/registers/`, completed `FRM-` forms |
   | Practice | Interview and demonstration — asking the process owner to perform the step and observing what it actually produces |

4. **Sample and trace.** Where records are numerous, the auditor samples and states the sample in
   the report. Vertical tracing is preferred: take one release, or one defect, and follow it end to
   end through requirements, design, change control, verification, release and any resulting
   corrective action.
5. **Test the negative.** For every control the QMS claims, the auditor asks what would happen if it
   failed and looks for the evidence that it did not. A control with no failure mode and no evidence
   is not a control.
6. **Confirm factual accuracy.** Each potential finding is put to the auditee before the report is
   issued, so that disagreement is about fact, not about surprise.
7. **Closing.** The auditor presents the findings, the classification of each, and the conclusion.

## 8. Classification of findings

| Classification | Definition | Required response |
|---|---|---|
| **Major nonconformity** | The absence of a required process or control; a total breakdown of one; or a nonconformity that has resulted, or is likely to result, in the delivery of nonconforming product to a customer — including any breach of data isolation or of the offline-deployment guarantee. Also: a systematic pattern of minor nonconformities in the same process. | Nonconformity record raised under `SOP-14` immediately. Containment considered within one working day. Corrective action with root cause and effectiveness verification required. |
| **Minor nonconformity** | A single lapse in an otherwise implemented process; a requirement met in practice but not evidenced; a record incomplete. Does not by itself put product conformity at risk. | Nonconformity record raised under `SOP-14`. Correction and, where the cause is likely to recur, corrective action. |
| **Observation** | A factual statement about a weakness that is not yet a nonconformity — a control that works but depends on one person remembering, or evidence that exists but is not retrievable. | Recorded. Reviewed at management review. Does not require a corrective action but may lead to one. |
| **Opportunity for improvement (OFI)** | A suggestion that would improve effectiveness or efficiency where no requirement is currently breached. | Recorded and considered under `SOP-14` §10 (continual improvement). Never used to soften something that is actually a nonconformity. |

A finding is written as: the **requirement**, the **evidence examined**, and the **discrepancy**
between them. "Poor configuration management" is not a finding. "Clause 8.5.1: the release
`__________` was deployed with no record of the source revision; `FRM-03` for that release has no
git SHA recorded" is a finding.

## 9. Reporting (9.2.2 d, 9.2.2 f)

The auditor completes `forms/FRM-06_Internal_Audit_Report.md` and issues it to the Managing
Director within **10 working days** of the closing meeting. The report contains: audit identifier,
scope, clauses and processes audited, criteria, auditor and the independence statement, dates,
evidence examined, the findings table, and a conclusion on **both** conformity and effectiveness —
these are different questions and both must be answered.

Reports are retained under `SOP-01` §8 and are an input to management review (`SOP-15` §5.8).

## 10. Follow-up (9.2.2 e)

1. Every major and minor nonconformity is raised as a record under `SOP-14` using
   `forms/FRM-05_Nonconformity_and_CAPA_Record.md`, logged in
   `registers/REG-04_Nonconformity_and_CAPA_Log.md`, with an owning role and a due date.
2. Corrections are made without undue delay. Corrective action follows the full `SOP-14` cycle.
3. **An audit finding is not closed when the action is done. It is closed when the auditor, or the
   Managing Director where no auditor is available, has verified with evidence that the action was
   effective** — see `SOP-14` §8. Closure without effectiveness evidence is itself a nonconformity.
4. Open findings are carried forward to the next audit event and re-examined there.

## 11. Records

| Record | Location | Retention |
|---|---|---|
| Audit programme and schedule | §4 and §4.1 of this document; revision history in Git | Life of the QMS + 3 years (`SOP-01` §8) |
| Audit plan | `docs/qms/records/audits/` (created at first audit) | 3 years (`SOP-01` §8) |
| Internal audit report, including findings and independence statement | Completed copy of `forms/FRM-06_Internal_Audit_Report.md`, stored under `docs/qms/records/audits/` | 3 years (`SOP-01` §8) |
| Nonconformities arising from audit | `registers/REG-04_Nonconformity_and_CAPA_Log.md` and completed `FRM-05` records | 3 years after closure (`SOP-01` §8) |
| Auditor competence and independence evidence | `registers/REG-05_Competence_and_Training_Record.md` for internal auditors; the engagement record for an external auditor | Duration of engagement + 3 years |
| Review of audit results | Management review minutes (`SOP-15`) | 3 years (`SOP-01` §8) |

## 12. Current state and gaps

**Stated plainly: no internal audit has yet been conducted at Ajace AI. There is no audit
programme in operation, no audit report, and no auditor appointed.** This procedure is the first
audit arrangement the organisation has had; all planned periods in §4.1 are blank because the
programme has not been approved.

**The impartiality requirement of 9.2.2 c cannot currently be met internally.** With one person,
any internally conducted audit is an audit of the auditor's own work. This is recorded as an open
nonconformity of the QMS itself, against clause 9.2.2 c, in
`registers/REG-04_Nonconformity_and_CAPA_Log.md`, and it remains open until an external or contract
auditor is engaged, or a competent second person joins the organisation. Until then any internal
exercise is titled a **self-assessment** and is not presented as satisfying clause 9.2 (§5).

| Gap | Current state | Consequence |
|---|---|---|
| No internal audit ever conducted | No audit records exist in the repository | Clause 9.2.1 unmet; the QMS has never been independently checked against evidence |
| No independent auditor available | One person in the organisation (`SOP-02` §4) | Clause 9.2.2 c unmet; recorded as an open nonconformity of the QMS |
| Audit programme not approved | Planned periods in §4.1 are placeholders | Coverage of all clauses over a cycle is defined but not scheduled |
| No `docs/qms/records/` directory yet | Created when the first audit or review generates a record | Record locations in §11 are defined in advance, not populated |
| No prior findings to weight the programme | 9.2.2 a requires results of previous audits to inform the programme; there are none | The first cycle is weighted on risk and on the defect history in `SOP-14` §9 instead |
| Much of the auditable evidence is in Git commit bodies rather than in records | `SOP-02` §4 internal issues; `SOP-01` §3 | Auditable, but only by an auditor willing and able to read a Git history; slows audit and raises the competence bar in §5 |

Each gap above is carried into `ISO9001_Gap_Analysis.md`.
