# SOP-14 — Nonconformity and Corrective Action

| Field | Value |
|---|---|
| Document ID | SOP-14 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Managing Director |
| Approved by | `________________________` |
| Approval date | `____ / ____ / ________` |
| ISO 9001:2015 clauses | 10.1, 10.2.1, 10.2.2, 10.3 |

---

## 1. Purpose

To define how Ajace AI reacts to a nonconformity, controls and corrects its consequences, decides
whether corrective action is needed, eliminates the cause so that it does not recur, verifies that
the action was effective, and uses what it learns to improve the quality management system (QMS)
and the product.

## 2. Scope

Every nonconformity affecting EchoMind Enterprise or the QMS, from any source: internal audit,
evaluation failure, test failure, build or deployment failure, customer complaint, the
organisation's own observation while building or operating the system, or the failure of an
externally provided component.

Also covers nonconforming outputs under clause 8.7 — a build, release, model, index or generated
artefact that does not meet requirements — and the continual improvement requirement of 10.3.

## 3. Definitions

| Term | Meaning here |
|---|---|
| **Nonconformity (NC)** | Non-fulfilment of a requirement. The requirement may be a customer requirement, a statutory or regulatory requirement, an ISO 9001:2015 requirement, or a requirement Ajace AI has set itself in this QMS or in a design input. |
| **Correction** | Action to eliminate the detected nonconformity itself. Fixing the broken thing. Does not address the cause. |
| **Containment** | Action to limit the consequences while the correction is being made — stopping a release, disabling a feature, isolating an index, notifying an affected party. |
| **Corrective action** | Action to eliminate the **cause** of a nonconformity so that it does not recur. |
| **Preventive action** | Not a separate clause in ISO 9001:2015. Its intent is met by the risk and opportunity process in `SOP-02`. |
| **Effectiveness verification** | Objective evidence, gathered after the corrective action was implemented, that the cause has actually been eliminated. |

**Correction is not corrective action.** A commit that fixes a bug is a correction. A commit that
fixes the bug *and* adds the test that would have caught it, *and* whose absence caused the class of
defect, is a correction plus corrective action. Most of the historical record at Ajace AI (§9) is
the first without the second being tracked.

## 4. Responsibilities

| Role | Responsibility |
|---|---|
| Anyone who observes a nonconformity | Raises it. Immediately, before deciding whether it matters. Judging severity is a later step performed on a record, not a filter applied before one exists. |
| Managing Director | Owns this procedure and `registers/REG-04_Nonconformity_and_CAPA_Log.md`. Decides severity, decides whether corrective action is required, assigns the owning role and the due date, authorises closure. Decides whether a customer must be notified. |
| Lead Engineer | Performs containment and correction. Conducts root-cause analysis. Implements corrective action and records the evidence in the commit. |
| Verifier | Verifies effectiveness (§8). Where more than one person exists, the verifier is not the person who implemented the action. |

Ajace AI currently has one person holding all of these roles. The consequence — that implementation
and verification of effectiveness are performed by the same person — is recorded in §11 and must be
stated on each record in the *verified by* field of `FRM-05`.

## 5. Raising a nonconformity (10.2.1)

Any of the following **shall** result in a nonconformity record:

| Source | Trigger |
|---|---|
| Internal audit (`SOP-13`) | Any major or minor nonconformity finding |
| Evaluation | A drop in the 52-question golden evaluation against the previous recorded run, or any failing item that was previously passing (`eval/run_eval.py`; the suite gate is binary — `return 0 if total_pass == len(results) else 1` at `eval/run_eval.py:281`) |
| Test failure | Any failing test in `backend/tests/` or `voice/tests/` that reaches `main` |
| Build or deployment failure | Any failure of a clean rebuild, or any container failing its healthcheck in a deployed environment |
| Customer complaint | Any complaint recorded under `SOP-08` on `forms/FRM-08_Customer_Feedback_and_Complaint_Record.md` |
| Own observation | Anything a contributor recognises as not meeting a requirement, including a latent defect found while working on something else |
| External provider failure | A model, image, package or service that does not meet what was relied on — including a pinned dependency that moves (`SOP-09`, `registers/REG-02_External_Documents_and_Providers.md`) |
| QMS itself | A QMS requirement that the organisation cannot currently meet — for example auditor impartiality under `SOP-13` §5 |

**How.** Complete `forms/FRM-05_Nonconformity_and_CAPA_Record.md` and add a line to
`registers/REG-04_Nonconformity_and_CAPA_Log.md`. NC identifiers are sequential and never reused:
`NC-YYYY-nnn`.

**Raise first, assess second.** A nonconformity that turns out on assessment to be trivial is closed
as trivial with the reasoning recorded. A nonconformity never raised leaves no trace at all, and the
pattern it belonged to cannot be seen later.

## 6. Correction, containment and control of nonconforming output (8.7, 10.2.1 a)

On raising the record, and **before** any analysis of cause:

1. **Determine the blast radius.** What is affected, and is any of it in a customer's hands? Record
   the affected releases, deployments, customers, indexes and data. Where nothing is deployed, say
   so on the record — an empty answer that was actually asked is evidence; a blank field is not.
2. **Contain.** Choose and record the containment action:

   | Situation | Containment |
   |---|---|
   | Defect is in an unreleased change | Do not release. Hold the release record (`FRM-03`) open. |
   | Defect is in a deployed release | Roll back per the rollback plan on that release's `FRM-03`, or disable the affected feature by configuration. Commit `2decb99` is the pattern for gating a feature off by default. |
   | Defect affects data isolation or data residency | Stop use of the affected path immediately. Treat as severity **Critical** and go to step 3 without waiting. |
   | Defect is in an externally provided component | Pin to the last known-good version, or remove the component from the runtime path. |

3. **Decide on notification.** Where a customer's data, a customer's compliance position or a
   customer's deployment availability is or may be affected, the Managing Director decides whether
   the customer must be notified, records the decision and its reasoning on `FRM-05`, and records
   the date the customer was informed. **A decision not to notify is recorded with the reason.**
4. **Correct.** Fix the detected nonconformity. Record the commit SHA on `FRM-05`.

Clause 8.7.2 requires that the nonconformity, the action taken, any concession obtained, and who
authorised the action are retained as documented information. `FRM-05` carries all four fields.

## 7. Evaluating the need for corrective action, and root-cause analysis (10.2.1 b, c, d)

### 7.1 Does this need corrective action?

Corrective action is required when the answer to any of the following is yes:

1. Could this nonconformity, or one materially like it, occur again?
2. Does a similar nonconformity exist elsewhere, or could it potentially occur elsewhere in the
   product? (Clause 10.2.1 d requires this question to be asked explicitly.)
3. Was the nonconformity latent — present for some time before it became visible?
4. Did the nonconformity reach, or nearly reach, a customer?
5. Is the severity Major or Critical?

Where the answer to all five is no, the record may be closed with correction only, and the reasoning
is written on the record. "No corrective action required" with no reasoning is not acceptable.

### 7.2 Root-cause analysis

Use **five whys**, and cross-check the result against the **cause categories** below. Both are
required: five whys finds a chain, the categories stop the chain from stopping too early in a
comfortable place.

**Five whys.** State the nonconformity as an observable fact, then ask why until the answer names
something the organisation can change. Record every step, not just the last one.

**Cause categories.** For each nonconformity, state which of these the root cause falls into:

| Category | Question to ask |
|---|---|
| Requirement | Was the requirement ever stated, and stated testably? |
| Design | Did the design allow the failure — a missing constraint, an unguarded path, a default that fails open? |
| Implementation | Was the design right and the code wrong? |
| Verification | Why did testing or evaluation not catch it? This question is asked for **every** nonconformity, whatever the other answers, because a verification gap is a separate cause with a separate action. |
| Process / control | Was a QMS control absent, or present and not followed? |
| External provider | Did a supplied component change or behave other than relied on? |
| Competence / capacity | Was the work done by someone without the knowledge or the time to do it correctly? |

**Stopping rule.** Stop at the first cause that is **actionable by this organisation** and whose
elimination would have prevented the nonconformity. Do not continue down to causes that are true but
useless.

> "Root cause: human error" is never an acceptable stopping point — it names the person, not the
> system that allowed the error to have an effect.
> "Root cause: one developer, bus factor 1" is true of everything here and actionable by almost
> nothing; it belongs in `registers/REG-03_Risk_Register.md`, not as the root cause of a specific
> defect.
> Stop instead at, for example: "the namespace predicate was applied on the dense path but not on
> the sparse path, and no test asserted isolation on the sparse path" — two causes, both actionable.

A nonconformity may have more than one root cause. Record all of them; each gets its own action.

## 8. Implementing action and verifying effectiveness (10.2.1 e, f, g)

1. **Plan the action.** Each corrective action has a description, an owning **role**, and a due
   date. Actions without an owner and a date are not actions.
2. **Implement.** Record the commit, branch or configuration change that delivers it.
3. **Update risks and opportunities (10.2.1 e).** Ask: was this risk in
   `registers/REG-03_Risk_Register.md`? If it was, was the score or the treatment wrong? If it was
   not, why was it not identified, and what does that say about the identification process in
   `SOP-02` §6? Record the register update, or record explicitly that none was needed.
4. **Change the QMS if needed (10.2.1 f).** Where the cause is a missing or inadequate control, the
   procedure that should have contained that control is amended under `SOP-01` §6.
5. **Verify effectiveness.**

> **This is the step that is most often skipped, and skipping it is the single most common reason a
> corrective-action system fails an audit.** A corrective action is **not closed** when the fix is
> committed. It is closed when evidence gathered *after* the action shows the cause is gone.

Effectiveness verification must state four things on `FRM-05`:

| Field | Requirement |
|---|---|
| Method | What was done to test the cause is gone — not what was done to fix it |
| Date | When verification was performed. It is a separate, later date from the implementation date; where it is the same day, the record says so. |
| Evidence | Objective and reproducible: a test that now fails without the fix and passes with it; a full evaluation run with its score; a count from a query; recorded output. A statement that "it works now" is not evidence. |
| Verified by | The role. Where implementer and verifier are the same person (§4), the record says so explicitly. |

Acceptable evidence in this organisation, by category:

| Cause category | Effectiveness evidence that counts |
|---|---|
| Isolation / retrieval correctness | A re-run of the 52-question golden evaluation with the score recorded, plus a direct query count over the affected index |
| Implementation defect | A new or amended test in `backend/tests/` or `voice/tests/` that fails on the pre-fix code, plus a recorded run |
| Build / dependency | A clean, from-scratch rebuild completing with all containers healthy |
| Runtime behaviour | Recorded artefacts from a real run — output files, logs, timings |
| Process / QMS | The changed procedure plus one instance of the process being run under the new version and producing the required record |

6. **Close.** The Managing Director authorises closure on `FRM-05` and updates `REG-04`. Closure
   without a completed effectiveness-verification block is itself a nonconformity against this
   procedure.
7. **Where verification shows the action was not effective**, the record is **not** closed. Re-open
   the root-cause analysis: the cause identified was not the cause. Record the second analysis on the
   same record so the reasoning trail survives.

## 9. Worked example

> ### EXAMPLE — ILLUSTRATION ONLY
>
> The commits and evaluation figures below are real and verifiable in this repository. The layout
> below is **not** a completed nonconformity record: no `NC-` record was raised at the time, no
> owner or due date was assigned, and nothing was logged in `REG-04`. It is reproduced here to show
> what a complete `FRM-05` looks like when populated, and to show that the underlying engineering
> discipline already exists. **It must not be treated as an actual QMS record.**

| Field | Content drawn from commit `4e27109` (2026-08-07) |
|---|---|
| Source | Own observation during evaluation; regression detected by the golden-question suite |
| Description | Retrieved content crossed a tenant namespace boundary on the sparse (keyword) transcript retrieval path. The namespace predicate was enforced on the dense path only. |
| Severity | Critical — impact 5 under `SOP-02` §7 (content crossing a tenant boundary) |
| Detection | Golden-question evaluation score fell from 50/52 to 42/52 as the corpus grew |
| Containment | Defect was in the development corpus, not a customer deployment |
| Root cause — design | The isolation constraint was implemented per-retrieval-path rather than enforced at a single chokepoint, so a second path could be added without it |
| Root cause — verification | No test asserted isolation on the sparse path |
| Correction | Namespace-enforcing wrapper applied so the predicate cannot be bypassed by a retrieval path |
| Effectiveness verification | Query over the affected index returned **0 out-of-namespace hits from 359**; golden evaluation re-run recovered to **49/52** |
| What is still missing to make this a conformant record | An owning role, a due date, a `REG-04` entry, a risk-register update, and a named verifier distinct from the implementer |

Three further real records of the same shape, each with correction + cause + verification present in
the commit body and each **not** tracked as a CAPA record at the time:

| Commit | Nonconformity | Verification evidence recorded |
|---|---|---|
| `558eaae` (2026-07-29) | `/api/transcribe/speak` returned null audio — wrong voice service port, and the failure was silent because it was swallowed by a debug-level `except` | Three real WAV outputs produced and checked |
| `724fb98` (2026-09-04) | Three build and runtime breakages surfaced only by a clean rebuild | From-scratch rebuild completed with all six containers healthy |
| `ebd232f` (2026-07-29) | Five retrieval defects found by the golden-question evaluation, including a namespace leak | Golden-question evaluation re-run |

Two further commits illustrate the **silent-failure** and **measured-worse** patterns that this
procedure exists to catch: `73f0b4f` (2026-07-29) — the voice WebSocket authentication gate was
silently non-functional because the wrong environment variable name was passed, so a control
believed to be operating was not; and `2decb99` (2026-07-30) — a feature was gated off by default
because measurement showed it performed worse than the baseline, which is a correct response to a
negative result rather than a defect.

**The pattern in the evidence.** 22 of 202 commits are `fix:` commits. `eval/paper/results/REGRESSION_AND_FIXES.json`
records a baseline, a regression, a root cause, the fixes and the non-regression evidence, and
states that the defect was **"PRE-EXISTING and latent"**, becoming visible only as the corpus grew.
That is precisely the case in which correction alone is insufficient: a latent defect implies a
verification gap, and the verification gap is a second cause requiring its own action.

## 10. Continual improvement (10.1, 10.3)

Ajace AI improves the suitability, adequacy and effectiveness of the QMS by:

| Input | Mechanism | Where it lands |
|---|---|---|
| Analysis of nonconformity data | Trends across `REG-04` — repeated cause categories, repeated affected components, actions closed without effectiveness evidence | Management review (`SOP-15` §5.6) |
| Evaluation and test results, including negative ones | `SOP-12`; the golden evaluation; `eval/paper/results/SUMMARY.json`, which records honest negative findings against the organisation's own published paper | Management review (`SOP-15` §5.5) |
| Audit findings and observations | `SOP-13` | Management review (`SOP-15` §5.8) |
| Opportunities for improvement raised by anyone | `REG-04`, marked as OFI rather than NC | Management review (`SOP-15` §5.11) |
| Risk register review | `SOP-02` §10 | Management review (`SOP-15` §5.10) |

Improvement opportunities that are accepted become actions with an owning role and a date, tracked
in the same register as corrective actions so that there is one list of outstanding commitments and
not two.

**Reporting a negative result is an improvement act, not a failure.** `eval/paper/results/SUMMARY.json`
recording E1 as contradicting the organisation's own paper, E2 as `not_run` with its reason, and E4
containment as real but silent, is the behaviour this procedure intends to preserve.

## 11. Records

| Record | Location | Retention |
|---|---|---|
| Nonconformity and corrective action record | Completed copy of `forms/FRM-05_Nonconformity_and_CAPA_Record.md`, stored under `docs/qms/records/capa/` | 3 years after closure (`SOP-01` §8) |
| Nonconformity and CAPA log | `registers/REG-04_Nonconformity_and_CAPA_Log.md` | 3 years after closure of the last entry (`SOP-01` §8) |
| Correction and its technical rationale | Git commit body (`SOP-01` §3) | Per `SOP-01` §8 |
| Effectiveness verification evidence | Cited on `FRM-05` by path, commit SHA or evaluation report filename (`eval/reports/`) | 3 years after closure |
| Customer notification decision and date | `FRM-05`; the customer-facing record on `forms/FRM-08_Customer_Feedback_and_Complaint_Record.md` | 3 years (`SOP-01` §8) |
| Risk register update arising from a nonconformity | `registers/REG-03_Risk_Register.md` | Life of the QMS + 3 years |
| Nonconforming output disposition and authorisation (8.7.2) | `FRM-05`, authorisation field | 3 years after closure |

## 12. Current state and gaps

**Stated plainly: corrective action has been practised at Ajace AI and is genuinely evidenced — but
it has never been tracked in a register with owners, due dates and effectiveness verification, until
this QMS.** The practice is real and inspectable: 22 of 202 commits are `fix:` commits; several
commit bodies contain a full cycle of detection, containment, root cause, fix and post-fix
verification (`4e27109`, `558eaae`, `724fb98`, `ebd232f`); and
`eval/paper/results/REGRESSION_AND_FIXES.json` is a structured record of baseline → regression →
root cause → fixes → non-regression evidence. What was missing is the management layer: a single
list, an owner per action, a date, and a deliberate, separately evidenced check that the action
worked.

| Gap | Current state | Consequence |
|---|---|---|
| No nonconformity register before this QMS | `registers/` held no CAPA log at the time of writing | Defects were fixed individually; trends across them were never analysed |
| No NC identifiers, owners or due dates | Evidence exists only as commit SHAs | No way to show an auditor an outstanding action, or that any action was ever overdue |
| Effectiveness verification is present in some commits and absent in others | Compare `4e27109` (0/359 out-of-namespace, eval 49/52) with the general run of `fix:` commits | Closure discipline is inconsistent because there was no closure step to be disciplined about |
| Implementer and verifier are the same person | One person in the organisation (`SOP-02` §4) | Independent verification of effectiveness is not achievable internally; each record must state this |
| No customer complaints process has yet been exercised | `SOP-08` and `FRM-08` are drafted; no complaint has been received | The customer-sourced branch of §5 is untested |
| No CI, so test and evaluation failures are not automatically detected | No `.github/` directory and no CI configuration in the repository | Detection depends on someone choosing to run `eval/run_eval.py` and the test suites; a regression can sit undetected, exactly as recorded in `eval/paper/results/REGRESSION_AND_FIXES.json` ("PRE-EXISTING and latent") |
| Container healthchecks cover 4 of 6 services | `docker-compose.yml` | Two services can fail without the failure being surfaced as a detection source under §5 |
| No release versioning to attach a nonconformity to | No git tags, no CHANGELOG; `frontend/package.json` is `"0.0.0"` | The *affected releases* field on `FRM-05` cannot currently be completed precisely; it must reference a commit SHA instead |
| The QMS's own known nonconformity — auditor impartiality | `SOP-13` §5 and §12 | Must be raised as `NC-____-___` in `REG-04` on adoption and remain open until resolved |

Each gap above is carried into `ISO9001_Gap_Analysis.md` and, where it is a risk rather than only a
gap, into `registers/REG-03_Risk_Register.md`.
