# SOP-05 — Design and Development Control

| Field | Value |
|---|---|
| Document ID | SOP-05 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Lead Engineer |
| Approved by | `________________________` |
| Approval date | `____ / ____ / ________` |
| ISO 9001:2015 clauses | 8.3.1, 8.3.2, 8.3.3, 8.3.4, 8.3.5, 8.3.6 |

---

## 1. Purpose

To define how Ajace AI plans, controls and records the design and development of EchoMind
Enterprise, so that what is built meets the requirements determined for it, and so that the evidence
of that is retained and can be produced.

## 2. Scope

The design and development of the EchoMind Enterprise platform in all its parts — the backend
services, the retrieval and generation pipeline, live transcription, the voice assistant, document
generation, the front end, and the deployment configuration — together with the per-customer
customisation of those parts.

It covers design work performed with the assistance of AI coding tools, which is a material part of
how this product is built and is treated explicitly in §7.

## 3. Responsibilities

| Role | Responsibility |
|---|---|
| Lead Engineer | Owns this procedure. Plans each piece of design work, states its inputs, performs and records verification and validation, and produces the design outputs. Is accountable for every line committed, including AI-generated lines. |
| Managing Director | Approves the transition of a design output to release. Confirms that customer, statutory and regulatory inputs are complete before build starts. Chairs design reviews. |
| Anyone contributing code | Applies this procedure to their own changes without exception, including small ones. |

With one person holding all roles, the acts of designing, reviewing and approving are performed by
the same person. §11 states what an auditor should take from that.

## 4. Design and development planning (8.3.2)

Before work starts on anything beyond a trivial correction, the Lead Engineer determines and records
the items below. For a small change, the record is the commit body. For a substantial change — a new
capability, a change to the retrieval pipeline, a change to the deployment model, or a
customer-specific customisation — the record is a design plan entered in
`registers/REG-07_Design_and_Change_Register.md`.

| 8.3.2 item | How it is determined here |
|---|---|
| (a) Nature, duration and complexity | Judged against the three change classes in §4.1 |
| (b) Process stages, including design reviews | §4.2 |
| (c) Verification and validation activities | §6, with the points at which each applies fixed in §4.2 |
| (d) Responsibilities and authorities | §3 |
| (e) Internal and external resource needs | GPU capacity on the reference platform; model weights; any new dependency, which must be assessed under `SOP-02` before it is introduced |
| (f) Interfaces between the people involved | Not currently material (one person); becomes material on the first additional contributor |
| (g) Involvement of customers and users | Through requirements captured under `SOP-04` §4, and through the golden-question sets that encode expected behaviour per vertical pack |
| (h) Requirements for subsequent provision | Deployment and operating documentation — `OFFLINE_DEPLOYMENT.md`, `docs/PUBLIC_DEPLOYMENT.md`, `docs/USER_MANUAL.md` |
| (i) Level of control expected by customers and interested parties | Regulated-sector customers expect grounding, isolation and data residency to be demonstrable, not asserted — hence the evaluation gate in §6.2 |
| (j) Documented information needed to demonstrate requirements were met | §10 |

### 4.1 Change classes

| Class | Definition | Planning record required |
|---|---|---|
| **A — Corrective** | Fixes defective behaviour without changing intended behaviour | Commit body stating cause, fix and verification |
| **B — Substantive** | Changes intended behaviour, the retrieval or generation path, the data model, the deployment topology, or any dependency | Design plan entry in `registers/REG-07`, plus a design review under §5 |
| **C — Customer customisation** | Configures or extends the platform for a named customer | Design plan entry linked to the requirements review record from `SOP-04` §5 |

A change that weakens grounding, abstention, tenant isolation, or the offline property of the
deployment is always Class B, however small the diff.

### 4.2 Stages and control points

| Stage | Exit condition | Control applied |
|---|---|---|
| 1. Inputs stated | The inputs in §5 are written down and are unambiguous, complete and not in conflict | Design review (Class B and C) |
| 2. Design and build | Working implementation exists | AI-output verification (§7) applies continuously |
| 3. Verification | The outputs meet the stated inputs | §6.1 — unit tests, golden-question suite, targeted measurement |
| 4. Validation | The result meets the intended use in the customer's context | §6.2 |
| 5. Release | Managing Director approves | Release record in `registers/REG-08_Release_and_Deployment_Register.md` |

## 5. Design and development inputs (8.3.3)

Inputs are stated before build begins and are recorded in the design plan. ISO 9001:2015 8.3.3
requires the organisation to consider each of the following; for this product each has a concrete
meaning, set out below.

### 5.1 Functional and performance requirements

Drawn from `SOP-04` §4 for a customer engagement, and from the product baseline in
`docs/CAPABILITIES.md` (284 lines) otherwise. Where a performance requirement is stated it must be
stated as a measurable figure with the conditions of measurement, because the harness in
`eval/paper/` records that unqualified performance intuitions have been wrong here:
`eval/paper/results/SUMMARY.json` (E1) records that retrieval accounted for 2.8% of grounded
response time and generation for approximately 97%, contradicting the assumption in the paper text.

### 5.2 Information derived from previous similar design activities

The pipeline documentation is the accumulated record of previous design activity and is treated as
an input, not as background reading: `docs/RAG_FLOW.md` (281 lines), `docs/RAG_AND_CHUNKING_EXPLAINED.md`
(315 lines), `docs/CHAT_AND_RAG_FLOW.md` (273 lines, per-message path and LLM call accounting),
`docs/CHAT_SESSION_AND_SUMMARY_FLOW.md`, `docs/CONVERSATION_AI_AND_WAKE_WORD_FLOW.md` (181 lines),
`docs/TRANSCRIPT_STORAGE_FLOW.md` (171 lines), `backend/app/transcribe/PROTOCOL.md` (the WebSocket
protocol specification) and `docs/RAG_AUDIT.md` (183 lines, a self-audit dated February 2026).

### 5.3 Statutory, regulatory and standards inputs

Obligations applicable to Ajace AI, and those the delivered system must not put the customer in
breach of — data protection, data residency, record retention, and professional-advice boundaries in
the legal and healthcare packs. Captured per engagement under `SOP-04` §4. ISO 9001:2015 itself
applies through this QMS: the commitments in `QP-01` §2.1 (grounding), §2.2 (data stays where the
customer put it) and §2.3 (honest statement of capability) are binding design inputs and may not be
traded away for a feature.

### 5.4 Consequences of failure — and what they require as inputs

This is the input category that matters most for this product, and it is stated in full.

**The consequence of failure for EchoMind Enterprise is an ungrounded or fabricated answer reaching a
user in a regulated sector.** A clinician, a lawyer, an analyst in a defence or finance context acts
on what the system tells them. A confident answer that is not supported by the customer's own
material is worse than no answer, because it is indistinguishable from a correct one at the point of
use. A secondary consequence of comparable severity is content crossing a tenant boundary between
vertical packs.

Because that is the consequence, the following are **mandatory design inputs for any work touching
retrieval, generation or the conversational path**, and a design that does not satisfy them is not
acceptable:

| Mandatory input | What it requires | Where the current implementation stands |
|---|---|---|
| **Grounding** | An assertion about the customer's material must be supported by retrieved content and must carry attribution | Citation handling in `backend/app/rag/citation_utils.py`; strict-citation retry path at `backend/app/rag/advanced.py:999`, which falls through to insufficient context when a second attempt still produces no citations |
| **Abstention** | Where retrieval is insufficient, the system says so deterministically rather than answering from the model's own parameters | `backend/app/rag/advanced.py:74` — a deterministic insufficient-context message, with the comment stating its purpose: "no hallucination fallback to general chat"; source type `"insufficient"` is a first-class outcome at `backend/app/rag/advanced.py:1519` |
| **No false abstention** | Abstention must not be so aggressive that valid cited answers are rejected | Recorded as a real defect already encountered and fixed — `backend/app/rag/advanced.py:919` notes that valid cited answers were previously rejected as "insufficient context" |
| **Tenant isolation** | Every retrieval path must apply the namespace predicate; there must be no path that bypasses it | `backend/app/rag/index.py:36` (`_ns_ok`), applied at `:681`, `:855`, `:885`, `:920`. Commit `4e27109` records a path that bypassed it and returned 10/10 out-of-namespace chunks until fixed |
| **Prompt-injection resistance** | Content retrieved from customer documents must not be able to redirect the system's instructions | Guards asserted by `backend/tests/test_prompt_guards.py:10`, `:17`, `:23`, `:39`; measured by `eval/paper/e4_injection.py` |
| **Persona boundaries** | Sector personas must carry the disclaimers their sector requires, consistently | `backend/tests/test_prompt_guards.py:29` asserts the lawyer disclaimer is consistent across variants; the conversational golden set asserts the converse — a greeting must carry no disclaimer (`eval/README.md`) |
| **Offline operation** | No runtime dependency on an outbound network call | `docker-compose.yml:170` disables the runtime Piper voice download from Hugging Face for offline deployments; ASR weights are pre-downloaded at build time (`backend/Dockerfile:34-41`) |

### 5.5 Adequacy of inputs

Inputs are reviewed for adequacy before build starts: they must be complete, unambiguous and not in
conflict with one another. Conflicts between inputs — most commonly between a customer's desire for
an always-helpful assistant and the abstention requirement in §5.4 — are resolved in favour of the
mandatory inputs, and the resolution is recorded in the design plan and communicated to the customer
under `SOP-04` §6.

## 6. Design and development controls (8.3.4)

Three distinct activities, which must not be conflated.

- **Review** asks: *is the design going to meet the requirements?* — a judgement, made by people.
- **Verification** asks: *does the output meet the input?* — a measurement against the specification.
- **Validation** asks: *does the result work for the intended use?* — a measurement against reality.

### 6.1 Verification

| Verification mechanism | What it covers | Where it lives | Gate |
|---|---|---|---|
| Unit test suites | Normalisation, prompt guards, RAG behaviour, transcript audit fixes, document-generation images, and named fix sets | `backend/tests/` — 40 test functions across 7 files | Run by the author; no automated gate (see §11) |
| Voice unit tests | Hold phrases, STT health, echo memory | `voice/tests/` — 38 test functions across 3 files | As above |
| Golden-question regression suite | 52 questions across the five vertical packs (7 each: bank, fmr, health, law, meetings, retail) plus 10 conversational items, scoring expected documents cited, expected facts present, forbidden strings absent, citation counts, and document precision | `eval/run_eval.py`, `eval/golden/*.jsonl`, `eval/README.md` | **Binary pass/fail.** `eval/run_eval.py:281` returns `0` only if every item passes, otherwise `1`. A JSON report is written to `eval/reports/` |
| Targeted measurement for the change at hand | Whatever demonstrates the specific claim of the change | Recorded in the commit body | Author judgement |
| Chunk coverage check | Ingestion completeness | `eval/test_chunk_coverage.py` | Run on demand |
| Voice end-to-end and cadence checks | Full-duplex behaviour and speech cadence | `eval/voice_e2e_test.py`, `eval/voice_cadence_test.py` | Run on demand |

**The golden-question suite is the primary verification instrument for the retrieval path**, because
it is the only mechanism that measures the mandatory inputs in §5.4 as a set rather than one at a
time. Its use as a real gate is demonstrated by commit `4e27109`, which reports the regression it
caught (50/52 → 42/52), the fix, and the post-fix score (49/52) together with a path audit (0/359
out-of-namespace hits).

**Adding a golden question is part of the design work**, not an afterthought. Every `expect_facts`
variant must be a string that verifiably appears in the source document — `eval/README.md` states
the rule as "mine the corpus, don't guess".

### 6.2 Validation

Validation asks whether the result works for the intended use. Three mechanisms:

| Mechanism | What it validates | Evidence |
|---|---|---|
| Behaviour on real material in a running deployment | That the capability works end to end on the reference hardware, not only in tests | Recorded in the commit body — commit `558eaae` is the pattern: "card mode 122 KB WAV, summary mode 541 KB WAV, and through the public site 83 KB WAV (RIFF header confirmed)" |
| The paper evaluation harness, E1–E10 | The architectural claims of the design: latency decomposition (E1), retrieval scaling (E2), permission and path isolation (E3, E3b), injection resistance (E4), grounding (E5), outcome and interaction quality (E6, E7), process (E8), weighting (E9) and full ablation (E10) | `eval/paper/` scripts; results in `eval/paper/results/` including `SUMMARY.json` and `REGRESSION_AND_FIXES.json` |
| Customer acceptance against the criteria agreed in `SOP-04` §5 | That the delivered customisation meets the customer's intended use | Customer acceptance record; none exists yet (`SOP-04` §11) |

**Validation results are reported as measured, including when they contradict the design intent.**
`eval/paper/results/SUMMARY.json` is the standard: E1 is recorded as contradicting §7.2 of the paper;
E3 is recorded as only partially supporting §6.1, because the content and wording side channels are
closed but a timing channel is not; E2 is recorded as `not_run` with the reason, rather than
estimated. A validation result that is inconvenient is still the result.

### 6.3 What a design review is, in this organisation

A design review is a scheduled, recorded examination of a design against its stated inputs, held
before the design is built (Class B and C work) and before release. It is not a code read-through
and it is not the same as verification.

A design review must:

1. examine the design against every input stated in §5, including each mandatory input in §5.4;
2. identify problems and determine the necessary actions, with an owning role and a date;
3. state explicitly what could go wrong if the design is wrong, in terms of the consequence in §5.4;
4. reach one of three outcomes: **proceed**, **proceed with the recorded actions**, or **do not
   proceed**;
5. leave a record.

**The record is `forms/FRM-01_Design_Review_Record.md`**, completed at the time of the review and
filed in `registers/REG-07_Design_and_Change_Register.md`. It names the participants, the date, the
design under review, the inputs it was reviewed against, the problems found, the actions with owners
and dates, and the outcome. It is not written retrospectively and it is not pre-filled.

Where the organisation has one person, the review is a self-review. It is still held, still recorded,
and the record states plainly that it was a self-review — that is more useful to an auditor than a
record that implies participants who were not there.

## 7. AI-assisted design and its mandatory verification control

**AI-assisted code generation is part of the design process at Ajace AI.** This is not a caveat; it
is a statement of how the product is built. Many substantive commits in this repository carry a
`Co-Authored-By: Claude …` trailer, including `558eaae` and `4e27109`, and the declaration is
deliberate.

**The control:**

> Verification of AI-generated output is a mandatory design control. It is not optional, it is not
> proportionate to the apparent size of the change, and it is not discharged by the code appearing
> to work. The human author reviews the change against the design inputs in §5 and records, in the
> commit body, what was verified and how.

Three specific requirements follow.

1. **Read every line before committing it.** An AI assistant produces plausible code; plausibility is
   precisely the failure mode this product exists to defend against, and the same standard applies
   to its own construction.
2. **Verify against the mandatory inputs, not only against the immediate task.** The defect in commit
   `4e27109` was a single retrieval path that did not apply the namespace predicate the other paths
   applied. It worked, it passed casual inspection, and it silently broke tenant isolation. The
   question to ask of any generated change touching retrieval is: *does this path apply `_ns_ok`?*
3. **Record the verification.** The commit body states what was checked and what the result was. A
   commit body that states only what was changed is not sufficient for a Class B change.

**Declaration.** Where AI assistance was material to a change, the `Co-Authored-By` trailer is
retained. Ajace AI does not remove it to make the development history appear wholly human-authored.
Customers and auditors are entitled to know how the software was built.

The competence dimension of this — that reviewing generated output requires more competence in the
area, not less — is in `SOP-03` §7.

## 8. Design and development outputs (8.3.5)

Outputs are produced in a form that can be verified against inputs, and are approved before release.

| 8.3.5 requirement | Output in this organisation |
|---|---|
| (a) Meet the input requirements | Source code in `backend/`, `voice/`, `frontend/`; deployment configuration in `docker-compose.yml`, `backend/Dockerfile`, `voice/Dockerfile` |
| (b) Adequate for the subsequent processes for provision of products and services | Deployment documentation: `OFFLINE_DEPLOYMENT.md`, `docs/PUBLIC_DEPLOYMENT.md`, `scripts/export_offline_bundle.sh`, `scripts/import_offline_bundle.sh`, `scripts/verify_offline_readiness.sh` |
| (c) Include or reference monitoring and measuring requirements, and acceptance criteria | Golden-question sets in `eval/golden/*.jsonl` are the acceptance criteria in executable form; the binary gate is `eval/run_eval.py:281`. Unit tests in `backend/tests/` and `voice/tests/` |
| (d) Specify the characteristics essential for the intended purpose and safe and proper provision | The architecture documentation in `docs/` listed in §5.2; the rationale comments carried inline where the configuration is easy to get wrong — `docker-compose.yml:7-11`, `:41-44`, `:69-72`, `:108-112`; the operating documentation `docs/USER_MANUAL.md` (1257 lines, 16 chapters) |

Outputs are approved before release by the Managing Director, recorded in
`registers/REG-08_Release_and_Deployment_Register.md`.

**Documentation is an output, not a by-product.** A Class B change that alters observable behaviour
and does not update the affected document in `docs/` is incomplete, and the design review outcome
for it is "proceed with the recorded actions" at best.

## 9. Design and development changes (8.3.6)

Changes made during, or subsequent to, the design and development of the platform are identified,
reviewed and controlled to the extent necessary to ensure no adverse impact on conformity to
requirements.

| Step | Requirement | Record |
|---|---|---|
| Identify | The change is classified A, B or C under §4.1 before work starts | Design plan or commit body |
| Review | Class B and C changes are reviewed under §6.3 | `forms/FRM-01_Design_Review_Record.md` |
| Assess impact | What else depends on what is being changed, and what could regress | Commit body; for Class B, the design plan |
| Verify | §6.1 applies. For anything touching retrieval, generation or the conversational path, the golden-question suite is run and the result recorded | `eval/reports/`, commit body |
| Authorise | Class B and C changes are authorised by the Managing Director before release | `registers/REG-08_Release_and_Deployment_Register.md` |
| Record the actions taken to prevent adverse impact | Stated in the commit body | Git history (`SOP-01` §3) |

**Where the change originates with the customer**, `SOP-04` §6 applies first; this section applies to
the technical change that follows.

**Rollback.** Where a change carries a credible risk of regression, the rollback path is stated
before the change is made. The existing practice is to state it inline — `docker-compose.yml:10-11`
and `:72` both record the exact rollback to apply if the change regresses. This practice is adopted
as a requirement for Class B changes.

## 10. Records

| Record | Location | Retention |
|---|---|---|
| Design plan, per Class B or C item | `registers/REG-07_Design_and_Change_Register.md` | Life of the product release + 3 years (SOP-01 §8) |
| Design review record | `forms/FRM-01_Design_Review_Record.md`, filed in REG-07 | As above |
| Design inputs, as stated before build | Design plan; for Class A, the commit body | As above |
| Verification results — unit tests | Run output; failures resolved before commit | Not currently retained (see §11) |
| Verification results — golden-question suite | `eval/reports/eval_<run_id>.json`, written by `eval/run_eval.py` | Life of the release + 3 years |
| Validation results — paper harness | `eval/paper/results/`, including `SUMMARY.json` and `REGRESSION_AND_FIXES.json` | As above |
| Rationale and verification for each change | Git commit body (`git log`) — the change record under `SOP-01` §3 | Per SOP-01 §8 |
| Design outputs | The repository itself, at the tagged or recorded commit | Per SOP-01 §8 |
| Release approval | `registers/REG-08_Release_and_Deployment_Register.md` | Life of the release + 3 years |
| Customer acceptance against agreed criteria | Filed under `SOP-04` §10 | 3 years after the engagement ends |

## 11. Current state and gaps

Design control at Ajace AI is **partially evidenced and materially incomplete**. The evidence that
does exist is genuine and unusually strong for an organisation of this size; the gaps are structural
and none of them is closed by this procedure alone.

### 11.1 What is genuinely in place

| Control | Evidence |
|---|---|
| Documented architecture | Nine design documents in `docs/` plus `backend/app/transcribe/PROTOCOL.md`, totalling roughly 2,900 lines, covering retrieval, chunking, chat flow, sessions, wake word, transcript storage and a self-audit |
| Design rationale and verification recorded per change | Commit bodies that state the defect mechanism, the fix, the blast radius and the measured verification — `558eaae` and `4e27109` are representative |
| Executable acceptance criteria for the retrieval path | 52 golden questions across the five vertical packs and a conversational set, with a binary pass/fail gate at `eval/run_eval.py:281` |
| Unit test coverage of named risk areas | 40 test functions in `backend/tests/`, 38 in `voice/tests/` |
| Independent-style validation of architectural claims | `eval/paper/` E1–E10, with results recorded including those that contradict the design intent |
| Rationale preserved where configuration is easy to get wrong | `docker-compose.yml:7-11`, `:41-44`, `:69-72`, `:108-112`, `backend/Dockerfile:30-32` |

### 11.2 What is missing

| Gap | Actual state | Why it matters |
|---|---|---|
| **No architecture decision records** | No ADR directory or equivalent exists. Decision rationale is distributed across commit bodies and inline comments, and only for decisions taken after the commit-body discipline was adopted — 80 of 202 commits use conventional prefixes, the convention having been adopted partway through | A decision cannot be reviewed if it cannot be found. Rationale predating the discipline is unrecoverable except from the person who made it |
| **No acceptance criteria recorded before build** | Golden questions exist, but they were written alongside or after the implementation, not stated as inputs beforehand. No design plan exists for any past change | 8.3.3 requires inputs to be determined before the design; today the inputs are inferred from the outputs |
| **No requirements or traceability document** | There is no document stating the platform's requirements, and no mapping from a requirement to the test that demonstrates it | It cannot currently be shown that every requirement is verified, only that many tests pass |
| **No design review records** | No design review has been held or recorded. `forms/FRM-01_Design_Review_Record.md` is created by this QMS and is unused | 8.3.4(a) is not met today |
| **No change-control register** | `registers/REG-07_Design_and_Change_Register.md` is created empty. Changes have been controlled through Git alone | Classification, review and authorisation of changes are not evidenced as distinct acts |
| **No CI gate** | There is no `.github/` directory, no CI configuration, and no CODEOWNERS file. The golden-question suite returns a CI-friendly exit code at `eval/run_eval.py:281`, but nothing runs it automatically | Verification depends entirely on the author choosing to run it. A regression can reach `main` unmeasured |
| **No independent review** | Development is direct-to-`main`. Only 2 of 202 commits are merges, both author-to-self on 2026-02-24. 200 of 202 commits had no pull-request review | There is no second pair of eyes on any design decision, including AI-generated ones |
| **Unit test results are not retained** | Test runs leave no stored artefact; only golden-eval runs write a report to `eval/reports/` | 8.3.4(f) requires documented information of the control activities; for unit testing there is none |
| **Design review, verification and approval are performed by the same person** | One developer, two Git identities, one email, 202 commits from 2026-02-05 to `ff29843` (2026-09-21) | An auditor should weigh the objective artefacts — evaluation reports, test suites, commit bodies — above any statement of review, because the review is self-administered |

### 11.3 Position for an auditor

Clauses 8.3.3 (inputs), 8.3.5 (outputs) and much of 8.3.4 (verification and validation) are
supported by real, inspectable evidence. Clause 8.3.2 (planning) and the review limb of 8.3.4 are
**not met** today: no design plan and no design review record exists for any past change. Clause
8.3.6 (changes) is partially met through Git history but is not evidenced as a controlled,
authorised act.

This procedure defines the control going forward. It does not retrospectively create planning or
review records for the 202 commits already made, and no attempt should be made to construct them.

All gaps above are carried into `ISO9001_Gap_Analysis.md` and, where they carry risk, into
`registers/REG-03_Risk_Register.md`.
