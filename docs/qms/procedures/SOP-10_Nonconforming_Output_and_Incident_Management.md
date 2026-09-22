# SOP-10 — Nonconforming Output and Incident Management

| Field | Value |
|---|---|
| Document ID | SOP-10 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Lead Engineer |
| Approved by | `________________________` |
| Approval date | `____ / ____ / ________` |
| ISO 9001:2015 clauses | 8.7.1, 8.7.2, 10.2 (interface only) |

---

## 1. Purpose

To define how Ajace AI identifies, contains, records and dispositions nonconforming output of the
EchoMind Enterprise platform, so that nonconforming output is not delivered or used unintentionally,
and so that the decision taken about it is recoverable afterwards by someone who was not there.

## 2. Scope

All output of the platform and of the organisation's delivery process: answers returned to a user,
transcripts, meeting reports, generated documents, voice replies, and the released software itself
(images, configuration, documentation). It covers defects found before release and defects found in
a running deployment.

It does **not** contain the corrective-action process. Determining root cause, deciding whether the
nonconformity could recur elsewhere, and verifying that a corrective action worked are done under
`SOP-14`. This procedure defines the point at which a nonconformity becomes a corrective action
(§9) and stops there.

## 3. Responsibilities

| Role | Responsibility |
|---|---|
| Lead Engineer | Owns this procedure. Classifies severity, decides containment, performs correction, records the nonconformity, and proposes the disposition. |
| Managing Director | Authorises any disposition that releases nonconforming output to a customer — acceptance under concession, and every decision to inform or not inform a customer. Reviews all S1 and S2 records at management review (`SOP-15`). |
| Anyone contributing code | Raises a nonconformity as soon as it is recognised, including one found incidentally while working on something else, and including one they caused themselves. |

Ajace AI currently has one person, who holds all three roles (`SOP-02` §3). Where this procedure
requires authorisation by a different role from the one that proposed the action, that separation
does not currently exist; the record must still be made, and the absence of independent
authorisation is a recorded gap (§11).

## 4. What counts as nonconforming output for this product

Nonconformity for a grounded AI platform is not primarily a crash. A crash is visible. **The severe
nonconformity for EchoMind Enterprise is an ungrounded, fabricated or misattributed answer reaching
a user**, because it is *invisible* — it is delivered in the same confident form as a correct
answer, to a user in a regulated sector who may act on it.

| Class | Definition | Why it matters here |
|---|---|---|
| **N1 — Grounding failure** | A fabricated fact; a fabricated or misattributed citation; an invented figure; or a confident answer where the retrieved material does not support one | The customer cannot tell it apart from a correct answer. In defence, legal, finance or healthcare use the consequence sits with the customer, not with us |
| **N2 — Tenant or namespace isolation failure** | Content from one knowledge-base namespace reaching a query scoped to another | A confidentiality breach in the customer's terms, not a retrieval bug in ours |
| **N3 — Silent failure** | An operation that fails but returns an empty or successful-looking result, with nothing raised to the user and nothing at ERROR in the logs | Undetectable without deliberate testing; erodes trust in every other output |
| **N4 — Loss of service** | A container crash-loop, a failed build, a module unavailable | Visible, and therefore self-reporting |
| **N5 — Degraded behaviour** | Latency outside the usable range, quality regression short of fabrication, truncated or malformed output | Usually measurable before a user reports it |
| **N6 — Documentation or release defect** | Shipped documentation that contradicts the running system | Misleads the operator, who then misconfigures the deployment |

Controls that exist today specifically against **N1**, cited so an auditor can read them:

| Control | Where |
|---|---|
| Cross-encoder relevance reranking of the retrieved candidate pool | `backend/app/rag/advanced.py:616-644`, using `backend/app/rag/reranker.py` |
| Relevance threshold below which no source is accepted as an answer basis | `backend/app/rag/advanced.py:1699-1712` (`settings.RAG_RELEVANCE_THRESHOLD`) |
| Deterministic not-found response instead of falling back to ungrounded general chat | `backend/app/rag/advanced.py:74-80` (`INSUFFICIENT_CONTEXT_MSG`) |
| Retrieved and recorded content fenced as untrusted data in every prompt | `backend/app/api/routes/chat.py:20` (`_fence_untrusted`); asserted by `backend/tests/test_prompt_guards.py` (5 tests over the analyser, compress, contextualiser, RAG and strict prompts, plus lawyer-persona disclaimer consistency) |
| Namespace predicate applied inside every index search path | `backend/app/rag/index.py:37-40` (`_ns_ok`) |

## 5. Severity scale and required response

Severity is assigned at the moment of detection, by the person who detected it, and is revised only
upwards until the record is closed. The impact levels correspond to the impact scale in `SOP-02` §7.

| Severity | Definition | Required response | Required record |
|---|---|---|---|
| **S1** | Incorrect, fabricated or misattributed content has reached, or could reach, a user (N1); or a tenant-isolation breach (N2); or customer data lost or exposed | Contain immediately — disable the affected path rather than leave it serving. Correction before any other work in that area. Always escalates to corrective action under `SOP-14`. Customer notification decision recorded by the Managing Director (§8.4) | Full record on `forms/FRM-05_Nonconformity_and_CAPA_Record.md`, logged in `registers/REG-04_Nonconformity_and_CAPA_Log.md`, plus verification evidence |
| **S2** | A module is unavailable, or a build or release is broken, or data is at risk but not yet lost (N3, N4) | Contain and correct before the next release touching the area. Escalates to corrective action if it has occurred before, or if the cause is a class of defect rather than an instance | Full record as above |
| **S3** | Degraded behaviour: latency, quality regression short of fabrication, malformed output (N5) | Correct in normal work. Escalates to corrective action if it recurs after correction | Log entry in `registers/REG-04`; full record only if escalated |
| **S4** | Cosmetic or documentation-only, with no effect on the correctness of output (N6, minor) | Correct in normal work | Log entry in `registers/REG-04` |

An S1 may not be closed by correction alone. A nonconformity whose severity cannot be agreed is
treated at the higher severity until evidence supports lowering it.

## 6. Detection channels

Every channel below exists today. What is recorded against each is what the channel actually
produces, not what it could produce.

| Channel | What it detects | Evidence path |
|---|---|---|
| Golden-question evaluation | N1 and N2 regressions — a retrieval item that stops being answered, a refusal item that stops refusing, a routing item that starts citing | `eval/run_eval.py`; 52 items across seven sets in `eval/golden/*.jsonl`; binary gate at `eval/run_eval.py:281` (`return 0 if total_pass == len(results) else 1`) |
| Paper harness | Grounding, isolation and injection-containment measurement at experiment scale | `eval/paper/`, results in `eval/paper/results/` |
| Unit tests | Prompt-guard and RAG behaviour regressions | `backend/tests/` — 40 test functions across 7 files |
| Container healthchecks | N4 — fatal CUDA/STT fault, LLM server not serving, embedding model absent | `backend/app/main.py:185-195`; `voice/app/server.py:74-82`; `docker-compose.yml` healthchecks for `trtllm` and `ollama` |
| Front-end crash reporting | N4 in the browser | `frontend/components/ErrorBoundary.tsx:27,36` posts to `/api/client-error`, logged at ERROR by `backend/app/main.py:198-206` |
| Error surfaced to the user over the transcription WebSocket | N3 and N4 during live transcription | 14 `{"type": "error"}` emission sites in `backend/app/transcribe/ws.py`, e.g. `:1031` and `:1069` |
| Activity log | Anomalous request volume or a pattern of failing status codes | `activity_log` table (`backend/app/core/db.py:17`), written by `backend/app/main.py:154-163`, read via `backend/app/core/audit.py:25,36` |
| User or customer report | Anything the above miss — in particular N1, which no automated channel catches in production | No in-product feedback channel exists (§11) |
| Own observation during development or deployment | Historically the largest source | Commit history; see the worked examples in §10 |

## 7. Procedure

1. **Raise.** Whoever detects the nonconformity opens a record on
   `forms/FRM-05_Nonconformity_and_CAPA_Record.md` and adds a line to
   `registers/REG-04_Nonconformity_and_CAPA_Log.md`. The record is opened *before* the fix, not
   written up afterwards from memory. The minimum at this point is: what was observed, where, when,
   and the class and severity from §4 and §5.
2. **Contain.** Stop the nonconforming output reaching anyone else. Containment is a distinct act
   from correction and is recorded separately — see §8.1.
3. **Determine the extent.** Establish how much output is affected, and since when. For N1 and N2
   this means asking whether the defect is new or latent: a defect can be present for months and
   become observable only when a threshold is crossed.
4. **Correct.** Fix the defect. The commit is the correction record (`SOP-01` §3) and must state
   the defect, the mechanism, the blast radius, the fix and the verification.
5. **Verify the correction.** Re-run the channel that detected it, plus the golden evaluation where
   retrieval, grounding or isolation is involved. Record the measured result, not the intent.
6. **Disposition.** Select and record one of the options in §8.
7. **Close or escalate.** Close under §9, or open a corrective action under `SOP-14`.

## 8. Disposition of nonconforming output (8.7.1)

ISO 9001:2015 8.7.1 requires that one or more of the following be applied. All four apply to this
product; which applies depends on whether the output has already reached a user.

### 8.1 Correction

The default. The defect is fixed and the output regenerated. For a stored artefact already produced
by a defective path — a transcript, a meeting report, a generated document — correction means the
artefact is regenerated or withdrawn, not only that the code is fixed. **A code fix alone does not
correct output already in the customer's hands.**

### 8.2 Segregation, containment, return or suspension of provision

Withholding the affected capability until it is correct. In this platform containment is
configuration, and the configuration points exist:

| Containment | Mechanism |
|---|---|
| Disable a retrieval path or feature that is producing ungrounded output | Feature flags in `backend/app/core/config.py`; the pattern is commit `2decb99`, which gated BookRAG off by default because it measured worse |
| Take a deployment off the public internet | Stop the `public` compose profile (`docker-compose.yml:218-228`) or disable the Cloudflare Access application (`docs/PUBLIC_DEPLOYMENT.md` §"Rolling back") |
| Suspend a module | Stop the owning container; the frontend degrades rather than the whole stack failing |
| Withhold a release | The release is not tagged or distributed under `SOP-08` |

### 8.3 Informing the customer

Required where nonconforming output has already been delivered. For N1 and N2 the customer cannot
detect the nonconformity themselves, so the obligation falls entirely on Ajace AI. The decision —
including a decision *not* to inform, with its reason — is made by the Managing Director and
recorded on the nonconformity record. There is no procedure for this today (§11).

### 8.4 Obtaining authorisation for acceptance under concession

Releasing or continuing to use output that is known to be nonconforming, because the alternative is
worse. Permitted only with a dated written authorisation by the Managing Director on the
nonconformity record, stating what is nonconforming, why acceptance is chosen, what the customer
has been told, and the date by which it will be corrected. **An S1 arising from N1 or N2 may not be
accepted under concession.** A customer whose data or answers are affected is told before, not
after.

## 9. Records required by 8.7.2, and closure

Every nonconformity record carries, as a minimum:

| Required by 8.7.2 | Field on `FRM-05` |
|---|---|
| Description of the nonconformity | What was observed, where, and the class from §4 |
| Description of the actions taken | Containment, correction, and the commit implementing it |
| Description of any concessions obtained | The authorisation text, the authorising role, the date |
| Identification of the authority deciding the action | Role, not a person's name, plus the commit or record that shows the decision |

Two further fields are required by this procedure because they are what makes the record usable:
the **severity and its justification**, and the **verification evidence** — a measurement taken
after the correction, not a statement of belief.

A record is closed when correction is verified, the disposition is recorded, and either no
corrective action is required or the corrective action has been opened under `SOP-14` and
cross-referenced. Closure is recorded in `registers/REG-04_Nonconformity_and_CAPA_Log.md`.

**Escalation to corrective action (10.2) is mandatory when any of the following is true:**

- severity is S1;
- the nonconformity has occurred before, in any form;
- the cause is a class of defect rather than a single instance — that is, the same mistake is
  possible elsewhere in the codebase and nothing prevents it;
- the nonconformity was latent and became observable only because conditions changed;
- a control that was believed to be in place turned out not to be working.

The last two criteria come directly from this organisation's own history (§10) and are the reason
they are written down.

## 10. Worked examples from the evidenced record

The following are real, already-recorded defects. They are reproduced here as the standard a
nonconformity record must meet — nothing below is invented, and each is traceable to a commit. They
predate this QMS and were therefore never recorded on `FRM-05`; recording them retrospectively is
part of seeding `registers/REG-04`.

**Example A — `4e27109` (2026-08-07), class N2, severity S1: tenant-isolation breach.**
`backend/app/rag/advanced.py` called `Bm25Index.search` directly on the sparse transcript path,
bypassing the `_ns_ok` namespace predicate (`backend/app/rag/index.py:37-40`) that every other
retrieval path applied. Under a tenant-scoped query the path returned 10 out of 10 out-of-namespace
chunks, which outranked in-namespace documents and silently zeroed citations. Extent: the golden
evaluation fell from 50/52 to 42/52 as the corpus grew. Correction: a
`search_transcript_only_sparse()` wrapper that enforces the predicate
(`backend/app/rag/index.py:925`), plus candidate-pool scaling with corpus size
(`backend/app/rag/index.py:44-57`). Verification: 0 out-of-namespace hits across 359 checks on all
five retrieval paths; golden evaluation 49/52. This is the model record: defect, mechanism, extent,
fix, measured verification.

**Example B — `558eaae` (2026-07-29), class N3, severity S2: silent audio failure.**
`/api/transcribe/speak` defaulted to `voice:8002` — a host port, not the container port. Every call
hit "Connection refused", the exception was swallowed at debug level, and the endpoint returned
`{"audio_b64": null}` with a 2xx status and nothing in the logs. Correction: the correct default
plus a configurable `VOICE_TTS_URL`; an empty 2xx now treated as a failure; the fallback logged at
WARNING rather than DEBUG; timeout raised from 15 s to 30 s. Verification: three real WAV outputs.
This is the canonical N3 — nothing detected it, because nothing could.

**Example C — `724fb98` (2026-09-04), class N4, severity S2: three breakages on a clean rebuild.**
A Python heredoc inside a `RUN ... \` continuation broke `docker compose build voice`; NeMo
installed from `@main` pulled `setuptools>=82`, removing `pkg_resources` and breaking `librosa` at
build time and `webrtcvad` at runtime; and the boardroom module stored raw JSON as the executive
summary when the LLM report JSON was truncated. Verification was a from-scratch rebuild: all six
containers healthy, chat, Silent Assistant, voice and boardroom passing, volumes preserved. The
mitigation for the dependency drift is pinned in place with its reason at
`backend/Dockerfile:32-33`.

**Example D — `73f0b4f` (2026-07-29): a control that was not working.**
The voice WebSocket authentication gate was silently non-functional because `AUTH_SECRET` was
passed where `VOICE_AUTH_SECRET` was expected (`docker-compose.yml:143-146`). A control believed to
be in place was not. This class always escalates under §9.

**Example E — `eval/paper/results/REGRESSION_AND_FIXES.json`: the nearest existing thing to a
nonconformity record.** It is structured as baseline → regression → root cause → fixes →
non-regression evidence, and states that the defect was "PRE-EXISTING and latent; it became
observable only when the corpus grew from 10.3k to 17.6k chunks". That sentence is why latency of
detection, not only date of detection, is a field on `FRM-05`.

**Example F — honest negative reporting.** `eval/paper/results/SUMMARY.json` records measurements
that contradict the organisation's own published paper: E1 `"contradicts_paper": "YES"`; E4
containment real but silent, with `report_rate` 0.000 in every arm; E3 the timing channel not
closed; E2 recorded as `"not_run"` with the reason given rather than omitted. An auditor should
read this as evidence of a culture of honest measurement: adverse findings about the organisation's
own claims are recorded in the organisation's own artefacts.

## 11. Records

| Record | Location | Retention |
|---|---|---|
| Nonconformity and CAPA log | `registers/REG-04_Nonconformity_and_CAPA_Log.md` | 3 years after closure (`SOP-01` §8) |
| Individual nonconformity record | `forms/FRM-05_Nonconformity_and_CAPA_Record.md`, completed per event | 3 years after closure |
| Correction, mechanism and verification | Git commit body (`git log`) | Per `SOP-01` §8 |
| Concession authorisation | On the nonconformity record, with authorising role and date | 3 years after closure |
| Customer notification, or the reasoned decision not to notify | On the nonconformity record | 3 years after closure |
| Evaluation output supporting a verification | `eval/reports/` — see the gap in §12 | Not currently retained |
| Review of S1 and S2 records | Management review minutes under `SOP-15` | 3 years |

## 12. Current state and gaps

**Stated plainly: Ajace AI has a strong, evidenced defect-correction practice and no incident
management system.** The corrections in §10 are real, specific and verified by measurement — the
weakness is not engineering rigour, it is that nothing aggregates, owns, ages or closes these
events, and nothing at all governs telling a customer.

| Gap | Current state | Consequence |
|---|---|---|
| No incident or nonconformity register before this QMS | `docs/qms/registers/` contained no register at the time of writing; `registers/REG-04` is created by this QMS and seeded only from the evidence in §10 | Defects were corrected but never counted, aged or reviewed as a set; recurrence cannot be demonstrated either way |
| No in-repo issue tracker | No `ISSUES.md`, no `TODO.md`, no `.github/` directory; a repository-wide search for `TODO`, `FIXME` and `HACK` returns zero matches | There is no list of known-open problems. An auditor asking "what is currently wrong?" cannot be answered from the repository |
| Known issues are recorded in configuration comments | For example `docker-compose.yml:7-11` (GB10 worker-spawn CUDA bug and its rollback plan), `:41-44` (`gpus: all` versus the `deploy` form breaking NVML, "Verified: same image works with --gpus all, fails via the deploy form"), `:86-89`, `:164-166` (onnxruntime CUDA EP unavailable on ARM), `backend/Dockerfile:32-33` (setuptools drift) | The content is real, specific and verified — but it is distributed across configuration files with no list, no owner, no status and no closure evidence. It is knowledge, not a record |
| No customer-notification procedure | Nothing in `docs/` defines who tells a customer what, within what time, after an N1 or N2 event | §8.3 is currently an obligation with no mechanism. This is the highest-priority gap in this procedure |
| No separation between proposer and authoriser | One person holds all roles (`SOP-02` §3) | Acceptance under concession is self-authorised. The record must still be written, and the absence noted at management review |
| No in-product feedback channel | The front end has a single `ErrorBoundary` (`frontend/App.tsx:156`) and no toast or notification component at all; there is no "report a bad answer" affordance | N1 — the most severe class — has no production detection channel whatsoever. It can only be found by evaluation, and the evaluation runs against a fixed question set |
| Raw exception text leaks to the client | Four bare `str(e)` passthroughs in `backend/app/transcribe/ws.py` at `:353`, `:905`, `:963` and `:1178` | Internal detail reaches a user in a regulated environment; the message is also useless to that user |
| Evaluation reports are not retained | `eval/.gitignore` excludes `reports/`, which is where `eval/run_eval.py:275-276` writes each run | Verification evidence cited in a nonconformity record cannot be produced later. See `SOP-12` §6 |
| No severity or classification has ever been assigned | The 22 `fix:` commits among 202 carry no severity field | Trend analysis by severity is not possible from the existing record and will begin only from adoption of this procedure |

Each gap above is carried into `registers/REG-03_Risk_Register.md` and into `ISO9001_Gap_Analysis.md`.
