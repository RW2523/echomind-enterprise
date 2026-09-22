# SOP-12 — Monitoring, Measurement and Analysis

| Field | Value |
|---|---|
| Document ID | SOP-12 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Lead Engineer |
| Approved by | `________________________` |
| Approval date | `____ / ____ / ________` |
| ISO 9001:2015 clauses | 9.1.1, 9.1.3, 7.1.5 (measurement resources) |

---

## 1. Purpose

To define what Ajace AI monitors and measures, by what method, when it is measured, when the
results are analysed, and how the analysis is evaluated and fed into management review. This is the
procedure that determines whether the quality management system and the product are actually
performing, as opposed to being believed to perform.

## 2. Scope

Measurement of the EchoMind Enterprise platform's quality characteristics, of the delivery process,
and of the effectiveness of the QMS itself. It covers measurements taken on Ajace AI's own
development and demonstration deployment.

It does **not** cover measurement inside a customer deployment. No telemetry leaves a customer
installation by design (`SOP-11` §7), so Ajace AI has no visibility of a customer's running system
and must not imply otherwise. Any measurement of a customer deployment is one the customer takes
and chooses to share.

Customer satisfaction measurement (9.1.2) is not in this procedure; it belongs with customer
communication under `SOP-04`.

## 3. Responsibilities

| Role | Responsibility |
|---|---|
| Lead Engineer | Owns this procedure. Takes the measurements in §4 at the stated frequency, records the result, and raises a nonconformity under `SOP-10` when a measurement crosses a threshold. Maintains the integrity of the measurement harness (§6). |
| Managing Director | Reviews the analysis at management review (`SOP-15`), decides whether the results show the QMS to be effective, and sets or revises thresholds and objectives in `QO-01`. |
| Anyone contributing code | Runs the applicable measurement before a change to the area it covers is merged, and records the result in the commit body. |

## 4. What is monitored and measured

Only real, existing sources appear below. Where a measurement is not currently taken on a schedule,
the Frequency column says so rather than stating an intention as fact.

| Metric | Source | Method | Frequency | Reviewed by |
|---|---|---|---|---|
| Golden-question pass count | 52 items across seven sets in `eval/golden/*.jsonl` (43 retrieval, 6 smalltalk, 2 refusal, 1 off-corpus) | `python3 eval/run_eval.py` against a running backend; binary gate at `eval/run_eval.py:281` returns non-zero unless every item passes | On demand today. **Required by this procedure:** before any release, and after any change to retrieval, prompting, chunking, embedding or namespacing | Lead Engineer; summary to management review |
| Citation document-precision (average) | Same run | Computed per item at `eval/run_eval.py:171` and averaged per set at `:271-272` | With every golden run | Lead Engineer |
| Median latency per question type | Same run | Printed per set at `eval/run_eval.py:273` | With every golden run | Lead Engineer |
| Unit-test pass rate | `backend/tests/` — 40 test functions across 7 files, including the 5 prompt-guard tests in `backend/tests/test_prompt_guards.py` | `pytest backend/tests` | On demand today. **Required:** before any release and before any change to prompt construction | Lead Engineer |
| Grounding, isolation and injection-containment measures | `eval/paper/` harness, experiments E1–E10; results in `eval/paper/results/*.json`, consolidated in `SUMMARY.json` | Scripted experiments, each writing a JSON result | Per experiment campaign; not scheduled | Lead Engineer; management review |
| Per-path namespace leakage | `eval/paper/e3b_path_isolation.py`, result in `eval/paper/results/e3b_path_isolation.json` | Counts out-of-namespace hits per retrieval path; last recorded result 0 of 359 across all five paths | Not scheduled. **Required:** after any change to a retrieval path or to `_ns_ok` | Lead Engineer |
| Container health state | Docker healthchecks: `docker-compose.yml:31-36` (trtllm `GET /v1/models`), `:60-66` (backend `GET /health`), `:130-136` (voice `GET /health`), `:248-252` (ollama `ollama list \| grep -q nomic-embed-text`) | `docker compose ps`; the healthcheck itself runs on a 15–30 s interval | Continuous while running; observed on demand | Lead Engineer |
| Voice per-turn latency | The `app.session.turn` logger (`voice/app/session.py:46`), raised to INFO at `voice/app/server.py:15`, emitting `[SPEC]`, `[ROUTE]` and `[TURN]` lines with `first_token_ms` and `endpoint_to_reply_ms` (`voice/app/session.py:2408, 2415, 2453`) | Read from `docker logs echomind-voice` | Emitted continuously; read on demand | Lead Engineer |
| Application errors | Backend logs at INFO to stdout (`backend/app/main.py:25-32`); front-end crashes posted to `/api/client-error` and logged at ERROR (`backend/app/main.py:198-206`) | `docker logs`; grep for ERROR | Continuous; read on demand | Lead Engineer |
| Errors surfaced to the user during live transcription | 14 `{"type": "error"}` emission sites in `backend/app/transcribe/ws.py` | Observed in the client; not aggregated anywhere | Not measured | — |
| Activity volume and per-user usage | `activity_log` table (`backend/app/core/db.py:17`), written by `backend/app/main.py:154-163` | `recent_activity(limit)` and `usage_summary()` at `backend/app/core/audit.py:25,36` | On demand | Lead Engineer |
| Defect count and defect rate | Git history | `git log --oneline` count; commits prefixed `fix:` as a share of all commits — currently 22 of 202 | Per management review | Managing Director |
| Change volume | Git history | Commits and files changed per period | Per management review | Managing Director |
| Nonconformities raised, closed and open, by severity | `registers/REG-04_Nonconformity_and_CAPA_Log.md` (created by this QMS; no history before adoption) | Count from the register | Per management review | Managing Director |

## 5. Product quality measures versus QMS effectiveness measures

ISO 9001:2015 9.1.1 requires both, and they answer different questions. Conflating them is the
usual way an organisation ends up measuring only the easy one.

**Product quality — "is the output correct?"**

| Measure | Question it answers |
|---|---|
| Golden-question pass count | Does the system still answer the questions it is supposed to answer, and still refuse the ones it is supposed to refuse? |
| Citation document-precision | Are the citations it offers actually the right documents? |
| Namespace leakage count | Does content cross a tenant boundary? |
| Prompt-guard test results | Can retrieved content still steer the model? |
| Voice per-turn latency | Is the interaction usable? |
| Container health state | Is the service available? |

**QMS effectiveness — "is the way we work producing correct output reliably?"**

| Measure | Question it answers |
|---|---|
| Nonconformities by severity, raised versus closed | Are we finding problems, and are we finishing them? |
| Proportion of nonconformities detected by evaluation rather than by a user | Are our controls catching defects before customers do? |
| Recurrence — nonconformities of a class already corrected | Are corrective actions actually working? |
| Defect rate (`fix:` commits as a share of all commits) | Is the rate of rework changing? |
| Age of open nonconformities | Are problems being carried indefinitely? |
| Proportion of releases with a recorded pre-release golden run | Is the release gate being applied, or bypassed? |
| Gaps closed from `ISO9001_Gap_Analysis.md` | Is the QMS maturing, or static? |

The second table cannot be populated from the existing record. Nothing in it has ever been counted,
because the register it depends on is created by this QMS (§8, and `SOP-10` §12). The first
measurement of QMS effectiveness will be taken at the first management review, and the absence of a
baseline is itself the first finding.

## 6. Integrity of the measuring instrument (7.1.5)

ISO 9001:2015 7.1.5 concerns the resources needed to ensure valid and reliable monitoring and
measurement results, including — where measurement traceability is a requirement — calibrated
equipment.

**There is no physical measuring equipment in the scope of this QMS.** Ajace AI does not manufacture
and does not measure physical characteristics. Clause 7.1.5.2 (measurement traceability to
international standards) therefore does not apply, and that is recorded here rather than left as an
unexplained omission.

**The equivalent control for this organisation is the integrity of the measurement harness itself.**
The golden question set, its expected answers, its thresholds and the scripts that execute it are
this organisation's measuring instrument. If the question set silently changes, every trend measured
against it is meaningless — exactly as an uncalibrated gauge makes every reading meaningless. The
instrument must therefore be identified, version-controlled and change-controlled.

| Instrument component | Location | Controlled? |
|---|---|---|
| Golden question sets and expected answers | `eval/golden/bank.jsonl`, `conversational.jsonl`, `fmr.jsonl`, `health.jsonl`, `law.jsonl`, `meetings.jsonl`, `retail.jsonl` | **Yes** — tracked in Git; every change is a commit with an author, a date and a diff (`SOP-01` §3) |
| Scoring logic and pass criteria | `eval/run_eval.py` | **Yes** — tracked in Git |
| Paper experiment scripts | `eval/paper/*.py` (E1, E3, E3b, E4, E5, E7–E10 plus the corpus builders) | **Yes** — tracked in Git |
| Recorded experiment results | `eval/paper/results/*.json`, including `SUMMARY.json`, `REGRESSION_AND_FIXES.json` and `config_manifest.json` | **Yes** — tracked in Git |
| Per-run golden evaluation reports | `eval/reports/eval_<run_id>.json`, written at `eval/run_eval.py:275-276` | **No** — `eval/.gitignore` excludes `reports/`. See §10 |

**Rules for changing the instrument.**

1. A change to a golden question, to an expected answer or to a pass criterion is a change to the
   instrument, not a change to the test data. It is made in its own commit, with the reason in the
   commit body, and never in the same commit as a change to the code being measured.
2. Adding items to the question set is expected as the product grows, and changes the denominator.
   A pass count is therefore never compared across a change to the set without stating the set size
   at both points. The figures recorded in commit `4e27109` — 50/52 falling to 42/52, and 49/52
   after the fix — are valid because the denominator was constant; the set today holds 52 items and
   any future comparison must state its own denominator.
3. An item may not be removed or weakened because it fails. A failing item is either a real defect
   (raise a nonconformity under `SOP-10`) or a wrong expectation (correct the expectation, in its
   own commit, with the reasoning).
4. The environment a measurement was taken in is part of the measurement. `eval/paper/results/config_manifest.json`
   is the pattern: it records the configuration the experiments were run under.

The organisation already applies rule 3 in substance. `eval/paper/results/SUMMARY.json` records
results that contradict Ajace AI's own published paper — E1 `"contradicts_paper": "YES"`, E4
containment real but silent with `report_rate` 0.000 in every arm, E3 the timing channel not closed,
and E2 recorded as `"not_run"` with a stated reason rather than omitted. Measurements that make the
organisation look worse are retained in the organisation's own artefacts. That is the behaviour this
section exists to preserve.

## 7. Methods

| Measurement | How it is taken, precisely |
|---|---|
| Golden evaluation | Bring the stack up; confirm the backend healthcheck is passing; run `python3 eval/run_eval.py` (optionally `--set <stem>` for one set). The exit code is the gate. Record the pass count, the per-type breakdown, the average doc-precision and the median latency in the commit or the release record |
| Unit tests | `pytest backend/tests`. Record pass/fail counts |
| Namespace isolation | `python3 eval/paper/e3b_path_isolation.py`; record out-of-namespace hits and the number of checks, as `0/359` was recorded |
| Container health | `docker compose ps`; every service that defines a healthcheck must read `healthy`. Note that `frontend` and `cloudflared` define none, so their state is `running`, not `healthy` — this is not evidence of health |
| Voice latency | Run a set of turns; read `[TURN]` lines from `docker logs echomind-voice`; record `first_token_ms` and `endpoint_to_reply_ms` |
| Activity volume | Call the endpoint backed by `usage_summary()` (`backend/app/core/audit.py:36`); record total events and events per user. Treat the figure as a lower bound: the logging middleware is best-effort and swallows its own failures (`backend/app/main.py:161-162`) |
| Defect and change counts | `git log --oneline \| wc -l`, and the count of commits whose subject begins `fix:` |

A measurement is only a record if the result is written down where it can be found again. A
measurement taken and observed on screen, with the number never recorded, has not been taken for the
purposes of this procedure.

## 8. Analysis and evaluation (9.1.3)

At each management review the Lead Engineer prepares, and the Managing Director evaluates, the
following analysis. Each line states what the organisation is trying to learn, not merely what it
counted.

| Analysis | Evaluated against |
|---|---|
| Golden-question pass count over time | Any decline is a product-quality regression, whatever its cause. A decline explained by corpus growth is still a decline, and `eval/paper/results/REGRESSION_AND_FIXES.json` records exactly that case: a defect "PRE-EXISTING and latent; it became observable only when the corpus grew from 10.3k to 17.6k chunks" |
| Citation document-precision trend | Whether answers are attributed to the right material. Computed today but not compared against a threshold (§10) |
| Namespace leakage results | Must be zero. Any non-zero result is an S1 nonconformity under `SOP-10` §5 |
| Nonconformities by severity, raised versus closed, and ageing | Whether the organisation finishes what it starts, and whether the severity mix is worsening |
| Detection channel mix | What share of nonconformities was found by evaluation, by test, by healthcheck, and by a person noticing. A rising share found by people is a warning about the automated controls |
| Recurrence | Any nonconformity of a class already corrected indicates a corrective action that did not work (`SOP-14`) |
| Effectiveness of risk treatments | Whether the evidence cited in `registers/REG-03_Risk_Register.md` is still true of the code today (`SOP-02` §8) |
| Adequacy of the measurement harness | Whether the golden set still covers what the product now does. A product that has grown features the question set does not exercise is measured by an instrument that no longer fits it |

The output of this analysis is an input to management review under `SOP-15` and, where it shows a
need, to change planning under `SOP-02` §9 and to the quality objectives in `QO-01`.

## 9. Records

| Record | Location | Retention |
|---|---|---|
| Golden evaluation result for a release | Release record under `SOP-08`, and the commit body | Life of the release + 3 years (`SOP-01` §8) |
| Per-run golden evaluation report | `eval/reports/eval_<run_id>.json` | **Not currently retained** — see §10 |
| Paper harness results | `eval/paper/results/*.json`, in Git | Life of the QMS + 3 years |
| Measurement taken to verify a correction | On the nonconformity record (`forms/FRM-05_Nonconformity_and_CAPA_Record.md`) and in the commit body | 3 years after closure |
| Analysis and evaluation of measurement results | Management review minutes under `SOP-15` | 3 years |
| Changes to the measurement harness | Git history of `eval/golden/*.jsonl`, `eval/run_eval.py` and `eval/paper/*.py` | Life of the QMS + 3 years |
| Container health observations | Not recorded — see §10 | — |

## 10. Current state and gaps

**Stated plainly: Ajace AI has unusually good measurement instruments and almost no measurement
process.** The golden evaluation, the paper harness and the per-turn timing logs are real, specific
and honest — the recorded results include findings that contradict the organisation's own published
claims. What is missing is that nothing runs on a schedule, nothing is retained as a series, and
nothing is compared against a threshold except the single binary golden gate.

| Gap | Current state | Consequence |
|---|---|---|
| No metrics infrastructure | No Prometheus, no OpenTelemetry, no metrics endpoint, and no structured or JSON logging anywhere in `backend/` or `voice/` | Nothing is measured continuously. Every number in §4 is obtained by a person running something by hand |
| No scheduled measurement | Nothing in the repository runs the golden evaluation, the tests or the isolation check on a timer or on merge; there is no `.github/` directory and no CI configuration (`SOP-02` §4) | Measurement depends entirely on someone remembering. A regression can sit undetected between runs, which is how the defect in commit `4e27109` grew from latent to observable |
| No trend data | No measurement series is retained for any metric | Analysis under §8 can currently compare only against whatever figures happen to be quoted in commit bodies. There is no baseline for the first management review |
| Evaluation reports are not retained | `eval/.gitignore` excludes `reports/`, the directory `eval/run_eval.py:275-276` writes to | The instrument is version-controlled but its readings are discarded. Verification evidence cited in a nonconformity record cannot be produced later (`SOP-10` §12) |
| Doc-precision is computed but not thresholded | Calculated at `eval/run_eval.py:171` and averaged at `:271-272`, but the pass gate at `:281` uses only the binary per-item result | Citation quality can degrade materially without failing the gate |
| No log rotation or retention of any kind | `docker-compose.yml` sets no `logging:` driver or options on any service, so the default `json-file` driver applies with no `max-size` and no `max-file` | Container logs grow unbounded until the disk fills, taking the deployment down. The voice per-turn timing data and the backend error log are also the only record of several measurements, and they are neither shipped nor retained |
| Health endpoints check almost nothing | `backend/app/main.py:185-195` returns 503 only on a fatal CUDA/STT fault and otherwise `{"ok": true}` — it does not check the database, the FAISS indexes, the LLM or the embedding service. `voice/app/server.py:74-82` has the same narrow scope | A backend can report healthy with an unreachable LLM, a missing index or a locked database. Health state is not evidence that the product works |
| No readiness endpoint and no dependency aggregation | Only the single `/health` liveness endpoint per service | Nothing distinguishes "the process is up" from "the service can serve a request" |
| Two services have no healthcheck at all | `frontend` (`docker-compose.yml:207`) and `cloudflared` (`:223`) define none | Their failure is invisible to `docker compose ps` |
| The activity log is not a reliable measurement source | The middleware is wrapped in a bare `except Exception: pass` (`backend/app/main.py:161-162`) and the insert swallows failures at debug level (`backend/app/core/audit.py:21-22`) | Usage figures are a lower bound of unknown tightness and must be reported as such |
| No pruning of `activity_log` | No endpoint and no task deletes from it | The table grows without limit; see also `SOP-11` §10 |
| No thresholds or objectives set | `QO-01` exists but no numeric threshold is bound to any metric in §4 apart from the binary golden gate | "Analysis and evaluation" under 9.1.3 currently has nothing to evaluate results against |
| Errors surfaced to users are not aggregated | The 14 `{"type": "error"}` emission sites in `backend/app/transcribe/ws.py` are counted nowhere | The organisation cannot say how often users hit an error, or which one |
| No measurement of a customer deployment | By design — no telemetry leaves a customer installation (`SOP-11` §7) | Field quality is invisible to Ajace AI. The compensating control must be an agreed reporting route with each customer under `SOP-04`, which does not yet exist |

Each gap above is carried into `registers/REG-03_Risk_Register.md` and into `ISO9001_Gap_Analysis.md`.
