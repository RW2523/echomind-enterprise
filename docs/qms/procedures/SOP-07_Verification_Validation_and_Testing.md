# SOP-07 — Verification, Validation and Testing

| Field | Value |
|---|---|
| Document ID | SOP-07 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Engineering Lead |
| Approved by | `________________________` |
| Approval date | `____ / ____ / ________` |
| ISO 9001:2015 clauses | 8.3.4, 8.6, 9.1.1 |

---

## 1. Purpose

To define how EchoMind Enterprise is verified against its design inputs and validated against its
intended use, what evidence each level of testing produces, when each level must be run, and what
constitutes a release-blocking failure. ISO 9001:2015 8.6 requires that planned arrangements are
satisfactorily completed before release, and that evidence of conformity with the acceptance
criteria is retained.

## 2. Scope

All verification and validation activity for the EchoMind Enterprise platform: unit tests,
stack-dependent regression checks, the golden retrieval evaluation, the research evaluation harness,
and manual live verification. It applies to the reference instance operated by Ajace AI. Acceptance
testing at a customer site is covered by SOP-08.

## 3. Responsibilities

| Role | Responsibility |
|---|---|
| Engineering Lead | Determines which verification levels a given change requires; declares a failure release-blocking or deferrable; owns this procedure |
| Developer | Runs the applicable levels; records the outcome in the commit body; does not mark a change complete on an unverified assumption |
| Release Manager | Confirms before release that the required levels were run against the build being released, and retains the evidence (SOP-08) |

## 4. Verification levels that exist today

Five levels of verification exist in the repository. Each covers something the others do not, and
each has a boundary that must be understood before its result is relied on.

### 4.1 Level 1 — Backend unit tests

`backend/tests/` holds seven modules and 40 test functions.

| Module | Tests | Covers |
|---|---|---|
| `test_critical_high_fixes.py` | 9 | Audio-format validation, path-traversal rejection, untrusted-content fencing and the prompt-injection guard, section-restriction disable, index alignment after `clear_doc` |
| `test_rag_advanced.py` | 9 | Advanced retrieval behaviour |
| `test_docgen_images.py` | 6 | Document Studio image generation |
| `test_prompt_guards.py` | 5 | Presence of the injection guard in the analyser, compressor, contextualiser, RAG and strict prompts; consistency of the legal disclaimer |
| `test_medium_low_fixes.py` | 4 | Lower-severity defect regressions |
| `test_normalize.py` | 4 | Text normalisation |
| `test_transcript_audit_fixes.py` | 3 | Transcript audit defect regressions |

Every module sets a temporary `ECHOMIND_DATA_DIR` before importing application code
(`os.environ.setdefault("ECHOMIND_DATA_DIR", tempfile.mkdtemp(...))`), so index loading never
touches `/data`. This makes the suite pure and independent of a running stack.

**What it does not cover.** No test exercises a live model, a real index, retrieval quality, the
GPU path, the HTTP layer end to end, or the frontend. Passing Level 1 says the guarded defect
classes have not regressed; it says nothing about whether the product answers correctly.

### 4.2 Level 2 — Voice unit tests

`voice/tests/` holds three modules and 38 test functions: `test_hold_phrases.py` (22),
`test_echo_memory.py` (12), `test_stt_health.py` (4). These are pure-stdlib module tests; the
docstring of `test_hold_phrases.py` states the invocation explicitly.

### 4.3 Level 3 — Stack-dependent regression checks

These are not pytest. They are scripts copied into a running container and executed there, because
they need the application and the real corpus.

| Check | Path | Assertion | Result |
|---|---|---|---|
| Chunking content-loss regression | `eval/test_chunk_coverage.py` | `MIN_COVERAGE = 0.98` (`:26`) per document, plus named canary facts that were each lost by a real regression at some point | Exit 0 / 1 — usable as a gate |
| Voice end-to-end speech loop | `eval/voice_e2e_test.py` | **None.** Prints transcript, reply and latencies for a human to read | Observation only |
| Voice cadence and latency | `eval/voice_cadence_test.py` | **None.** Measurement output only | Observation only |

The two voice scripts are instruments, not tests. Their output must be read and judged by a person;
they cannot fail a release on their own.

### 4.4 Level 4 — Golden retrieval evaluation

`eval/run_eval.py` (285 lines) is the product's acceptance test for grounded answering. It requires
the running stack and drives it over HTTP at `http://localhost:3000/api` by default (`:46`).

- **52 questions** across seven tracked JSONL files in `eval/golden/`: `bank` 7, `conversational`
  10, `fmr` 7, `health` 7, `law` 7, `meetings` 7, `retail` 7.
- **Item types:** `retrieval`, `smalltalk`, `refusal`, `offcorpus` — so routing behaviour (a small-
  talk turn or an off-corpus question must not cite) is graded alongside retrieval.
- **Checks per item:** expected citation count; expected documents actually cited; **all** fact
  groups matched (any variant within a group); forbidden strings absent. An item passes only if
  every applicable check passes — `"passed": all(checks.values())` (`:191`).
- **Suite gate:** `return 0 if total_pass == len(results) else 1` (`:281`). The suite is
  all-or-nothing: 51 of 52 exits non-zero exactly like 0 of 52.
- **Not thresholded:** document precision is computed per item (`:171`) but never compared against a
  limit, so a degradation in citation precision that still satisfies the expected-document check
  does not fail anything.
- `--judge` adds LLM-as-judge grading via Ollama; this is supplementary and is not part of the gate.

### 4.5 Level 5 — Research evaluation harness

`eval/paper/` is a second, independent harness covering experiments E1–E10, with results tracked in
`eval/paper/results/`. It is not a release gate; it exists to measure and to test the claims made in
the project's own paper.

Its value to the QMS is that it is **honest about its own limits**, which is the behaviour
QP-01 §2.3 requires. `eval/paper/results/SUMMARY.json` records:

- `E2` as `"not_run"` with the reason stated (only 421 of 17,514 chunks are content, far below the
  specification's floor, so a corpus-scale sweep would measure transcript noise);
- `E6`, `E8` and `E9` as `"not_run"`, each with a reason — for `E9`, that synthesising the pairwise
  judgements would fabricate the very weights the experiment exists to measure;
- `E1` as contradicting the paper: `"contradicts_paper": "YES — §7.2 argues the retrieval terms
  dominate T_grounded. Measured, they are 2.8% of it"`;
- `E4` as contradicting the paper: containment is real but silent, `report_rate` 0.000 in every arm;
- `E3b` as having found, before its fix, a namespace breach in one of four retrieval paths that
  also degraded the golden score from 50/52 to 42/52.

These records are evidence of an honest measurement culture and are to be retained as design and
development records under SOP-01 §8.

### 4.6 Level 6 — Manual live verification

The de facto final check is a person exercising chat, Silent Assistant live checks, a voice turn and
a boardroom analyse/export against the running stack, and recording the result in the commit body.
`724fb98` is the model: *"Verified on a from-scratch rebuild: all six containers healthy; chat,
Silent Assistant live checks, voice turn, and boardroom analyse/export all pass. Data/model volumes
preserved."* `73f0b4f` similarly records the two handshake cases it checked and their outcomes.

This is a real control and the commit body is a real record. Its weakness is that it is unstructured
and unrepeatable: no checklist defines what "all pass" covered.

## 5. Which levels are required, and when

| Change type | L1 backend | L2 voice | L3 chunking | L4 golden eval | L6 manual |
|---|---|---|---|---|---|
| Retrieval, ranking, chunking, prompting or grounding (SOP-06 S-1) | Required | — | Required | **Required** | Required |
| Model swap or upgrade (S-2) | Required | Required | Required | **Required** | Required |
| Voice pipeline, TTS, STT, cadence | — | Required | — | — | Required, including `eval/voice_e2e_test.py` output |
| Authentication, namespace isolation, permission filter (S-4) | Required | Required | — | Required | Required, both positive and negative case |
| Container topology, base images, volumes (S-5) | Required | Required | Required | Required | Required, on a from-scratch rebuild |
| Frontend only | — | — | — | — | Required |
| Golden set or eval criteria change (S-6) | — | — | — | Required, before and after | — |
| Documentation only | — | — | — | — | — |

A change that touches more than one row requires the union of the rows.

## 6. Verification checklist

The commands below are the real invocations. Where a command is not documented in the repository,
the table says so rather than implying it is established practice.

| # | Command | What it proves | Pass criterion |
|---|---|---|---|
| V-1 | `python3 -m pytest tests -q` run from `backend/`, or with `backend/` on `PYTHONPATH` (the modules import `app.rag.…`) | The 40 guarded backend defect classes have not regressed | Zero failures, zero collection errors. **The working command is documented nowhere in the repository** — see §9 |
| V-2 | `python3 -m pytest voice/tests/test_hold_phrases.py -q` (from the repository root) | Hold-phrase selection behaviour | Zero failures |
| V-3 | `python3 -m pytest voice/tests -q` | All 38 voice tests | Zero failures |
| V-4 | `docker cp eval/test_chunk_coverage.py echomind-backend:/tmp/t.py && docker exec echomind-backend python3 /tmp/t.py` | No document lost content during chunking; named canary facts survive | Exit 0; every document at or above `MIN_COVERAGE` 0.98 |
| V-5 | `python3 eval/run_eval.py` (stack running; `--base` defaults to `http://localhost:3000/api`) | Grounded answering, citation behaviour, routing and refusal across 52 golden items | Exit 0, i.e. 52/52. **Never yet achieved** — see §9 |
| V-6 | `python3 eval/run_eval.py --set law` | One namespace in isolation, for fast iteration | All items in that set pass |
| V-7 | `python3 eval/voice_e2e_test.py` | A full speech loop completes through the live voice WebSocket | Human judgement — the script makes no assertions |
| V-8 | `python3 eval/voice_cadence_test.py` | Cadence and latency figures | Human judgement — no assertions |
| V-9 | `./scripts/verify_offline_readiness.sh` | The stack is configured to run with no network access | Exit 0. See SOP-08 §5 |
| V-10 | `docker compose ps` and each service's health state | All services that have a healthcheck report healthy | `trtllm`, `backend`, `voice`, `ollama` healthy. `frontend` and `cloudflared` have no healthcheck and cannot be judged this way |
| V-11 | Manual: chat answer with citation, Silent Assistant live check, one voice turn, boardroom analyse and export | The user-visible paths work on the build being released | Recorded in the commit body or release record, naming what was exercised |

## 7. Release-blocking failures

A failure is **release-blocking** if any of the following holds. A release-blocking failure is fixed
or the release does not happen; it is never waived informally.

1. Any V-1 to V-4 check fails or errors during collection.
2. V-5 regresses relative to the last recorded run — that is, an item that passed in the previous
   retained report now fails. (Because V-5 has never returned exit 0, "exit 0" cannot yet be the
   criterion; regression against the last retained report is the criterion until it can be.)
3. Any failure in a `refusal` or `offcorpus` golden item, or any citation on a `smalltalk` item.
   These are the behaviours QP-01 §2.1 treats as the highest-severity defect class, and they are
   release-blocking individually regardless of the aggregate score.
4. Any cross-namespace retrieval hit. The `E3b` finding in `eval/paper/results/SUMMARY.json`
   documents a real breach of this invariant in one of four retrieval paths, undetected by
   answer-level evaluation.
5. A service with a healthcheck fails to reach healthy on a from-scratch `docker compose build` and
   `up`.
6. V-9 fails, for any release intended for an offline or air-gapped installation.

A failure that is not release-blocking may be deferred, but only with the reason recorded in the
release record and a corresponding entry raised for correction.

## 8. Recording results

1. The outcome of every verification level run for a change is stated in the **commit body** —
   what was run, and what it showed. This is the organisation's existing practice and it is
   retained.
2. For a release, the checklist in §6 is completed on `forms/FRM-03_Release_Record.md` (SOP-08).
3. **An evaluation run's JSON report is a record and is retained.** `eval/run_eval.py:276-279`
   writes `eval/reports/eval_<run_id>.json` containing the run identifier, the base URL, the score
   and the per-item result. Going forward, the report for any run used to support a release is
   committed to the repository as evidence under 8.6.
4. Records are retained per SOP-01 §8: design and development records for the life of the product
   release plus three years.

## 9. Honest assessment of the verification regime

An auditor reading §4 to §8 would reasonably ask what is actually enforced. The answer is: nothing
is enforced automatically. The following statements are all true of the repository as it stands.

**9.1 There is no continuous integration.** `.github/` does not exist. No hook, no pipeline and no
build step runs any test. Every check in §6 runs only because a person chooses to run it.

**9.2 No quality tooling is configured.** There is no ESLint, no ruff, no mypy and no pre-commit
configuration anywhere in the tree. `frontend/package.json:8` defines the build as `vite build` with
no `tsc` step, and `frontend/tsconfig.json` sets `noEmit: true` and `skipLibCheck: true` and does
not set `strict` — so a TypeScript type error does not fail the frontend build.

**9.3 There is no pytest configuration.** No `pytest.ini`, `setup.cfg`, `tox.ini` or root
`pyproject.toml` exists at repository, backend or voice level (the only `pyproject.toml` in the tree
belongs to the vendored `nemotron_asr` package). Consequently there is no declared test path, no
import mode and no marker registry.

**9.4 The backend test command is undocumented.** The backend modules import `app.rag.…`, so they
require `backend/` on the import path; no README, script or docstring in the repository states the
invocation. The voice command appears only inside a test docstring
(`voice/tests/test_hold_phrases.py`). A newcomer cannot run the backend suite without working it
out from the imports.

**9.5 There is no record of a passing test run.** The only evidence of test execution in the tree is
three stale pytest caches, none of which is tracked in Git:

| Cache | What it records |
|---|---|
| `backend/.pytest_cache` | Ten failing tests across `tests/test_rules_library.py`, `tests/test_session_notes.py` and `tests/test_assistant_suggestions.py` — **none of these modules exists in `backend/tests/`**, and none was ever committed on any branch |
| `voice/.pytest_cache` | Three failing tests in `tests/test_semantic_router.py` and `tests/test_server_security.py` — again, modules that do not exist in the tree |
| `.pytest_cache` (repository root) | `backend/tests/test_normalize.py` recorded as failing at file level, i.e. a collection error |

The last recorded pytest activity therefore shows failures, against test modules that are no longer
present. There is no artefact anywhere in the repository showing a suite passing.

**9.6 The golden suite has never passed.** Twenty-three report files exist in `eval/reports/`, with
scores ranging from 27/52 to a best of 50/52 on 2026-07-30; the most recent is 49/52 on 2026-08-06.
No run has reached 52/52, so `eval/run_eval.py` has never exited 0. Its own gate has never been
satisfied.

**9.7 Evaluation reports are not retained as records.** `eval/.gitignore` contains `reports/`, so
every one of those 23 reports is untracked. They exist on one developer's machine and in no version
history. This directly conflicts with §8.3 above and with ISO 9001:2015 8.6's requirement to retain
evidence of conformity. Until `eval/.gitignore` is changed, or reports used for releases are copied
to a tracked location, the organisation has no retained verification evidence at all.

**9.8 The published score is stale.** `README.md:126` states *"Current: 48/52"*. The most recent
retained report is 49/52 dated 2026-08-06, and it predates commits `724fb98`, `15f3c96` and
`ff29843` — all of which changed retrieval, transcript or voice behaviour. The figure in the README
therefore describes neither the last measured run nor the current HEAD. QP-01 §2.3 requires
published figures to be reproducible from recorded evidence; this one is not.

## 10. Records

| Record | Where it lives | Retention |
|---|---|---|
| Verification outcome for each change | Commit body in Git history | Life of the product release + 3 years (SOP-01 §8) |
| Golden evaluation report per run | `eval/reports/eval_<run_id>.json` — **currently untracked** (`eval/.gitignore`); to be committed for any run supporting a release | Life of the release + 3 years |
| Research harness results | `eval/paper/results/` (tracked) | As above |
| Release verification checklist | `forms/FRM-03_Release_Record.md` | Life of the release + 3 years |
| Golden question set as an acceptance criterion | `eval/golden/*.jsonl` (tracked) | Life of the product |

## 11. Current state and gaps

| # | Requirement of this procedure | Present state | Gap |
|---|---|---|---|
| G-1 | Verification runs automatically on change | No CI; `.github/` does not exist | **Open.** Everything in §6 is manual and discretionary |
| G-2 | Test invocation is documented | Backend command documented nowhere; voice command only in a docstring | **Open.** Fix by adding a test section to the README or a `Makefile`, and a pytest configuration |
| G-3 | Evidence of a passing test run is retained | No such evidence exists; three stale caches record failures against deleted modules | **Open.** A clean, recorded run of V-1 to V-3 is the first action on approval of this procedure |
| G-4 | Golden suite exit code usable as a gate | Never reached 52/52; best 50/52 | **Open.** Until then §7.2 uses regression against the last retained report |
| G-5 | Evaluation reports retained as records | `eval/.gitignore` excludes `reports/`; all 23 reports untracked | **Open.** Highest-priority documentation fix in this procedure |
| G-6 | Published performance figures are current and evidenced | `README.md:126` quotes 48/52; latest retained report is 49/52 and predates three behaviour-changing commits | **Open.** Re-run and re-quote, or remove the figure |
| G-7 | Static analysis and type checking | No ESLint, ruff, mypy or pre-commit; `vite build` runs no `tsc`; `tsconfig.json` is not `strict` | **Open** |
| G-8 | Automated assertions on the voice path | `voice_e2e_test.py` and `voice_cadence_test.py` make no assertions | **Open.** Voice regressions are detectable only by a person reading output |
| G-9 | Citation-precision threshold | Computed at `eval/run_eval.py:171`, never compared against a limit | **Open.** README claims a 0.98 precision criterion that the harness does not enforce |
| G-10 | Frontend verification | No frontend tests of any kind exist | **Open** |

All gaps above are to be carried into `ISO9001_Gap_Analysis.md`.
