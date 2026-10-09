# EchoMind Law — Phase 1 gate report (S1.12)

| Field | Value |
|---|---|
| Date | 2026-10-09 |
| Branch | `law/s1.12-gate` (from `feature/echomind-law` after S1.11) |
| Code under test | `30fb2f2` (S1.12 tooling and fixes). Later commits on the branch change documentation only |
| Answer model | `nvidia/Qwen3-30B-A3B-FP4`, TensorRT-LLM on the host (`localhost:8355`; shared with production, used read-only as approved) |
| Embeddings | `nomic-embed-text` (Ollama on the host, `localhost:11435`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, loaded in-process from the local cache (offline) |
| Speech | Not installed on the development host (NeMo, Kokoro). Capture is tested with a stand-in transcriber |
| Data | Fictional only (demo world, Northwind and Voss-Lindqvist fixtures) |
| Production impact | None: nothing ran against production containers, volumes or ports |

**Spec exit criteria:**
- **Zero unauthorized matter disclosure:** met in every test listed below.
- **Zero recording without consent:** met (T03).
- **Owner sign-off:** required (🔒), at the end of this report.

## 1. Results by spec test

| Spec test | What was run | Result |
|---|---|---|
| **T01** matter isolation | 507 backend tests (policy matrix, matters API, walls, retrieval canaries, every legacy route closed) plus browser journeys 1–6 | **PASS** |
| **T02** revocation | 9 backend tests: an open answer stream, the next query, ingestion mid-job, live capture, the socket gate, sessions, revoked and expired memberships, and a download after removal (new) | **PASS** |
| **T03** consent | 11 backend tests plus the browser meeting journey with a fake microphone | **PASS** |
| **T04** legal holds | 9 backend tests on every deletion path, including a schema rollback (new guard, D-032) | **PASS** |
| **T08** prompt injection | 23 backend tests: a hidden-text PDF and an instruction-laden Word file, run against a model that obeys anything. Plus the real-model probe (D-031): the strongest injection was obeyed 10 of 18 times without the new defences and 0 of 36 with them | **PASS** (with the residuals in D-031) |
| **T09** export redaction | 12 backend tests plus the browser export journey | **PASS** |
| **T11** offline egress | Capture mode (D-033): a full browser run with every lookup and connection attempt recorded in the backend and watched from outside. Final run 13/13: all 20 attempts went to this machine (embeddings `127.0.0.1:11435` and its `localhost` lookup ×8, model `127.0.0.1:8355` ×2); 0 elsewhere in 5,036 outside snapshots. Earlier capture runs agree: uvloop run, 4,555 snapshots; asyncio run, 6,437 snapshots; 0 off-machine in both | **PARTIAL**: capture PASS; the enforced network-namespace run needs root (🔒) |
| **T13** upgrade, rollback, restore | 15 backend tests: upgrade from the S1.0 schema keeps legacy data; rollback keeps held records and refuses to drop them, the audit trail or source permissions; encrypted backup and restore with tampered, cut, wrong-key and traversal archives refused. Plus `dr_drill.sh` on browser-suite data: 148 events with the chain and anchor verified, schema at 0012, and 2 of 2 document versions opening with the master key | **PASS** |
| **T14** legacy smoke | Standard edition on a fresh database with the real models. API: 10 of 10 (sign-in, documents, knowledge chat plain and streamed, the voice answer path, transcripts, Boardroom sessions, Document Studio PDF and PPTX, live-transcription socket). Browser: 9 of 9 (every view, chat answer, Document Studio, the Boardroom switch, 5 verticals). Voice service boots. Backend legacy tests 50 of 50; voice 151 of 151 | **PASS**. One pre-existing defect, also on `main`: after signing in, chat cannot send until a reload. It is marked as an expected failure and offered as a separate task |
| **T15** source permissions | 13 backend tests plus the browser connector journey | **PASS** (local connector; a real DMS is blocked by P-010) |
| Audit chain | 23 backend tests; chain verification in the restore drill | **PASS** |
| Stability | `stability_runs.sh 10`: the full browser suite, 10 times in a row | **PASS**: 10 of 10 runs, 130 of 130 journeys (4.5–4.7 min each). With S1.11's 5 runs after D-030, that makes 15 consecutive clean runs with no stall |

**Not in Phase 1** (by plan):
- T05, T06: T06 is covered for retrieval; the version-conflict answer comes in S2.2.
- T07 (S3.4), T10 (S3.3), T12 (S2.8 load test).

## 2. Suites at this commit

| Suite | Result |
|---|---|
| Backend (Law + legacy) | **816 passed** |
| Voice | **151 passed** |
| Frontend unit (Vitest) and type check | **72 passed**; `tsc` clean |
| Consolidated security suite (`scripts/law/security_suite.sh`) | Every group passed (table above) |
| Browser, Law (`frontend/e2e`, 13 journeys) | **13 passed in each of 10 consecutive runs** |
| Browser, standard edition (`playwright.legacy.config.ts`) | 9 passed |

## 3. Found and fixed during the gate

- **The answer model obeys planted instructions** (T08, D-031). It now:
  - withholds AI-directed text from the model;
  - repeats the rule after the sources;
  - dims uncited sentences;
  - strips leaked think tags.

  Splitting answers into sentences had a bypass, a sentence starting in lowercase, which is fixed and tested.
- **A schema rollback could delete held records, the audit trail, or source-restricted documents' permissions.** It is now refused (D-032).
- **No encrypted backup existed for Law data.** `app.legal.backup` adds one (D-032). The legacy backup scripts would find production's container on this host; they are documented as never to be run from this clone.
- **The egress recorder was blind under uvloop.** The egress run uses the asyncio loop (D-033).
- **A race in the browser sign-in helper.** A stale error alert could win the "signed in?" check, which caused one failed meetings run. The helper now decides from the server's reply. A test proves a refused code keeps the code step and the next code works.
- **A stale FEATURE_MATRIX row:** SEC-02 still said MISSING although it was built in S1.2–S1.5.

## 4. Threat-model review

THREAT_MODEL.md was reviewed in full at this gate. Changes:
- **T-06 injection:** the measured behaviour and the new layers, with the residuals.
- **T-19 egress:** capture result; enforced mode pending.
- **T-20 held data:** the rollback guard.
- **T-21 backups:** encrypted archives.
- **T-05, T-08, T-15:** unchanged since S1.8–S1.11 and still accurate.

Residual risks the owner should know:

1. **Injection, deny-list:** an injection with novel wording can pass the deny-list. The person still sees uncited sentences dimmed. An injected claim that the model cites to the injected passage is shown with that passage's warning, but is not dimmed.
2. **Injection, voice:** voice reads answers aloud as they stream, so it cannot show the marks.
3. **T11 enforcement:** egress was observed, not enforced. Until the namespace run, it is evidence that nothing was sent, not proof that nothing could be.
4. **Backups:** they keep purged data until they rotate.
5. **Untested on this host:** speech recognition and Boardroom diarised analysis (no speech service natively).
6. **Unproven stall fix:** the S1.11 backend stall is fixed by D-030 but the exact SQLite lock is unproven. The evidence is 15 consecutive clean full runs.

## 5. Waiting for approval (🔒)

| Item | Why |
|---|---|
| Enforced T11 run in a network namespace | Needs root on this shared host |
| First `scripts/law/compose.sh up` | Shared host with production; needed for the speech service, a T14 run inside containers, and the in-container backup |
| A real DMS provider (P-010) | Provider, credentials, network approval |
| Counsel items P-001 to P-015 | Policy values, all at safe defaults |
| Pushing to GitHub or opening PRs | Not done |

## 6. Reproduce

```bash
scripts/law/security_suite.sh                       # backend security groups, one line per spec test
cd backend && LLM_BASE_URL=http://localhost:8355/v1 LLM_MODEL=nvidia/Qwen3-30B-A3B-FP4 \
  python ../scripts/law/injection_probe.py --trials 3                 # T08 live (add --no-defences to compare)
scripts/law/egress_test.sh                          # T11 capture mode over the full browser suite
scripts/law/dr_drill.sh                             # T13 drill on the browser-suite data folder
python3 scripts/law/legacy_smoke.py                 # T14 API smoke (standard edition on :8130)
cd frontend && npx playwright test -c playwright.legacy.config.ts    # T14 browser smoke
scripts/law/stability_runs.sh 10                    # the Law browser suite 10 times in a row
```

## 7. Sign-off

| Role | Name | Decision | Date |
|---|---|---|---|
| Owner (Ajace AI) | Owner, in the build session | ☑ **Phase 1 accepted** ("approved, merge S1.12 and start S2.1"). Still 🔒 and not approved by this: the root-run enforced T11, the first `compose.sh up`, pushes or PRs, network integrations and counsel items | 2026-10-09 |
