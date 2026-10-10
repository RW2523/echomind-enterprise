# EchoMind Law — Phase 2 gate report (S2.8)

| Field | Value |
|---|---|
| Date | 2026-10-10 |
| Branch | `law/s2.8-gate` (from `feature/echomind-law` after S2.7, `331c475`) |
| Code under test | `e0f6415` (the gate tooling). Later commits on the branch change only the Evaluations page headlines, the reports' "+changes" check, the evaluations journey and documentation |
| Answer model | `nvidia/Qwen3-30B-A3B-FP4`, TensorRT-LLM on the host (`localhost:8355`; shared with production, used read-only as approved) |
| Embeddings | `nomic-embed-text` (Ollama on the host, `localhost:11435`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, in-process from the local cache (offline) |
| Speech | Not on the development host (about 13 GB free, swap full, production on the GPU). Capture is tested with scripted speech |
| Data | Fictional only |
| Production impact | None: nothing ran against production containers, volumes or ports |

**Exit criteria (IMPLEMENTATION_PLAN §7, S2.8):**

| Criterion | State |
|---|---|
| Legal evaluation suite: golden, held-out, canaries, latency | **Done.** One run of every suite on the code under test (§2) |
| Evaluation runs recorded | **Done.** Migration 0022, import command, Evaluations page (D-044) |
| Latency benchmark on the target hardware | **Done** for this host: the suites' latency (§2) and the load-test smoke (§5) |
| Lawyer review session with the scoring rubric | **Ready, not held (🔒).** Rubric, a 67-item packet from this run's held-out outputs, and the scorer are in place |
| Load-test window | **Waiting (🔒).** The script runs; only a 30-second smoke ran |
| Owner sign-off | **Required (🔒)**, at the end of this report |

## 1. What Phase 2 delivered

| Slice | Backlog | What a lawyer gets | Decision |
|---|---|---|---|
| S2.1 | RAG-01 | Scans read by OCR; Word read as Word shows it (numbering, tracked changes, comments); hidden text left out and listed; clauses and defined terms; a reader that opens at the exact sentence; status per version | D-034, D-035 |
| S2.2 | RAG-01 | Answers from signed versions by default, warning where a draft differs; every quote checked against its source; statuses and abstention reasons; Evidence and "Copy with citations" | D-036–D-038 |
| S2.3 | RAG-02 | Two versions clause by clause (renumbered, moved, added, removed, exact word diff); explanations quoting both wordings; conflicts across documents with the same parties | D-039 |
| S2.4 | LIVE-01 | A verbatim transcript: roster speakers, wall-clock times, visible gaps, corrections as revisions, matter vocabulary, finalised transcripts in research | D-040 |
| S2.5 | LIVE-02 | Silent Assist packs for client calls, negotiations and boards: evidence-relative alerts with exact quotes; confirm or dismiss with a reason; telemetry | D-041 |
| S2.6 | MEET-01 | Agenda and board-pack research; minutes from the whole transcript, every item tied to the words said; approval by a second lawyer; filing | D-042 |
| S2.7 | — | Research notes sent for review; one Reviews queue; exports bound to the text they were planned on; approvals carried into exports | D-043 |
| S2.8 | OPS-02 | Evaluation records, the gate run, the load test, the lawyer review packet | D-044 |

## 2. Evaluation results (one run on `e0f6415`)

The system code was the same for every suite. Held-out sets were run once and nothing was changed after seeing them. Each line is a recorded evaluation run (Evaluations page).

| Suite | Set | Result | Slice result for comparison |
|---|---|---|---|
| Research answers (T05, T06) | Tuning (70) | Answers 60/64; abstentions 4/6; false abstentions 1; **T06 leaks 0**; citation precision 0.932; quotes verified 1.0; T05 conflicts 6/7; multi-source 1/2; p50/p95 2.51/5.77 s | 62/64; 4/6; 0; 0; 0.912; 0.979; 7/7; 2/2 (S2.2) |
| Research answers | Held-out (32) | Answers **28/29**; abstentions **0/3**; false abstentions 0; **T06 leaks 0**; citation precision 0.949; quotes verified 0.974; T05 1/1; p50/p95 2.47/6.15 s | 27/29; 1/3; 1; 0; 0.902; 0.905 (S2.2) |
| Compare and conflicts | Tuning | Alignment all correct; 13 explanations, 19/19 quotes verified, both sides quoted 6/6; conflicts: planted 5/5, precision 1.0 | Same (S2.3) |
| Compare and conflicts | Held-out | Alignment 10/10; 11 explanations, 18/18 quotes verified; conflicts: planted 4/5 (a cross-reference not proposed), precision 1.0 | Same (S2.3) |
| Silent Assist | Tuning (3 calls) | Recall 1.0 (8/8 critical); precision 1.0; false alerts 0; max latency 4.3 s | Same (S2.5) |
| Silent Assist | Held-out (3 calls) | Recall 1.0 (6/6 critical); precision 1.0; false alerts 0; max latency 2.5 s | Same (S2.5) |
| Minutes | Tuning (26 turns) | Recall 1.0; owners 3/3; deadlines 3/3; verified 1.0; **invented 0**; 40 s | Same (S2.6) |
| Minutes | Held-out (12 turns) | Recall 6/7 (an open point minuted as discussion); owners 3/3; deadlines **2/3** ("next week" not captured); verified 1.0; **invented 0**; 21 s | 6/7; 3/3; 3/3 (S2.6) |
| Transcription harness | Scripted (4 scenarios) | WER 0; entities, negations and speakers 1.0. Real audio needs the speech stack (🔒) | Same (S2.4) |
| Prompt injection (T08) | Probe, 63 answers | 1 obeyed, flagged on screen as uncited; **0 obeyed and unflagged** | 0 of 36 (S1.12) |

**Reading these honestly:**
- **Abstention is the weakest measure (D-038).** On the three held-out questions the documents cannot answer, the model said so and then added related, supported context, which counts as answering. The reviewer still sees "not stated" and the evidence.
- **Differences from the slice runs are model variance**, in both directions (research golden down 2, held-out up 1; one minutes deadline missed). They were not tuned away.
- **These are small, fictional sets.** They show the mechanisms work; the lawyer review measures quality.

## 3. Spec tests

| Spec test | What was run | Result |
|---|---|---|
| **T01** matter isolation | Security suite group (507 tests) plus browser journeys 1–6; Phase 2 adds: compare and scans only over visible documents; notes only for people who can see every quoted document; the Reviews queue only over matters you can open | **PASS** |
| **T02** revocation | Security suite group (9) | **PASS** |
| **T03** consent | Security suite group (11) plus the meeting journey; transcripts keep the final text at a stop, never after a withdrawal (S2.4) | **PASS** |
| **T04** legal holds | Security suite group (9); holds also keep minutes and their approvals (S2.6) | **PASS** |
| **T05** versions | Signed versions by default, drafts on request, conflicts reported: 29 backend tests, journey 15, and 7 of 8 conflict items in the evaluations | **PASS** |
| **T06** forbidden matter | 0 leaks across the 102 evaluation questions; backend canaries | **PASS** |
| **T08** prompt injection | Security suite group (23); minutes and notes withhold AI-addressed text; the probe: 0 obeyed and unflagged | **PASS** (residual in D-031) |
| **T09** export redaction | Security suite group (12); exports are now bound to the text their redactions were planned on (S2.7, a real gap closed) | **PASS** |
| **T11** offline egress | Capture mode, as at Phase 1; the enforced namespace run needs root (🔒) | **PARTIAL** |
| **T12** load | Load-test smoke (§5). The measured run needs the window (🔒) | **PARTIAL** |
| **T13** upgrade, rollback, restore | Security suite group (15); rollbacks below 0018–0021 refuse to delete corrections, review decisions and research notes | **PASS** |
| **T15** source permissions | Security suite group (13) | **PASS** |

## 4. Suites at this commit

| Suite | Result |
|---|---|
| Backend (Law and legacy) | **971 passed** |
| Voice | **151 passed** |
| Frontend unit (Vitest) and type check | **106 passed**; `tsc` clean |
| Security suite (`scripts/law/security_suite.sh`) | All 16 groups pass (10 from Phase 1, 6 new) |
| Browser, Law (21 journeys) | **21 passed in each of 5 consecutive runs** (`stability_runs.sh 5`: 105 of 105 journeys, 7.9–8.0 min each) |

## 5. Load test (T12) — smoke only

`scripts/law/load_test.py --smoke` on a fresh seeded development backend, for 30 seconds. One person asked research questions while another searched and a third streamed a consented recording in real time:

| Operation | n | p50 | p95 | max |
|---|---|---|---|---|
| Research answer (real model) | 11 | 2.94 s | 6.14 s | 6.14 s |
| Search (real embeddings) | 137 | 0.21 s | 0.24 s | 0.65 s |
| Capture socket open | 1 | 0.06 s | — | — |
| Capture stop | 1 | 0.01 s | — | — |

- 300 audio frames streamed in real time with no lag.
- No failures and no database-busy errors.
- Backend memory went from 670 to 760 MB.

The first smoke attempt failed on the capture socket because of the script's own WebSocket header argument; it was fixed in the script, and that report was discarded.

**What the measured run adds (🔒):** several minutes at two research users, with the production model's load as it is in the window. Speech recognition under load needs the speech stack.

## 6. Found and fixed in Phase 2

- **The export gateway rendered the version current at approval**, not the one its redactions were planned on: a new version or a corrected transcript could have been released under a plan made for other text. Exports are now bound to their source's hash (S2.7).
- **The legacy Boardroom** read only the first 6,000 characters of a transcript, searched the whole knowledge base and verified nothing. It stays closed; the Law minutes read the whole transcript and check every quote (S2.6).
- **The final words at a normal stop were discarded** (pre-existing). They are now kept, but never after a withdrawal of consent (S2.4).
- **A comparison hint passed AI-addressed text to the model** (S2.3). It is now withheld and escaped.
- **Silent Assist read "agreement" and "signed" as agreeing to a figure** (prefix matching), and a decision showed a user id instead of a name (S2.5).
- **A just-filed document opened in the reader before it was read** and showed a misleading error. It now waits and opens (S2.6).
- **Agenda attribution compared clocks from different sources.** The item is now stored on each segment (S2.6).

## 7. Threat-model review

The THREAT_MODEL rows changed in Phase 2: T-01 (queue and notes), T-06 (injection through documents, transcripts, comparisons, minutes and notes), T-08 (consent and the final text at a stop), and T-09 (exports bound to their source; approvals carried). No new threat class was found. The residuals stay as recorded:
- D-031: a novel injection phrasing can pass the deny-list; it is shown dimmed if uncited.
- D-033: egress is observed, not enforced, without root.

## 8. Known limits

- **Abstention:** see §2 and D-038.
- **Silent Assist thresholds** are engineering defaults until a lawyer signs off P-008, and the screen says so.
- **Transcription accuracy on real audio** is unmeasured (needs the speech stack, 🔒). Speakers are marked by hand.
- **Small evaluation sets.** The lawyer review is the quality measure.
- **One known pre-existing issue on `main`:** Knowledge Chat in the standard edition cannot send right after sign-in until a reload.

## 9. Waiting for approval (🔒)

- The **lawyer review session** (rubric `releases/phase2-rubric.md`; packet `eval/law/lawyer_review/packet.html` and `sheet.csv`; then `score.py`), and the owner's confirmation of the proposed bars.
- A **load-test window**, then `load_test.py --seconds 600 --research-users 2 --window-approved "<who, when>"`.
- The **first `compose.sh up`** on the shared host (the speech stack, real-audio WER, Tesseract in the image).
- **P-008** Silent Assist thresholds, and the other counsel items P-001…P-015.
- The **root-run enforced T11**.
- Any **push or PR** to GitHub.

## 10. Reproduce

```
cd backend && python3 ../eval/law/run_gate.py                       # every suite, about 8 minutes on this host
cd backend && python -m app.legal.evaluations import ../eval/law/reports/<reports>.json
bash scripts/law/security_suite.sh
bash scripts/law/stability_runs.sh 5
python3 scripts/law/load_test.py --smoke                            # with scripts/law/e2e_backend.sh running
python3 eval/law/lawyer_review/make_packet.py
```

## 11. Sign-off

Phase 2 is complete on `law/s2.8-gate` except for the items in §9. **Owner:** approve the Phase 2 gate (merge S2.8 and start Phase 3, S3.1), and say whether the lawyer review and the load-test window are to be scheduled before the pilot.
