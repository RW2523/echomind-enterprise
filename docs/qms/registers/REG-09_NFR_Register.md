# REG-09 — Non-Functional Requirements Register (EchoMind Product)

| Field | Value |
|---|---|
| Document ID | REG-09 |
| Revision | 1.0 |
| Status | **LIVE REGISTER** |
| Owner | Lead Engineer, EchoMind Product (Richard Watson Stephen Amudha) |
| ISO 9001:2015 clauses | 8.2.2 (determining requirements), 8.3.3 (design inputs), 8.3.4 (design controls), 9.1.1 (monitoring and measurement) |
| Raised in response to | **AFR Stage 2 audit finding #6 (16-Oct-2025, clause 8.2.2, Minor NC)** — *"No evidence could be seen for recording and monitoring of non functional requirements"* |
| Applies to | The EchoMind Enterprise codebase in this repository (first commit 2026-02-05) |
| Reviewed | 2026-10-07 — Alexander Peter (EchoMind Project Lead); QA Sheryl Nazareth |

---

## 1. Why this register exists, and what it does and does not claim

The 2025 finding was that non-functional requirements were **discussed but not recorded as controlled
records**. That diagnosis was accurate and it remains the accurate diagnosis for the codebase in this
repository: NFRs here are measured **extensively and rigorously** — latency decomposition, tenant
isolation, injection resistance, grounding accuracy, barge-in responsiveness — but until this register
they existed only as experiment result files and commit messages, not as a maintained NFR record with
targets and status.

**This register therefore records measurements that already exist.** Every figure below is transcribed
from a stored result file or a reproducible harness, and the source is cited so any row can be checked.
Nothing here is estimated.

**Two honest qualifications an auditor should be told directly:**

1. **Most of these NFRs never had a formally agreed target.** They were measured, and the measurement
   drove engineering decisions, but no one wrote down "p95 must be under X" beforehand. Where that is
   the case the Target column says `not formally set` rather than inventing a threshold retrospectively.
   Setting them is action **A-1** below.
2. **The 2025 corrective action described a different toolchain.** It cited an NFR tracker in Confluence
   and metrics from *"build v1.3.27 captured automatically via Jenkins"*. This repository contains no
   Confluence, Jira or Jenkins integration and no v1.3.x versioning. Either that evidence belongs to a
   predecessor codebase, or it does not describe this one. **This is flagged for the audit rather than
   papered over** — see §5.

---

## 2. The register

Targets approved 2026-10-07 by Anita Johan (EM26-17 §3; `records/approvals/2026-10-07_AR-2026-001_document_approval.md`). Latest values as recorded in EM26-17.

### 2.1 Performance

| ID | Requirement | Target | Measured | Method / source | Date | Owner | Status |
|---|---|---|---|---|---|---|---|
| NFR-P01 | Grounded chat answer latency | median ≤ 4 s; p95 ≤ 16 s | median 3.31 s; p95 15.17 s (n=30) | Paper harness E1 — eval/paper/results/e1_latency.json | 2026-08-07 | Richard Watson Stephen Amudha | Meets |
| NFR-P02 | Retrieval share of answer latency | Monitor only (no target) | 2.8 % | E1 decomposition | 2026-08-07 | Richard Watson Stephen Amudha | Monitored |
| NFR-P03 | Tenant permission-filter overhead | ≤ 1 ms | 0.08 ms | E1 | 2026-08-07 | Richard Watson Stephen Amudha | Meets |
| NFR-P04 | Voice: end of speech → first audible reply (dual-loop) | median ≤ 1.0 s | median 636 ms; p95 894 ms (n=10) | E10 ablation — e10_ablation.json; paper | 2026-08-07 | Richard Watson Stephen Amudha | Meets |
| NFR-P05 | Voice first reply after speculative-reply work | median ≤ 1.0 s | 0.14–0.49 s on GPU (12 turns, 2 personas); on CPU since 2026-10-06 — to re-measure | voice_e2e session test; commit ff29843 | 2026-09-21 | Richard Watson Stephen Amudha | Meets (GPU) — re-measure on CPU |
| NFR-P06 | Final speech-to-text decode time | ≤ 300 ms (GPU) / ≤ 1.6 s (CPU mode) | 70–130 ms on GPU; CPU mode since 2026-10-06 — to re-measure | commit ff29843 | 2026-09-21 | Richard Watson Stephen Amudha | Re-measure |
| NFR-P07 | Barge-in: interrupt → assistant audio stops | median ≤ 1.0 s | median 2.13 s, success 1.00 (n=8) | E7 — e7_interaction.json | 2026-08-07 | Richard Watson Stephen Amudha | Does not meet — action A-17-03 |

### 2.2 Accuracy

| ID | Requirement | Target | Measured | Method / source | Date | Owner | Status |
|---|---|---|---|---|---|---|---|
| NFR-Q01 | Citation precision | ≥ 0.98 (set in QO-01) | 0.979 (n=36) | E5 — e5_grounding.json | 2026-08-07 | Richard Watson Stephen Amudha | At boundary |
| NFR-Q02 | Citation recall | ≥ 0.80 | 0.826 (n=43) | E5 | 2026-08-07 | Richard Watson Stephen Amudha | Meets |
| NFR-Q03 | Fact support rate | ≥ 0.80 | 0.815 (75/92) | E5 | 2026-08-07 | Richard Watson Stephen Amudha | Meets |
| NFR-Q04 | Abstention when the corpus cannot answer | ≥ 0.80 | 0.783 (n=23) | E5 | 2026-08-07 | Richard Watson Stephen Amudha | Does not meet — action A-17-04 |
| NFR-Q05 | Golden regression suite (52 questions) | ≥ 49/52 (baseline 2026-08-06) | Blocked — corpus absent (9/52 by construction, 2026-09-22); run now aborts with the missing list | eval/run_eval.py; NC-2026-008 | 2026-10-07 | Richard Watson Stephen Amudha | Blocked — action A-17-05 |
| NFR-Q06 | Small talk answered without document citations | 100 % of acceptance probes | Pass on v1.4.1 (after fix e0ed4d0) | TC-EM-RAG-103 run 4 | 2026-10-06 | Richard Watson Stephen Amudha | Meets |

### 2.3 Security

| ID | Requirement | Target | Measured | Method / source | Date | Owner | Status |
|---|---|---|---|---|---|---|---|
| NFR-S01 | No cross-tenant leakage in answers | 0 leaks | 0 / 50 probes | E3 — e3_permission.json | 2026-08-07 | Kishan Haravu Pradeep | Meets |
| NFR-S02 | No out-of-namespace retrieval hits | 0 hits | 0 / 359 (all paths) after fix 4e27109 | e3b_path_isolation.json | 2026-08-07 | Kishan Haravu Pradeep | Meets |
| NFR-S03 | Prompt-injection success rate | ≤ 5 % with defences | 10.2 % undefended → 5.1 % with evidence envelope | E4 — e4_injection.json; paper | 2026-08-07 | Richard Watson Stephen Amudha | At boundary |
| NFR-S04 | Injection guards in every prompt path | All paths | 5 unit tests, run by CI on every push | test_prompt_guards.py; CI | 2026-10-07 | Richard Watson Stephen Amudha | Meets |
| NFR-S05 | Public instance requires login | Unauthenticated API → 401, always | 401 from GitHub's network every 6 h; was off 2026-07-30 → 10-06 (NC-2026-009) | public-gate-check workflow run 37561803121 | 2026-10-07 | Kishan Haravu Pradeep | Meets since 2026-10-06 |
| NFR-S06 | Logout ends the session | Replayed token → 401 | 401 (fix e0ed4d0) | TC-EM-AUTH-101 AC-101.7 run 4 | 2026-10-06 | Kishan Haravu Pradeep | Meets |
| NFR-S07 | Only supported document types accepted | Binary/unsupported → 415 | 415 (fix e0ed4d0) | TC-EM-DOC-102 DOC-102.5 run 4 | 2026-10-06 | Kishan Haravu Pradeep | Meets |
| NFR-S08 | Export gateway removes PII and keys | No sensitive value in redacted copy | Pass incl. sk-/pk- keys (fix e0ed4d0) | TC-EM-EXP-108 run 4 | 2026-10-06 | Kishan Haravu Pradeep | Meets |

### 2.4 Reliability

| ID | Requirement | Target | Measured | Method / source | Date | Owner | Status |
|---|---|---|---|---|---|---|---|
| NFR-R01 | Health checks that exercise the function | All long-running services | 4 of 6 services; Ollama check now makes a real embedding (2026-10-07); frontend/cloudflared none | docker-compose.yml; NC-2026-016 | 2026-10-07 | Richard Watson Stephen Amudha | Partial — action A-17-06 |
| NFR-R02 | Automatic recovery from a fatal GPU fault | Unattended recovery | /health 503 + watchdog + restart policy | backend/app/main.py; docker-compose.yml | 2026-09-22 | Richard Watson Stephen Amudha | Meets |
| NFR-R03 | Bounded logs | 50 MB × 5 per service | Applied to all six containers | docker inspect; REG-07 | 2026-10-06 | Richard Watson Stephen Amudha | Meets |
| NFR-R04 | Customer data recoverable | Restore tested ≤ 6 months; nightly off-site copy | Restore verified 2026-09-22; checksum verified 2026-10-06; off-site schedule not installed | TC-EM-BKP-113 | 2026-10-06 | Kishan Haravu Pradeep | Partial — action A-17-07 |
| NFR-R05 | Reproducible builds | All dependencies pinned | NeMo pinned (60ce9407); ollama/cloudflared images unpinned | Dockerfiles; REG-02 | 2026-09-22 | Richard Watson Stephen Amudha | Partial |

### 2.5 Maintainability

| ID | Requirement | Target | Measured | Method / source | Date | Owner | Status |
|---|---|---|---|---|---|---|---|
| NFR-M01 | Automated regression on every change | Every push | CI: backend 50, voice 131, type-check; proven to fail on a broken test | CI runs 37561379960 / 37561804619 | 2026-10-07 | Richard Watson Stephen Amudha | Meets |
| NFR-M02 | Running instance traceable to source | Version + commit reported | 1.4.1 / f1b8445 on backend, voice and front end | /api/version; TC-EM-REL-106 | 2026-10-06 | Richard Watson Stephen Amudha | Meets |

### 2.6 Data protection

| ID | Requirement | Target | Measured | Method / source | Date | Owner | Status |
|---|---|---|---|---|---|---|---|
| NFR-D01 | Customer data never leaves the perimeter | Absolute | All inference local; no cloud AI path enabled | docker-compose.yml; REG-02 | 2026-10-07 | Kishan Haravu Pradeep | Meets |
| NFR-D02 | Encryption at rest | Decision required | None (SQLite, uploads, indexes unencrypted) | REG-03 R-13 | 2026-09-21 | Kishan Haravu Pradeep | Gap — decision A-17-08 |
| NFR-D03 | Retention bounded per data class | Per class, in days (decision) | No policy | REG-03 R-06 | 2026-09-21 | Kishan Haravu Pradeep | Gap — decision A-17-08 |

## 3. NFRs deliberately NOT measured, and why

Recording what was *not* done, with the reason, is itself required evidence under 9.1.1. These are
transcribed verbatim from the stored results rather than quietly omitted.

| Experiment | Status | Stated reason |
|---|---|---|
| E2 corpus-scale retrieval sweep | `not_run` | Only 421 of 17,514 chunks are content; far below the 1,000-doc / 50k-chunk scale the method requires |
| E6 outcome correctness | `not_run` | Requires two independent annotators with Cohen's kappa; no annotator panel available. An LLM-only rubric would not be a valid substitute |
| E8 process efficiency | `not_run` | Measures tool-call recovery; EchoMind has no tool-calling layer, so the metric is undefined |
| E9 AHP weight elicitation | `not_run` | Requires ≥3 panels × ≥3 practitioners; synthesising pairwise judgements would fabricate the result |

> This is the organisation's measurement culture working as intended, and it is worth showing the
> auditor: where an experiment could not be run properly it is reported as not run, never estimated.
> The same results file also records findings that **contradict the project's own published paper**
> (E1 retrieval share, E4 containment being silent rather than reportable).

---

## 4. How NFR failures have actually been caught and corrected

Evidence that NFR monitoring produces action, not just numbers:

| NFR | What happened | Outcome |
|---|---|---|
| NFR-S02 | Golden evaluation fell 50/52 → 42/52. Root cause: one retrieval path bypassed the namespace predicate, returning 10/10 out-of-namespace chunks under a tenant-scoped query. **Pre-existing and latent**; became visible only as the corpus grew. | Fixed (`4e27109`), all five paths audited, **0/359** out-of-namespace hits, suite restored to 49/52. Logged as `REG-04` NC-2026-001. |
| NFR-Q05 | A retrieval feature measured *worse* than the baseline | Disabled by default (`2decb99`) rather than retained on expectation |
| NFR-R05 | A moving-branch dependency broke build and runtime | Root-caused, re-pinned, verified on a from-scratch rebuild (`724fb98`); permanently pinned 2026-09-22 |
| NFR-P05/P06 | Voice first-reply latency and STT decode time | Re-engineered; measured improvement ~2× and ~10× respectively (`ff29843`, `c2a6ed5`) |

---

## 5. Open actions on this register

| Ref | Action | Priority | Owner | Due |
|---|---|---|---|---|
| ~~A-1~~ | ~~Agree a target for every NFR~~ — **done 2026-10-07**: targets approved in EM26-17 §3 | — | Anita Johan | done |
| A-2 | ~~Re-run the golden evaluation against HEAD~~ — **done 2026-09-22; result 9/52, corpus absent.** Superseded by A-8 | — | — | done |
| **A-8** | **Restore the evaluation corpus and re-measure** (`REG-04` NC-2026-008). Until this is done the organisation has **no current measurement of retrieval quality** and cannot evidence NFR-Q01–Q05 | **Critical — before 7 Oct dry run** | Richard Watson Stephen Amudha | 2026-10-13 |
| ~~A-3~~ | ~~Reconcile the 2025 closure evidence~~ — **addressed 2026-10-07**: the 2025 product is the predecessor; equivalent controls for the rebuilt product in EM26-00 §2 and EM26-17 §2 | — | Alexander Peter | done |
| A-4 | Add health checks to `frontend` and `cloudflared` (NFR-R01) | Medium | Richard Watson Stephen Amudha | 2026-10-31 |
| A-5 | Improve NFR-Q04 abstention accuracy (0.78) — it underwrites the product's central claim | Medium | Richard Watson Stephen Amudha | 2026-11-30 |
| A-6 | Investigate NFR-P07 barge-in stop time (~2.1 s) against a target | Medium | Richard Watson Stephen Amudha | 2026-11-30 |
| A-7 | Set the retention policy (NFR-D03) and implement it | Medium | Anita Johan / Kishan Haravu Pradeep | 2026-10-31 |

## 6. Review

This register is reviewed at each release (SOP-08), at each management review (SOP-15), and whenever an
NFR measurement moves outside its agreed target.

| Review date | Reviewed by | Changes |
|---|---|---|
| 2026-10-07 | Alexander Peter (review), Anita Johan (approval) | Targets set and approved — EM26-17 |
