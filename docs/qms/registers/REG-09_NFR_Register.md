# REG-09 — Non-Functional Requirements Register (EchoMind Product)

| Field | Value |
|---|---|
| Document ID | REG-09 |
| Revision | 1.0 |
| Status | **LIVE REGISTER** |
| Owner | Lead Engineer, EchoMind Product |
| ISO 9001:2015 clauses | 8.2.2 (determining requirements), 8.3.3 (design inputs), 8.3.4 (design controls), 9.1.1 (monitoring and measurement) |
| Raised in response to | **AFR Stage 2 audit finding #6 (16-Oct-2025, clause 8.2.2, Minor NC)** — *"No evidence could be seen for recording and monitoring of non functional requirements"* |
| Applies to | The EchoMind Enterprise codebase in this repository (first commit 2026-02-05) |

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

Status key: **Met** — measured against an agreed target and satisfied · **Measured** — measured, no
agreed target · **Monitored** — observed continuously, no discrete measurement · **Gap** — not measured.

### 2.1 Performance

| ID | Requirement | Target | Measured | Method / source | Date | Status |
|---|---|---|---|---|---|---|
| NFR-P01 | End-to-end grounded answer latency (chat) | `not formally set` | median **3,313 ms**, p95 **15,171 ms** (concurrency 1, warm cache, n=30/cell) | `eval/paper/results/e1_latency.json` (harness `eval/paper/e1_latency.py`) | 2026-08 | Measured |
| NFR-P02 | Retrieval share of total answer latency | `not formally set` | **2.8%** of grounded latency | same | 2026-08 | Measured — notable: contradicts the project's own published assumption, recorded honestly in `SUMMARY.json` |
| NFR-P03 | Tenant permission filter overhead | Negligible | **0.08 ms** | same | 2026-08 | Met |
| NFR-P04 | Voice: end of user speech → first audible reply | `not formally set` | median **635.8 ms**, p95 **894.0 ms** (n=10, arm `dual_i1`) | `eval/paper/results/e10_ablation.json` | 2026-08 | Measured |
| NFR-P05 | Voice: first audible reply after the speculative-reply and GPU-STT work | `not formally set` | **0.14–0.49 s** across 12 turns, two personas | Session test `voice_e2e_v3.py`; commit `c2a6ed5` / `ff29843` | 2026-09-22 | Measured — ~2× improvement on NFR-P04 |
| NFR-P06 | Final speech-to-text decode time | `not formally set` | **~70–130 ms** (GPU, CUDA graphs disabled), previously ~1.0–1.6 s (CPU) | Commit `ff29843`; measured in-session | 2026-09-21 | Measured |
| NFR-P07 | Barge-in: user interrupt → assistant audio stops | `not formally set` | median **2,131 ms**, p95 **2,140.6 ms**, success **1.00** (n=8) | `eval/paper/results/e7_interaction.json` | 2026-08 | Measured — slowest NFR on record; candidate for a target and improvement |

### 2.2 Accuracy and grounding *(the product's primary quality characteristic)*

| ID | Requirement | Target | Measured | Method / source | Date | Status |
|---|---|---|---|---|---|---|
| NFR-Q01 | Citation precision — cited passages genuinely support the claim | ≥ 0.98 (set in `QO-01` QO-1, 2026-09-21) | **0.9792** (95% CI 0.938–1.02, n=36) | `eval/paper/results/e5_grounding.json` | 2026-08 | Met (at the target boundary) |
| NFR-Q02 | Citation recall | `not formally set` | **0.8256** (CI 0.713–0.938, n=43) | same | 2026-08 | Measured |
| NFR-Q03 | Fact support rate — required gold facts present in the answer | `not formally set` | **0.8152** (75 of 92 fact groups) | same | 2026-08 | Measured |
| NFR-Q04 | Abstention accuracy — correctly declines when the corpus cannot answer | `not formally set` | **0.7826** (CI 0.581–0.903, n=23) | same | 2026-08 | Measured — **weakest accuracy figure; directly tied to the product's central claim** |
| NFR-Q05 | Functional regression suite (52 golden questions, binary gate) | 52/52 (gate at `eval/run_eval.py:281`) | best **50/52** (2026-07-30), most recent **49/52** (2026-08-06) | `eval/reports/*.json` — 24 runs now retained as records | 2026-08-06 | **Gap** — never reached 52/52; and the last run predates current HEAD |

### 2.3 Security and tenant isolation

| ID | Requirement | Target | Measured | Method / source | Date | Status |
|---|---|---|---|---|---|---|
| NFR-S01 | No cross-tenant content leakage in answers | Zero | **0 / 50 probes** (CI 0–0.0714) | `eval/paper/results/e3_permission.json` | 2026-08 | Met |
| NFR-S02 | No out-of-namespace hits on any retrieval path | Zero | **0 / 359** hits across all retrieval paths | `eval/paper/results/e3b_path_isolation.json`; fix commit `4e27109` | 2026-08-07 | Met — **after** a real defect; see §4 |
| NFR-S03 | Prompt-injection containment | `not formally set` | attack success **0.102** with no defence (baseline arm), 98 attack documents | `eval/paper/results/e4_injection.json` | 2026-08 | Measured — partial; defended arms recorded in the file |
| NFR-S04 | Injection guards present in every prompt path | All paths | 5 unit tests asserting guards in analyzer / compress / contextualizer / RAG / strict prompts | `backend/tests/test_prompt_guards.py` | continuous | Monitored |

### 2.4 Reliability and availability

| ID | Requirement | Target | Measured | Method / source | Date | Status |
|---|---|---|---|---|---|---|
| NFR-R01 | Service liveness monitored | All long-running services | Health checks on **4 of 6** services (`trtllm`, `backend`, `voice`, `ollama`); `frontend` and `cloudflared` have none | `docker-compose.yml` | continuous | **Partial gap** |
| NFR-R02 | Automatic recovery from a fatal GPU fault | Unattended recovery | `/health` returns 503 on a poisoned CUDA context; in-process watchdog exits; `restart: unless-stopped` recreates with a fresh context | `backend/app/main.py:100-104`; `docker-compose.yml` | continuous | Monitored |
| NFR-R03 | Disk exhaustion from unbounded logs | Bounded | Log rotation 50 MB × 5 on all six services | `docker-compose.yml` (2026-09-22) | 2026-09-22 | Met — **applies on next `docker compose up -d`** |
| NFR-R04 | Customer data recoverable after loss | Tested restore within 6 months | Restore exercised into a clean volume: `integrity_check ok`, 32 tables, 351 MB archive | `scripts/backup_data.sh` / `restore_data.sh` | 2026-09-22 | Met |
| NFR-R05 | Build reproducibility | Pinned dependencies | `nemo_toolkit` pinned to commit `60ce9407`, previously the moving branch `@main` which caused a production outage | `backend/Dockerfile`, `voice/Dockerfile`; outage commit `724fb98` | 2026-09-22 | Met |

### 2.5 Data protection *(ISO 9001 8.5.3 customer property)*

| ID | Requirement | Target | Measured | Source | Status |
|---|---|---|---|---|---|
| NFR-D01 | Customer data never leaves the customer perimeter | Absolute | All inference local; no cloud AI provider enabled; the single cloud image path (`DOCGEN_IMAGE_BACKEND=nim`) is disabled | `docker-compose.yml`; `REG-02` §4 | Monitored |
| NFR-D02 | Encryption at rest | — | **None.** SQLite, uploads and FAISS indexes are unencrypted | `REG-03` risk R-13 | **Gap — accepted, disclosed** |
| NFR-D03 | Data retention bounded | Per data class | **No retention policy exists**; auto-store grows the corpus continuously (97.6% of chunks were auto-saved transcript at one measurement) | `REG-03` risk R-06; `corpus_stats.json` | **Gap** |

---

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
| **A-1** | **Agree a target for every `not formally set` NFR above.** Measurement without an agreed threshold cannot show conformity, only activity. This is the substantive remainder of finding #6. | **High — before 14 Oct** | `________` | `________` |
| A-2 | Re-run the golden evaluation against current HEAD and record the score (NFR-Q05 figures predate HEAD by six weeks and three feature commits) | **High — before 14 Oct** | `________` | `________` |
| A-3 | Reconcile the 2025 closure evidence (Confluence / Jenkins / build v1.3.27) with this codebase, or state plainly to the auditor that the product has been rebuilt since | **High — before 7 Oct dry run** | `________` | `________` |
| A-4 | Add health checks to `frontend` and `cloudflared` (NFR-R01) | Medium | `________` | `________` |
| A-5 | Improve NFR-Q04 abstention accuracy (0.78) — it underwrites the product's central claim | Medium | `________` | `________` |
| A-6 | Investigate NFR-P07 barge-in stop time (~2.1 s) against a target | Medium | `________` | `________` |
| A-7 | Set the retention policy (NFR-D03) and implement it | Medium | `________` | `________` |

## 6. Review

This register is reviewed at each release (SOP-08), at each management review (SOP-15), and whenever an
NFR measurement moves outside its agreed target.

| Review date | Reviewed by | Changes |
|---|---|---|
| `________` | `________` | Initial — set targets per action A-1 |
