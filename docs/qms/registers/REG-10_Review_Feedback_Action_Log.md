# REG-10 — Deliverable Review Feedback and Action Log (EchoMind Product)

| Field | Value |
|---|---|
| Document ID | REG-10 |
| Revision | 1.0 |
| Status | **LIVE REGISTER** |
| Owner | Lead Engineer, EchoMind Product |
| ISO 9001:2015 clauses | 9.1.1, 8.3.4 (design review), 10.2 |
| Raised in response to | **AFR Stage 2 audit finding #7 (16-Oct-2025, clause 9.1.1, Observation)** — *"No actions were available from review feedback of deliverables"* |

---

## 1. Purpose and honest scope

Finding #7 was that review comments were captured informally but produced no tracked actions with
owners, due dates and closure evidence. This register closes that loop for the EchoMind codebase.

**What the repository genuinely evidences:** review of deliverables happens and produces action. Commit
bodies routinely carry *defect → root cause → correction → verification*, and a structured adversarial
review of the largest recent change set produced 40 findings, of which 22 survived independent
verification and were fixed before release.

**What it did not evidence until now:** any of that as a *register* — a list with an owner, a date and
a closure state that someone could audit without reading 204 commit messages.

The entries in §3 are **retrospective**, transcribed from real records and dated by their commits. They
are marked `R`. The first contemporaneously raised entry will be RF-2026-004.

---

## 2. How a deliverable review runs

| Step | Who | Record |
|---|---|---|
| 1. Deliverable reaches review (design, substantive change, release candidate) | Lead Engineer | `FRM-01` design review record |
| 2. Review performed — self-review, structured adversarial review, or evaluation-harness run | Reviewer | Findings listed with severity |
| 3. Each finding becomes a row here with an owner and a due date | Reviewer | This register |
| 4. Finding verified as real before work starts — a plausible finding is not automatically a true one | Reviewer | Verdict recorded |
| 5. Fix implemented | Lead Engineer | Commit sha |
| 6. **Effectiveness verified** — the check that actually closes it | Lead Engineer | Evidence cited in the row |
| 7. Row closed | Owner | Closure date |

Findings that turn out to be real defects in released software are additionally raised as
nonconformities in `REG-04` under SOP-10.

---

## 3. Log

### RF-2026-001 (R) — Adversarial review: voice speculative reply, tool routing, GPU STT

| Field | Entry |
|---|---|
| Deliverable | Voice pipeline change set — speculative replies, LLM tool routing, dynamic hold phrases, GPU final STT |
| Review type | Structured adversarial review across five lenses (concurrency, failure modes, behavioural regression, backend contract, GPU/ASR), each finding then independently verified by two refuters instructed to refute it |
| Date | 2026-09-21 |
| Findings raised | **40** |
| Findings confirmed after verification | **22** (18 refuted — a plausible-but-wrong finding is not actioned) |
| Severity spread | 1 high, 12 medium, 9 low after correction |
| Actions taken | All 22 addressed before release. Examples: abort blocking the event loop (socket shutdown before close); WebSocket close not aborting in-flight generation; first-phrase hold released only on a token gap; hold phrase racing the answer; a mid-word text-splitting defect |
| Effectiveness verification | 12/12 spoken turns across two personas; barge-in verified; char-offset integrity 0 bad; Silent Assistant unaffected; 0 CUDA errors across ~15 sessions |
| Evidence | Commit `ff29843`; workflow journal `subagents/workflows/wf_cc7bac0b-db9/` |
| Status | **Closed**, effectiveness verified |

### RF-2026-002 (R) — Evaluation-harness review of the paper experiment campaign

| Field | Entry |
|---|---|
| Deliverable | Experiment campaign E1–E10 and the claims in the conference paper |
| Review type | Measurement against the paper's own stated claims |
| Date | 2026-08 |
| Findings raised | 4 substantive, including two that **contradict the organisation's own published claims** |
| Detail | E1 — the paper argues retrieval terms dominate grounded latency; measured they are 2.8%. E4 — the paper claims injection is degraded to a *reportable* event; measured report rate is 0.000, so containment is real but **silent**. E3 — the timing channel is not closed. E2/E6/E8/E9 — recorded `not_run` with stated reasons rather than estimated |
| Actions taken | Findings recorded verbatim in `eval/paper/results/SUMMARY.json`. Paper claims to be corrected in any future version |
| Effectiveness verification | Not applicable until a future publication |
| Evidence | `eval/paper/results/SUMMARY.json` |
| Status | **Open — action pending** (also `REG-04` NC-2026-007) |

### RF-2026-003 (R) — Golden-evaluation review of retrieval quality

| Field | Entry |
|---|---|
| Deliverable | Retrieval pipeline |
| Review type | 52-question golden evaluation run against the live stack |
| Date | 2026-07-29 → 2026-08-07 |
| Findings raised | Five retrieval defects in one review (`ebd232f`), plus a later tenant-isolation regression detected as 50/52 → 42/52 |
| Actions taken | Five fixes; then a namespace-enforcing wrapper and corpus-scaled candidate pool (`4e27109`) |
| Effectiveness verification | All five retrieval paths audited: **0 out-of-namespace hits across 359 checked**; suite restored to 49/52 |
| Evidence | Commits `ebd232f`, `4e27109`; `eval/paper/results/REGRESSION_AND_FIXES.json` |
| Status | **Closed**, effectiveness verified. Escalated to `REG-04` NC-2026-001 and NC-2026-002 |

---

## 4. Contemporaneous entries

| Ref | Date | Deliverable | Review type | Finding | Severity | Owner | Due | Closed (effectiveness verified) |
|---|---|---|---|---|---|---|---|---|
| RF-2026-004 | | | | | | | | |

---

## 5. Observations

| Ref | Observation | Action |
|---|---|---|
| **O-1** | Review rigour is high but **concentrated in one person**, who is also the author of the work being reviewed. Independent review is not achievable internally at current headcount — the same constraint that prevents impartial internal audit (SOP-13 §4). The adversarial-review method in RF-2026-001 is a partial mitigation, not a substitute for a second reviewer. | Carried as risk R-03 in `REG-03` |
| **O-2** | Of the three retrospective entries, **two record explicit effectiveness verification and one does not** (RF-2026-002, where verification is not yet applicable). | Tracked by objective QO-5 |
| **O-3** | Review findings that are *refuted* are as valuable as those confirmed — 18 of 40 in RF-2026-001. Recording the refutation prevents re-raising them. | Retained in the workflow journal |
