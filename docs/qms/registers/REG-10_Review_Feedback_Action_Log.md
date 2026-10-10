# REG-10 — Deliverable Review Feedback and Action Log (EchoMind Product)

| Field | Value |
|---|---|
| Document ID | REG-10 |
| Revision | 1.0 |
| Status | **LIVE REGISTER** |
| Owner | Lead Engineer, EchoMind Product (Richard Watson Stephen Amudha) |
| ISO 9001:2015 clauses | 9.1.1, 8.3.4 (design review), 10.2 |
| Raised in response to | **AFR Stage 2 audit finding #7 (16-Oct-2025, clause 9.1.1, Observation)** — *"No actions were available from review feedback of deliverables"* |
| Reviewed | 2026-10-07 — Alexander Peter (EchoMind Project Lead); QA Sheryl Nazareth. 2026-10-10 — client and design reviews CR-2026-01…09 entered (§3b) |

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

## 3a. Review log 2026 (summary of EM26-18)

All actions — 40 from engineering and quality reviews (below) and 15 from client and design reviews (§3b), 55 in total — with owner, due date and closure evidence, are in `evidence-2026/18_Review_Actions/EM26-18_Deliverable_Review_Feedback_and_Action_Log.docx` and on the project board. Reviewed by Alexander Peter, approved by Anita Johan on 2026-10-07.

| Review | Date | Deliverable | Performed by | Findings | Actions | Closed | Record |
|---|---|---|---|---|---|---|---|
| RV-2026-01 | 2026-07-29 → 08-07 | Retrieval pipeline | Richard Watson Stephen Amudha | 5 defects + 1 isolation regression | 2 | 2 | ebd232f, 4e27109 — REG-10 RF-2026-003 |
| RV-2026-02 | 2026-08-07 | Research paper experiment campaign (E1–E10) | Richard Watson Stephen Amudha | 4 substantive (2 contradict published claims) | 1 | 0 | eval/paper/results/SUMMARY.json — RF-2026-002 |
| RV-2026-03 | 2026-09-21 | Voice change set (speculative reply, tool routing, GPU STT) | Richard Watson Stephen Amudha (AI-assisted, independent verification) | 40 raised; 22 confirmed; 18 refuted | 22 | 22 | ff29843 — RF-2026-001 |
| RV-2026-04 | 2026-09-22 | Golden evaluation re-run for audit evidence | Richard Watson Stephen Amudha | Corpus absent (9/52) | 2 | 1 | NC-2026-008; eval_20260922-135322.json |
| RV-2026-05 | 2026-09-28 | Repository health check | Richard Watson Stephen Amudha | 5 failing tests, 20 type errors | 2 | 2 | 5ddf8a6, 736ea61 — NC-2026-010 |
| RV-2026-06 | 2026-10-06 | October audit pack | Richard Watson Stephen Amudha | 6 statements wrong (incl. no login on the public site) | 2 | 2 | 855bd80, 1d7f154 — NC-2026-009 |
| RV-2026-07 | 2026-10-06 | Release candidate (13 acceptance test cases) | Richard Watson Stephen Amudha | 4 defects; re-tests found 2 more | 5 | 3 | e0ed4d0, 3feb5db, f1b8445 — NC-2026-012…016 |
| RV-2026-08 | 2026-10-07 | Public instance access logs | Richard Watson Stephen Amudha | Ungated 68 days; 64 scanner probes; no data change outside engineering work | 4 | 2 | records/capa/2026-10-07_NC-2026-009_access_log_review.md |

## 3b. Client and design reviews 2026 (added 2026-10-10)

The 2026 deliverables came from client requirements and client feedback on each version shown to them.
Months are the EchoMind team's account, checked against the commit record (where they differed, the
commit date is used). Clients are identified as Client A–D; the names are held by the EchoMind
Project Lead and appear only in the controlled copy of EM26-18 given to the auditor.

**Project history**

| When | Event, requirement or feedback | Source | Review / action | Evidence |
|---|---|---|---|---|
| 2025 | EchoMind (predecessor) delivered a live transcript and a chatbot with RAG on a PC we built for the client, as the client required | 2025 client project | — | 2025 evidence pack; EM26-00 §0 |
| 2026-01 | Portable AI supercomputers (NVIDIA DGX Spark) reach the market; the team decides to move the product to one | Management / engineering decision | CR-2026-01 | This log |
| 2026-02 | First DGX Spark bought; the 2025 code ported in full and running on it | Engineering | AC-20 | 4b90b36, c751d79 (2026-02-05) |
| 2026-03 | Requirements from Client A and Client B (Bahrain): (1) personal AI assistant, (2) conversation assistance, (3) financial knowledge base on the DoD Financial Management Regulation (DoD 7000.14-R) | Clients A and B | CR-2026-02 | 95d3512 (2026-03-08) |
| 2026-03 | Conversation assistance v1 on the Kyutai Unmute stack — client: not natural and slow | Clients A and B | CR-2026-03 | 4b90b36 (Moshi adapter), bff6cf2 (Kyutai STT) |
| 2026-03 | DoD FMR is a very large document: BookRAG and PageIndex tried; hierarchical (section → chunk) chunking adopted for performance | Engineering | AC-22 | 31f1205 … 5057a30 (2026-03-03 → 06); 23ac2f9 (03-07); 2decb99 (07-30) |
| 2026-03 → 04 | Conversation assistance v2 on NVIDIA PersonaPlex — client: must work with the customer's large data in spoken conversation | Clients A and B | CR-2026-04 | QASC 2026 paper (539493f) |
| 2026-04 | Own dual-loop conversation architecture built and pushed to the code | Engineering | AC-24 | ddc0aa3, d89aeda (2026-04-08) |
| 2026-04 | Alexander Peter asks how a unit will work once deployed on a client site; proposes a hotspot — at every boot the unit raises its own Wi-Fi hotspot and serves a web page from which the applications open | Project Lead design review | CR-2026-05 | Deployment unit configuration; 0818cdc (2026-04-10) |
| 2026-04 | Client A reports live-transcript accuracy and speed problems — better speech model with fallback across models; chat moved from Ollama to TensorRT-LLM | Client A | CR-2026-06 | d89aeda (04-08), b6a70d8 (06-26), 80375c6 (07-29) |
| 2026-05 | Client C (law firm) likes conversation and live transcript; asks for a mode that listens, pulls up data while people speak and shows it on screen without replying — the Silent Assistant | Client C | CR-2026-07 | ec9b180 (05-11), f75121b (06-27), 7f2658a (08-16) |
| 2026-05 | Group discussions: a diarisation model adopted after much trial and error in model selection — Boardroom | Engineering | CR-2026-08 | f540976 (05-14), cf463aa (05-19), e1a337b (05-27) |
| 2026-06 | Client D (Gulf region) asks for a document studio — PPT and PDF created locally inside the application, with and without a template, with generated images | Client D | CR-2026-09 | 2d23f40 (06-20), 66356f9 (06-27) |

**Reviews**

| Review | Date | Deliverable | Review type | Performed by | Findings | Actions | Closed | Record |
|---|---|---|---|---|---|---|---|---|
| CR-2026-01 | 2026-01 | Platform for the next product generation | Management / engineering review of the 2025 appliance against new hardware | EchoMind team | A single-PC appliance limits capacity and portability; a portable AI supercomputer is now available | 1 | 1 | Decision recorded here; 4b90b36 |
| CR-2026-02 | 2026-03 | Requirements — Client A; Client B (Bahrain) | Requirements review with the clients | EchoMind team with the clients | 3 requirements: personal AI assistant; conversation assistance; DoD FMR financial knowledge base | 2 | 2 | 95d3512, 18715c7 |
| CR-2026-03 | 2026-03 | Conversation assistance v1 (Kyutai Unmute stack) | Client demonstration review | Clients A and B | Not natural; too slow | 1 | 1 | 4b90b36, bff6cf2 |
| CR-2026-04 | 2026-03 → 04 | Conversation assistance v2 (NVIDIA PersonaPlex) | Client demonstration review | Clients A and B | Fast, but cannot use the customer's large knowledge base in spoken conversation | 1 | 1 | ddc0aa3; QASC 2026 paper |
| CR-2026-05 | 2026-04 | Field deployment of the unit | Design review | Alexander Peter (EchoMind Project Lead) | No way to reach a deployed unit on a site without a network | 2 | 1 | Deployment unit; 0818cdc |
| CR-2026-06 | 2026-04 | Live transcription | Client feedback review | Client A | Accuracy and speed below need (e.g. 'DoD FMR' heard as 'DODFMA') | 3 | 3 | d89aeda, b6a70d8, 80375c6 |
| CR-2026-07 | 2026-05 | Conversation and live transcript demonstration | Client review | Client C (law firm) | Needs a listen-only assistant that pulls supporting data onto the screen while people speak | 2 | 2 | ec9b180, f75121b, 7f2658a |
| CR-2026-08 | 2026-05 | Group discussions (Boardroom) | Design review and model trials | EchoMind team | Speaker separation (diarisation) needed; model chosen by trial | 2 | 2 | f540976, cf463aa, e1a337b |
| CR-2026-09 | 2026-06 | Document creation | Client requirement review | Client D (Gulf region) | PPT and PDF made locally, with and without a template, with images | 1 | 1 | 2d23f40, 66356f9 |

**Actions**

| Action | From | Action | Owner | Due | Status | Closure evidence |
|---|---|---|---|---|---|---|
| AC-20 | CR-2026-01 | Buy the first DGX Spark and port the 2025 code to it (ARM64) | EchoMind team | 2026-02 | Closed | 4b90b36, c751d79 (2026-02-05) — 2025 code running on the DGX Spark |
| AC-21 | CR-2026-02 | Personal AI assistant: financial-assistant behaviour, conversation memory, personas | Richard Watson Stephen Amudha | 2026-04 | Closed | 95d3512, e59fd55 (2026-03-08); 518fb8e (2026-04-08); c6b834d (2026-06-27) |
| AC-22 | CR-2026-02 | DoD FMR knowledge base: try BookRAG and PageIndex; adopt hierarchical section → chunk retrieval with table-of-contents routing and regulation-tuned query classification | Richard Watson Stephen Amudha | 2026-03 | Closed | 31f1205, 045b1b1 (03-03); 18715c7, cfe3360 (03-05); 5057a30 (03-06). BookRAG 23ac2f9 (03-07) measured 43/52 vs 50/52 and was switched off (2decb99, 07-30); PageIndex evaluated, not adopted |
| AC-23 | CR-2026-03 | Replace the Unmute stack with NVIDIA PersonaPlex | Richard Watson Stephen Amudha | 2026-03 | Closed — superseded by AC-24 | Trial compared in the QASC 2026 paper; trial code not kept |
| AC-24 | CR-2026-04 | Build our own dual-loop conversation architecture grounded in the customer's knowledge base | Richard Watson Stephen Amudha | 2026-04 | Closed | ddc0aa3, d89aeda (04-08); fbbadd0 (05-28); 5ce8bbe, cbc57d1 (08-07) — 636 ms median first audio vs 866 ms single loop |
| AC-25 | CR-2026-05 | Hotspot: unit raises a Wi-Fi hotspot at every boot and serves a landing page that opens the applications | Alexander Peter | 2026-04 | Closed | Built on the deployment unit; offline preparation 0818cdc (2026-04-10) |
| AC-26 | CR-2026-05 | Bring the hotspot configuration under version control and add it to the deployment guide | Alexander Peter | 2026-10-31 | Open | — |
| AC-27 | CR-2026-06 | Replace Whisper with Nemotron streaming speech recognition for live transcription | Richard Watson Stephen Amudha | 2026-04-08 | Closed | d89aeda, 172b94c, ddc0aa3 |
| AC-28 | CR-2026-06 | Accurate final transcript with Parakeet-TDT and automatic fallback to the streaming model | Richard Watson Stephen Amudha | 2026-06-26 | Closed | b6a70d8, 3a6b1f9 — fixes 'DoD FMR' → 'DODFMA' |
| AC-29 | CR-2026-06 | Move chat generation from Ollama to TensorRT-LLM for speed | Richard Watson Stephen Amudha | 2026-07-29 | Closed | 06e788a (03-20) integration; 80375c6 (07-29) Qwen3-30B-A3B NVFP4 on TensorRT-LLM; Ollama kept for embeddings and rollback |
| AC-30 | CR-2026-07 | Silent Assistant: runs with the live transcript; an assistant and an agent check the content continuously and surface data and pointers on screen without replying | Richard Watson Stephen Amudha | 2026-05-11 | Closed | ec9b180 (2026-05-11) |
| AC-31 | CR-2026-07 | Law rule pack, always-surface mode, namespace awareness; sentence-level checks with quoted proof | Richard Watson Stephen Amudha | 2026-08-16 | Closed | f75121b (06-27), 7f2658a (08-16) |
| AC-32 | CR-2026-08 | Select a diarisation model by trial; build Boardroom (group discussion, speaker-separated transcript, report) | Richard Watson Stephen Amudha | 2026-05-27 | Closed | f540976 (05-14), cf463aa (05-19), e1a337b (05-27); VibeVoice-ASR adopted (REG-02 M-06) |
| AC-33 | CR-2026-08 | Harden Boardroom: report salvage when the model output is truncated | Richard Watson Stephen Amudha | 2026-09-04 | Closed | b08cf25 (06-18), 724fb98 (09-04) |
| AC-34 | CR-2026-09 | Evaluate image-generation and document models; build Document Studio with PDF and PPTX export, on-device images, with and without a template | Richard Watson Stephen Amudha | 2026-06-27 | Closed | 2d23f40 (06-20), 66356f9 (06-27); SDXL-Turbo on device (REG-02) |

## 4. Contemporaneous entries

| Ref | Date | Deliverable | Review type | Finding | Severity | Owner | Due | Closed (effectiveness verified) |
|---|---|---|---|---|---|---|---|---|
| RF-2026-004 | | | | | | | | |

---

## 5. Observations

| Ref | Observation | Action |
|---|---|---|
| **O-1** | Review rigour is high but **concentrated in one person** (from 2026-10-07: independent review of deliverables by the EchoMind Project Lead, Alexander Peter), who is also the author of the work being reviewed. Independent review is not achievable internally at current headcount — the same constraint that prevents impartial internal audit (SOP-13 §4). The adversarial-review method in RF-2026-001 is a partial mitigation, not a substitute for a second reviewer. | Carried as risk R-03 in `REG-03` |
| **O-2** | Of the three retrospective entries, **two record explicit effectiveness verification and one does not** (RF-2026-002, where verification is not yet applicable). | Tracked by objective QO-5 |
| **O-3** | Review findings that are *refuted* are as valuable as those confirmed — 18 of 40 in RF-2026-001. Recording the refutation prevents re-raising them. | Retained in the workflow journal |
