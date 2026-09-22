# REG-04 — Nonconformity and Corrective Action Log

| Field | Value |
|---|---|
| Document ID | REG-04 |
| Revision | 1.0 |
| Status | **LIVE REGISTER** (the register is live; the procedure governing it, SOP-14, is draft) |
| Owner | Quality representative |
| ISO 9001:2015 clauses | 8.7.2, 10.2.2 |
| Governing procedures | SOP-10 (nonconforming output), SOP-14 (corrective action) |

---

## How this register is used

One row per nonconformity. Raise a row as soon as a nonconformity is identified — not when it is
fixed. A row is closed only when the effectiveness of the corrective action has been **verified with
evidence** (SOP-14 §7); "the code was changed" is not effectiveness verification.

Severity scale (defined in SOP-10 §3):

| Severity | Meaning |
|---|---|
| **S1** | Incorrect, fabricated or misattributed content reached a user, or a tenant-isolation breach, or customer data loss/exposure |
| **S2** | Loss of a module or service, or customer data placed at risk |
| **S3** | Degraded behaviour, incorrect internal state, or a control that is silently not working |
| **S4** | Cosmetic or documentation-only |

---

## Section A — Retrospective entries (transcribed from the Git record on 2026-09-21)

> **Provenance.** These rows were **not** raised contemporaneously in this register, which did not
> exist. They are transcribed from commit messages in the project's Git history, which do record the
> defect, the root cause, the fix and — in most cases — the verification performed. Dates are the
> commit dates. They are entered here because they are genuine, verifiable records of nonconformity
> and corrective action, and because a register that begins empty would misrepresent a project with
> a real corrective-action track record. Nothing in these rows is inferred: where a commit does not
> record something, the cell says so.
>
> Entries are marked `R` (retrospective). The first contemporaneously raised entry will be NC-2026-008.

### NC-2026-001 (R) — Tenant isolation breach on the sparse transcript retrieval path

| Field | Entry |
|---|---|
| Source | Golden-question evaluation regression (50/52 → 42/52) during the paper experiment campaign |
| Date raised | Not recorded separately; fixed 2026-08-07 |
| Severity | **S1** — cross-tenant data exposure in retrieval results |
| Description | `advanced.py` called `Bm25Index.search` directly for the sparse transcript path, bypassing the `_ns_ok` namespace predicate applied by every other retrieval path. Under a tenant-scoped query it returned 10/10 out-of-namespace chunks, which also outranked in-namespace documents and silently zeroed citations on vertical queries. |
| Root cause | One retrieval path was added without routing through the shared namespace-enforcing wrapper; the control was implemented per-path rather than centrally. The defect was **pre-existing and latent** — it became observable only when the corpus grew from 10.3k to 17.6k chunks and the leaked hits began to outrank real ones. |
| Correction | `search_transcript_only_sparse()` wrapper enforcing the predicate; namespace candidate pool scaled with corpus size instead of a fixed 512. |
| Corrective action | All five retrieval paths audited for the same class of defect. |
| Effectiveness verification | **Yes — 0 out-of-namespace hits across 359 checked, all five paths; golden evaluation restored 42 → 49/52.** |
| Customer impact / notification | No customer deployment was affected at the time (pre-commercial); no notification made. |
| Evidence | commit `4e27109` (2026-08-07); `eval/paper/results/REGRESSION_AND_FIXES.json` |
| Status | **Closed**, effectiveness verified |

### NC-2026-002 (R) — Five retrieval defects found by the golden-question evaluation

| Field | Entry |
|---|---|
| Source | Golden-question evaluation |
| Date raised | Not recorded; fixed 2026-07-29 |
| Severity | **S1** — includes a keyword-grep namespace leak (a second tenant-isolation hole) |
| Description | Five distinct retrieval defects identified by the evaluation suite, including a fallback grep path that did not filter by namespace. |
| Root cause | Recorded per-defect in the commit body. The common theme is the same as NC-2026-001: a control applied per-path rather than centrally. |
| Correction | Five fixes in one commit. |
| Effectiveness verification | Partially recorded — the evaluation suite was the detector and was re-run; a per-defect verification statement is not recorded for all five. |
| Evidence | commit `ebd232f` (2026-07-29) |
| Status | **Closed** (effectiveness evidence incomplete — see §C observation O-1) |

### NC-2026-003 (R) — Voice WebSocket authentication gate silently non-functional

| Field | Entry |
|---|---|
| Source | Own observation |
| Date raised | Not recorded; fixed 2026-07-29 |
| Severity | **S2** — a security control believed to be active was not active |
| Description | The voice service's WebSocket auth gate was passed `AUTH_SECRET` where it expected `VOICE_AUTH_SECRET`, so every handshake validation failed open/closed incorrectly and the gate did not work as intended. |
| Root cause | Cross-service configuration coupling with no test covering it; the two services must share a secret under different variable names. |
| Correction | Correct variable passed; the coupling is now documented inline at `docker-compose.yml:142-144`. |
| Effectiveness verification | Not recorded as an explicit verification step. |
| Evidence | commit `73f0b4f` (2026-07-29) |
| Status | **Closed** (effectiveness evidence incomplete — see §C observation O-1) |

### NC-2026-004 (R) — Silent null-audio failure from an incorrect service port

| Field | Entry |
|---|---|
| Source | Own observation (feature did not work; nothing in the logs) |
| Date raised | Not recorded; fixed 2026-07-29 |
| Severity | **S3** — feature silently unavailable, no diagnostic signal |
| Description | `/api/transcribe/speak` defaulted to `http://voice:8002/speak`, but 8002 is only the host port mapping; on the Docker network the voice container listens on 8000. Every call hit "Connection refused", the error was swallowed by a debug-level `except`, and the endpoint silently returned `{"audio_b64": null}` with nothing in the logs. |
| Root cause | Two causes: (a) host-port/container-port confusion in a default value; (b) **an exception handler that logged at DEBUG, converting a hard failure into a silent one.** The second is the more important cause and the more generalisable. |
| Correction | Correct default plus a configurable `VOICE_TTS_URL`; an empty 2xx body now treated as failure; the fallback logged at WARNING instead of DEBUG "so a future misconfiguration is visible"; timeout raised 15 s → 30 s. |
| Effectiveness verification | **Yes — three real outputs confirmed: card mode 122 KB WAV, summary mode 541 KB WAV, and via the public site 83 KB WAV with the RIFF header confirmed.** |
| Evidence | commit `558eaae` (2026-07-29) |
| Status | **Closed**, effectiveness verified |

### NC-2026-005 (R) — Retrieval feature shipped enabled despite measuring worse

| Field | Entry |
|---|---|
| Source | Evaluation measurement |
| Date raised | Not recorded; addressed 2026-07-30 |
| Severity | **S3** |
| Description | The BookRAG hierarchical retrieval path was active by default but measured worse than the standard path. |
| Root cause | A feature was enabled on expectation rather than on measurement. |
| Correction | Gated behind `RAG_ENABLE_BOOKRAG`, default off. |
| Effectiveness verification | The measurement that prompted the change is the evidence. |
| Evidence | commit `2decb99` (2026-07-30) |
| Status | **Closed**. Noted as a positive example: a measured negative result was acted on rather than rationalised. |

### NC-2026-006 (R) — Three build and runtime breakages surfaced by a clean rebuild

| Field | Entry |
|---|---|
| Source | Own observation during a from-scratch rebuild |
| Date raised | Not recorded; fixed 2026-09-04 |
| Severity | **S2** — two of the three prevented the service from building or starting |
| Description | (a) A Python heredoc placed inside a `RUN … \` continuation in `voice/Dockerfile` caused every `docker compose build voice` to fail with "unknown instruction: fi". (b) `nemo_toolkit[asr]` installed from `git@main` pulled `setuptools>=82`, which removed `pkg_resources`; librosa failed at build (backend) and webrtcvad at runtime (voice crash-loop on boot). (c) Boardroom stored raw JSON text as the executive summary when the LLM's report JSON was truncated. |
| Root cause | (a) and (c) are local defects. (b) is a **supplier-risk realisation**: a production dependency installed from a moving Git branch rather than a release. |
| Correction | Heredoc restructured; `setuptools>=70,<82` re-pinned as the last install step in both Dockerfiles; a salvage parser added for truncated report JSON. |
| Corrective action | The underlying supplier risk — NeMo installed from `@main` — is **mitigated but not removed**. It remains open as risk R-01 in `REG-03`. |
| Effectiveness verification | **Yes — verified on a from-scratch rebuild: all six containers healthy; chat, Silent Assistant live checks, voice turn, and boardroom analyse/export all pass; data and model volumes preserved.** |
| Latent condition exposed | Running containers had previously been updated by `docker cp` rather than by rebuild, so the images had drifted from source and the breakages were invisible until a clean rebuild. This is addressed as a control in SOP-06 §6. |
| Evidence | commit `724fb98` (2026-09-04) |
| Status | **Closed** for the three defects; **root supplier risk remains open** (REG-03 R-01) |

### NC-2026-007 (R) — Published claims contradicted by the organisation's own measurements

| Field | Entry |
|---|---|
| Source | Internal experiment campaign (`eval/paper/`, experiments E1–E10) |
| Date raised | Recorded in `eval/paper/results/SUMMARY.json` |
| Severity | **S3** — affects the accuracy of published claims, not the product |
| Description | Measurements contradicted statements in the organisation's own conference paper: E1 — the paper's §7.2 argues the retrieval terms dominate the grounded-response time; measured, they are 2.8% of it. E4 — the paper claims the lattice degrades an injection attempt to a *reportable* event; the measured report rate is 0.000 in every arm, so containment is real but silent. E3 — the timing channel is not closed. E2 — recorded as `"not_run"` with a stated reason rather than estimated. |
| Root cause | Claims were reasoned rather than measured at the time of writing. |
| Correction | The findings are recorded verbatim in the results file, including the contradictions. |
| Corrective action | Pending — the claims should be corrected in any future version of the paper, and the "measure before claiming" rule is now QP-01 §2.3. |
| Effectiveness verification | Not applicable until a future publication. |
| Evidence | `eval/paper/results/SUMMARY.json` |
| Status | **Open — action pending** |

---

## Section B — Contemporaneous entries

### NC-2026-008 — Golden evaluation corpus absent; the quality instrument cannot measure

| Field | Entry |
|---|---|
| Date raised | 2026-09-22 |
| Raised by | `________` (detected by re-running the evaluation for audit evidence) |
| Source | Scheduled measurement ahead of the October surveillance audit |
| Severity | **S2** — the organisation's primary quality instrument is inoperable, so product conformity cannot currently be evidenced (ISO 9001:2015 9.1.1) |
| Description | `python3 eval/run_eval.py` against HEAD returned **9/52**, down from 49/52 (2026-08-06). All **43 retrieval questions failed with zero citations** and document precision 0.00. Non-corpus categories were unaffected: smalltalk 6/6, refusal 2/2, off-corpus 1/1. |
| Investigation | **Not a code regression.** Retrieval works when exercised directly (`retrieve_reranked` returns 6 ranked hits). The cause is the corpus: the knowledge base holds only **5 Meridian Bank demo PDFs**, while the golden set expects **15 documents** — the DoD FMR volumes (`01_01`, `06a_02`, `14_02`, `14_03`) and the vertical demo documents (Product Catalog, Dealership Inventory, Formulary, KYC-AML, Playbook, Visit Note, Company Policy Handbook, Q3 Product Strategy, Banking Products). Their source files are absent from `/data/uploads` (5 files present), so they were removed rather than merely de-indexed. Every retrieval question therefore fails by construction. |
| Root cause | The evaluation corpus is **not version-controlled and not reproducible**. The golden question set is tracked in git (`eval/golden/*.jsonl`), but the documents those questions are asked *about* live only in a mutable runtime volume. Any re-ingest, clear-down or environment rebuild silently invalidates the entire instrument, with no signal until someone runs it. Contributing factor: auto-store has grown the corpus to 9,602 documents of which only 5 are content, so a content document's absence is invisible in aggregate counts. |
| Immediate correction | None yet — the source documents must be recovered or re-obtained. |
| Corrective action (proposed) | (a) Treat the evaluation corpus as a controlled configuration item: store the source documents, or a manifest with checksums, under version control alongside the questions. (b) Add a corpus pre-flight check to `run_eval.py` that asserts every `expect_docs` entry is present and **aborts with a clear message** rather than reporting a misleading score. (c) Re-ingest and re-measure. |
| Effectiveness verification | Pending — the suite must return a score comparable to the 2026-08-06 baseline on a restored corpus. |
| Customer impact | None. This is an internal measurement capability; no customer deployment is affected. |
| Evidence | `eval/reports/eval_20260922-135322.json` (retained); this investigation |
| Status | **OPEN** |
| Audit relevance | Must be disclosed at the 7 October dry run. It is a genuine gap, but it is also evidence of the management system working: a scheduled measurement detected an instrument failure that aggregate statistics concealed. |

---

## Section B (continued) — summary table

| NC ID | Date raised | Source | Severity | Description | Owner | Status | Closed (effectiveness verified) |
|---|---|---|---|---|---|---|---|
| NC-2026-008 | 2026-09-22 | Measurement | S2 | Golden evaluation corpus absent — instrument inoperable | `________` | **Open** | — |

*(Add rows using `forms/FRM-05_Nonconformity_and_CAPA_Record.md`; keep the full record in the form and summarise it here.)*

---

## Section C — Observations arising from the retrospective review

| Ref | Observation | Action |
|---|---|---|
| **O-1** | Of the six retrospective defect entries, **three record explicit effectiveness verification and three do not.** Verification of effectiveness is the control most often skipped. | SOP-14 §7 makes effectiveness verification a mandatory closure condition. Tracked by objective QO-5. |
| **O-2** | Two of the six (NC-2026-001, NC-2026-002) are the **same class of defect**: a namespace/tenant-isolation control applied per retrieval path rather than centrally. A recurrence of a defect class is exactly what corrective action is meant to prevent. | Raise a corrective action to enforce namespace filtering structurally, so a new retrieval path cannot bypass it. To be added to Section B on adoption. |
| **O-3** | Two defects (NC-2026-003, NC-2026-004) were **silent failures** — a control or feature that appeared to work but did not. Silent failure is a distinct and dangerous class for this product. | SOP-10 §3 names silent failure explicitly; SOP-12 requires that a failure path produce an observable signal. |
| **O-4** | No nonconformity in the retrospective set was ever assigned an owner or a due date, because no register existed. | This register, from Section B onward. |

---

## Statistics

| Metric | Value | As at |
|---|---|---|
| Commits on `main` | 202 | 2026-09-21 (`ff29843`) |
| Commits prefixed `fix:` | 22 | 2026-09-21 |
| Retrospective NCs recorded here | 7 | 2026-09-21 |
| Of which S1 | 2 | |
| Closed with effectiveness verified | 3 | |
| Open | 2 (NC-2026-007, NC-2026-008) | |
