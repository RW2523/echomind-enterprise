# ISO 9001:2015 Surveillance Audit 2026 — EchoMind Product Evidence Pack

| Field | Value |
|---|---|
| Audit | Ajace Inc. ISO 9001:2015 surveillance audit |
| Dry run | **7 October 2026** (technical dry run with Alexander Peter) |
| Final audit | **14 October 2026** |
| Certificate expires | 28 October 2026 |
| Area | **EchoMind Product** |
| Auditee | `________________` (2025 auditee for this area: Alexander Peter) |
| Prepared by | `________________` |
| Prepared | 2026-09-22 |
| Surveillance window | 16 October 2025 → 14 October 2026 |
| Corporate QMS | Ajace Inc. QMS, MR/Quality: Sheryl Nazareth. **This pack covers the EchoMind Product area only** and feeds the corporate QMS; it does not replace it. |

---

## 0. Read this first — one thing to resolve before the dry run

The 2025 closure evidence for **finding #6** states:

> *"Verified 28-Oct-2025: NFR Tracker active in Confluence under /projects/EchoMind/quality/NFR-monitoring.
> Performance and scalability metrics from build v1.3.27 captured automatically via Jenkins reports."*

and for **finding #7**:

> *"Jira dashboard shows closed actions with supporting evidence attachments."*

**None of that toolchain exists in the current EchoMind codebase.** A search of this repository finds no
Confluence reference, no Jira integration, no Jenkins, and no v1.3.x versioning. The codebase in this
repository begins on **2026-02-05** — i.e. it postdates the 2025 audit entirely, and all 204 commits in
the surveillance window belong to it.

There are two possible readings and the team must agree which is true **before 7 October**:

| Reading | What we tell the auditor |
|---|---|
| **A — the product was rebuilt.** The audited 2025 product is a predecessor; this is a new implementation. | The corrective actions were effective for the product they applied to; the product has since been re-engineered, and equivalent controls have been re-established for the new codebase (§2 below). This is an honest and defensible position. |
| **B — Confluence/Jira/Jenkins are live corporate tooling** outside this repository and still hold EchoMind NFR and action records. | Produce those records at the audit. This pack then supplements them rather than replacing them. |

**Do not let the auditor discover this.** A surveillance auditor's standard move is to test whether last
year's corrective actions are still effective. If the answer is "the evidence describes a system we no
longer use" and we have not said so first, a Minor NC becomes a **repeat finding**, which escalates to
**Major**. Stating it first, with the replacement controls already in place, is the difference between a
strength and a major nonconformity.

> **Action:** Alexander/Richard to confirm A or B, and Sheryl to be briefed, before the 7 Oct dry run.

---

## 1. Closure of 2025 findings — EchoMind Product

Three of the ten 2025 findings touch this area.

### Finding #6 — clause 8.2.2, **Minor NC**
> *"No evidence could be seen for recording and monitoring of non functional requirements"*

| | |
|---|---|
| **Status for the current codebase** | **Addressed — evidence available** |
| **What is produced** | **`REG-09` Non-Functional Requirements Register** — 20 NFRs across performance, accuracy/grounding, security/isolation, reliability, and data protection, each with the measured value, the method, the date and the source file |
| **Substantive evidence behind it** | Latency decomposition (n=30/cell): grounded answer median **3,313 ms**, p95 15,171 ms, retrieval share **2.8%**, permission filter **0.08 ms** · Voice first-reply median **635.8 ms** → **0.14–0.49 s** after the September work · Tenant isolation **0/50** content leaks and **0/359** out-of-namespace retrieval hits · Citation precision **0.979** · Injection resistance across 98 attack documents · Abstention accuracy 0.783 |
| **Where it lives** | `eval/paper/results/*.json` (tracked in git), `eval/reports/*.json` (24 runs, now retained as records), `docs/qms/registers/REG-09_NFR_Register.md` |
| **Honest residual** | Most NFRs were measured **without a formally agreed target beforehand**. Measurement without a threshold demonstrates activity, not conformity. Setting targets is action A-1 in REG-09 and should be done **before 14 October**. |

### Finding #7 — clause 9.1.1, **Observation**
> *"No actions were available from review feedback of deliverables"*

| | |
|---|---|
| **Status for the current codebase** | **Addressed — evidence available** |
| **What is produced** | **`REG-10` Deliverable Review Feedback and Action Log** |
| **Strongest single exhibit** | A structured adversarial review of the September voice change set: **40 findings raised, 22 confirmed after independent verification, 18 refuted, all 22 actioned before release**, with effectiveness verified (12/12 spoken turns, two personas, 0 regressions). The refutations are retained too — a plausible finding that is not real should not be actioned, and recording why is part of the control. |
| **Also evidenced** | Golden-evaluation review that found five retrieval defects and a tenant-isolation regression, each root-caused and verified (`ebd232f`, `4e27109`) |
| **Honest residual** | Review is rigorous but performed by the **same person who wrote the work**. Independent review is not achievable at current headcount — the same constraint that limits internal audit. Carried as risk R-03. |

### Finding #10 — clause 9.3.2, **Minor NC** *(Senior Management, with EchoMind content)*
> *"Unavailability of Blue Yonder skills and delay in Hardware by a supplier in EchoMind project — these two issues/risks were not available in any of the management review meeting minutes"*

| | |
|---|---|
| **Status** | **EchoMind side ready; corporate side owned by Senior Management** |
| **What this area now provides** | **`REG-03` Risk Register** — 14 EchoMind risks with evidence, likelihood/impact and treatment, in a form that can be tabled directly at management review. Supplier risk is explicitly covered: R-01 (a dependency on a moving upstream branch that **caused a real production outage**) is the direct analogue of the 2025 hardware-supplier finding, and it is now **treated** — pinned to a verified commit on 2026-09-22, residual 20 → 4. |
| **What the auditor will check** | That EchoMind risks now actually **appear in management review minutes**. That is a corporate action (SOP-15 input checklist), not a repository one. |
| **Action** | Ensure the next management review minutes record EchoMind project risks from `REG-03`. Owner: Senior Management / MR. |

---

## 2. Sheryl's eight review points — answers for this area

### 2.1 Closure of last year's findings
See §1. Two EchoMind findings addressed with new registers; one corporate finding supported with a risk register.

### 2.2 New products, services and significant activities since the last audit

**The entire EchoMind Enterprise platform in this repository is new within the surveillance window** — first commit 2026-02-05, 204 commits to date. Delivered in the window:

| Capability | Evidence |
|---|---|
| Knowledge Chat — retrieval over customer documents with citations | `docs/RAG_FLOW.md`; golden eval |
| Live Transcription + Silent Assistant — real-time fact-checking of speech against the knowledge base | `backend/app/transcribe/`, `backend/app/silent_assistant/` |
| Boardroom — meeting capture, diarised transcript, AI report | `backend/app/boardroom/` |
| Voice Conversation — full-duplex speech, sub-second replies, barge-in | `voice/` |
| Document Studio — document generation with on-device image generation | `backend/app/docgen/` |
| Five vertical packs (health, law, meetings, retail, bank) with isolated knowledge-base namespaces | `frontend/packs.ts`, `frontend/verticals/` |
| Migration to Qwen3-30B-A3B NVFP4 on TensorRT-LLM | `80375c6` |
| Secure offline export gateway with PII/secret scan and redaction | `d19177d` |
| Multi-tenant per-tenant scoping | `ef19d68` |
| Public deployment via Cloudflare Tunnel + Access | `docs/PUBLIC_DEPLOYMENT.md` |
| A peer-reviewed conference paper (QASC 2026) with a reproducible experiment harness | `docs/paper/`, `eval/paper/` |

### 2.3 Process improvements — AI and technology
*This is the strongest section for this area; Sheryl asked for it specifically.*

| Improvement | Effect | Evidence |
|---|---|---|
| **Golden-question evaluation harness** — 52 curated questions, binary pass gate, run against the live stack | Retrieval quality is measured, not asserted. It has caught real defects including a tenant-isolation breach | `eval/run_eval.py`, `eval/golden/`, 24 retained reports |
| **AI-assisted development, declared** | A material share of commits are AI-assisted and say so via `Co-Authored-By` trailers. Human review and verification of generated work is a mandatory control, and commit bodies evidence it | Git history; `SOP-05` |
| **Structured adversarial review** | 40 findings on one change set; each independently verified before action; 18 refuted | `REG-10` RF-2026-001 |
| **Research-grade experiment harness (E1–E10)** | Independent measurement of latency, isolation, injection resistance and grounding | `eval/paper/results/` |
| **Automated data backup with verified restore** | Customer data recoverable; restore proven, not assumed | `scripts/backup_data.sh`, `restore_data.sh` |
| **Release identification and traceability** | A running instance can now report the exact source revision it was built from | `scripts/release.sh`, `/api/version`, `CHANGELOG.md` |
| **Speculative reply + GPU speech recognition** | Voice first-reply latency roughly halved; final STT decode ~10× faster | `ff29843`, NFR-P05/P06 |

### 2.4 Process changes since the last audit

| Change | Date | Why |
|---|---|---|
| Release identification introduced — semantic version, annotated git tag, build identifier reported at runtime, generated changelog | 2026-09-22 | No traceability from a deployed instance to its source existed (ISO 8.5.2) |
| Data backup and restore procedure introduced and tested | 2026-09-22 | The only volume holding customer data had no backup |
| Container log rotation | 2026-09-22 | Unbounded logs would fill the disk and stop the database |
| Dependency pinned to an immutable commit | 2026-09-22 | A moving upstream branch caused a production outage |
| Evaluation reports retained as quality records | 2026-09-22 | Results were previously discarded, so no measurement history existed |
| NFR register and review-action log introduced | 2026-09-22 | Closure of findings #6 and #7 |

### 2.5 Metrics / KPI changes

**Seven quality objectives now exist for this area** (`QO-01`), where previously there were none:
grounded answers or none · verified before release · traceability from instance to source · customer
data backed up and restorable · nonconformities closed with verified effectiveness · dependencies
pinned and licensed · the management system actually operating.

| Objective | Baseline | Status |
|---|---|---|
| QO-4 data recoverable | No procedure existed | **Met** — restore verified 2026-09-22 |
| QO-6 dependencies pinned | ≥5 unpinned | **Partly met** — highest-risk one pinned; licences outstanding |
| QO-1 citation precision ≥ 0.98 | 0.9792 (2026-08-06) | At the boundary; **needs re-measuring against HEAD** |
| QO-2 verified before release | 0% | Mechanism now exists; no release record yet produced |
| QO-3 instance traceable to source | No mechanism | Mechanism built 2026-09-22; first tagged release outstanding |

### 2.6 Quality manual and procedures
Product-level documentation now exists at `docs/qms/` — a manual, policy, objectives, 15 procedures,
13 forms and 10 registers, with a clause-by-clause mapping and an honest gap analysis. **It is at draft
status and is not yet approved**, which is stated on every document. It supplements, and must be
reconciled with, the corporate Ajace QMS — see §4.

### 2.7 Organisational and process risks
`REG-03` holds 14 risks derived from code and configuration evidence, each with likelihood, impact,
existing controls and treatment. Four have been treated within the window (supplier pinning, backup,
log rotation, evaluation retention). Ratings are marked **proposed** pending management confirmation.

### 2.8 Previous audit items
Covered in §1.

---

## 3. What the auditor is most likely to ask, and where the evidence is

| Question | Answer | Where |
|---|---|---|
| "Show me which source revision this running instance was built from." | `GET /api/version` returns version, commit and build date; `CHANGELOG.md` and the annotated tag link it to the change set | `scripts/release.sh`, `backend/app/main.py` |
| "How do you know a change did not break retrieval quality?" | 52-question golden evaluation with a binary gate, plus 78 unit tests | `eval/`, `backend/tests/`, `voice/tests/` |
| "Show me a non-functional requirement and its measurement." | `REG-09`, any row | `docs/qms/registers/REG-09_NFR_Register.md` |
| "Show me review feedback that produced an action and was closed." | `REG-10` RF-2026-001 — 22 confirmed findings actioned, effectiveness verified | `docs/qms/registers/REG-10_Review_Feedback_Action_Log.md` |
| "How is customer data protected and recoverable?" | `SOP-11`; backup with a verified restore | `scripts/backup_data.sh`, `REG-09` NFR-R04 |
| "Show me a nonconformity, its root cause and evidence the fix worked." | `REG-04` NC-2026-001 — tenant isolation breach, root cause, fix, **0/359 verified** | `docs/qms/registers/REG-04_Nonconformity_and_CAPA_Log.md` |
| "How do you control third-party components?" | `SOP-09`, `REG-02` — 11 models, 6 base images, 7 services inventoried | `docs/qms/registers/REG-02_External_Documents_and_Providers.md` |
| "What are the risks in this area and how are they monitored?" | `REG-03` | `docs/qms/registers/REG-03_Risk_Register.md` |

---

## 4. Known weaknesses — raise these ourselves

An auditor finds these in the first hour. Declaring them, with an owner and a date, converts a finding
into evidence that the management system is working.

| # | Weakness | Clause | Our position |
|---|---|---|---|
| W-1 | The 2025 closure evidence names a toolchain not present in this codebase | 10.2 | §0 — resolve before the dry run |
| W-2 | Most NFRs have no formally agreed target | 8.2.2 | REG-09 action A-1, due before 14 Oct |
| W-3 | The golden evaluation has never reached 52/52, and the last recorded run (49/52, 2026-08-06) predates current HEAD | 9.1.1 | Re-run before the audit — REG-09 A-2 |
| W-4 | No release has yet been cut with the new mechanism | 8.5.2, 8.6 | Cut `v1.4.0` before the dry run so a real release record exists |
| W-5 | Product-level QMS documents are draft and unapproved | 7.5 | Either approve them or present them explicitly as supporting documentation under the corporate QMS |
| W-6 | Independent review and internal audit are not achievable at current headcount | 9.2.2 c | Declared; corporate-level decision |
| W-7 | No data-retention policy; no encryption at rest | 8.5.3 | Declared and risk-assessed (R-06, R-13) |
| W-8 | Application authentication is off by default and WebSocket endpoints are not covered by the auth middleware | 8.5.3 | Declared; deployment is gated by Cloudflare Access |

---

## 5. Before 7 October — dry run readiness

| # | Task | Owner | Status |
|---|---|---|---|
| 1 | **Resolve §0** — decide reading A or B and brief Sheryl | Alexander / Richard | ☐ |
| 2 | **Set NFR targets** (REG-09 A-1) | Lead Engineer | ☐ |
| 3 | **Re-run the golden evaluation against HEAD**, retain the report, update QO-1/NFR-Q05 | Lead Engineer | ☐ |
| 4 | **Cut release `v1.4.0`** with `scripts/release.sh` so a tagged release and a completed `FRM-03` exist | Lead Engineer | ☐ |
| 5 | Rebuild images with build args so `/api/version` reports the real build, then verify on the running stack | Lead Engineer | ☐ |
| 6 | Install the backup timer with an off-host destination | Lead Engineer | ☐ |
| 7 | Decide whether the product-level QMS documents are approved or presented as supporting material (W-5) | Richard / Sheryl | ☐ |
| 8 | Confirm EchoMind risks are tabled at the next management review (finding #10) | Senior Management | ☐ |
| 9 | Walk this pack end to end with Alexander | Both | ☐ |

## 6. Evidence index

| Evidence | Path |
|---|---|
| NFR register | `docs/qms/registers/REG-09_NFR_Register.md` |
| Review feedback and action log | `docs/qms/registers/REG-10_Review_Feedback_Action_Log.md` |
| Nonconformity and CAPA log | `docs/qms/registers/REG-04_Nonconformity_and_CAPA_Log.md` |
| Risk register | `docs/qms/registers/REG-03_Risk_Register.md` |
| External providers | `docs/qms/registers/REG-02_External_Documents_and_Providers.md` |
| Quality objectives | `docs/qms/QO-01_Quality_Objectives.md` |
| Clause mapping | `docs/qms/ISO9001_Clause_Mapping.md` |
| Gap analysis | `docs/qms/ISO9001_Gap_Analysis.md` |
| Release procedure and changelog | `scripts/release.sh`, `CHANGELOG.md` |
| Backup and restore | `scripts/backup_data.sh`, `scripts/restore_data.sh` |
| Measurement results | `eval/paper/results/`, `eval/reports/` |
| Test suites | `backend/tests/` (40), `voice/tests/` (38) |
| Architecture documentation | `docs/` |
