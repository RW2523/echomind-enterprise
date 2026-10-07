# QM-01 — Quality Manual

| Field | Value |
|---|---|
| Document ID | QM-01 |
| Revision | 2.0 |
| Status | **APPROVED** |
| Organisation | AJACE Inc. — 14159 Robert Paris Ct., Suite A, Chantilly, VA 20151, USA |
| Owner | Managing Director (Anita Johan) |
| Prepared by | Richard Watson Stephen Amudha (Lead Engineer) — 2026-10-07 |
| Reviewed by | Alexander Peter (EchoMind Project Lead) — 2026-10-07 |
| Quality assurance | Sheryl Nazareth (QA / Management Representative) — 2026-10-07 |
| Approved by | Anita Johan (Managing Director) |
| Approval date | 2026-10-07 |
| Approval record | `records/approvals/2026-10-07_AR-2026-001_document_approval.md` |
| ISO 9001:2015 clauses | 4.1, 4.2, 4.3, 4.4, 5, 6, 7, 8, 9, 10 (overview) |

> ISO 9001:2015 does not require a quality manual. This one exists so that the EchoMind Product area
> has a single document that says what its management system is, what it covers, how it fits within
> AJACE Inc.'s certified corporate QMS, and — candidly — what is operating today (section 14).

---

## 1. The organisation

**AJACE Inc.** (product brand *Ajace AI*) designs, builds, customises, deploys and supports
**EchoMind Enterprise** — a private, on-premises artificial-intelligence workspace. The platform
provides knowledge chat over the customer's own documents with citations, live transcription with a
Silent Assistant that checks what is said against that knowledge base, meeting capture and reporting,
a full-duplex voice assistant, and document generation. Every model it uses — language, embedding,
speech-to-text, text-to-speech and image generation — is open-weight and served locally. The
reference deployment runs the entire platform on a single NVIDIA DGX Spark (Grace-Blackwell GB10).

AJACE Inc.'s commercial model for EchoMind is **customisation of a building block**, not sale of a
fixed product. AJACE Inc. does not sell hardware; it establishes the customer's problem statement and
advises on the hardware and cloud arrangement that suits the customer's security posture.

| | |
|---|---|
| Registered name | AJACE Inc. |
| Registered address | 14159 Robert Paris Ct., Suite A, Chantilly, VA 20151, USA |
| Principal place of business | 14159 Robert Paris Ct., Suite A, Chantilly, VA 20151, USA |
| Company registration number | `________________________` (to be added from the incorporation record) |
| Corporate QMS | AJACE Inc. ISO 9001:2015 QMS — certification audit 2025-10-16, scope *Product Development and Technology Consulting Services*; Management Representative: Sheryl Nazareth |
| EchoMind Product area — role holders | Five named roles (§6.1); engineering delivered principally by the Lead Engineer |

## 2. Relationship to the AJACE Inc. corporate QMS

This manual and the documents under `docs/qms/` form the **product-level** quality system of the
EchoMind Product area. They operate **within** AJACE Inc.'s certified corporate QMS and feed it; they
do not replace it.

| Process | Owned by | Where |
|---|---|---|
| Product requirements, design and development, verification, release, deployment, customer data handling, nonconformity and corrective action for EchoMind | EchoMind Product area | This QMS — SOP-04 to SOP-14 |
| Internal audit programme and auditors | Corporate (MR) | Corporate QMS; SOP-13 describes how EchoMind is audited within it |
| Management review | Senior management | Corporate QMS; SOP-15 and `EM26-02d` provide the EchoMind input |
| Recruitment, training records, competence evidence | Corporate HR / Training team | Corporate QMS; `REG-05` lists the EchoMind role holders |
| Approved supplier list, procurement | Corporate (Business Development & Procurement) | Corporate QMS; `REG-02` lists the EchoMind-specific providers |
| Customer satisfaction and past-performance feedback | Corporate | Corporate QMS; `REG-06` records EchoMind-specific feedback |

## 3. Context of the organisation (4.1)

| External issues | Why it matters to quality |
|---|---|
| Customers operate in regulated sectors (banking, legal, healthcare, government, defence) | Their own compliance obligations flow through to us as product requirements — data residency, auditability, accuracy of assertions |
| Data sovereignty and AI-privacy concern is the reason customers choose on-premises AI | Our value proposition rests on data never leaving the customer's perimeter; a breach of that is an existential quality failure, not a defect |
| Rapid movement in open-weight models and inference runtimes | Capability improves quickly, but supplier interfaces and behaviour change under us — see the NeMo dependency history in SOP-09 and NC-2026-006 |
| Specialised accelerator hardware (NVIDIA GB10 / Grace-Blackwell) is new and its software stack is immature | Real, evidenced defects have originated in the platform below us (CUDA graph conflicts, GPU memory sharing — NC-2026-016) |
| Rising expectation — and emerging regulation — that AI systems are transparent about what they know | Grounding, citation and honest abstention are becoming compliance features, not differentiators |
| Air-gapped and offline operation is a customer requirement | Everything must be pre-cached and reproducible offline; we cannot rely on a network at run time |

| Internal issues | Why it matters to quality |
|---|---|
| Engineering is concentrated in the Lead Engineer | Knowledge and delivery risk (REG-03 R-03); mitigated by written rationale, the GitHub project board, and independent review by the EchoMind Project Lead |
| Development is AI-assisted and says so in its commit record | Verification of generated work is a mandatory control, not an optional one (SOP-05) |
| Strong measurement culture — a 52-question golden evaluation, unit suites run by CI, a 13-case acceptance suite, and a research harness that records findings contradicting our own published claims | A genuine strength the QMS is built on |
| Release discipline established in 2026: versioned releases (v1.4.0, v1.4.1), changelog, runtime build identity, CI on every push | Traceability (8.5.2) and release control (8.6) are now operating; release records must be authorised before deployment |
| The EchoMind product was rebuilt in 2026; the 2025 product (separate repositories, Streamlit UI, RTX 4090 appliance) is its predecessor | 2025 audit evidence describes the predecessor; equivalent controls are re-established for the rebuilt product (`AUDIT_PACK_2026_EchoMind.md` §0) |

## 4. Interested parties and their requirements (4.2)

| Interested party | What they require from us |
|---|---|
| Customers (regulated enterprises — banking, law firms and others) | Answers grounded in their own material; data that never leaves their perimeter; tenant isolation; deployability offline; support and customisation; evidence they can show their own auditors |
| Customers' end users | Accurate, attributable answers; honest "I could not find that"; a system that does not invent |
| Customers' own regulators and auditors | Demonstrable control over data handling and over the accuracy of assertions the system makes |
| AJACE Inc. senior management and corporate QMS (MR) | Product-area evidence that feeds the corporate QMS, management review and certification |
| Certification body | Conformity with ISO 9001:2015, evidenced by records |
| Model providers (NVIDIA, Qwen, Nomic, Microsoft, Stability AI, Rhasspy, hexgrad) | Compliance with model licences and gated-model terms |
| Infrastructure and platform providers (Cloudflare, Hugging Face, GitHub, NVIDIA NGC, Docker Hub, Ubuntu mirror operators) | Acceptable use; our dependence on their availability is a supplier risk |
| Academic and conference reviewers (QASC 2026) | Claims supported by reproducible evidence; honest reporting of what was and was not measured |

Requirements are monitored through customer communication (SOP-04), supplier review (SOP-09) and
management review (SOP-15).

## 5. Scope of the quality management system (4.3)

> **Scope statement.** The design, development, customisation, deployment and support of the
> EchoMind Enterprise on-premises artificial-intelligence software platform, and the related advisory
> services concerning hardware and deployment architecture, provided by AJACE Inc. from
> 14159 Robert Paris Ct., Suite A, Chantilly, VA 20151, USA. This product-area scope falls within
> AJACE Inc.'s certified scope *Product Development and Technology Consulting Services*.

**Included:** requirements capture and review; design and development of the platform and of
per-customer customisations; verification and validation; configuration, build and release;
deployment (on-premises and air-gapped) and the reference public instance; handling of customer
data held by the platform; support, defect handling and corrective action.

**Not included:** manufacture or supply of hardware (AJACE Inc. advises on hardware but neither
manufactures nor resells it); the customer's own operation of their infrastructure, network and
physical security once deployed on their premises.

### 5.1 Applicability of requirements

Every requirement of ISO 9001:2015 clauses 4–10 applies, with one qualification:

| Clause | Determination |
|---|---|
| **7.1.5.2 Measurement traceability** | **Not applicable in its literal form.** No physical measuring equipment requiring calibration against international or national standards is used. The measuring instruments are software: the golden question set, the evaluation scripts and their thresholds. The equivalent control — version control and change control over those instruments, plus a pre-flight check that the evaluation corpus is present — is specified in SOP-12 §5. |
| 8.3 Design and development | **Fully applicable.** The organisation designs the product; this is its core process. |
| 8.5.1 f) Validation of processes whose output cannot be verified by subsequent monitoring | **Applicable.** The output of a generative model cannot be exhaustively verified; this is why the golden evaluation, the grounding gate and the abstention behaviour exist. See SOP-07. |

No requirement is excluded on the grounds that it is inconvenient.

## 6. The QMS and its processes (4.4)

The product-area system is organised as six processes. Each has an owning procedure.

```
                    ┌──────────────────────────────────────────┐
   Customer ───────▶│ P1  Understand and agree requirements    │  SOP-04
   requirement      └───────────────────┬──────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │ P2  Design and develop                   │  SOP-05, SOP-03
                    └───────────────────┬──────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │ P3  Verify and validate                  │  SOP-07
                    └───────────────────┬──────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │ P4  Control configuration and release    │  SOP-06, SOP-08
                    └───────────────────┬──────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │ P5  Deploy, operate and protect data     │  SOP-08, SOP-11
                    └───────────────────┬──────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │ P6  Measure, correct and improve         │  SOP-10, SOP-12,
                    └───────────────────┬──────────────────────┘  SOP-13, SOP-14, SOP-15
                                        │
     Supporting throughout: SOP-01 documented information · SOP-02 context and risk
                            SOP-09 external providers
                                        ▼
                             Improvement feeds back to P1–P5
```

| Process | Inputs | Outputs | Principal measures |
|---|---|---|---|
| P1 Requirements | Customer need, sector obligations | Agreed, reviewed requirements | Requirements reviewed before commitment |
| P2 Design & develop | Requirements, architecture, risk, NFRs (`EM26-17`) | Source, design documentation, rationale | Design reviews held; rationale recorded |
| P3 Verify & validate | Built software | Test and evaluation results | CI pass rate; acceptance suite; golden evaluation score |
| P4 Configure & release | Verified change | Built images, release record | Traceability source → image → instance |
| P5 Deploy & protect | Release, customer environment | Running instance; protected customer data | Health state; external login check; data-handling conformity |
| P6 Measure & improve | All of the above | Nonconformities, actions, reviews | NC closure with effectiveness verified (`REG-04`); review actions closed (`EM26-18`) |

## 7. Leadership (5)

Top management — the **Managing Director, Anita Johan** — is accountable for the effectiveness of the
QMS, establishes the quality policy (**QP-01**) and objectives (**QO-01**), ensures the customer focus
expressed in QP-01 §2, provides resources, and authorises releases. Customer focus (5.1.2) is
maintained by determining customer and regulatory requirements (SOP-04), addressing risks that can
affect conformity (REG-03), and monitoring accuracy of the product's answers (REG-09).

### 7.1 Roles, responsibilities and authorities (5.3)

| Role | Responsibility | Held by |
|---|---|---|
| Managing Director | QMS effectiveness; policy and objectives; release authorisation; management review; approval of QMS documents | **Anita Johan** |
| EchoMind Project Lead | Independent review of design, deliverables and QMS documents; programme direction | **Alexander Peter** |
| Lead Engineer | Design, development, verification, configuration and release execution; preparation of QMS documents | **Richard Watson Stephen Amudha** |
| Quality representative (QA / MR) | Maintaining the QMS, the registers, the audit programme and corrective actions; link to the corporate QMS | **Sheryl Nazareth** |
| Data custodian | Customer property and data handling per SOP-11; data-protection NFRs | **Kishan Haravu Pradeep** |

Where one person holds several roles, the roles remain distinct in the records: an entry is made
against the role that is acting. Independence where required — review of the Lead Engineer's work,
internal audit (9.2.2 c) — is provided by the Project Lead and the corporate audit programme (SOP-13 §4).

### 7.2 Organisation chart

```
                     Managing Director — Anita Johan
                                  │
             ┌────────────────────┼─────────────────────────────┐
             │                    │                             │
  QA / MR — Sheryl Nazareth   EchoMind Project Lead —      Data custodian —
  (corporate QMS link)        Alexander Peter              Kishan Haravu Pradeep
                                  │
                       Lead Engineer — Richard Watson Stephen Amudha
```

## 8. Planning (6)

Risks and opportunities are identified and treated under **SOP-02** and tracked in **REG-03 Risk
Register**. Quality objectives, their measures and targets are in **QO-01**; non-functional targets in
**EM26-17 / REG-09**. Changes to the QMS and to the product are planned under **SOP-02 §7** and
**SOP-06**, and recorded in **REG-07**.

## 9. Support (7)

| Clause | How it is met |
|---|---|
| 7.1.2 People | Role holders in §7.1; competence evidence held by the corporate training team and summarised in `REG-05` |
| 7.1.3 Infrastructure | NVIDIA DGX Spark reference host; Docker Compose stack (six services); GitHub repository, GitHub Actions CI and the project board; backups (`scripts/backup_data.sh`) |
| 7.1.4 Environment for operation | Controlled host access; production configuration recorded and drift-checked (`SOP-06` §10.1) |
| 7.1.5 Monitoring and measuring resources | Version-controlled evaluation instruments (`eval/`), CI, acceptance suite, external login check — `EM26-17` §5 |
| 7.1.6 Organisational knowledge | `docs/` architecture documentation, commit rationale, the QASC 2026 paper, lessons learnt (`EM26-16`), the QMS itself |
| 7.2–7.3 Competence and awareness | `SOP-03`; corporate onboarding and training |
| 7.4 Communication | Internal: GitHub project board and issues, email/Teams, review records (`EM26-18`). External: customer communication under `SOP-04`; audit communication through the MR |
| 7.5 Documented information | `SOP-01`; the Git repository is the control mechanism; `REG-01` is the document register |

## 10. Operation (8)

| Clause | Procedure |
|---|---|
| 8.1 Operational planning and control | QM-01 §6, SOP-06 |
| 8.2 Requirements for products and services | SOP-04 |
| 8.3 Design and development | SOP-05 |
| 8.4 Externally provided processes, products and services | SOP-09 |
| 8.5.1 Control of provision | SOP-08 |
| 8.5.2 Identification and traceability | SOP-06 |
| 8.5.3 Property belonging to customers | SOP-11 |
| 8.5.4 Preservation | SOP-08, SOP-11 |
| 8.5.5 Post-delivery activities | SOP-08, SOP-10 |
| 8.5.6 Control of changes | SOP-06 |
| 8.6 Release of products and services | SOP-07, SOP-08 |
| 8.7 Control of nonconforming outputs | SOP-10 |

## 11. Performance evaluation (9)

Monitoring, measurement, analysis and evaluation: **SOP-12**, **EM26-17**. Customer satisfaction:
**SOP-04** and the corporate process. Internal audit: **SOP-13** within the corporate programme.
Management review: **SOP-15** within corporate management review.

## 12. Improvement (10)

Nonconformity and corrective action: **SOP-14**, logged in **REG-04** and tracked on the project
board. Review feedback and actions: **EM26-18**. Continual improvement is an output of management
review (SOP-15), of the analysis in SOP-12 and of the lessons learnt (EM26-16).

## 13. Structure of the documented information

```
docs/qms/
├── README.md                          index and how to use the system
├── QM-01_Quality_Manual.md            this document
├── QP-01_Quality_Policy.md            5.2
├── QO-01_Quality_Objectives.md        6.2
├── ISO9001_Clause_Mapping.md          clause → document → evidence in this repository
├── ISO9001_Gap_Analysis.md            readiness assessment and roadmap
├── AUDIT_PACK_2026_EchoMind.md        surveillance-audit evidence for the EchoMind Product area
├── ADOPTION_GUIDE.md · COMPLETION_CHECKLIST.md · INTAKE_FORM.md
├── ISO9001_EchoMind_2026.zip          the whole set as one download
├── procedures/   SOP-01 … SOP-15
├── forms/        FRM-00 … FRM-12      blank templates
├── registers/    REG-01 … REG-10      live records
├── records/      approvals · releases · capa · audits · management-review · runtime
└── evidence-2026/                     2026 evidence pack (Word records, test runs, screenshots)
```

## 14. Implementation status

### 14.1 What is operating

- **Documents approved** on 2026-10-07 (approval record AR-2026-001).
- **Releases identified and traceable** — v1.4.0 and v1.4.1; every running service reports its version and commit.
- **Verification enforced** — CI runs the unit suites (backend 50, voice 131 cases) and the type-check on every push; a 13-case acceptance suite was executed four times on 2026-10-06; all executable cases pass on v1.4.1.
- **Nonconformities handled end to end** — 16 recorded in REG-04, 11 closed with effectiveness verified, 5 open with owners and actions.
- **Review feedback actioned** — 8 reviews, 40 actions, 34 closed with evidence (EM26-18).
- **Non-functional requirements recorded and monitored** — 31 NFRs with targets (EM26-17).
- **External monitoring** — the public instance's login is checked from outside every six hours; runtime security settings are drift-checked.
- **Engineering evidence** — architecture documentation in `docs/`, a 1,257-line user manual, the QASC 2026 paper and its reproducible harness.

### 14.2 What is not yet operating

- **Internal audit of the EchoMind area** — to be scheduled in the corporate audit programme (SOP-13).
- **Management review minutes covering EchoMind** — input prepared (EM26-02d); to be recorded at the next corporate review.
- **Second measurement of the quality objectives**, so that a trend exists (QO-01).
- **Customer-related records** (requirement reviews, satisfaction, post-delivery) — no customer engagement of the rebuilt product yet.
- **Golden evaluation** — blocked until the evaluation corpus is restored (NC-2026-008).

### 14.3 Constraints an auditor will raise

| Constraint | Clause affected | Where it is addressed |
|---|---|---|
| Engineering concentrated in one engineer | 7.1.6, 9.2.2 c | Independent review by the Project Lead; corporate internal audit; REG-03 R-03 |
| Public instance ran without login 2026-07-30 → 2026-10-06 | 8.5.3 | NC-2026-009 — corrected; external check and drift check in place |
| No off-site scheduled backup yet | 8.5.3, 8.5.4 | NFR-R04 action A-17-07 |
| No data-retention policy; no encryption at rest | 8.5.3 | NFR-D02/D03 action A-17-08 |
| No LICENSE/NOTICE, model-licence verification or SBOM | 8.4.2 | SOP-09; board tickets #55, #56 |
| GPU shared with another stack; voice and embeddings on CPU | 7.1.3 | NC-2026-016; decision on GPU sharing (#60) |

## 15. Revision history and distribution

| Date | Revision | Change | Prepared by | Reviewed by |
|---|---|---|---|---|
| 2026-09-21 | 1.0 | First draft | Lead Engineer | — |
| 2026-10-07 | 2.0 | AJACE Inc. details and scope; relationship to the corporate QMS; named roles and organisation chart; support section; structure and implementation status updated | Richard Watson Stephen Amudha | Alexander Peter |

**Distribution:** Managing Director, EchoMind Project Lead, Lead Engineer, QA/MR, Data custodian.
The controlled copy is this file in Git; printed or Word copies are uncontrolled.
