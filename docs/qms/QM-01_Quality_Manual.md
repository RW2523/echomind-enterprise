# QM-01 — Quality Manual

| Field | Value |
|---|---|
| Document ID | QM-01 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Managing Director |
| Approved by | `________________________` |
| Approval date | `____ / ____ / ________` |
| ISO 9001:2015 clauses | 4.1, 4.2, 4.3, 4.4, 5, 6, 7, 8, 9, 10 (overview) |

> **Read this first.** ISO 9001:2015 does not require a quality manual. This one exists because a
> small organisation benefits from a single document that says what the management system is, what
> it covers, and — candidly — how much of it is operating today. Sections 1 to 12 describe the
> system. **Section 13 states its actual implementation status, which is: authored, not yet
> operating.** Do not present this manual to a customer or a certification body without section 13.

---

## 1. The organisation

**Ajace AI** designs, builds, customises, deploys and supports **EchoMind Enterprise** — a private,
on-premises artificial-intelligence workspace. The platform provides knowledge chat over the
customer's own documents with citations, live transcription with real-time fact-checking against
that knowledge base, meeting capture and reporting, a full-duplex voice assistant, and document
generation. Every model it uses — language, embedding, speech-to-text, text-to-speech and image
generation — is open-weight and served locally. The reference deployment runs the entire platform
on a single NVIDIA DGX Spark (GB10).

Ajace AI's commercial model is **customisation of a building block**, not sale of a fixed product.
Ajace AI does not sell hardware; it establishes the customer's problem statement and advises on the
hardware and cloud arrangement that suits the customer's security posture.

| | |
|---|---|
| Registered name | `________________________` |
| Registered address | `________________________` |
| Principal place of business | `________________________` |
| Company registration number | `________________________` |
| Personnel at revision 1.0 | 1 (see §13.3 — this is material to several clauses) |

## 2. Context of the organisation (4.1)

| External issues | Why it matters to quality |
|---|---|
| Customers operate in regulated sectors (defence, government, legal, finance, healthcare) | Their own compliance obligations flow through to us as product requirements — data residency, auditability, accuracy of assertions |
| Data sovereignty and AI-privacy concern is the reason customers choose on-premises AI | Our entire value proposition rests on data never leaving the customer's perimeter; a breach of that is an existential quality failure, not a defect |
| Rapid movement in open-weight models and inference runtimes | Capability improves quickly, but supplier interfaces and behaviour change under us — see the NeMo `main` dependency in SOP-09 |
| Specialised accelerator hardware (NVIDIA GB10 / Grace-Blackwell) is new and its software stack is immature | Real, evidenced defects have originated in the platform below us (CUDA graph conflicts, NVML behaviour in containers) |
| Rising expectation — and emerging regulation — that AI systems are transparent about what they know | Grounding, citation and honest abstention are becoming compliance features, not differentiators |
| Air-gapped and offline operation is a customer requirement | Everything must be pre-cached and reproducible offline; we cannot rely on a network at run time |

| Internal issues | Why it matters to quality |
|---|---|
| Very small organisation, currently one person | Single point of failure for knowledge and delivery; impartial internal audit is not currently possible (SOP-13) |
| Development is AI-assisted and says so in its commit record | Verification of generated work is a mandatory control, not an optional one (SOP-05) |
| Strong measurement culture already in place — a 52-question golden evaluation, unit suites, and a paper harness that records findings contradicting our own published claims | A genuine strength to build the QMS on rather than replace |
| Weak release discipline — no versioning, no CI, no change register | The largest cluster of nonconformities against ISO 9001 (§13, Gap Analysis) |
| Deep domain knowledge held largely in one person's head, commit bodies and `docs/` | Organisational knowledge (7.1.6) is a live risk |

## 3. Interested parties and their requirements (4.2)

| Interested party | What they require from us |
|---|---|
| Customers (regulated enterprises) | Answers grounded in their own material; data that never leaves their perimeter; tenant isolation; deployability offline; support and customisation; evidence they can show their own auditors |
| Customers' end users | Accurate, attributable answers; honest "I could not find that"; a system that does not invent |
| Customers' own regulators and auditors | Demonstrable control over data handling and over the accuracy of assertions the system makes |
| Ajace AI owner / personnel | A system that makes quality repeatable rather than heroic; knowledge that survives a person being unavailable |
| Model providers (NVIDIA, Qwen, Nomic, Microsoft, Stability AI, Rhasspy, hexgrad) | Compliance with model licences and gated-model terms |
| Infrastructure and platform providers (Cloudflare, Hugging Face, GitHub, NVIDIA NGC, Docker Hub, Ubuntu mirror operators) | Acceptable use; our dependence on their availability is a supplier risk |
| Academic and conference reviewers | Claims supported by reproducible evidence; honest reporting of what was and was not measured |

Requirements are monitored through customer communication (SOP-04), supplier review (SOP-09) and
management review (SOP-15).

## 4. Scope of the quality management system (4.3)

> **Scope statement.** The design, development, customisation, deployment, and support of the
> EchoMind Enterprise on-premises artificial-intelligence software platform, and the related
> advisory services concerning hardware and deployment architecture, provided by Ajace AI from
> `________________________`.

**Included:** requirements capture and review; design and development of the platform and of
per-customer customisations; verification and validation; configuration, build and release;
deployment (on-premises and air-gapped) and the reference public instance; handling of customer
data held by the platform; support, defect handling and corrective action.

**Not included:** manufacture or supply of hardware (Ajace AI advises on hardware but neither
manufactures nor resells it); the customer's own operation of their infrastructure, network and
physical security once deployed on their premises.

### 4.1 Applicability of requirements

Every requirement of ISO 9001:2015 clauses 4–10 applies, with one qualification:

| Clause | Determination |
|---|---|
| **7.1.5.2 Measurement traceability** | **Not applicable in its literal form.** No physical measuring equipment requiring calibration against international or national standards is used. The organisation's measuring instruments are software: the golden question set, the evaluation scripts and their thresholds. The equivalent control — version control and change control over those instruments so that a measurement result means the same thing over time — is specified in SOP-12 §5. |
| 8.3 Design and development | **Fully applicable.** The organisation designs the product; this is its core process. |
| 8.5.1 f) Validation of processes whose output cannot be verified by subsequent monitoring | **Applicable.** The output of a generative model cannot be exhaustively verified; this is precisely why the golden evaluation, the grounding gate and the abstention behaviour exist. See SOP-07. |

No requirement is excluded on the grounds that it is inconvenient.

## 5. The QMS and its processes (4.4)

The management system is organised as six processes. Each has an owning procedure.

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
| P2 Design & develop | Requirements, architecture, risk | Source, design documentation, rationale | Design reviews held; rationale recorded |
| P3 Verify & validate | Built software | Test and evaluation results | Unit pass rate; golden evaluation score |
| P4 Configure & release | Verified change | Built images, release record | Traceability source → image → instance |
| P5 Deploy & protect | Release, customer environment | Running instance; protected customer data | Health state; data-handling conformity |
| P6 Measure & improve | All of the above | Nonconformities, actions, reviews | NC closure with effectiveness verified |

## 6. Leadership (5)

Top management (the Managing Director) is accountable for the effectiveness of the QMS, establishes
the quality policy (**QP-01**) and objectives (**QO-01**), ensures the customer focus expressed in
QP-01 §2, and provides resources. In an organisation of one, the practical meaning of clause 5 is
that the same person who writes the code also owns the decision to stop and fix it — and that the
management system exists to make that decision evidence-based rather than discretionary.

### 6.1 Roles, responsibilities and authorities (5.3)

| Role | Responsibility | Currently held by |
|---|---|---|
| Managing Director | QMS effectiveness; policy and objectives; release authorisation; management review; approval of QMS documents | `________________` |
| Lead Engineer | Design, development, verification, configuration and release execution | `________________` |
| Quality representative | Maintaining the QMS, the registers, the audit programme and corrective actions | `________________` |
| Data custodian | Customer property and data handling per SOP-11 | `________________` |

Where one person holds several of these roles, the roles remain distinct in the records: an entry is
made against the role that is acting. Where a role's independence is required and cannot be provided
internally — notably internal audit (9.2.2 c) — SOP-13 §4 sets out how it is obtained.

## 7. Planning (6)

Risks and opportunities are identified and treated under **SOP-02** and tracked in
**REG-03 Risk Register**. Quality objectives, their measures and targets are in **QO-01**. Changes
to the QMS and to the product are planned under **SOP-02 §7** and **SOP-06**.

## 8. Support (7)

Resources, competence, awareness and organisational knowledge are addressed in **SOP-03**.
Documented information is controlled under **SOP-01**, using the product's Git repository as the
control mechanism.

## 9. Operation (8)

| Clause | Procedure |
|---|---|
| 8.1 Operational planning and control | QM-01 §5, SOP-06 |
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

## 10. Performance evaluation (9)

Monitoring, measurement, analysis and evaluation: **SOP-12**. Customer satisfaction: **SOP-04**.
Internal audit: **SOP-13**. Management review: **SOP-15**.

## 11. Improvement (10)

Nonconformity and corrective action: **SOP-14**, logged in **REG-04**. Continual improvement is an
output of management review (SOP-15) and of the analysis in SOP-12.

## 12. Structure of the documented information

```
docs/qms/
├── README.md                       index and how to use the system
├── ADOPTION_GUIDE.md               what must happen for this QMS to become real
├── QM-01_Quality_Manual.md         this document
├── QP-01_Quality_Policy.md         5.2
├── QO-01_Quality_Objectives.md     6.2
├── ISO9001_Clause_Mapping.md       clause → document → evidence in this repository
├── ISO9001_Gap_Analysis.md         honest readiness assessment and roadmap
├── procedures/   SOP-01 … SOP-15
├── forms/        FRM-00 … FRM-10   blank templates
└── registers/    REG-01 … REG-06   live records
```

## 13. Implementation status — read this before relying on anything above

### 13.1 What is real today

The engineering evidence base is genuinely strong for an organisation this size, and the QMS is
built on it rather than beside it:

- A **52-question golden evaluation suite** with a binary pass gate (`eval/run_eval.py:281`) and
  version-controlled question sets in `eval/golden/`.
- **78 unit test functions** across `backend/tests/` (40) and `voice/tests/` (38).
- A **second, independent evaluation harness** (`eval/paper/`, experiments E1–E10) whose results
  file records findings that *contradict the organisation's own published paper*, and marks an
  experiment `"not_run"` rather than estimating it.
- **Substantial design rationale and verification evidence in commit bodies** — several commits
  record the defect, the root cause, the fix and the verification performed.
- **Architecture documentation** in `docs/` covering retrieval, chat, voice, transcription and
  deployment, plus a 1,257-line user manual.
- **Self-recovering service design** — health endpoints, a CUDA-fault watchdog and restart policies.

### 13.2 What is not real yet

This QMS was authored on `2026-09-21`. At revision 1.0:

- **No document in this set has been approved.** All are `DRAFT — not yet approved`.
- **No internal audit has been conducted** (SOP-13).
- **No management review has been held** (SOP-15).
- **No quality objective has a baseline measurement recorded against it** beyond the historical
  evaluation figures cited in QO-01, which predate HEAD.
- The registers are seeded from documented historical evidence where that evidence genuinely
  exists, and are otherwise empty. Seeded entries are marked as retrospective.

### 13.3 Constraints an auditor will raise immediately

| Constraint | Clause affected | Where it is addressed |
|---|---|---|
| One person in the organisation; internal audit cannot be impartial | 9.2.2 c | SOP-13 §4 — treated as an open nonconformity of the QMS, with options |
| No CI, so no verification is automatically enforced | 8.6, 9.1.1 | SOP-07, Gap Analysis G-02 |
| No versioning: no tags, no changelog, no build identifier; a running instance cannot be traced to a source revision | 8.5.2, 8.6 | SOP-06, SOP-08, Gap Analysis G-01 |
| No backup procedure for `echomind_data`, the only volume holding customer data | 8.5.3, 8.5.4 | SOP-11, Gap Analysis G-03 |
| No data-retention policy | 8.5.3 | SOP-11, Gap Analysis G-04 |
| No LICENSE/NOTICE, model-licence inventory or SBOM | 8.4.2 | SOP-09, Gap Analysis G-05 |
| Application authentication off by default; WebSocket endpoints outside the auth middleware | 8.5.3 | SOP-11, Gap Analysis G-06 |

These are stated here, in the manual, deliberately. An auditor will find them within an hour; a
management system that has already found them, owns them and has dated actions against them is in a
materially better position than one that has not.
