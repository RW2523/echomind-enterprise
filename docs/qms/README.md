# Quality Management System — EchoMind Enterprise

**Ajace AI · ISO 9001:2015 · Revision 1.0 · authored 2026-09-21**

> ### Status: documented, not yet adopted
>
> Every document in this set is **`DRAFT — not yet approved`**. No document has been signed, no
> internal audit has been conducted, and no management review has been held. This is an accurate
> statement of position, not a disclaimer.
>
> **Do not present this set as an implemented or certified quality management system.** Read
> [`ADOPTION_GUIDE.md`](ADOPTION_GUIDE.md) first — it explains exactly what is needed to make it
> real, and it is short.

---

## What this is

A complete ISO 9001:2015 quality management system written specifically for the design,
development, deployment and support of **EchoMind Enterprise** — Ajace AI's private, on-premises AI
workspace.

It is not a template with the company name substituted in. Every procedure describes what this
project actually does, and cites the file and line that proves it. Where a control does not exist,
the procedure says so in its *Current state and gaps* section rather than describing an aspiration
in the present tense. The registers are seeded with real, dated evidence drawn from the repository's
own history — including seven nonconformities transcribed from Git commits and fourteen risks
derived from code and configuration.

## Start here

| If you are… | Read, in this order |
|---|---|
| **The owner, adopting this** | [`ADOPTION_GUIDE.md`](ADOPTION_GUIDE.md) → [`QP-01`](QP-01_Quality_Policy.md) → [`QM-01`](QM-01_Quality_Manual.md) → [`QO-01`](QO-01_Quality_Objectives.md) → [`REG-03`](registers/REG-03_Risk_Register.md) |
| **An auditor** | [`ISO9001_Clause_Mapping.md`](ISO9001_Clause_Mapping.md) → [`ISO9001_Gap_Analysis.md`](ISO9001_Gap_Analysis.md) → [`QM-01 §13`](QM-01_Quality_Manual.md) (implementation status) |
| **A customer's procurement or assurance team** | [`QP-01`](QP-01_Quality_Policy.md) → [`SOP-11`](procedures/SOP-11_Customer_Property_and_Data_Handling.md) (your data) → [`SOP-07`](procedures/SOP-07_Verification_Validation_and_Testing.md) (how we verify) → [`ISO9001_Gap_Analysis.md`](ISO9001_Gap_Analysis.md) |
| **An engineer working on the product** | [`SOP-05`](procedures/SOP-05_Design_and_Development_Control.md), [`SOP-06`](procedures/SOP-06_Configuration_and_Change_Management.md), [`SOP-07`](procedures/SOP-07_Verification_Validation_and_Testing.md), [`SOP-08`](procedures/SOP-08_Release_and_Deployment.md) |

## The document set

### Core

| ID | Document | Covers |
|---|---|---|
| — | [Adoption Guide](ADOPTION_GUIDE.md) | What must happen for this to become a real management system |
| — | [**Audit Pack 2026**](AUDIT_PACK_2026_EchoMind.md) | **Surveillance audit 14 Oct 2026** — evidence for the EchoMind Product area, mapped to the 2025 findings |
| — | [**Intake Form**](INTAKE_FORM.md) | **Start here** — the 43 details only you can supply, as a form to fill in |
| — | [Completion Checklist](COMPLETION_CHECKLIST.md) | Every outstanding item — your information, your decisions, licences to verify, engineering work, and the records that must accumulate |
| QM-01 | [Quality Manual](QM-01_Quality_Manual.md) | Context, interested parties, scope, process map, implementation status |
| QP-01 | [Quality Policy](QP-01_Quality_Policy.md) | 5.2 — the organisation's quality commitments |
| QO-01 | [Quality Objectives](QO-01_Quality_Objectives.md) | 6.2 — seven measurable objectives with baselines and targets |
| MAP-01 | [Clause Mapping](ISO9001_Clause_Mapping.md) | Every ISO 9001 clause → document → objective evidence in this repository |
| GAP-01 | [Gap Analysis](ISO9001_Gap_Analysis.md) | Honest readiness assessment and a prioritised roadmap |

### Procedures

| ID | Procedure | Clauses |
|---|---|---|
| SOP-01 | [Control of Documented Information](procedures/SOP-01_Control_of_Documented_Information.md) | 7.5 |
| SOP-02 | [Context, Interested Parties and Risk](procedures/SOP-02_Context_Interested_Parties_and_Risk.md) | 4.1, 4.2, 6.1, 6.3 |
| SOP-03 | [Competence, Training and Awareness](procedures/SOP-03_Competence_Training_and_Awareness.md) | 7.1.2, 7.1.6, 7.2, 7.3 |
| SOP-04 | [Customer Requirements and Communication](procedures/SOP-04_Customer_Requirements_and_Communication.md) | 8.2, 9.1.2 |
| SOP-05 | [Design and Development Control](procedures/SOP-05_Design_and_Development_Control.md) | 8.3 |
| SOP-06 | [Configuration and Change Management](procedures/SOP-06_Configuration_and_Change_Management.md) | 8.1, 8.5.2, 8.5.6 |
| SOP-07 | [Verification, Validation and Testing](procedures/SOP-07_Verification_Validation_and_Testing.md) | 8.3.4, 8.6, 9.1.1 |
| SOP-08 | [Release and Deployment](procedures/SOP-08_Release_and_Deployment.md) | 8.5.1, 8.5.4, 8.6 |
| SOP-09 | [External Providers and Open Source](procedures/SOP-09_External_Providers_and_Open_Source.md) | 8.4 |
| SOP-10 | [Nonconforming Output and Incident Management](procedures/SOP-10_Nonconforming_Output_and_Incident_Management.md) | 8.7 |
| SOP-11 | [Customer Property and Data Handling](procedures/SOP-11_Customer_Property_and_Data_Handling.md) | 8.5.3, 8.5.4 |
| SOP-12 | [Monitoring, Measurement and Analysis](procedures/SOP-12_Monitoring_Measurement_and_Analysis.md) | 9.1.1, 9.1.3 |
| SOP-13 | [Internal Audit](procedures/SOP-13_Internal_Audit.md) | 9.2 |
| SOP-14 | [Nonconformity and Corrective Action](procedures/SOP-14_Nonconformity_and_Corrective_Action.md) | 10.1, 10.2, 10.3 |
| SOP-15 | [Management Review](procedures/SOP-15_Management_Review.md) | 9.3 |

### Forms — blank templates

`FRM-00` document template · `FRM-01` design review · `FRM-02` change request · `FRM-03` release
record · `FRM-04` supplier evaluation · `FRM-05` nonconformity and CAPA · `FRM-06` internal audit
report · `FRM-07` management review minutes · `FRM-08` customer feedback and complaint ·
`FRM-09` competence and training · `FRM-10` risk assessment · `FRM-11` requirements review ·
`FRM-12` customer satisfaction review.
→ [`forms/`](forms/)

### Registers — live records

| ID | Register | State at revision 1.0 |
|---|---|---|
| REG-01 | [Document Control](registers/REG-01_Document_Control_Register.md) | Complete index; all documents unapproved |
| REG-02 | [External Documents and Providers](registers/REG-02_External_Documents_and_Providers.md) | Seeded — 11 models, 6 base images, 7 services. **Licence column incomplete.** |
| REG-03 | [Risk and Opportunity Register](registers/REG-03_Risk_Register.md) | Seeded — 14 evidenced risks, 4 opportunities. **Ratings proposed, not confirmed.** |
| REG-04 | [Nonconformity and CAPA Log](registers/REG-04_Nonconformity_and_CAPA_Log.md) | Seeded — 7 retrospective entries transcribed from Git history |
| REG-05 | [Competence and Training](registers/REG-05_Competence_and_Training_Record.md) | Empty |
| REG-06 | [Customer Feedback and Complaints](registers/REG-06_Customer_Feedback_and_Complaints_Log.md) | Empty |
| REG-07 | [Design and Change Register](registers/REG-07_Design_and_Change_Register.md) | Empty |
| REG-08 | [Release and Deployment Register](registers/REG-08_Release_and_Deployment_Register.md) | Mechanism now exists (`scripts/release.sh`); first release outstanding |
| REG-09 | [NFR Register](registers/REG-09_NFR_Register.md) | **Seeded — 20 NFRs with real measurements.** Closes 2025 finding #6 |
| REG-10 | [Review Feedback and Action Log](registers/REG-10_Review_Feedback_Action_Log.md) | **Seeded — 3 retrospective reviews.** Closes 2025 finding #7 |

### Records

Completed records are filed under [`records/`](records/) — `audits/`, `capa/`,
`management-review/` and `releases/`. All four are empty at revision 1.0.

## How documented information is controlled

There is no separate document-management system. **Git is the control mechanism** — every document
is versioned, attributed and timestamped by the commit that changed it, and replicated to every
clone. `git log --follow -- docs/qms/<file>` is the revision history of any document in this set.
The reasoning, and how this satisfies clause 7.5.3, is in
[`SOP-01 §3`](procedures/SOP-01_Control_of_Documented_Information.md).

## The three things an auditor will find first

Stated here deliberately, because a system that has already found its own gaps is in a better
position than one that has not:

1. **Nothing is approved, audited or reviewed** — the QMS is documented but not operating
   (clauses 5.2, 9.2, 9.3).
2. **No release traceability** — no tags, no version, no build identifier; a running instance cannot
   be traced to its source revision (clause 8.5.2).
3. **No backup of the only volume holding customer data** (`echomind_data`) — the highest-consequence
   gap in the system, and the first thing to fix (clauses 8.5.3, 8.5.4).

All three, and eighteen more, are in [`ISO9001_Gap_Analysis.md`](ISO9001_Gap_Analysis.md) with
priority, consequence and effort.

## What is genuinely strong

For balance, and because these are the foundations the system is built on:

- **Documented-information control is properly solved** by Git — versioned, attributed, replicated.
- **A real, version-controlled measuring instrument**: 52 golden questions across seven domains with
  a binary pass gate (`eval/run_eval.py:281`), plus 78 unit test functions.
- **Measurement honesty**: `eval/paper/results/SUMMARY.json` records findings that contradict the
  organisation's own published paper, and marks an unrun experiment `"not_run"` rather than
  estimating it. That culture is the hardest part of a QMS to create and it already exists here.
- **Root-cause discipline** in commit records — several fixes document defect, cause, correction and
  verification.
- **Quality by design in the product**: relevance gating, citation filtering, honest abstention and
  injection guards asserted by tests.
