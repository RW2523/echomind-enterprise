# REG-01 — Document Control Register

| Field | Value |
|---|---|
| Document ID | REG-01 |
| Revision | 1.0 |
| Status | **LIVE REGISTER** |
| Owner | Managing Director |
| ISO 9001:2015 clauses | 7.5.1, 7.5.2, 7.5.3 |
| Governing procedure | SOP-01 |

---

## 1. Purpose

The index of controlled QMS documents, their current revision, owner and approval state. Revision
history itself is not duplicated here — it is the Git history of each file (SOP-01 §3).

## 2. Controlled documents

> **Every document is at Revision 1.0 with status `DRAFT — not yet approved`.** Approval is the
> single action that turns this document set into a management system; see `ADOPTION_GUIDE.md`.
> Complete the *Approved by* and *Approval date* columns only when the named person has actually
> approved the document.

| ID | Title | Rev | Status | Owner (role) | Approved by | Approval date |
|---|---|---|---|---|---|---|
| QM-01 | Quality Manual | 1.0 | DRAFT | Managing Director | `________` | `________` |
| QP-01 | Quality Policy | 1.0 | DRAFT | Managing Director | `________` | `________` |
| QO-01 | Quality Objectives | 1.0 | DRAFT | Managing Director | `________` | `________` |
| SOP-01 | Control of Documented Information | 1.0 | DRAFT | Managing Director | `________` | `________` |
| SOP-02 | Context, Interested Parties and Risk | 1.0 | DRAFT | Managing Director | `________` | `________` |
| SOP-03 | Competence, Training and Awareness | 1.0 | DRAFT | Managing Director | `________` | `________` |
| SOP-04 | Customer Requirements and Communication | 1.0 | DRAFT | Managing Director | `________` | `________` |
| SOP-05 | Design and Development Control | 1.0 | DRAFT | Lead Engineer | `________` | `________` |
| SOP-06 | Configuration and Change Management | 1.0 | DRAFT | Lead Engineer | `________` | `________` |
| SOP-07 | Verification, Validation and Testing | 1.0 | DRAFT | Lead Engineer | `________` | `________` |
| SOP-08 | Release and Deployment | 1.0 | DRAFT | Lead Engineer | `________` | `________` |
| SOP-09 | External Providers and Open Source | 1.0 | DRAFT | Lead Engineer | `________` | `________` |
| SOP-10 | Nonconforming Output and Incident Management | 1.0 | DRAFT | Quality representative | `________` | `________` |
| SOP-11 | Customer Property and Data Handling | 1.0 | DRAFT | Data custodian | `________` | `________` |
| SOP-12 | Monitoring, Measurement and Analysis | 1.0 | DRAFT | Quality representative | `________` | `________` |
| SOP-13 | Internal Audit | 1.0 | DRAFT | Quality representative | `________` | `________` |
| SOP-14 | Nonconformity and Corrective Action | 1.0 | DRAFT | Quality representative | `________` | `________` |
| SOP-15 | Management Review | 1.0 | DRAFT | Managing Director | `________` | `________` |

## 3. Forms (blank templates)

| ID | Title | Rev |
|---|---|---|
| FRM-00 | QMS Document Template | 1.0 |
| FRM-01 | Design Review Record | 1.0 |
| FRM-02 | Change Request | 1.0 |
| FRM-03 | Release Record | 1.0 |
| FRM-04 | Supplier Evaluation | 1.0 |
| FRM-05 | Nonconformity and CAPA Record | 1.0 |
| FRM-06 | Internal Audit Report | 1.0 |
| FRM-07 | Management Review Minutes | 1.0 |
| FRM-08 | Customer Feedback and Complaint Record | 1.0 |
| FRM-09 | Competence and Training Record | 1.0 |
| FRM-10 | Risk Assessment Record | 1.0 |
| FRM-11 | Requirements Review Record | 1.0 |
| FRM-12 | Customer Satisfaction Review | 1.0 |

## 4. Registers (live records)

| ID | Title | State at revision 1.0 |
|---|---|---|
| REG-01 | Document Control Register | This document |
| REG-02 | External Documents and Providers | Seeded from the actual dependency inventory; **licence column incomplete** |
| REG-03 | Risk and Opportunity Register | Seeded with 14 evidenced risks and 4 opportunities; **ratings proposed, not confirmed** |
| REG-04 | Nonconformity and CAPA Log | Seeded with 7 retrospective entries transcribed from Git history |
| REG-05 | Competence and Training Record | Empty |
| REG-06 | Customer Feedback and Complaints Log | Empty |
| REG-07 | Design and Change Register | Empty |
| REG-08 | Release and Deployment Register | Empty — **cannot be completed until release identification exists (G-01)** |

## 4a. Completed records

Completed records are filed as dated Markdown files under `docs/qms/records/`:

| Directory | Holds | Form used |
|---|---|---|
| `records/audits/` | Internal audit reports | FRM-06 |
| `records/capa/` | Nonconformity and corrective action records | FRM-05 |
| `records/management-review/` | Management review minutes | FRM-07 |
| `records/releases/` | Release records | FRM-03 |

All four are empty at revision 1.0.

## 5. Supporting engineering documentation relied on as QMS evidence

These are not QMS documents, but the QMS cites them as evidence. They are controlled by the same
repository.

| Document | Evidence for |
|---|---|
| `docs/CAPABILITIES.md`, `docs/RAG_FLOW.md`, `docs/CHAT_AND_RAG_FLOW.md`, `docs/CONVERSATION_AI_AND_WAKE_WORD_FLOW.md`, `docs/TRANSCRIPT_STORAGE_FLOW.md` | Design outputs (8.3.5) |
| `backend/app/transcribe/PROTOCOL.md` | Interface specification |
| `docs/USER_MANUAL.md` | Post-delivery information (8.5.5) |
| `OFFLINE_DEPLOYMENT.md`, `docs/PUBLIC_DEPLOYMENT.md` | Deployment control (8.5.1) |
| `eval/golden/*.jsonl`, `eval/run_eval.py` | The measuring instrument (9.1.1; QM-01 §4.1) |
| `eval/paper/results/SUMMARY.json`, `REGRESSION_AND_FIXES.json` | Measurement results and a nonconformity record |
| Git commit history | Design rationale, verification evidence, change history (8.3.2, 8.3.4, 8.5.6) |
