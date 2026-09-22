# SOP-01 — Control of Documented Information

| Field | Value |
|---|---|
| Document ID | SOP-01 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Managing Director |
| Approved by | `________________________` |
| Approval date | `____ / ____ / ________` |
| ISO 9001:2015 clauses | 7.5.1, 7.5.2, 7.5.3 |

---

## 1. Purpose

To define how Ajace AI creates, approves, distributes, changes, retains and disposes of the
documented information required by the quality management system (QMS) and by ISO 9001:2015.

## 2. Scope

All QMS documents and records listed in `registers/REG-01_Document_Control_Register.md`, plus the
engineering documentation held in this repository that the QMS relies on as evidence.

## 3. Principle: the repository is the document control system

Ajace AI is a software organisation with an existing, enforced version-control discipline. Rather
than maintain a parallel document system that would drift from reality, the QMS uses **Git** as its
control mechanism. This satisfies ISO 9001:2015 7.5.3 as follows.

| ISO 9001:2015 7.5.3 requirement | How Git satisfies it |
|---|---|
| Available and suitable for use | Documents live in `docs/qms/` in the product repository; every engineer has a working copy |
| Adequately protected (loss of confidentiality, improper use, loss of integrity) | Repository access is controlled by the Git hosting account; history is immutable in practice — content cannot be silently altered because every change is a commit with an author and timestamp |
| Distribution, access, retrieval, use | `git clone` / `git pull`; `docs/qms/README.md` is the index |
| Storage and preservation | Remote origin plus every clone is a replica |
| Control of changes (version control) | Commit history is the revision history; `git log --follow <file>` is the change record for any document |
| Retention and disposition | Section 8 of this procedure |
| Identification of documents of external origin | Section 7 |

**Consequence:** the Git commit that changes a QMS document *is* the change record. No separate
change log needs to be maintained inside each document, and none should be — two records of the
same change will disagree eventually.

## 4. Document identification

Every controlled document carries a header table with: Document ID, Revision, Status, Owner,
Approved by, Approval date, and the ISO 9001:2015 clauses it addresses.

| Prefix | Meaning | Location |
|---|---|---|
| `QM-` | Quality Manual | `docs/qms/` |
| `QP-` | Policy | `docs/qms/` |
| `QO-` | Quality objectives | `docs/qms/` |
| `SOP-` | Procedure | `docs/qms/procedures/` |
| `FRM-` | Form / template (blank) | `docs/qms/forms/` |
| `REG-` | Register (live record) | `docs/qms/registers/` |

Filenames follow `<ID>_<Title_With_Underscores>.md`. IDs are never reused: a withdrawn document's
ID is retired.

## 5. Revision and status

**Revision numbering.** `MAJOR.MINOR`. Increment MINOR for clarification that does not change what
anyone must do. Increment MAJOR when the required actions, responsibilities or records change —
a MAJOR increment requires re-approval and re-communication.

**Status** is one of:

| Status | Meaning |
|---|---|
| `DRAFT — not yet approved` | Authored but carrying no authority. Must not be presented as an approved control. |
| `APPROVED` | Signed by the named approver on the stated date; in force. |
| `SUPERSEDED BY <ID>` | Replaced; retained for retention period only. |
| `WITHDRAWN` | No longer part of the QMS; retained for retention period only. |

> Every document in this initial set is `DRAFT — not yet approved`. That is deliberate and
> accurate: the QMS has been authored but not yet adopted. See `ADOPTION_GUIDE.md`.

## 6. Creation, review and approval

1. The author drafts or amends the document on a Git branch.
2. The document Owner reviews it for technical accuracy — does it describe what we actually do?
3. Top management approves it by completing the *Approved by* and *Approval date* fields and
   setting Status to `APPROVED`, in a commit made or explicitly authorised by that person.
4. The change is merged to `main`. The merge commit is the distribution event.
5. Where the change is MAJOR, the Owner communicates it to affected personnel and records that
   communication in `registers/REG-05_Competence_and_Training_Record.md`.

**Approval must be a human act.** Signature and date fields are completed by the named person.
They are never pre-filled, back-dated, or filled in on someone's behalf.

## 7. Documents of external origin

Documents originating outside Ajace AI that the QMS depends on — the ISO 9001:2015 standard text,
model cards and licences for the open-weight models, third-party library licences, customer-supplied
specifications — are identified in `registers/REG-02_External_Documents_and_Providers.md` with their
source and the version/date in use. They are not edited. Where a copy is held in the repository it
is stored unmodified.

## 8. Retention and disposition

| Record type | Minimum retention | Disposition |
|---|---|---|
| QMS documents (policies, procedures, manual) | Life of the QMS + 3 years after supersession | Retained in Git history; no action needed |
| Management review minutes | 3 years | As above |
| Internal audit reports | 3 years | As above |
| Nonconformity / corrective action records | 3 years after closure | As above |
| Design and development records (design reviews, verification/validation results) | Life of the product release + 3 years | As above |
| Release records | Life of the release + 3 years | As above |
| Customer feedback and complaint records | 3 years | As above |
| Competence and training records | Duration of employment + 3 years | As above |
| Customer data held in the deployed product | Per the customer contract | Per `SOP-11`; **not** in Git |

Because Git history is append-only in practice, deletion of a record requires an explicit,
justified history rewrite authorised by top management — which is expected to occur only to remove
information committed in error (for example a credential or personal data). Such an action is
itself recorded as a nonconformity under `SOP-14`.

## 9. What must never be committed

| Never commit | Why | Control |
|---|---|---|
| Credentials, API tokens, private keys | Irreversible disclosure; history is replicated | `.gitignore` excludes `.env`, `.env.local`, `.env.*.bak` |
| Customer documents, transcripts or audio | Customer property (8.5.3); not ours to replicate | Held only in the deployment's data volumes |
| Personal data of identifiable individuals | Data protection obligations | Review before commit |
| Signed legal instruments personal to an individual | Not organisational records | Kept outside version control |

If any of the above is committed in error, raise a nonconformity under `SOP-14` immediately; rotate
the exposed credential before anything else.

## 10. Records generated by this procedure

| Record | Location |
|---|---|
| Document control register | `registers/REG-01_Document_Control_Register.md` |
| External documents and providers register | `registers/REG-02_External_Documents_and_Providers.md` |
| Revision history of every document | Git commit history (`git log --follow -- <path>`) |
