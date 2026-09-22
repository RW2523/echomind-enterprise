# FRM-00 — QMS Document Template

| Field | Value |
|---|---|
| Document ID | FRM-00 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Managing Director |
| ISO 9001:2015 clauses | 7.5.2 |

Copy the block below when creating a new QMS document. Keep the heading order; delete sections
that genuinely do not apply rather than leaving them empty with "N/A".

---

```markdown
# <ID> — <Title>

| Field | Value |
|---|---|
| Document ID | <ID> |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | <role, not a person's name> |
| Approved by | `________________________` |
| Approval date | `____ / ____ / ________` |
| ISO 9001:2015 clauses | <clauses> |

## 1. Purpose
Why this document exists, in two or three sentences.

## 2. Scope
What it covers and, where it matters, what it does not.

## 3. Responsibilities
Table of role → responsibility. Roles, never individuals — individuals change.

## 4. Procedure
The actual steps, numbered. Each step says who does what, when, and what evidence it leaves.

## 5. Records
Table of record → where it lives → retention (cross-reference SOP-01 §8).

## 6. Current state and gaps
Honest statement of what is implemented today versus what this procedure requires, with the
gap carried into ISO9001_Gap_Analysis.md. Never describe an aspiration in the present tense.
```

---

## House rules for every QMS document

1. **Describe what actually happens.** If a control is not yet in place, say so in §6 and record it
   as a gap. A procedure that describes an imaginary process is worse than no procedure: it fails
   the first audit and it misleads our own people.
2. **Roles, not names.** Personnel change; documents should not need reissuing when they do.
3. **Cite evidence by path.** `backend/app/main.py:185` is verifiable. "The system has health
   checks" is not.
4. **No fabricated records.** Approval names, signatures, dates, audit findings and review minutes
   are completed by the people who actually did the thing, at the time they did it.
5. **Present tense for what is in place; explicit future tense for what is planned.**
6. **British English, plain language, no marketing tone.** An auditor is the reader.
