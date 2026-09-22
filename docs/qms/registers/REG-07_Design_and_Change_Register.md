# REG-07 — Design and Change Register

| Field | Value |
|---|---|
| Document ID | REG-07 |
| Revision | 1.0 |
| Status | **LIVE REGISTER — currently empty** |
| Owner | Lead Engineer |
| ISO 9001:2015 clauses | 8.3.2, 8.3.4, 8.3.6, 8.5.6 |
| Governing procedures | SOP-05 (design), SOP-06 (change) |

---

## 1. Purpose

One row per significant design item or change: what was planned, what review it received, and what
verification closed it. ISO 9001:2015 requires records of design planning (8.3.2), design controls
(8.3.4) and design changes (8.3.6).

## 2. What must be registered

Not every commit. Only changes meeting the significance criteria in SOP-06 §5 — in summary, anything
that changes intended behaviour, the retrieval or generation path, the data model, the deployment
topology, authentication or data handling, a model, or a customer-visible interface.

Routine changes remain controlled by Git alone; the commit is the record.

## 3. Register

| Ref | Date raised | Class | Title | Design inputs / acceptance criteria | Review (date, FRM-01 ref) | Verification evidence | Implemented (commit) | Status |
|---|---|---|---|---|---|---|---|---|
| DC-2026-001 | | | | | | | | |

Use `forms/FRM-01_Design_Review_Record.md` for the review and `forms/FRM-02_Change_Request.md` for
the change, then summarise here.

## 4. Current state

Empty. Historically, design and change control has been exercised through Git alone — and the commit
bodies are unusually good, several recording rationale, root cause and verification. What they do not
record is a design review as a distinct act with participants and a decision, or acceptance criteria
agreed *before* the work. That is the gap this register closes (Gap Analysis G-16, G-18).
