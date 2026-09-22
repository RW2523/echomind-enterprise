# FRM-01 — Design Review Record

| Field | Value |
|---|---|
| Document ID | FRM-01 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Managing Director |
| ISO 9001:2015 clauses | 8.3.4 |

## When to use this form

At each planned design and development review stage under `SOP-05`, and before any design output is
released for build or deployment. One record per review of one design item. A review that produces
no record did not happen as far as the QMS is concerned.

---

## Design review record

| Field | Entry |
|---|---|
| Design review reference | `DR-________-____` |
| Design item | `________________________________________` |
| Design stage | ☐ Concept  ☐ Detailed design  ☐ Pre-implementation  ☐ Pre-release |
| Review date | `____ / ____ / ________` |
| Participants (role, and name where a name is required) | `________________________________________` |
| Related change request (`FRM-02`) | `________________` |
| Related release record (`FRM-03`) | `________________` |

### 1. Design inputs reviewed (8.3.3)

| # | Input | Source (path / record / customer requirement ref) |
|---|---|---|
| 1 | `________________________________` | `________________________________` |
| 2 | `________________________________` | `________________________________` |
| 3 | `________________________________` | `________________________________` |

### 2. Requirements and acceptance criteria

| # | Requirement | Acceptance criterion (measurable) | Met? |
|---|---|---|---|
| 1 | `________________________` | `________________________` | ☐ Yes ☐ No ☐ Partial |
| 2 | `________________________` | `________________________` | ☐ Yes ☐ No ☐ Partial |
| 3 | `________________________` | `________________________` | ☐ Yes ☐ No ☐ Partial |

### 3. Results of the review (8.3.4 b)

`________________________________________________________________________________`

### 4. Problems identified and necessary actions (8.3.4 c)

| # | Problem | Action required | Owning role | Due date |
|---|---|---|---|---|
| 1 | `____________________` | `____________________` | `____________` | `____/____/________` |
| 2 | `____________________` | `____________________` | `____________` | `____/____/________` |

### 5. Verification — does the output meet the input? (8.3.4 c / 8.3.4 d)

| Field | Entry |
|---|---|
| Method | `________________________________________` |
| Performed by (role) | `________________` |
| Date | `____ / ____ / ________` |
| Evidence (path, commit SHA, report filename) | `________________________________________` |
| Result | ☐ Pass ☐ Fail ☐ Pass with actions |

### 6. Validation — does the output meet the intended use? (8.3.4 d)

| Field | Entry |
|---|---|
| Method | `________________________________________` |
| Performed by (role) | `________________` |
| Date | `____ / ____ / ________` |
| Evidence | `________________________________________` |
| Result | ☐ Pass ☐ Fail ☐ Pass with actions ☐ Not applicable — reason: `____________` |

### 7. Conclusion

| Field | Entry |
|---|---|
| Conclusion | ☐ Proceed  ☐ Proceed with the actions in §4  ☐ Do not proceed |
| Reasoning | `________________________________________` |
| Authorised by (role) | `________________` |
| Date | `____ / ____ / ________` |

---

## Completion notes

- **Verification and validation are different questions.** Verification asks whether the output
  meets the input that was specified. Validation asks whether the resulting product meets the
  intended use in the customer's hands. A design can pass verification and fail validation.
- **Acceptance criteria must be measurable.** "Retrieval is accurate" cannot be reviewed.
  "Golden-question evaluation ≥ the previous recorded score, with zero out-of-namespace hits" can.
- **Cite evidence by path or SHA**, per the house rules in `FRM-00`. An assertion that something
  was checked is not evidence that it was.
- **Validation "not applicable"** requires a reason. It is applicable more often than it feels.
- If §4 contains actions, the conclusion cannot be a plain *Proceed*; use *Proceed with actions*
  and track them to completion before the next stage.
- Where a problem in §4 is a non-fulfilment of a requirement, also raise a nonconformity under
  `SOP-14` on `FRM-05`.
