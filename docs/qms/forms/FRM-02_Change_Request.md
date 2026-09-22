# FRM-02 — Change Request

| Field | Value |
|---|---|
| Document ID | FRM-02 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Managing Director |
| ISO 9001:2015 clauses | 8.5.6, 8.3.6 |

## When to use this form

For any change to a released or deployed configuration item — application code in a deployed
release, container images, model versions, deployment configuration, or a design output already
approved. Routine development on an unreleased branch does not need a change request; changing
something a customer is running does.

---

## Change request

| Field | Entry |
|---|---|
| Change ID | `CR-________-____` |
| Raised by (role) | `________________` |
| Date raised | `____ / ____ / ________` |
| Related NC (`FRM-05`) | `________________` |
| Related design review (`FRM-01`) | `________________` |

### 1. Description of the change

`________________________________________________________________________________`

### 2. Reason for the change

☐ Defect correction  ☐ Corrective action  ☐ Customer requirement  ☐ Security
☐ Dependency or model update  ☐ Performance  ☐ Improvement  ☐ Other: `____________`

`________________________________________________________________________________`

### 3. Classification

| Class | Criteria | Select |
|---|---|---|
| **Major** | Changes behaviour a customer relies on; touches data isolation, data residency, authentication or retention; changes a model or a core dependency; requires customer notification or re-validation | ☐ |
| **Minor** | Behaviour-preserving fix or improvement within an existing design; no customer-visible contract change | ☐ |
| **Administrative** | Documentation, comments, formatting; no runtime effect | ☐ |

A **Major** change requires a design review (`FRM-01`) before authorisation.

### 4. Affected configuration items

| Item | Identifier / path | Current version | Target version |
|---|---|---|---|
| `________________` | `________________` | `____________` | `____________` |
| `________________` | `________________` | `____________` | `____________` |

### 5. Impact assessment

| Dimension | Impact | Assessed by (role) |
|---|---|---|
| Requirements — does any stated requirement change? | `____________________________` | `____________` |
| Customers — who is affected, and is notification required? | `____________________________` | `____________` |
| Data — migration, retention, or loss risk? | `____________________________` | `____________` |
| Security and isolation — auth, tenancy, residency? | `____________________________` | `____________` |
| Externally provided components — licence or version effect? | `____________________________` | `____________` |
| Risk register — entry to add or amend (`REG-03`)? | `____________________________` | `____________` |

### 6. Verification plan (what will be run, before authorisation is given)

| # | Check | Expected result |
|---|---|---|
| 1 | `________________________________` | `________________________` |
| 2 | `________________________________` | `________________________` |
| 3 | `________________________________` | `________________________` |

### 7. Authorisation

| Field | Entry |
|---|---|
| Decision | ☐ Approved  ☐ Approved with conditions  ☐ Rejected  ☐ Deferred |
| Conditions / reasoning | `________________________________________` |
| Authorised by (role) | `________________` |
| Date | `____ / ____ / ________` |

### 8. Implementation

| Field | Entry |
|---|---|
| Branch | `________________________` |
| Commit SHA(s) | `________________________` |
| Implemented by (role) | `________________` |
| Date | `____ / ____ / ________` |

### 9. Post-implementation verification

| # | Check from §6 | Actual result | Evidence (path / SHA / report) | Pass? |
|---|---|---|---|---|
| 1 | `____________________` | `________________` | `________________` | ☐ Yes ☐ No |
| 2 | `____________________` | `________________` | `________________` | ☐ Yes ☐ No |
| 3 | `____________________` | `________________` | `________________` | ☐ Yes ☐ No |

### 10. Closure

| Field | Entry |
|---|---|
| All checks passed | ☐ Yes ☐ No — action: `________________` |
| Customer notified (where §5 required it) | ☐ Yes, on `____/____/________`  ☐ Not required |
| Release record raised (`FRM-03`) | `________________` |
| Closed by (role) | `________________` |
| Closure date | `____ / ____ / ________` |

---

## Completion notes

- **Classification drives the rest of the form.** Get §3 wrong and the controls that follow are
  wrong. When in doubt between Major and Minor, take Major — the cost is one design review.
- **§5 must be answered, not skipped.** "No impact" is a valid entry only where it was actually
  considered; a blank is not the same answer.
- **§6 is written before authorisation, §9 after implementation.** Writing both afterwards defeats
  the point: the plan exists so that the result can be compared against an expectation set in
  advance.
- Until the repository carries git tags and a version, record target versions in §4 as commit SHAs.
- A change that fails §9 does not close. Either fix it and re-verify, or revert and raise a
  nonconformity under `SOP-14`.
