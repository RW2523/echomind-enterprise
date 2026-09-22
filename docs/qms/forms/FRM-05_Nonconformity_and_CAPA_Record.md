# FRM-05 — Nonconformity and Corrective Action Record

| Field | Value |
|---|---|
| Document ID | FRM-05 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Managing Director |
| ISO 9001:2015 clauses | 8.7.2, 10.2.1, 10.2.2 |

## When to use this form

For every nonconformity raised under `SOP-14` §5, from any source. One record per nonconformity.
Raise the record first and assess severity on it — do not filter before a record exists. Every
record is also listed in `registers/REG-04_Nonconformity_and_CAPA_Log.md`.

---

## Nonconformity record

| Field | Entry |
|---|---|
| NC ID | `NC-________-____` |
| Date raised | `____ / ____ / ________` |
| Raised by (role) | `________________` |
| Source | ☐ Internal audit ☐ Evaluation ☐ Test failure ☐ Build/deploy failure ☐ Customer complaint ☐ Own observation ☐ External provider ☐ QMS itself |
| Source reference (audit ref / report / complaint ref / commit) | `________________________` |
| Severity | ☐ Critical ☐ Major ☐ Minor ☐ Observation |

### 1. Description — the requirement, the evidence, the discrepancy

| Field | Entry |
|---|---|
| Requirement not fulfilled | `________________________________________` |
| What was observed | `________________________________________` |
| Evidence (path / SHA / report / log) | `________________________________________` |

### 2. Blast radius

| Field | Entry |
|---|---|
| Affected components / paths | `________________________________________` |
| Affected releases (`FRM-03` refs or commit SHAs) | `________________________` |
| Affected deployments / customers | `________________________` (enter `none` where genuinely none) |
| Customer data affected? | ☐ No ☐ Yes — describe: `________________________` |
| Customer notification required? | ☐ Yes ☐ No — reason: `________________________` |
| Customer informed on | `____ / ____ / ________` |

### 3. Immediate correction and containment (10.2.1 a)

| Field | Entry |
|---|---|
| Containment action | `________________________________________` |
| Containment applied on | `____ / ____ / ________` by (role) `________________` |
| Correction (the fix itself) | `________________________________________` |
| Correction commit SHA | `________________________` |
| Disposition of nonconforming output | ☐ Corrected ☐ Withheld ☐ Reverted ☐ Released under concession (§8) |
| Authorised by (role) | `________________` |

### 4. Is corrective action required? (10.2.1 b, d)

| Question | Answer |
|---|---|
| Could this recur? | ☐ Yes ☐ No |
| Does or could a similar nonconformity exist elsewhere? | ☐ Yes — where: `____________` ☐ No |
| Was the nonconformity latent before becoming visible? | ☐ Yes ☐ No |
| Did it reach, or nearly reach, a customer? | ☐ Yes ☐ No |
| Is severity Major or Critical? | ☐ Yes ☐ No |
| **Corrective action required** | ☐ Yes ☐ No — reasoning: `________________________` |

### 5. Root-cause analysis (10.2.1 c)

**Five whys** — record every step.

| Step | Statement |
|---|---|
| Observed fact | `________________________________________` |
| Why 1 | `________________________________________` |
| Why 2 | `________________________________________` |
| Why 3 | `________________________________________` |
| Why 4 | `________________________________________` |
| Why 5 | `________________________________________` |

**Cause categories** — tick every category that applies and state the cause in each.

| Category | Applies | Root cause statement |
|---|---|---|
| Requirement | ☐ | `________________________________` |
| Design | ☐ | `________________________________` |
| Implementation | ☐ | `________________________________` |
| **Verification** (asked every time) | ☐ | `________________________________` |
| Process / control | ☐ | `________________________________` |
| External provider | ☐ | `________________________________` |
| Competence / capacity | ☐ | `________________________________` |

### 6. Corrective action(s) (10.2.1 d)

| # | Action | Owning role | Due date | Completed (SHA / ref) | Date completed |
|---|---|---|---|---|---|
| 1 | `____________________` | `____________` | `____/____/______` | `____________` | `____/____/______` |
| 2 | `____________________` | `____________` | `____/____/______` | `____________` | `____/____/______` |

### 7. Risks, opportunities and QMS updates (10.2.1 e, f)

| Field | Entry |
|---|---|
| Was this risk already in `REG-03`? | ☐ Yes — ref `________` ☐ No |
| If no, why was it not identified? | `________________________________________` |
| Risk register update made | ☐ Yes — ref `________` ☐ Not required — reason: `____________` |
| QMS document change needed | ☐ Yes — document `________`, change: `____________` ☐ No |

### 8. Effectiveness verification — a record does not close without this (10.2.1 g)

| Field | Entry |
|---|---|
| Method (how the *cause* was tested as gone — not how it was fixed) | `________________________________________` |
| Date verified | `____ / ____ / ________` |
| Evidence (test that fails pre-fix, evaluation score, query count, recorded output) | `________________________________________` |
| Verified by (role) | `________________` |
| Same person as implementer? | ☐ No ☐ Yes — recorded as a limitation |
| Outcome | ☐ Effective ☐ Not effective — re-open §5 and record the second analysis below |
| Second analysis (where not effective) | `________________________________________` |

### 9. Closure

| Field | Entry |
|---|---|
| §8 completed and outcome Effective | ☐ Yes ☐ No — record remains open |
| Closure authorised by (role) | `________________` |
| Closure date | `____ / ____ / ________` |
| `REG-04` updated | ☐ Yes |

---

## Completion notes

- **Correction is not corrective action.** §3 stops the bleeding. §5 and §6 stop it happening again.
  A record with §3 completed and §6 empty must justify that in §4.
- **The verification category in §5 is asked for every nonconformity**, whatever else caused it. If
  a defect reached this point, something did not catch it, and that is a separate cause with a
  separate action.
- **Do not stop the five whys at "human error"** — that names a person, not the system that let the
  error have an effect. Do not stop at "bus factor 1" either: true of everything here, actionable
  for almost nothing, and it belongs in `REG-03`.
- **§8 is the step most often skipped.** Implementation evidence is not effectiveness evidence.
  "The tests pass" is effectiveness evidence only if a test exists that would have failed before
  the fix.
- Where implementer and verifier are the same person — currently unavoidable — say so in §8 rather
  than leaving the field implying independence that does not exist.
- A record whose §8 outcome is *Not effective* stays open. The identified cause was not the cause.
