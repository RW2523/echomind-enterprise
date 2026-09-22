# FRM-10 — Risk Assessment Record

| Field | Value |
|---|---|
| Document ID | FRM-10 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Managing Director |
| ISO 9001:2015 clauses | 6.1.1, 6.1.2 |

## When to use this form

To assess a single risk or opportunity in detail — on identification, on material change, or when a
risk already in `registers/REG-03_Risk_Register.md` needs fuller analysis than a register line
holds. The register remains the summary list; this form is the working. Scales below are reproduced
from `SOP-02` §7 so the form is self-contained.

---

## Risk record

| Field | Entry |
|---|---|
| Risk ID | `R-________-____` |
| Date assessed | `____ / ____ / ________` |
| Assessed by (role) | `________________` |
| Type | ☐ Risk ☐ Opportunity |
| Category | ☐ Product quality ☐ Data protection / isolation ☐ Availability ☐ Supply chain ☐ Legal / licence ☐ Resource / capacity ☐ Process / QMS ☐ Other: `________` |
| Source of identification | ☐ Engineering work ☐ Incident ☐ Evaluation result ☐ Management review ☐ Context review ☐ Audit ☐ Customer |

### 1. Description — cause → event → consequence

| Field | Entry |
|---|---|
| Cause | `________________________________________` |
| Event | `________________________________________` |
| Consequence | `________________________________________` |

### 2. Affected interested parties (`SOP-02` §5)

| Interested party | How they are affected |
|---|---|
| `________________________` | `________________________________` |
| `________________________` | `________________________________` |

### 3. Assessment scales

**Likelihood — chance of materialising within the next 12 months:**

| Level | Name | Meaning |
|---|---|---|
| 1 | Rare | No known occurrence; would require an unusual combination of events |
| 2 | Unlikely | Plausible but not expected |
| 3 | Possible | Expected roughly once in the period, or has happened once before |
| 4 | Likely | Expected more than once in the period |
| 5 | Almost certain | Occurring now, or certain to occur unless treated |

**Impact — worst credible consequence:**

| Level | Name | Meaning for EchoMind Enterprise |
|---|---|---|
| 1 | Negligible | Internal inconvenience; no customer visibility |
| 2 | Minor | Degraded feature; workaround exists; no data affected |
| 3 | Moderate | Feature unavailable to a customer, or a published claim shown to be unsupported |
| 4 | Major | Ungrounded or fabricated content reaches a user in a regulated sector; or a deployment unavailable for a working day; or recoverable data loss |
| 5 | Severe | Customer data leaves the customer's perimeter, or crosses a tenant boundary, or is irrecoverably lost |

**Bands (score = likelihood × impact) and required response:**

| Score | Band | Required response |
|---|---|---|
| 1–4 | Low | Accept and record. Review at management review. |
| 5–9 | Medium | Treatment planned with a named owning role and a target date. |
| 10–14 | High | Treatment planned and started before the next release touching the area. Monthly review. |
| 15–25 | Critical | New work in the affected area stops until the score is below 15, or a dated written acceptance is recorded. |

### 4. Inherent rating (before treatment)

| Likelihood | Impact | Score | Band | Justification |
|---|---|---|---|---|
| `____` | `____` | `____` | `__________` | `________________________________` |

### 5. Existing controls

| # | Control | Evidence (file path, line, commit SHA, procedure §) | Operating? |
|---|---|---|---|
| 1 | `________________________` | `________________________________` | ☐ Yes ☐ No ☐ Partial |
| 2 | `________________________` | `________________________________` | ☐ Yes ☐ No ☐ Partial |
| 3 | `________________________` | `________________________________` | ☐ Yes ☐ No ☐ Partial |

### 6. Treatment decision (6.1.2)

| Option | Select | Reasoning |
|---|---|---|
| Avoid — stop the activity creating the risk | ☐ | `________________________` |
| Reduce — add a control | ☐ | `________________________` |
| Transfer / share — the risk properly sits with the customer or a supplier | ☐ | `________________________` |
| Accept — cost of treatment exceeds exposure | ☐ | `________________________` |

**Where Accept is selected and impact is 4 or 5**, a dated written acceptance is required:

| Field | Entry |
|---|---|
| Residual risk accepted by (role and name) | `________________________` |
| Date | `____ / ____ / ________` |
| Reason for acceptance | `________________________________________` |

### 7. Planned actions

| # | Action | Owning role | Target date | Complete | Evidence |
|---|---|---|---|---|---|
| 1 | `____________________` | `__________` | `____/____/______` | ☐ | `____________` |
| 2 | `____________________` | `__________` | `____/____/______` | ☐ | `____________` |

### 8. Residual rating (after treatment)

| Likelihood | Impact | Score | Band | Evidence the treatment is in place |
|---|---|---|---|---|
| `____` | `____` | `____` | `__________` | `________________________________` |

### 9. Review

| Field | Entry |
|---|---|
| Review date | `____ / ____ / ________` |
| Additional review trigger | `________________________________________` |
| `REG-03` entry created / updated | ☐ Yes — ref `________` |
| Carried to management review | `MR-________` |

---

## Completion notes

- **Write the description as cause → event → consequence.** "Data loss" is not a risk statement;
  "no backup exists for the only volume holding customer data (cause), so a volume failure (event)
  destroys all customer content irrecoverably (consequence)" can be scored and treated.
- **Impact is the worst *credible* consequence, not the worst imaginable** and not the average.
- **§5 requires evidence, not assurance.** A control that cannot be cited by path, line or procedure
  section is not yet a control — `SOP-02` §8 is explicit that "we are careful about X" does not
  count.
- **Residual rating must be justified by the treatment, not by optimism.** If the evidence column in
  §8 is empty, the residual score is the inherent score.
- **An impact of 5 may not be accepted without a written, dated acceptance**, whatever the
  likelihood (`SOP-02` §7).
- Opportunities use the same form: the "consequence" is the benefit forgone if it is not taken, and
  the treatment options reduce to pursue or decline, with reasoning.
