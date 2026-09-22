# FRM-04 — Supplier / External Provider Evaluation

| Field | Value |
|---|---|
| Document ID | FRM-04 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Managing Director |
| ISO 9001:2015 clauses | 8.4.1, 8.4.2 |

## When to use this form

Before an externally provided product, service or component is relied on in EchoMind Enterprise or
in its build, and at each re-evaluation date thereafter. Covers open-weight models, container base
images, software packages, hosted services and distribution channels. One record per provider or
component. Summarised in `registers/REG-02_External_Documents_and_Providers.md`.

---

## Evaluation record

| Field | Entry |
|---|---|
| Evaluation reference | `SUP-________-____` |
| Provider / component | `________________________________________` |
| Type | ☐ Model ☐ Base image ☐ Package / library ☐ Hosted service ☐ Distribution mirror ☐ Other: `________` |
| Used for | `________________________________________` |
| Version or pin in use | `________________________________________` |
| Pin mechanism (file and line) | `________________________________________` |
| Evaluated by (role) | `________________` |
| Date of evaluation | `____ / ____ / ________` |

### 1. Licence and obligations

| Field | Entry |
|---|---|
| Licence name and version | `________________________` |
| Licence source (URL or path to retained copy) | `________________________` |
| Permits commercial use | ☐ Yes ☐ No ☐ Conditional: `____________` |
| Permits on-premises redistribution to a customer | ☐ Yes ☐ No ☐ Conditional: `____________` |
| Attribution / NOTICE obligation | `________________________` |
| Copyleft or share-alike obligation | ☐ None ☐ Yes: `____________` |
| Use restrictions (field-of-use, acceptable-use, export) | `________________________` |
| Obligations discharged where and how | `________________________` |

### 2. Criticality

| Field | Entry |
|---|---|
| Criticality | ☐ Critical — product does not function without it  ☐ Important — a feature is lost  ☐ Low — replaceable |
| Effect of this component failing or being withdrawn | `________________________________________` |
| Known alternative | `________________________________________` |

### 3. Selection criteria and assessment (8.4.1)

| Criterion | Requirement | Assessment | Met? |
|---|---|---|---|
| Fitness for the intended use | `____________________` | `____________________` | ☐ Yes ☐ No ☐ Partial |
| Licence compatibility with on-premises commercial deployment | `____________________` | `____________________` | ☐ Yes ☐ No ☐ Partial |
| Version stability — released version, not a moving branch | `____________________` | `____________________` | ☐ Yes ☐ No ☐ Partial |
| Offline operability — no runtime call outside the customer perimeter | `____________________` | `____________________` | ☐ Yes ☐ No ☐ Partial |
| Security posture — known vulnerabilities, provenance of artefacts | `____________________` | `____________________` | ☐ Yes ☐ No ☐ Partial |
| Maintenance and support — activity, release cadence, responsiveness | `____________________` | `____________________` | ☐ Yes ☐ No ☐ Partial |
| Other | `____________________` | `____________________` | ☐ Yes ☐ No ☐ Partial |

### 4. Verification performed before use (8.4.3 / 8.6)

| # | Verification | Result | Evidence (path / SHA / report) |
|---|---|---|---|
| 1 | `________________________________` | ☐ Pass ☐ Fail | `________________` |
| 2 | `________________________________` | ☐ Pass ☐ Fail | `________________` |
| 3 | `________________________________` | ☐ Pass ☐ Fail | `________________` |

### 5. Risks and mitigations

| # | Risk | Likelihood / impact / score | Mitigation in place (cite by path) | `REG-03` ref |
|---|---|---|---|---|
| 1 | `____________________` | `____` / `____` / `____` | `____________________` | `________` |
| 2 | `____________________` | `____` / `____` / `____` | `____________________` | `________` |

### 6. Decision

| Field | Entry |
|---|---|
| Decision | ☐ Approved for use  ☐ Approved with conditions  ☐ Not approved |
| Conditions | `________________________________________` |
| Approved by (role) | `________________` |
| Date | `____ / ____ / ________` |
| Re-evaluation due | `____ / ____ / ________` |
| Re-evaluation trigger (in addition to the date) | `________________________________________` |

---

## Completion notes

- **Version stability is a real criterion here, not a formality.** A dependency installed from a
  moving branch rather than a release can change under the build with no action by anyone;
  `backend/Dockerfile:19` installs `nemo_toolkit[asr]` from `git+…NeMo.git@main`, and that has
  already produced a real build and runtime outage (commit `724fb98`). Where a moving reference
  cannot be avoided, mark the criterion *Partial* and record the compensating control in §5.
- **Offline operability matters more than for most products.** Any component that reaches outside
  the customer perimeter at runtime undermines the reason the product exists. Test it, do not assume
  it — a model or library may fetch on first use.
- **Record the licence obligation *and* how it is discharged.** "Apache-2.0" alone is not a
  completed §1; the NOTICE obligation still has to land somewhere.
- **Criticality drives re-evaluation frequency.** Critical components: at least annually and on
  every version change. Low: at the scheduled date.
- Re-evaluations are new records, not edits to this one, so the history is visible.
