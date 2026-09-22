# FRM-07 — Management Review Minutes

| Field | Value |
|---|---|
| Document ID | FRM-07 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Managing Director |
| ISO 9001:2015 clauses | 9.3.2, 9.3.3 |

## When to use this form

At every management review held under `SOP-15` — quarterly, minimum annually, plus any additional
review triggered under `SOP-15` §4. The headings below mirror the `SOP-15` §5 agenda, which mirrors
the 9.3.2 input list. Work down them in order. Every heading is answered; "nothing to report" is a
valid answer and a blank is not.

---

## Review details

| Field | Entry |
|---|---|
| Review reference | `MR-________-____` |
| Date | `____ / ____ / ________` |
| Period under review (dates, and commit range) | `____/____/______` to `____/____/______` — `________..________` |
| Attendees (role, name) | `________________________________________` |
| Chaired by | `________________` |
| Evidence pack assembled before the review | ☐ Yes ☐ No |

## 9.3.2 inputs

### a — Status of actions from previous reviews

| Action ref | Description | Owning role | Due | State | Note |
|---|---|---|---|---|---|
| `________` | `________________` | `__________` | `____/____/______` | ☐ Complete ☐ In progress ☐ Overdue | `____________` |

Findings: `________________________________________`

### b — Changes in external and internal issues (`SOP-02` §4, §5)

| Question | Finding |
|---|---|
| New customer sector or first production customer? | `________________________` |
| Hardware or model platform change? | `________________________` |
| Regulatory or legal change affecting customers? | `________________________` |
| Dependency supply-chain change? | `________________________` |
| Change in the number of people in the organisation? | `________________________` |
| `SOP-02` §4/§5 amended as a result? | ☐ Yes ☐ No |

### c1 — Customer satisfaction and feedback from interested parties

Source: `FRM-08` records, direct communication, reviewer feedback.
Findings: `________________________________________`
Where none: reason — ☐ no customers ☐ no one asked ☐ route exists, produced nothing

### c2 — Extent to which quality objectives are met (`QO-01`)

| Objective | Target | Measured value | Measurement date | Source | Met? |
|---|---|---|---|---|---|
| `____________` | `________` | `________` | `____/____/______` | `____________` | ☐ Yes ☐ No ☐ Not measured |

### c3 — Process performance and product conformity (`SOP-12`)

| Measure | This period | Previous | Direction | Evidence |
|---|---|---|---|---|
| Golden-question evaluation | `____`/52 | `____`/52 | ☐ Up ☐ Down ☐ Same ☐ Not run | `____________` |
| Backend unit tests | `____________` | `____________` | | `____________` |
| Voice unit tests | `____________` | `____________` | | `____________` |
| Clean from-scratch rebuild performed | ☐ Yes ☐ No | | | `____________` |
| Healthchecked services healthy | `____` of `____` | | | `____________` |
| Voice per-turn timing | `____________` | `____________` | | `____________` |

Items that regressed: `________________________________________`

### c4 — Nonconformities and corrective actions (`REG-04`)

| Field | Entry |
|---|---|
| Raised this period (by source and severity) | `________________________` |
| Closed this period | `________________________` |
| Open | `________________________` |
| Open and overdue | `________________________` |
| **Closed without documented effectiveness verification** | `________________________` |
| Recurring cause categories | `________________________` |

### Standing item — auditor impartiality nonconformity (`SOP-13` §5)

| Field | Entry |
|---|---|
| Progress towards external/contract auditor or second competent person | `________________________` |
| Interim self-assessment position still acceptable? | ☐ Yes ☐ No — `____________` |

### c6 — Audit results (`SOP-13`, `FRM-06`)

| Field | Entry |
|---|---|
| Audits conducted against the programme | `________________________` |
| Scheduled audits not conducted, and why | `________________________` |
| Findings by classification | Major `____` Minor `____` Observation `____` OFI `____` |
| Follow-up state | `________________________` |
| Clauses in the cycle still unaudited | `________________________` |

### c5 — Monitoring and measurement results (`SOP-12`)

| Field | Entry |
|---|---|
| Defined measurements actually taken | `________________________` |
| Measurements not taken, and why | `________________________` |
| Measurements no longer useful | `________________________` |

### c7 — Performance of external providers (`REG-02`, `SOP-09`, `FRM-04`)

| Field | Entry |
|---|---|
| Provider failures this period | `________________________` |
| Components whose version moved when relied on not to | `________________________` |
| Licence position changes | `________________________` |
| Providers due re-evaluation | `________________________` |

### d — Adequacy of resources

| Resource | Finding |
|---|---|
| People — capacity, competence, bus factor | `________________________` |
| Infrastructure — GPU, storage, backup | `________________________` |
| Tooling — CI, automation | `________________________` |
| Time | `________________________` |

### e — Effectiveness of actions on risks and opportunities (`REG-03`)

| Risk ref | Treatment | Recurred? | Cited evidence still true today? | Residual score | Action |
|---|---|---|---|---|---|
| `________` | `____________` | ☐ Yes ☐ No | ☐ Yes ☐ No | `____` | `____________` |

### f — Opportunities for improvement

| Ref | Opportunity | Decision | Reason |
|---|---|---|---|
| `________` | `____________________` | ☐ Accept ☐ Defer ☐ Reject | `____________` |

## 9.3.3 outputs

### a — Improvement opportunities accepted

| # | Action | Owning role | Target date | How completion is recognised |
|---|---|---|---|---|
| 1 | `____________________` | `__________` | `____/____/______` | `____________________` |

### b — Changes needed to the QMS

| # | Document ID | Change required | Owning role | Target date |
|---|---|---|---|---|
| 1 | `____________` | `____________________` | `__________` | `____/____/______` |

### c — Resource needs

| # | Resource | Needed for | Decision | Owning role | Target date |
|---|---|---|---|---|---|
| 1 | `____________` | `____________` | ☐ Approved ☐ Deferred ☐ Declined | `__________` | `____/____/______` |

### Conclusion

| Field | Entry |
|---|---|
| Is the QMS suitable, adequate, effective and aligned with strategic direction? | `________________________________________` |
| Decisions to change nothing, and why | `________________________________________` |
| Minutes recorded by | `________________` |
| Next review date | `____ / ____ / ________` |

---

## Completion notes

- **Every heading is answered.** An omitted heading is a gap in the review; a heading answered
  "nothing to report, because `____`" is evidence that the question was asked.
- **Cite the source for every input.** A finding with no source is a recollection. `SOP-15` §3
  explains why this matters more, not less, in a one-person organisation.
- **An objective with no measurement is reported *Not measured*, never *Met*.**
- **c4 asks specifically about records closed without effectiveness verification** because that is
  the most common failure of a CAPA system, and it is invisible unless counted.
- **Outputs need an owning role and a date.** Without both, they are comments, not decisions.
- Store the completed minutes under `docs/qms/records/management-review/` and retain for 3 years
  (`SOP-01` §8).
