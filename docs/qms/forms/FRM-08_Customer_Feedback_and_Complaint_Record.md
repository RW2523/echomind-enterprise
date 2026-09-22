# FRM-08 — Customer Feedback and Complaint Record

| Field | Value |
|---|---|
| Document ID | FRM-08 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Managing Director |
| ISO 9001:2015 clauses | 8.2.1, 9.1.2, 10.2 |

## When to use this form

For every enquiry, item of feedback or complaint received from a customer, a prospective customer
or an end user of a deployed system — however it arrives, including verbally. One record per item.
Completed records are an input to management review (`SOP-15` §5.3) and are the organisation's
evidence of monitoring customer perception under clause 9.1.2.

---

## Record

| Field | Entry |
|---|---|
| Reference | `CF-________-____` |
| Date received | `____ / ____ / ________` |
| Received by (role) | `________________` |
| Customer / organisation | `________________________________________` |
| Contact (name, role) | `________________________________________` |
| Channel | ☐ Email ☐ Call ☐ Meeting ☐ In-product ☐ Written ☐ Third party ☐ Other: `________` |
| Type | ☐ Enquiry ☐ Feedback — positive ☐ Feedback — improvement suggestion ☐ **Complaint** |
| Affected deployment / release (`FRM-03` ref or SHA) | `________________________` |

### 1. What was said

Record in the customer's own terms before interpreting it.

`________________________________________________________________________________`

`________________________________________________________________________________`

### 2. Immediate response

| Field | Entry |
|---|---|
| Acknowledged to customer on | `____ / ____ / ________` |
| Acknowledged by (role) | `________________` |
| Immediate response given | `________________________________________` |
| Interim workaround offered | `________________________________________` |

### 3. Assessment

| Question | Answer |
|---|---|
| Is this a non-fulfilment of a requirement? | ☐ Yes ☐ No |
| If yes, nonconformity raised (`FRM-05`) | `NC-________-____` |
| Does it indicate a risk not in `REG-03`? | ☐ Yes — ref `________` ☐ No |
| Does it indicate a requirement not captured under `SOP-04`? | ☐ Yes ☐ No |
| Is it an opportunity for improvement? | ☐ Yes — logged in `REG-04` as `________` ☐ No |
| Severity to the customer | ☐ Critical ☐ Major ☐ Minor ☐ Informational |

### 4. Action taken

| # | Action | Owning role | Due date | Completed | Evidence (SHA / ref) |
|---|---|---|---|---|---|
| 1 | `____________________` | `__________` | `____/____/______` | `____/____/______` | `____________` |
| 2 | `____________________` | `__________` | `____/____/______` | `____/____/______` | `____________` |

### 5. Communication back to the customer

| Field | Entry |
|---|---|
| Customer informed of the outcome on | `____ / ____ / ________` |
| Informed by (role) | `________________` |
| What was communicated | `________________________________________` |
| Customer response | `________________________________________` |

### 6. Satisfaction follow-up (9.1.2)

| Field | Entry |
|---|---|
| Follow-up performed on | `____ / ____ / ________` |
| Method | ☐ Direct question ☐ Call ☐ Email ☐ Review meeting ☐ Other: `____________` |
| Is the customer satisfied with the resolution? | ☐ Yes ☐ Partially ☐ No |
| Customer's own words | `________________________________________` |
| Further action needed | ☐ No ☐ Yes — `________________________` |

### 7. Closure

| Field | Entry |
|---|---|
| All actions in §4 complete | ☐ Yes ☐ No |
| Linked NC (where raised) closed | ☐ Yes ☐ No ☐ Not applicable |
| Closed by (role) | `________________` |
| Closure date | `____ / ____ / ________` |
| Carried to management review `MR-________` | ☐ Yes |

---

## Completion notes

- **Record it before deciding whether it counts.** Something logged as an enquiry that turns out to
  be a complaint is recoverable; something never logged is not. Positive feedback is recorded too —
  clause 9.1.2 is about perception, not only about failure.
- **§1 in the customer's own words.** Paraphrasing into engineering terms at the point of capture
  loses the information that makes the trend visible later.
- **A complaint about a regulated-sector deployment is treated as Major at minimum** until assessed
  otherwise, because the customer's exposure is not limited to the defect.
- **§3 is the link into the rest of the QMS.** Most complaints are nonconformities; where the
  answer is No, the record should be able to say why.
- **Closing with the customer and closing the record are different events.** §5 is the first, §7 is
  the second, and §6 sits between them — do not close before asking whether the resolution actually
  satisfied them.
- Verbal feedback counts. Write it down on this form the same day, with the channel set to Call or
  Meeting.
