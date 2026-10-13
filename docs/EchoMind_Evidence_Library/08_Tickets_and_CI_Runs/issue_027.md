# #27 — NC-2026-007: Published claims contradicted by the organisation's own measurements

| Field | Value |
|---|---|
| State | OPEN |
| Created | 2026-10-07T01:41:24Z |
| Closed | — |
| Labels | bug, nonconformity, retrospective, S3, ISO 8.7, ISO 10.2 |
| URL | https://github.com/RW2523/echomind-enterprise/issues/27 |

**Record:** `REG-04` NC-2026-007 · **Severity:** S3

**Date raised:** Recorded in `eval/paper/results/SUMMARY.json`

**Description:** Measurements contradicted statements in the organisation's own conference paper: E1 — the paper's §7.2 argues the retrieval terms dominate the grounded-response time; measured, they are 2.8% of it. E4 — the paper claims the lattice degrades an injection attempt to a *reportable* event; the measured report rate is 0.000 in every arm, so containment is real but silent. E3 — the timing channel is not closed. E2 — recorded as `"not_run"` with a stated reason rather than estimated.

**Root cause:** Claims were reasoned rather than measured at the time of writing.

**Correction:** The findings are recorded verbatim in the results file, including the contradictions.

**Corrective action:** Pending — the claims should be corrected in any future version of the paper, and the "measure before claiming" rule is now QP-01 §2.3.

**Effectiveness verification:** Not applicable until a future publication.

**Evidence:** `eval/paper/results/SUMMARY.json`

**Status:** **Open — action pending**

---
*Transcribed to this board on 2026-10-06 from the git record and `docs/qms/registers`. Work before that date was not tracked on a board; the dates above come from the commits and registers, not from this ticket's creation time.*
