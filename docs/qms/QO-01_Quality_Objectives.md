# QO-01 — Quality Objectives

| Field | Value |
|---|---|
| Document ID | QO-01 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved.** Targets are proposed and require management agreement. |
| Owner | Managing Director |
| Approved by | `________________________` |
| Approval date | `____ / ____ / ________` |
| Review frequency | At every management review (SOP-15); at minimum annually |
| ISO 9001:2015 clauses | 6.2.1, 6.2.2, 9.1.1, 9.1.3 |

---

## 1. Purpose

To set measurable quality objectives consistent with the quality policy (QP-01), and to record how
each will be achieved, measured and reviewed, as ISO 9001:2015 6.2.2 requires.

## 2. How to read the baseline column

**A baseline is a measurement that was actually taken.** Where no measurement exists, the baseline
reads `not yet measured` — it is not estimated, and no target is treated as met until a real
measurement supports it. Two baselines below come from recorded evaluation runs; the rest do not
exist yet, which is itself the honest starting position of a management system at revision 1.0.

Historical measurements available at the time of writing:

| Measurement | Value | Date | Source |
|---|---|---|---|
| Golden evaluation, best recorded | 50 / 52 | 2026-07-30 | `eval/reports/eval_20260730-020659.json` |
| Golden evaluation, most recent recorded | 49 / 52 | 2026-08-06 | `eval/reports/eval_20260806-143249.json` |
| Document-citation precision at that run | 0.98 | 2026-08-06 | same report |

> Both figures **predate the current HEAD** by roughly six weeks and three substantial feature
> commits. They are the last known values, not the present values. `README.md` quotes 48/52, which
> matches neither and is stale. Re-measuring is objective QO-2's first action.

## 3. Objectives

### QO-1 — Grounded answers, or none

*Policy link: QP-01 §2.1. An ungrounded or fabricated answer is the most serious product defect.*

| | |
|---|---|
| **Objective** | No release ships with a known grounding failure — a fabricated fact, a fabricated citation, an invented figure, or an assertion the retrieved material does not support. |
| **Measure** | (a) Count of severity-S1 grounding nonconformities in `REG-04` per release; (b) `forbid_facts` check failures in the golden evaluation; (c) citation precision reported by `eval/run_eval.py`. |
| **Target** | (a) zero at release; (b) zero; (c) ≥ 0.98 |
| **Baseline** | (a) not yet measured (REG-04 begins with this QMS); (b) not separately reported today — the suite reports only pass/fail per item; (c) 0.98 at 2026-08-06 |
| **How achieved** | Cross-encoder relevance gate and citation filtering already in the retrieval path; abstention behaviour; prompt-guard unit tests (`backend/tests/test_prompt_guards.py`); golden evaluation before release (SOP-07) |
| **Monitored** | Per release, and at each management review |
| **First action** | Make `forbid_facts` failures a separately reported figure in the evaluation output so this objective is directly measurable rather than inferred. |

### QO-2 — Verified before release

*Policy link: QP-01 §2.5. Nothing ships unverified.*

| | |
|---|---|
| **Objective** | Every release is preceded by a recorded, retained verification run: unit suites and the 52-question golden evaluation. |
| **Measure** | Percentage of releases with a completed `FRM-03 Release Record` carrying test and evaluation results, and a retained evaluation report. |
| **Target** | 100% |
| **Baseline** | 0% — no release record has ever been produced. (Retention of evaluation reports was fixed on 2026-09-22; 24 historical reports are now held as records, so the *retention* half of this objective is in place and only the release record itself is missing.) |
| **How achieved** | SOP-07 verification levels; SOP-08 release procedure; ~~retain evaluation reports~~ done 2026-09-22 |
| **Monitored** | Per release |
| **First action** | Re-run the golden evaluation against HEAD, retain the report, and record the true current score. Until that is done, no score should be quoted anywhere, including in `README.md`. |

### QO-3 — Traceability from a running instance to its source

*Policy link: QP-01 §2.4 and §2.5. ISO 9001:2015 8.5.2.*

| | |
|---|---|
| **Objective** | Any deployed instance can be traced to the exact source revision, image set and model versions it is running. |
| **Measure** | Can the running system report its build identifier, and does that identifier resolve to a git tag and image digests? |
| **Target** | Yes, for 100% of releases |
| **Baseline** | **No.** No git tags exist; `frontend/package.json` is `"version": "0.0.0"`; backend, voice and frontend images are untagged; no endpoint reports a build identifier |
| **How achieved** | Release identification scheme in SOP-08 §6 |
| **Monitored** | Per release |
| **First action** | Adopt the scheme in SOP-08 §6 and tag the current HEAD as the first identified release. |

### QO-4 — Customer data is protected and recoverable

*Policy link: QP-01 §2.2. ISO 9001:2015 8.5.3, 8.5.4.*

| | |
|---|---|
| **Objective** | Customer data held by a deployment is backed up on a defined schedule, and a restore has been demonstrated. |
| **Measure** | (a) A documented, tested backup procedure for the `echomind_data` volume exists; (b) date of the most recent successful restore test. |
| **Target** | (a) yes; (b) a successful restore test within the last 6 months |
| **Baseline** | (a) **yes, as of 2026-09-22** — `scripts/backup_data.sh` and `scripts/restore_data.sh`; (b) **restore exercised 2026-09-22** into a throwaway volume: 351 MB archive, `PRAGMA integrity_check` returned `ok`, 32 tables present |
| **Status** | **Target met.** Next restore test due within 6 months of 2026-09-22. Outstanding: schedule the backup (it is manual today) and decide the off-host destination. |
| **How achieved** | SOP-11 §8 |
| **Monitored** | At each management review |
| **First action** | ~~Write and test the backup and restore procedure~~ — done 2026-09-22. Now: put it on a schedule and send the archive off-host; an archive beside the data it protects survives nothing. |

### QO-5 — Nonconformities are closed with verified effectiveness

*Policy link: QP-01 §2.5. ISO 9001:2015 10.2.*

| | |
|---|---|
| **Objective** | Every nonconformity raised is closed only after its corrective action's effectiveness has been verified with evidence. |
| **Measure** | (a) Percentage of closed `REG-04` entries with a completed effectiveness-verification field; (b) count of recurrences of a previously closed nonconformity. |
| **Target** | (a) 100%; (b) zero |
| **Baseline** | Not yet measured — no register existed. The historical record is encouraging: several past fixes do record verification (for example commit `4e27109` verified 0/359 out-of-namespace hits across all five retrieval paths after a tenant-isolation fix), but this was practice, not a tracked control. |
| **How achieved** | SOP-14 |
| **Monitored** | At each management review |

### QO-6 — Supplier dependencies are pinned, licensed and known

*Policy link: QP-01 §2.4. ISO 9001:2015 8.4.*

| | |
|---|---|
| **Objective** | Every externally provided component is version-pinned, its licence is recorded, and the inventory is complete. |
| **Measure** | (a) Count of unpinned runtime dependencies; (b) `REG-02` completeness — every model, base image and service has a licence entry; (c) an SBOM or equivalent exists. |
| **Target** | (a) zero moving-branch or `:latest` dependencies in a released image; (b) 100%; (c) yes |
| **Baseline** | (a) at least five — `nemo_toolkit[asr] @ git+…NeMo.git@main` (a moving branch, which caused a real build-and-runtime outage, commit `724fb98`), `torchvision` unpinned, `accelerate` with no upper bound, `ollama/ollama:latest`, `cloudflare/cloudflared:latest`; (b) 0% — no licence inventory exists; (c) no |
| **How achieved** | SOP-09 |
| **Monitored** | At each management review, and whenever an image is rebuilt |
| **First action** | Pin NeMo to a release or a commit sha. It is the single dependency with a demonstrated production impact. |

### QO-7 — The management system actually operates

*ISO 9001:2015 9.2, 9.3 — an objective about the QMS itself, appropriate at revision 1.0.*

| | |
|---|---|
| **Objective** | The QMS is approved, audited and reviewed on schedule rather than existing only as documents. |
| **Measure** | (a) All QMS documents approved and dated; (b) internal audit conducted in the period; (c) management review held in the period. |
| **Target** | (a) 100% within `____` of adoption; (b) at least one audit per 12 months covering all clauses; (c) at least one review per 12 months, quarterly preferred |
| **Baseline** | (a) 0% — every document is `DRAFT — not yet approved`; (b) none conducted; (c) none held |
| **How achieved** | ADOPTION_GUIDE.md, SOP-13, SOP-15 |
| **Monitored** | At each management review |

## 4. Summary

| Ref | Objective | Target | Baseline | Status |
|---|---|---|---|---|
| QO-1 | No grounding failures at release | 0 S1; precision ≥ 0.98 | precision 0.98 (2026-08-06, stale) | Partly measurable today |
| QO-2 | Verified and recorded before release | 100% | 0% | Not started |
| QO-3 | Instance traceable to source | 100% | No mechanism exists | Not started |
| QO-4 | Customer data backed up and restorable | Tested within 6 months | Restore verified 2026-09-22 | **Met** — needs scheduling |
| QO-5 | NCs closed with verified effectiveness | 100% | Not measured | Not started |
| QO-6 | Dependencies pinned and licensed | 0 unpinned; 100% licensed | ≥5 unpinned; 0% licensed | Not started |
| QO-7 | QMS operating, not just written | Approved, audited, reviewed | Nothing approved | Not started |

## 5. Records

| Record | Location |
|---|---|
| This document and its revisions | Git history of `docs/qms/QO-01_Quality_Objectives.md` |
| Measurement results against objectives | SOP-12; `registers/REG-04`; retained evaluation reports |
| Review of objectives | Management review minutes, `forms/FRM-07` |
