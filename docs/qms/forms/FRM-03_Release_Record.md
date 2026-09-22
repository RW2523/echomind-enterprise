# FRM-03 — Release Record

| Field | Value |
|---|---|
| Document ID | FRM-03 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Managing Director |
| ISO 9001:2015 clauses | 8.6, 8.5.1 |

## When to use this form

Once per release of EchoMind Enterprise to any environment a customer or an external party can
reach, including a demonstration deployment. Completed **before** the release is authorised; the
post-deployment section is completed after. Clause 8.6 requires evidence of conformity with
acceptance criteria and traceability to the person authorising release.

---

## Release record

| Field | Entry |
|---|---|
| Release identifier | `REL-________-____` |
| Release name / tag | `________________` |
| Source revision (git SHA) | `________________________________________` |
| Git tag (or `none — see completion notes`) | `________________` |
| Prepared by (role) | `________________` |
| Date prepared | `____ / ____ / ________` |

### 1. Contents

| Component | Image / artefact | Tag or digest |
|---|---|---|
| Backend | `________________________` | `________________` |
| Frontend | `________________________` | `________________` |
| Voice | `________________________` | `________________` |
| Inference runtime | `________________________` | `________________` |
| Other | `________________________` | `________________` |

| Model | Identifier | Version / revision | Licence checked (`FRM-04`) |
|---|---|---|---|
| Chat | `________________________` | `____________` | ☐ Yes ☐ No |
| Embedding | `________________________` | `____________` | ☐ Yes ☐ No |
| STT | `________________________` | `____________` | ☐ Yes ☐ No |
| TTS | `________________________` | `____________` | ☐ Yes ☐ No |

### 2. Change summary

| Change ID (`FRM-02`) or commit | Description | Classification |
|---|---|---|
| `________________` | `____________________________` | `____________` |
| `________________` | `____________________________` | `____________` |

### 3. Verification evidence (8.6)

| # | Check | Acceptance criterion | Result | Evidence (path / report filename) |
|---|---|---|---|---|
| 1 | Backend unit tests | All pass | ☐ Pass ☐ Fail — `____` | `________________` |
| 2 | Voice unit tests | All pass | ☐ Pass ☐ Fail — `____` | `________________` |
| 3 | Golden-question evaluation (`eval/run_eval.py`) | Score `____`/52, not below previous recorded run | ☐ Pass ☐ Fail | `________________` |
| 4 | Clean from-scratch rebuild | Completes; all containers start | ☐ Pass ☐ Fail ☐ Not performed | `________________` |
| 5 | Container healthchecks | All healthchecked services healthy | ☐ Pass ☐ Fail | `________________` |
| 6 | Manual functional check — chat / RAG | `________________________` | ☐ Pass ☐ Fail | `________________` |
| 7 | Manual functional check — transcription / voice | `________________________` | ☐ Pass ☐ Fail | `________________` |
| 8 | Isolation check (namespace / tenancy) | Zero out-of-namespace results | ☐ Pass ☐ Fail | `________________` |
| 9 | Other | `________________________` | ☐ Pass ☐ Fail | `________________` |

### 4. Known issues accepted into this release

| # | Issue | NC ref (`FRM-05`) | Why acceptable | Accepted by (role) |
|---|---|---|---|---|
| 1 | `____________________` | `____________` | `____________________` | `____________` |
| 2 | `____________________` | `____________` | `____________________` | `____________` |

### 5. Release authorisation (8.6)

| Field | Entry |
|---|---|
| All §3 checks passed, or exceptions recorded in §4 | ☐ Yes ☐ No — release withheld |
| Decision | ☐ Release  ☐ Release with the exceptions in §4  ☐ Do not release |
| Authorised by (role) | `________________` |
| Date | `____ / ____ / ________` |

### 6. Deployment

| Field | Entry |
|---|---|
| Target environment / customer | `________________________` |
| Deployment method | `________________________` |
| Deployed by (role) | `________________` |
| Date and time | `____ / ____ / ________  ____:____` |

### 7. Post-deployment verification

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | All services running and healthy | ☐ Pass ☐ Fail | `________________` |
| 2 | Smoke test of primary user journey | ☐ Pass ☐ Fail | `________________` |
| 3 | No data loss or migration error | ☐ Pass ☐ Fail | `________________` |
| Verified by (role) / date | `________________` / `____/____/________` | | |

### 8. Rollback plan

| Field | Entry |
|---|---|
| Rollback target (SHA / image tags) | `________________________` |
| Rollback method | `________________________________________` |
| Data migration reversible? | ☐ Yes ☐ No — implication: `________________` |
| Maximum acceptable time to roll back | `____________` |
| Rollback tested? | ☐ Yes, on `____/____/________`  ☐ No |

---

## Completion notes

- **§3 is completed before §5, from a real run.** Copying the previous release's results forward is
  a falsified record.
- **Record the golden-evaluation score as a number, even when it passes.** The trend matters more
  than the single result, and `SOP-15` §5.5 reports it.
- **§4 is the concession record required by clause 8.7.2** where a known nonconformity is shipped.
  Every entry needs an NC reference and a named authorising role — not a shrug.
- **Git tags and versioning:** the repository currently has no tags and `frontend/package.json` is
  `"0.0.0"`. Until that changes, the source revision SHA in the header is the authoritative
  identifier and the tag field is completed `none`.
- **A rollback plan that has never been tested is a hypothesis.** Record honestly whether it has
  been.
