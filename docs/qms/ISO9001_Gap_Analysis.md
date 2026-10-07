# ISO 9001:2015 — Gap Analysis and Readiness Assessment

| Field | Value |
|---|---|
| Document ID | GAP-01 |
| Revision | 1.0 |
| Status | **Assessment as at 2026-09-21, HEAD `ff29843`; rows re-rated on 2026-10-07 at document approval carry that date** |
| Assessed by | Documentation review of the repository against ISO 9001:2015 clauses 4–10 |
| Method | Evidence-based: every finding cites a file, a command output or the demonstrable absence of an artefact |
| Owner | Managing Director (Anita Johan) |

---

## 1. Summary

**Update 2026-10-07.** The document set was approved by the Managing Director on 2026-10-07
(`records/approvals/2026-10-07_AR-2026-001_document_approval.md`), roles are named, the policy is
signed, risk ratings and owners are confirmed, and the registers are live. The first internal audit
(Nov 2026 – Jan 2027) and the first management review are scheduled but have not yet taken place;
those two remain the open items for clauses 9.2 and 9.3.

**Readiness at the original assessment (2026-09-21): not ready.** The management system was documented but not
operating — nothing approved, no audit, no management review. That alone is a major nonconformity
against clauses 9.2 and 9.3 regardless of anything else.

**Readiness of the underlying engineering practice: better than the paperwork suggests.** The
product has a version-controlled 52-question evaluation suite with a binary pass gate, 78 unit test
functions, a second research-grade evaluation harness whose results contradict the organisation's own
published claims where the measurements say so, and commit records that routinely carry root cause
and verification. Several ISO 9001 clauses are substantially satisfied in substance and fail only on
the record-keeping.

The gaps cluster in three places, in order of severity:

| Cluster | Clauses | Nature |
|---|---|---|
| **A. The QMS is not operating** | 5.2, 6.2, 9.2, 9.3, 10.2 | Documented, never run |
| **B. Release and configuration control** | 8.5.2, 8.5.6, 8.6 | No versioning, no traceability, no enforcement |
| **C. Customer data lifecycle** | 8.5.3, 8.5.4 | No backup, no retention, no encryption at rest |

Cluster C contains the only gap that is a live operational risk rather than a compliance one, and it
should be closed first on that basis alone.

## 2. Clause-by-clause assessment

Ratings: **Conformant** · **Partial** — the substance exists but the record or rigour does not ·
**Gap** — not in place.

| Clause | Requirement | Rating | Evidence / finding | Ref |
|---|---|---|---|---|
| 4.1 | Context of the organisation | Met (2026-10-07) | QM-01 §3 and SOP-02 §4, approved by the Managing Director | — |
| 4.2 | Interested parties | Met (2026-10-07) | QM-01 §4 and SOP-02 §5, reviewed and approved | — |
| 4.3 | Scope of the QMS | Met (2026-10-07) | QM-01 §5 — scope with site address, within the AJACE Inc. certified scope; justified non-applicability of 7.1.5.2 | — |
| 4.4 | QMS and its processes | Partial | Six processes mapped in QM-01 §6 with inputs, outputs and measures; operating since 2026-10-07 — first audit and management review cycle not yet complete | — |
| 5.1 | Leadership and commitment | Partial (2026-10-07) | Policy approved and signed, roles named, documents approved by the Managing Director; first management review not yet held | G-21 |
| 5.2 | Quality policy | Met (2026-10-07) | QP-01 signed by Anita Johan (Managing Director); communicated to the role holders (`REG-05`) | ~~G-07~~ closed |
| 5.3 | Roles and authorities | Met (2026-10-07) | Named role holders and organisation chart in QM-01 §7.1–7.2 | G-08 closed |
| 6.1 | Risks and opportunities | Met (2026-10-07) | REG-03: 14 evidenced risks and 4 opportunities; ratings confirmed at approval; owners and due dates set | ~~G-09~~ closed |
| 6.2 | Quality objectives | Partial | QO-01 with 7 objectives, approved 2026-10-07; baselines to be re-measured for the first management review | G-10 |
| 6.3 | Planning of changes | Partial | SOP-02 §7; historically changes were planned in commit bodies, sometimes very well, but never against criteria | — |
| 7.1.1–7.1.4 | Resources, people, infrastructure, environment | Partial | Infrastructure well documented (compose, Dockerfiles, deployment docs); people resource is one person | G-08 |
| 7.1.5.1 | Monitoring and measuring resources | **Conformant** | The measuring instrument is the evaluation harness; `eval/golden/*.jsonl` and `eval/run_eval.py` are version-controlled, and results are now retained as records (2026-09-22, 24 historical reports committed) | ~~G-12~~ closed |
| 7.1.5.2 | Measurement traceability | Not applicable | No physical measuring equipment; justified in QM-01 §4.1 | — |
| 7.1.6 | Organisational knowledge | Partial | Unusually good written rationale in `docs/` and commit bodies — a real strength — but concentrated in one person | G-08 |
| 7.2 | Competence | Partial (2026-10-07) | REG-05 competence matrix completed for the five role holders; training records accumulate from here | G-13 |
| 7.3 | Awareness | Met (2026-10-07) | Policy awareness recorded for the role holders in `REG-05` | ~~G-07~~ closed |
| 7.4 | Communication | Partial | Internal: GitHub board, email/Teams, review records (QM-01 §9); external: SOP-04 — no customer engagement yet | G-14 |
| 7.5 | Documented information | **Conformant** (2026-10-07) | SOP-01 uses Git as the control mechanism — versioned, attributed, timestamped, replicated. All controlled documents approved 2026-10-07 (`REG-01`, AR-2026-001) | — |
| 8.1 | Operational planning and control | Partial | Deployment and build are well controlled and documented; planning criteria are not written down | — |
| 8.2.1 | Customer communication | Gap | No feedback channel defined, no log. REG-06 created and empty. | G-15 |
| 8.2.2–8.2.3 | Determining and reviewing requirements | Gap | No requirements-review record exists for any engagement | G-15 |
| 8.2.4 | Changes to requirements | Gap | No mechanism | G-15 |
| 8.3.1–8.3.2 | Design planning | Partial | Design happens and is documented in `docs/`, but stages, reviews and acceptance criteria are not planned in advance | G-16 |
| 8.3.3 | Design inputs | Partial | Implicit in the code and prompts (grounding, abstention, isolation are clearly deliberate design inputs) but never written as inputs | G-16 |
| 8.3.4 | Design controls | **Partial — strong substance** | Verification genuinely happens: 78 unit test functions, the 52-question golden evaluation, the E1–E10 harness, and manual verification recorded in commit bodies. **Design reviews now recorded (2026-10-07): 8 reviews, 40 actions in `REG-10`.** Before that, no design review records existed. | G-16 |
| 8.3.5 | Design outputs | Conformant | Source, architecture documents in `docs/`, `PROTOCOL.md`, user manual | — |
| 8.3.6 | Design changes | Partial | Every change is in Git with rationale; no impact assessment against requirements, because requirements are not recorded | G-16 |
| 8.4.1 | Control of external providers | Partial | Controlled in practice by pinning and offline pre-caching; **no selection or evaluation criteria**, no re-evaluation | G-05 |
| 8.4.2 | Type and extent of control | Partial | Strong for most packages (14/20 backend exact-pinned); the one dependency installed from a moving Git branch, which caused a real outage (`724fb98`), was pinned to a commit on 2026-09-22 (`51fd3bc`); licence inventory still open | G-05 |
| 8.4.3 | Information for external providers | Not applicable in substance | No subcontracted work; providers are upstream | — |
| 8.5.1 | Control of production and service provision | Partial | Deployment procedures documented and repeatable (`OFFLINE_DEPLOYMENT.md`, `scripts/`); release authorisation now recorded on `FRM-03` (v1.4.0, v1.4.1) but given after deployment — authorise before deployment from the next release | G-01 |
| 8.5.2 | **Identification and traceability** | Met (2026-10-06) | Tags `v1.4.0`/`v1.4.1`, `CHANGELOG.md`, `package.json` `1.4.1`, version-tagged images, build identity at `/api/version`, voice `/health`, `/build.json`; `REG-08` | ~~G-01~~ closed |
| 8.5.3 | **Property belonging to customers** | **Partial — with a critical gap** | Isolation is designed and enforced (namespace predicate, tenant forcing) but has failed twice in the same way; **no backup of the only customer-data volume**; no retention policy; no encryption at rest; auth off by default and WebSockets outside the auth middleware | **G-03, G-04, G-06** |
| 8.5.4 | Preservation | **Partial** | Model volumes preserved by design; customer data now has a backup and restore procedure with a verified restore (2026-09-22) — outstanding: it is manual and on-host | ~~G-03~~ closed; see G-22 |
| 8.5.5 | Post-delivery activities | Partial | User manual and troubleshooting exist; no support process, SLA or escalation defined | G-17 |
| 8.5.6 | Control of changes | Partial | Git is the mechanism and commit discipline is good; change register `REG-07` and significance criteria (SOP-06 §6) now in place; impact assessment against requirements still to mature | ~~G-18~~ closed |
| 8.6 | **Release of products and services** | Partial (2026-10-07) | Release records with retained verification evidence for v1.4.0 and v1.4.1 (`records/releases/`); authorised by the Managing Director after deployment (deviation recorded); CI runs the unit tests on every push. `scripts/verify_offline_readiness.sh` is still called by nothing. | G-02 |
| 8.7 | Control of nonconforming outputs | Partial | Handled well in practice (22 `fix:` commits with root cause); no register until REG-04, no disposition or concession process, no customer-notification procedure | G-19 |
| 9.1.1 | Monitoring and measurement | Partial | Real measurement exists and is good; non-functional requirements recorded and monitored against approved targets in `REG-09` (2025 finding #6 closed, EM26-17); evaluation results retained; not yet trended | G-12 |
| 9.1.2 | Customer satisfaction | Gap | No method defined, no data | G-15 |
| 9.1.3 | Analysis and evaluation | Gap | No periodic analysis; the paper harness is the closest thing and it was a one-off campaign | G-12 |
| 9.2 | **Internal audit** | **Gap** | Programme approved 2026-10-07 (SOP-13 §4: Nov 2026 clauses 4–7, Dec 2026 clause 8, Jan 2027 clauses 9–10), through the AJACE Inc. corporate audit programme (MR Sheryl Nazareth); none conducted yet | **G-20** |
| 9.3 | **Management review** | **Gap** | None held yet; quarterly cadence approved (SOP-15 §4) | **G-21** |
| 10.1 | Improvement | Partial | Improvement clearly happens (a feature was disabled because it measured worse — `2decb99`); not systematic | — |
| 10.2 | Nonconformity and corrective action | Partial | Corrective action is practised and sometimes verified; **of 6 retrospective defect records, 3 record effectiveness verification and 3 do not**; no register before REG-04 | G-19 |
| 10.3 | Continual improvement | Gap | No mechanism; depends on 9.3 which has not happened | G-21 |

## 3. Prioritised gap list

Priority reflects **consequence if left open**, not audit severity — the two differ, and where they
differ the operational consequence wins.

### Priority 1 — close before the next customer deployment

| Ref | Gap | Clause | Consequence if left | Effort |
|---|---|---|---|---|
| ~~G-03~~ | **CLOSED 2026-09-22** — `scripts/backup_data.sh` / `restore_data.sh`; restore verified into a throwaway volume (integrity_check ok, 32 tables). Superseded by G-22 below. | 8.5.3, 8.5.4 | — | done |
| ~~G-11~~ | **CLOSED 2026-09-22** — bounded `json-file` logging (50 MB x 5) on all six services. Applies on the next `docker compose up -d`; running containers keep their original config until recreated. *Applied on the reference host 2026-10-06 (verified).* | 8.5.1 | — | done |
| ~~G-22~~ | **CLOSED 2026-09-22** — nightly systemd timer (`scripts/install_backup_timer.sh`) and off-host replication via `BACKUP_REMOTE`, tested; a failed off-host copy aborts the run rather than silently leaving one on-host copy. **Operator must set `BACKUP_REMOTE` and install the timer on each deployment.** *Not yet installed on the reference host (verified 2026-10-06).* | 8.5.4 | — | done |
| **G-06** | Auth off by default; WebSocket endpoints outside the auth middleware (`backend/app/main.py:157`); tenant isolation only enforced when auth is on; CORS `*` | 8.5.3 | Cross-tenant exposure in any deployment not perfectly network-isolated | 2–3 days |

### Priority 2 — close before a certification attempt

| Ref | Gap | Clause | Note | Effort |
|---|---|---|---|---|
| ~~G-07~~ | **CLOSED 2026-10-07** — QP-01 signed by the Managing Director; awareness recorded in `REG-05` | 5.1, 5.2, 7.3 | — | done |
| **G-21** | No management review | 9.3 | Major nonconformity at any audit | 2 hours + cadence |
| **G-20** | No internal audit; impartiality unachievable internally | 9.2 | Major nonconformity. Needs an external or second auditor. | 1–2 days/year, external |
| ~~G-01~~ | **CLOSED 2026-10-06** — release identification adopted with v1.4.0 (tag, changelog, build identity at runtime, image tags, `REG-08`, `FRM-03` records) | 8.5.2, 8.6 | — | — |
| **G-02** | *Partly closed 2026-10-07* — release records and CI exist; authorise before deployment and wire `verify_offline_readiness.sh` into the release | 8.6 | Authorisation given after deployment for v1.4.0/v1.4.1 | 0.5 day |
| **G-05** | No licence inventory, no LICENSE/NOTICE, no SBOM; a gated model is baked into shipped images | 8.4.2 | Legal exposure; regulated buyers will ask | 2–3 days |
| **G-04** | No data-retention policy; unbounded auto-store growth (97.6% of the corpus was auto-saved transcript at one measurement) | 8.5.3 | Storage cost, retrieval quality degradation, and no answer to "how long do you keep our data?" | 1–2 days |

### Priority 3 — close during the first QMS cycle

| Ref | Gap | Clause |
|---|---|---|
| ~~G-09~~ | **CLOSED 2026-10-07** — ratings confirmed, owners and dates set in `REG-03` | 6.1 |
| G-10 | Objectives lack current baselines — re-measure the evaluation against HEAD | 6.2, 9.1.1 |
| G-12 | Measurement not scheduled and no trend data yet. Retention of results is now in place. | 9.1.1, 9.1.3 |
| G-15 | No customer communication, requirements-review or satisfaction mechanism | 8.2, 9.1.2 |
| G-16 | *Partly closed 2026-10-07* — review log of 8 deliverable reviews with 40 actions (`REG-10`, EM26-18) and 31 non-functional requirements with targets (`REG-09`, EM26-17); requirement-to-test traceability still partial | 8.3 |
| ~~G-18~~ | **CLOSED** — `REG-07` change register and SOP-06 §6 significance criteria | 8.5.6 |
| G-19 | No nonconformity disposition/concession process; no customer-notification procedure; effectiveness verification inconsistent | 8.7, 10.2 |
| G-13 | *Partly closed 2026-10-07* — competence matrix in `REG-05`; training records to accumulate | 7.2 |
| G-08 | *Partly closed 2026-10-07* — roles named and held by separate people (QM-01 §7); engineering knowledge still concentrated in the Lead Engineer | 5.3, 7.1.6 |
| G-14 | External communication undefined | 7.4 |
| G-17 | No defined support process or escalation | 8.5.5 |

## 4. What is already strong

It would be a false picture to list only gaps. These are genuine strengths an auditor should be
shown, and several are unusual in a small software organisation:

| Strength | Evidence |
|---|---|
| **Documented information control is properly solved** | Git as the control mechanism — versioned, attributed, timestamped, replicated, tamper-evident. Better than most manual document systems. |
| **A real, version-controlled measuring instrument** | 52 golden questions across 7 domains in `eval/golden/`, a binary suite gate at `eval/run_eval.py:309`, citation-precision measurement |
| **Measurement honesty** | `eval/paper/results/SUMMARY.json` records findings that contradict the organisation's own published paper, and marks an experiment `"not_run"` rather than estimating it. This is the single best cultural evidence in the repository. |
| **Root-cause discipline** | Commit bodies routinely carry defect → root cause → fix → verification. `4e27109` verified 0/359 out-of-namespace hits after a tenant-isolation fix; `724fb98` verified a from-scratch rebuild of all six containers. |
| **Quality-by-design in the product itself** | Relevance gating, citation filtering, honest abstention, injection guards asserted by unit tests — the product's own design embodies the quality policy |
| **Self-recovering operation** | Health endpoints, CUDA-fault watchdog, restart policies on all six services |
| **Architecture documentation** | ~1,900 lines of flow and architecture documentation plus a 1,257-line user manual |

## 5. Honest bottom line

If an auditor arrived tomorrow, the outcome would be a **major nonconformity against clauses 9.2 and
9.3** (no audit, no management review) and a further major against **8.6** (no evidence of release
verification), plus a set of minors. That is the expected result for a system documented on day one,
and it is not a reflection on the engineering.

The realistic path is: adopt (ADOPTION_GUIDE.md §3), close Priority 1 on operational grounds, then
accumulate three to six months of genuine operating records. The records cannot be compressed —
which is precisely why they are worth something.
