# ISO 9001:2015 — Gap Analysis and Readiness Assessment

| Field | Value |
|---|---|
| Document ID | GAP-01 |
| Revision | 1.0 |
| Status | **Assessment — accurate as at 2026-09-21, HEAD `ff29843`** |
| Assessed by | Documentation review of the repository against ISO 9001:2015 clauses 4–10 |
| Method | Evidence-based: every finding cites a file, a command output or the demonstrable absence of an artefact |
| Owner | Managing Director |

---

## 1. Summary

**Readiness for certification audit today: not ready.** The management system is documented but not
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
| 4.1 | Context of the organisation | Partial | Determined and written in QM-01 §2, but not previously documented and not yet reviewed by management | — |
| 4.2 | Interested parties | Partial | QM-01 §3; never formally reviewed | — |
| 4.3 | Scope of the QMS | Partial | QM-01 §4 with a justified non-applicability for 7.1.5.2; awaiting approval | — |
| 4.4 | QMS and its processes | Partial | Six processes mapped in QM-01 §5 with inputs, outputs and measures; not yet operating | — |
| 5.1 | Leadership and commitment | Gap | No approved policy, no review, no resourcing decisions recorded. Cannot be evidenced until adoption. | G-07 |
| 5.2 | Quality policy | Partial | QP-01 written; **unsigned and uncommunicated** | G-07 |
| 5.3 | Roles and authorities | Partial | Defined in QM-01 §6.1; one person holds all of them | G-08 |
| 6.1 | Risks and opportunities | Partial | REG-03 seeded with 14 evidenced risks and 4 opportunities; **ratings proposed, not confirmed**; no treatment owners or dates | G-09 |
| 6.2 | Quality objectives | Partial | QO-01 with 7 objectives; only two have real baselines, both stale | G-10 |
| 6.3 | Planning of changes | Partial | SOP-02 §7; historically changes were planned in commit bodies, sometimes very well, but never against criteria | — |
| 7.1.1–7.1.4 | Resources, people, infrastructure, environment | Partial | Infrastructure well documented (compose, Dockerfiles, deployment docs); people resource is one person | G-08 |
| 7.1.5.1 | Monitoring and measuring resources | **Conformant** | The measuring instrument is the evaluation harness; `eval/golden/*.jsonl` and `eval/run_eval.py` are version-controlled, and results are now retained as records (2026-09-22, 24 historical reports committed) | ~~G-12~~ closed |
| 7.1.5.2 | Measurement traceability | Not applicable | No physical measuring equipment; justified in QM-01 §4.1 | — |
| 7.1.6 | Organisational knowledge | Partial | Unusually good written rationale in `docs/` and commit bodies — a real strength — but concentrated in one person | G-08 |
| 7.2 | Competence | Gap | No competence matrix, no records. REG-05 created and empty. | G-13 |
| 7.3 | Awareness | Gap | Policy not yet communicated to anyone | G-07 |
| 7.4 | Communication | Partial | Internal communication is a single person; external is undefined | G-14 |
| 7.5 | Documented information | **Conformant in mechanism** | SOP-01 uses Git as the control mechanism — versioned, attributed, timestamped, replicated. Strong. Weakness is that the documents it controls are unapproved. | — |
| 8.1 | Operational planning and control | Partial | Deployment and build are well controlled and documented; planning criteria are not written down | — |
| 8.2.1 | Customer communication | Gap | No feedback channel defined, no log. REG-06 created and empty. | G-15 |
| 8.2.2–8.2.3 | Determining and reviewing requirements | Gap | No requirements-review record exists for any engagement | G-15 |
| 8.2.4 | Changes to requirements | Gap | No mechanism | G-15 |
| 8.3.1–8.3.2 | Design planning | Partial | Design happens and is documented in `docs/`, but stages, reviews and acceptance criteria are not planned in advance | G-16 |
| 8.3.3 | Design inputs | Partial | Implicit in the code and prompts (grounding, abstention, isolation are clearly deliberate design inputs) but never written as inputs | G-16 |
| 8.3.4 | Design controls | **Partial — strong substance** | Verification genuinely happens: 78 unit test functions, the 52-question golden evaluation, the E1–E10 harness, and manual verification recorded in commit bodies. **No design review records exist.** | G-16 |
| 8.3.5 | Design outputs | Conformant | Source, architecture documents in `docs/`, `PROTOCOL.md`, user manual | — |
| 8.3.6 | Design changes | Partial | Every change is in Git with rationale; no impact assessment against requirements, because requirements are not recorded | G-16 |
| 8.4.1 | Control of external providers | Partial | Controlled in practice by pinning and offline pre-caching; **no selection or evaluation criteria**, no re-evaluation | G-05 |
| 8.4.2 | Type and extent of control | Partial | Strong for most packages (14/20 backend exact-pinned); **one dependency installed from a moving Git branch, which caused a real outage** (`724fb98`) | G-05 |
| 8.4.3 | Information for external providers | Not applicable in substance | No subcontracted work; providers are upstream | — |
| 8.5.1 | Control of production and service provision | Partial | Deployment procedures documented and repeatable (`OFFLINE_DEPLOYMENT.md`, `scripts/`); no release authorisation step | G-01 |
| 8.5.2 | **Identification and traceability** | **Gap** | No tags, no changelog, `package.json` is `0.0.0`, images untagged, no build identifier. A running instance cannot be traced to a source revision. | **G-01** |
| 8.5.3 | **Property belonging to customers** | **Partial — with a critical gap** | Isolation is designed and enforced (namespace predicate, tenant forcing) but has failed twice in the same way; **no backup of the only customer-data volume**; no retention policy; no encryption at rest; auth off by default and WebSockets outside the auth middleware | **G-03, G-04, G-06** |
| 8.5.4 | Preservation | **Partial** | Model volumes preserved by design; customer data now has a backup and restore procedure with a verified restore (2026-09-22) — outstanding: it is manual and on-host | ~~G-03~~ closed; see G-22 |
| 8.5.5 | Post-delivery activities | Partial | User manual and troubleshooting exist; no support process, SLA or escalation defined | G-17 |
| 8.5.6 | Control of changes | Partial | Git is the mechanism and commit discipline is good; no change register, no significance criteria, no impact assessment | G-18 |
| 8.6 | **Release of products and services** | **Gap** | No release records, no authorisation, no retained verification evidence. `scripts/verify_offline_readiness.sh` exists and is called by nothing. | **G-02** |
| 8.7 | Control of nonconforming outputs | Partial | Handled well in practice (22 `fix:` commits with root cause); no register until REG-04, no disposition or concession process, no customer-notification procedure | G-19 |
| 9.1.1 | Monitoring and measurement | Partial | Real measurement exists and is good; not scheduled, not retained, not trended | G-12 |
| 9.1.2 | Customer satisfaction | Gap | No method defined, no data | G-15 |
| 9.1.3 | Analysis and evaluation | Gap | No periodic analysis; the paper harness is the closest thing and it was a one-off campaign | G-12 |
| 9.2 | **Internal audit** | **Gap** | None conducted. Impartiality (9.2.2 c) cannot currently be met internally — one person. | **G-20** |
| 9.3 | **Management review** | **Gap** | None held | **G-21** |
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
| ~~G-11~~ | **CLOSED 2026-09-22** — bounded `json-file` logging (50 MB x 5) on all six services. Applies on the next `docker compose up -d`; running containers keep their original config until recreated. | 8.5.1 | — | done |
| ~~G-22~~ | **CLOSED 2026-09-22** — nightly systemd timer (`scripts/install_backup_timer.sh`) and off-host replication via `BACKUP_REMOTE`, tested; a failed off-host copy aborts the run rather than silently leaving one on-host copy. **Operator must set `BACKUP_REMOTE` and install the timer on each deployment.** | 8.5.4 | — | done |
| **G-06** | Auth off by default; WebSocket endpoints outside the auth middleware (`backend/app/main.py:135`); tenant isolation only enforced when auth is on; CORS `*` | 8.5.3 | Cross-tenant exposure in any deployment not perfectly network-isolated | 2–3 days |

### Priority 2 — close before a certification attempt

| Ref | Gap | Clause | Note | Effort |
|---|---|---|---|---|
| **G-07** | Policy unapproved and uncommunicated | 5.1, 5.2, 7.3 | One signature and one communication record | 1 hour |
| **G-21** | No management review | 9.3 | Major nonconformity at any audit | 2 hours + cadence |
| **G-20** | No internal audit; impartiality unachievable internally | 9.2 | Major nonconformity. Needs an external or second auditor. | 1–2 days/year, external |
| **G-01** | No release identification or traceability | 8.5.2, 8.6 | Cannot tell a customer what they are running | 1 day |
| **G-02** | No release records; verification not enforced or retained | 8.6 | Cannot evidence that anything was verified before release | 1 day + CI |
| **G-05** | No licence inventory, no LICENSE/NOTICE, no SBOM; a gated model is baked into shipped images | 8.4.2 | Legal exposure; regulated buyers will ask | 2–3 days |
| **G-04** | No data-retention policy; unbounded auto-store growth (97.6% of the corpus was auto-saved transcript at one measurement) | 8.5.3 | Storage cost, retrieval quality degradation, and no answer to "how long do you keep our data?" | 1–2 days |

### Priority 3 — close during the first QMS cycle

| Ref | Gap | Clause |
|---|---|---|
| G-09 | Risk ratings proposed but not confirmed; no treatment owners or dates | 6.1 |
| G-10 | Objectives lack current baselines — re-measure the evaluation against HEAD | 6.2, 9.1.1 |
| G-12 | Measurement not scheduled and no trend data yet. Retention of results is now in place. | 9.1.1, 9.1.3 |
| G-15 | No customer communication, requirements-review or satisfaction mechanism | 8.2, 9.1.2 |
| G-16 | No design review records, no recorded design inputs or acceptance criteria, no requirement-to-test traceability | 8.3 |
| G-18 | No change register or significance criteria | 8.5.6 |
| G-19 | No nonconformity disposition/concession process; no customer-notification procedure; effectiveness verification inconsistent | 8.7, 10.2 |
| G-13 | No competence records | 7.2 |
| G-08 | Bus factor 1; roles not separable; organisational knowledge concentrated | 5.3, 7.1.6 |
| G-14 | External communication undefined | 7.4 |
| G-17 | No defined support process or escalation | 8.5.5 |

## 4. What is already strong

It would be a false picture to list only gaps. These are genuine strengths an auditor should be
shown, and several are unusual in a small software organisation:

| Strength | Evidence |
|---|---|
| **Documented information control is properly solved** | Git as the control mechanism — versioned, attributed, timestamped, replicated, tamper-evident. Better than most manual document systems. |
| **A real, version-controlled measuring instrument** | 52 golden questions across 7 domains in `eval/golden/`, a binary suite gate at `eval/run_eval.py:281`, citation-precision measurement |
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
