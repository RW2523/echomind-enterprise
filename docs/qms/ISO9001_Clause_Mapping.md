# ISO 9001:2015 — Clause Mapping

| Field | Value |
|---|---|
| Document ID | MAP-01 |
| Revision | 1.0 |
| Status | Reference document — accurate as at 2026-09-21, HEAD `ff29843` |
| Owner | Managing Director |
| Purpose | For an auditor: where each requirement is addressed, and what objective evidence exists in this repository |

---

## How to use this

Three columns matter to an auditor and they are kept separate on purpose:

- **QMS document** — where the requirement is *specified*.
- **Objective evidence in the repository** — what actually exists that can be inspected. This is the
  column that decides an audit.
- **State** — `Conformant` / `Partial` / `Gap` / `N/A`, consistent with `ISO9001_Gap_Analysis.md`.

Where the evidence column says *none*, that is the finding. Nothing is claimed that cannot be opened
and read.

---

## Clause 4 — Context of the organisation

| Clause | Requirement | QMS document | Objective evidence in this repository | State |
|---|---|---|---|---|
| 4.1 | Understanding the organisation and its context | QM-01 §2; SOP-02 §4 | `README.md` (market and deployment posture); `docker-compose.yml` inline rationale on GB10 platform constraints (`:7-11`, `:41-44`, `:86-89`, `:164-166`) | Partial |
| 4.2 | Interested parties and their requirements | QM-01 §3; SOP-02 §5 | `docs/PUBLIC_DEPLOYMENT.md`; `REG-02` provider inventory | Partial |
| 4.3 | Determining the scope of the QMS | QM-01 §4 | Scope statement with a justified non-applicability for 7.1.5.2 | Partial |
| 4.4 | QMS and its processes | QM-01 §5 | Six-process map with inputs, outputs and measures | Partial |

## Clause 5 — Leadership

| Clause | Requirement | QMS document | Objective evidence | State |
|---|---|---|---|---|
| 5.1.1 | Leadership and commitment | QM-01 §6 | None yet — no approvals, no review, no resourcing record | Gap |
| 5.1.2 | Customer focus | QP-01 §2.1–2.3 | Product design embodies it: relevance gating, citation filtering, honest abstention (`backend/app/rag/advanced.py` insufficient-context path); injection guards asserted by `backend/tests/test_prompt_guards.py` | Partial |
| 5.2.1 | Establishing the quality policy | QP-01 | Written, **unsigned** | Partial |
| 5.2.2 | Communicating the quality policy | QP-01 §4; SOP-03 §6 | None — not yet communicated | Gap |
| 5.3 | Roles, responsibilities and authorities | QM-01 §6.1 | Roles defined; one person holds all of them | Partial |

## Clause 6 — Planning

| Clause | Requirement | QMS document | Objective evidence | State |
|---|---|---|---|---|
| 6.1.1–6.1.2 | Actions to address risks and opportunities | SOP-02 §6; `REG-03` | `REG-03` seeded with 14 risks and 4 opportunities, each citing repository evidence; commit `724fb98` is a realised supplier risk; ratings **proposed, not confirmed** | Partial |
| 6.2.1–6.2.2 | Quality objectives and planning to achieve them | QO-01 | 7 objectives with measures and targets; real baselines exist for two (evaluation 49/52 and citation precision 0.98, both 2026-08-06 and stale) | Partial |
| 6.3 | Planning of changes | SOP-02 §7; SOP-06 | Change rationale recorded in commit bodies; no criteria or register | Partial |

## Clause 7 — Support

| Clause | Requirement | QMS document | Objective evidence | State |
|---|---|---|---|---|
| 7.1.1–7.1.2 | Resources, people | QM-01 §8; SOP-03 | One person; no competence record | Partial |
| 7.1.3 | Infrastructure | SOP-08 | `docker-compose.yml` (6 services, healthchecks on 4, restart policies on all); `docker/` Dockerfiles; `scripts/` build, prepare, export/import | Conformant |
| 7.1.4 | Environment for operation | SOP-08 | Offline/air-gapped operation documented in `OFFLINE_DEPLOYMENT.md` | Conformant |
| 7.1.5.1 | Monitoring and measuring resources | SOP-12 §5 | `eval/golden/*.jsonl` (52 questions, version-controlled) and `eval/run_eval.py` — the measuring instrument is controlled; **results are not retained** (`eval/.gitignore` excludes `reports/`) | Partial |
| 7.1.5.2 | Measurement traceability | QM-01 §4.1 | No physical measuring equipment — non-applicability justified | N/A |
| 7.1.6 | Organisational knowledge | SOP-03 §7 | `docs/` architecture set (~1,900 lines); `docs/USER_MANUAL.md` (1,257 lines); commit bodies carrying rationale | Partial |
| 7.2 | Competence | SOP-03; `REG-05` | None — register empty | Gap |
| 7.3 | Awareness | SOP-03 §6 | None | Gap |
| 7.4 | Communication | SOP-04 §6 | Internal: single person. External: undefined | Partial |
| 7.5.1–7.5.2 | Documented information: creation and update | SOP-01 §4–6 | This document set; `FRM-00` template | Conformant |
| 7.5.3.1–7.5.3.2 | Control of documented information | SOP-01 §3 | **Git is the control mechanism** — versioned, attributed, timestamped, replicated. `git log --follow -- <path>` is the revision record for any document. External documents identified in `REG-02` §6. | Conformant |

## Clause 8 — Operation

| Clause | Requirement | QMS document | Objective evidence | State |
|---|---|---|---|---|
| 8.1 | Operational planning and control | QM-01 §5; SOP-06 | Build and deploy are repeatable and documented (`scripts/build.sh`, `scripts/prepare_offline.sh`, `OFFLINE_DEPLOYMENT.md`) | Partial |
| 8.2.1 | Customer communication | SOP-04 §6; `REG-06` | None — register empty | Gap |
| 8.2.2 | Determining requirements | SOP-04 §4 | None recorded | Gap |
| 8.2.3 | Review of requirements | SOP-04 §5 | None recorded | Gap |
| 8.2.4 | Changes to requirements | SOP-04 §5.4 | None | Gap |
| 8.3.1 | Design and development — general | SOP-05 §4 | Design happens and is documented; stages not planned in advance | Partial |
| 8.3.2 | Design planning | SOP-05 §5 | None recorded | Gap |
| 8.3.3 | Design inputs | SOP-05 §6 | Implicit and clearly deliberate (grounding, abstention, tenant isolation) but never recorded as inputs | Partial |
| 8.3.4 | Design controls (review, verification, validation) | SOP-05 §7; SOP-07 | **Strong in substance:** 78 unit test functions (`backend/tests/` 40, `voice/tests/` 38); 52-question golden evaluation with a binary gate (`eval/run_eval.py:281`); the E1–E10 harness in `eval/paper/`; manual verification recorded in commit bodies. **No design review records.** | Partial |
| 8.3.5 | Design outputs | SOP-05 §8 | Source; `docs/RAG_FLOW.md`, `docs/CHAT_AND_RAG_FLOW.md`, `docs/CONVERSATION_AI_AND_WAKE_WORD_FLOW.md`, `docs/TRANSCRIPT_STORAGE_FLOW.md`, `docs/CAPABILITIES.md`; `backend/app/transcribe/PROTOCOL.md`; `docs/USER_MANUAL.md` | Conformant |
| 8.3.6 | Design changes | SOP-05 §9; SOP-06 | Complete change history in Git with rationale; no impact assessment against recorded requirements | Partial |
| 8.4.1 | Control of externally provided items | SOP-09 §4; `REG-02` | Pinning: backend 14/20 exact; offline pre-caching of all models into volumes/images; **`nemo_toolkit[asr]` from a moving branch** (`backend/Dockerfile:19`, `voice/Dockerfile:18`) | Partial |
| 8.4.2 | Type and extent of control | SOP-09 §5 | `setuptools>=70,<82` corrective re-pin (`backend/Dockerfile:32`, `voice/Dockerfile:79`) following the `724fb98` outage; **no licence inventory, no LICENSE/NOTICE, no SBOM** | Partial |
| 8.4.3 | Information for external providers | SOP-09 §6 | No subcontracted work — providers are upstream only | N/A |
| 8.5.1 | Control of provision | SOP-08 §4 | `OFFLINE_DEPLOYMENT.md`; `scripts/export_offline_bundle.sh` / `import_offline_bundle.sh`; healthchecks on trtllm, backend, voice, ollama | Partial |
| 8.5.2 | **Identification and traceability** | SOP-06 §7 | **None.** No git tags; `frontend/package.json` `"version": "0.0.0"`; backend/voice/frontend images untagged; no build identifier surfaced by any endpoint | **Gap** |
| 8.5.3 | **Property belonging to customers** | SOP-11 | Isolation designed and enforced: `_ns_ok` predicate (`backend/app/rag/index.py:36`), tenant forcing (`backend/app/api/routes/chat.py:25-33`), applied at every retrieval path after the `4e27109` audit (0/359 out-of-namespace hits). **But:** no backup of `echomind_data`; no retention policy; no encryption at rest; auth off by default; WebSockets outside the auth middleware (`backend/app/main.py:135`) | **Partial — critical gaps** |
| 8.5.4 | Preservation | SOP-08 §9; SOP-11 §8 | Model volumes preserved across rebuilds (verified in `724fb98`); **customer data has no backup procedure** | Gap |
| 8.5.5 | Post-delivery activities | SOP-08 §10 | `docs/USER_MANUAL.md` §13 troubleshooting; `OFFLINE_DEPLOYMENT.md:129-160` five named failure modes; no support process or escalation defined | Partial |
| 8.5.6 | Control of changes | SOP-06 §5 | Git history; no change register or significance criteria | Partial |
| 8.6 | **Release of products and services** | SOP-07 §7; SOP-08 §5 | **No release records, no authorisation step, no retained verification evidence.** `scripts/verify_offline_readiness.sh` exists but is invoked by nothing. | **Gap** |
| 8.7.1–8.7.2 | Control of nonconforming outputs | SOP-10; `REG-04` | 22 `fix:` commits of 202, several with full root-cause analysis; `eval/paper/results/REGRESSION_AND_FIXES.json`; `REG-04` seeded with 7 retrospective entries. No disposition/concession process, no customer-notification procedure. | Partial |

## Clause 9 — Performance evaluation

| Clause | Requirement | QMS document | Objective evidence | State |
|---|---|---|---|---|
| 9.1.1 | Monitoring, measurement, analysis, evaluation | SOP-12 §4 | Golden evaluation (best 50/52 on 2026-07-30; most recent 49/52 on 2026-08-06); citation precision 0.98; unit suites; per-turn voice latency logging (`voice/app/session.py:46`); healthchecks; `activity_log` table. **Not scheduled, not retained, not trended.** | Partial |
| 9.1.2 | Customer satisfaction | SOP-04 §7; `REG-06` | None | Gap |
| 9.1.3 | Analysis and evaluation | SOP-12 §7 | `eval/paper/results/SUMMARY.json` — a genuine analysis campaign, including findings that contradict the organisation's own published paper and one experiment marked `"not_run"` rather than estimated. One-off, not periodic. | Partial |
| 9.2.1–9.2.2 | **Internal audit** | SOP-13 | **None conducted.** Impartiality (9.2.2 c) cannot be met internally with one person — recorded as an open nonconformity in SOP-13 §4. | **Gap** |
| 9.3.1–9.3.3 | **Management review** | SOP-15 | **None held.** | **Gap** |

## Clause 10 — Improvement

| Clause | Requirement | QMS document | Objective evidence | State |
|---|---|---|---|---|
| 10.1 | Improvement — general | SOP-14 §9 | Commit `2decb99` disabled a feature by default because it measured worse — improvement driven by measurement rather than opinion | Partial |
| 10.2.1 | Reacting to nonconformity | SOP-10 §5; SOP-14 §4 | Practised: `4e27109`, `558eaae`, `724fb98`, `ebd232f`, `73f0b4f` | Partial |
| 10.2.2 | Retaining documented information on nonconformities | SOP-14 §8; `REG-04` | Commit bodies and `eval/paper/results/REGRESSION_AND_FIXES.json`; `REG-04` now formalises it. **Of 6 retrospective records, 3 verify effectiveness and 3 do not.** | Partial |
| 10.3 | Continual improvement | SOP-15 §7 | Depends on management review, which has not happened | Gap |

---

## Mandatory documented information — checklist

ISO 9001:2015 names specific documented information. This is what an auditor will ask for by name.

### Documents required to be maintained

| Requirement | Clause | Where | Present |
|---|---|---|---|
| Scope of the QMS | 4.3 | QM-01 §4 | ☑ (unapproved) |
| Quality policy | 5.2.2 | QP-01 | ☑ (unsigned) |
| Quality objectives | 6.2.1 | QO-01 | ☑ (unapproved) |
| Information to support process operation | 4.4.2 a) | SOP-01…SOP-15 | ☑ (unapproved) |

### Records required to be retained

| Requirement | Clause | Where | Present |
|---|---|---|---|
| Fitness of monitoring and measuring resources | 7.1.5.1 | `eval/golden/`, `eval/run_eval.py` version-controlled | ☑ instrument / ☐ results not retained |
| Competence evidence | 7.2 d) | `REG-05` | ☐ empty |
| Evidence of conformity to product requirements / release | 8.6 | — | ☐ **none** |
| Results of requirements review | 8.2.3.2 | — | ☐ none |
| Design and development records (inputs, controls, outputs, changes) | 8.3.3–8.3.6 | Commit history; `docs/`; test and evaluation results | ◩ partial — no review records |
| Evaluation and re-evaluation of external providers | 8.4.1 | `REG-02` | ◩ inventory present, evaluation absent |
| Traceability | 8.5.2 | — | ☐ **none** |
| Customer property lost/damaged/unsuitable | 8.5.3 | `REG-04`, SOP-11 §9 | ☑ mechanism, no occurrences recorded |
| Change control records | 8.5.6 | Git history | ◩ mechanism, no register |
| Nonconforming output and actions taken | 8.7.2 | `REG-04` | ☑ 7 retrospective entries |
| Monitoring and measurement results | 9.1.1 | `eval/paper/results/`; evaluation reports **untracked** | ◩ partial |
| Internal audit programme and results | 9.2.2 | `REG-01`, SOP-13 | ☐ **none** |
| Management review results | 9.3.3 | — | ☐ **none** |
| Nonconformity and corrective action results | 10.2.2 | `REG-04` | ☑ partial (3 of 6 verify effectiveness) |

**Legend:** ☑ present · ◩ partially present · ☐ absent
