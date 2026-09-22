# SOP-02 — Context, Interested Parties and Risk

| Field | Value |
|---|---|
| Document ID | SOP-02 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Managing Director |
| Approved by | `________________________` |
| Approval date | `____ / ____ / ________` |
| ISO 9001:2015 clauses | 4.1, 4.2, 6.1, 6.3 |

---

## 1. Purpose

To define how Ajace AI determines the issues and interested parties relevant to the quality
management system (QMS), how it identifies, assesses, treats and reviews the resulting risks and
opportunities, and how it plans changes to the QMS and to the product.

## 2. Scope

The whole QMS and the whole of the EchoMind Enterprise platform: the software in this repository,
the deployment pattern it is shipped under, and the customisation and advisory services Ajace AI
delivers around it. It does not cover the risk management that a customer performs on its own
infrastructure and its own data; where the two meet, the boundary is stated in §5 of this procedure.

## 3. Responsibilities

| Role | Responsibility |
|---|---|
| Managing Director | Owns this procedure. Determines context and interested parties. Accepts or rejects residual risk. Approves the risk register at each management review. |
| Lead Engineer | Identifies technical risks arising from the architecture, the dependency chain and the deployment model. Proposes and implements treatments. Records verification of a treatment in the commit that applies it. |
| Anyone contributing code | Raises a risk as soon as it is recognised, including risks found while fixing something else. |

Ajace AI currently has one person, who holds all three roles. This is itself a recorded risk
(§6 and `registers/REG-03_Risk_Register.md`).

## 4. Determining the context of the organisation (4.1)

The Managing Director determines the external and internal issues relevant to Ajace AI's purpose
and to its ability to achieve the intended results of the QMS. The determination is reviewed at
each management review under `SOP-15`, and immediately on any material change — a new regulated
sector, a change of hardware platform, a change in the legal treatment of AI systems, or the
addition of a second person to the organisation.

**External issues currently determined:**

| Issue | Why it matters to quality |
|---|---|
| Customers are in regulated sectors — defence, government, legal, finance, healthcare | Their own regulators constrain what they may deploy and where data may reside. A defect in tenant isolation or in grounding is a compliance event for the customer, not only a bug for us. |
| Data sovereignty is the reason the product exists | The platform is built to run fully offline on customer-controlled hardware. Any design decision that reintroduces an outbound dependency undermines the core value proposition. |
| The AI model supply chain moves quickly and without stability guarantees | Weights, inference runtimes and ASR toolkits change under us. `backend/Dockerfile:19` installs `nemo_toolkit[asr]` from `git+https://github.com/NVIDIA/NeMo.git@main` — a moving branch, not a release. |
| Hardware platform maturity | The reference platform is NVIDIA DGX Spark (GB10). Driver- and kernel-level defects have directly broken the stack; the rationale and the verification are recorded inline at `docker-compose.yml:7-11` and `docker-compose.yml:41-44`. |
| Ajace AI does not sell hardware | We advise on hardware and cloud fit by security posture. We depend on the customer procuring suitable hardware, and on our advice being sound. |
| Emerging AI-specific regulation and customer assurance expectations | Customers increasingly ask for a management system, not only a product. This QMS is the response. |

**Internal issues currently determined:**

| Issue | Why it matters to quality |
|---|---|
| One developer, two Git identities, one email | 202 commits from 2026-02-05 to `ff29843` (2026-09-21). Concentration of all knowledge and all capacity in one person. |
| Development is direct-to-main | There is no `.github/` directory, no CI configuration and no CODEOWNERS file in the repository. Only 2 of 202 commits are merges, both author-to-self on 2026-02-24. There is therefore no independent review gate. |
| A material share of development is AI-assisted and declared | Many substantive commits carry a `Co-Authored-By: Claude …` trailer (for example `558eaae`, `4e27109`). Verification of that output is a control, not an option — see `SOP-05` §7. |
| Strong written engineering rationale exists, in the wrong places for an auditor | Design reasoning lives in commit bodies and in inline comments (`docker-compose.yml:7-11`, `41-44`, `69-72`, `108-112`) rather than in decision records. |
| The product is pre-/early-commercial | Live at echomind-ajace.com behind Cloudflare Access (`docs/PUBLIC_DEPLOYMENT.md` step 4, marked MANDATORY). There is no customer base over which to spread risk, and no contract register. |

## 5. Interested parties and their requirements (4.2)

The Managing Director determines the interested parties relevant to the QMS and the requirements of
those parties that Ajace AI must monitor. The table below is the current determination. It is
reviewed at each management review.

| Interested party | What they require from Ajace AI | How the requirement is monitored |
|---|---|---|
| Customer organisations in defence, government, legal, finance and healthcare | Data never leaves their perimeter; answers grounded in their own material with attribution; tenant isolation between vertical packs; deployability on hardware they control; an auditable supplier | Requirements review under `SOP-04`; grounding and isolation measured by the golden-question harness (`eval/run_eval.py`) and by the paper harness (`eval/paper/`) |
| End users inside those organisations (analysts, clinicians, lawyers, meeting participants) | Answers that are correct or honestly absent; the system to say "I could not find that" rather than fill a gap; usable live transcription and voice; documentation they can follow | Abstention behaviour implemented at `backend/app/rag/advanced.py:74` and asserted by golden items of type `offcorpus` and `refusal` (`eval/README.md`); `docs/USER_MANUAL.md` (1257 lines) |
| Ajace AI itself — the Managing Director / Lead Engineer as owner and sole worker | Sustainable workload; the ability to be away without the product failing; knowledge retained outside one head | This procedure; `SOP-03` §6; the bus-factor risk in `registers/REG-03_Risk_Register.md` |
| NVIDIA — GPU hardware, driver, CUDA, TensorRT-LLM, NeMo ASR | Stable, released versions; compatibility of the driver, the runtime and the model kernels | `docker-compose.yml:5` (TensorRT-LLM image tag), `backend/Dockerfile:19` and `voice/Dockerfile:18` (NeMo from `@main`), `docker-compose.yml:186` (Parakeet-TDT final ASR) |
| Open-weight model providers (Qwen chat weights; `nomic-embed-text` embeddings; Piper TTS voices) | Licence terms that permit on-premises commercial deployment; model behaviour stable across versions | `docker-compose.yml:73-76`; licences to be recorded in `registers/REG-02_External_Documents_and_Providers.md` |
| Infrastructure and platform suppliers (Docker, Ollama, Cloudflare Tunnel and Access, Hugging Face as a weight distribution channel) | Availability where used; and, for an offline deployment, the ability to be removed from the runtime path entirely | `docker-compose.yml:223-227` (cloudflared, `public` profile only); `docker-compose.yml:170` (runtime Piper voice download from Hugging Face disabled for offline) |
| Open-source Python and JavaScript library maintainers | Correct attribution and licence compliance; our defect reports where we find defects | `registers/REG-02_External_Documents_and_Providers.md` |
| Conference and academic reviewers (QASC 2026) | Claims that are reproducible from recorded evidence; experiments reported as run only when run | `eval/paper/results/SUMMARY.json`, which records E2 as `not_run` with the reason, and records where measurements contradict the paper text |
| Regulators of the customers' sectors (indirect party) | That the product does not put the customer in breach — data residency, record-keeping, professional-advice boundaries | Requirements capture under `SOP-04` §4; the lawyer-persona disclaimer consistency test at `backend/tests/test_prompt_guards.py:29` |

Requirements of interested parties become inputs to design (`SOP-05` §4) and to the quality
objectives in `QO-01`.

## 6. Risk and opportunity: how they are identified (6.1.1)

Risks and opportunities are identified from five sources. The first three are continuous; the last
two are periodic.

1. **Engineering work.** Anything a contributor recognises while building, debugging or deploying.
   The historical record shows this is where most real risks have surfaced — for example the
   dependency drift recorded at `backend/Dockerfile:30-32`.
2. **Incidents.** Any production or build failure. Commit `724fb98` records a real outage of this
   kind: NeMo installed from `@main` pulled `setuptools>=82`, which removed `pkg_resources`,
   breaking `librosa` at build time and `webrtcvad` at runtime.
3. **Evaluation results.** A drop in the golden-question suite, or a paper-harness experiment that
   contradicts an assumption, is treated as a risk signal (`eval/paper/results/SUMMARY.json`
   records both an outright contradiction in E1 and a partial one in E3).
4. **Management review** under `SOP-15`, which reviews the whole register.
5. **Context review** under §4 and §5 of this procedure.

Opportunities are recorded in the same register and in the same way. An opportunity for this
organisation is typically a change that removes a class of risk rather than a commercial upside —
for example, moving from SQLite to Postgres removes the single-writer scaling wall described in
`docs/POSTGRES_MIGRATION.md` (whose header states "Status: planned, not executed").

## 7. Risk assessment scale

Each risk is scored on likelihood and impact. Score = likelihood × impact.

**Likelihood — the chance of the risk materialising within the next 12 months:**

| Level | Name | Meaning |
|---|---|---|
| 1 | Rare | No known occurrence; would require an unusual combination of events |
| 2 | Unlikely | Plausible but not expected |
| 3 | Possible | Expected roughly once in the period, or has happened once before |
| 4 | Likely | Expected more than once in the period |
| 5 | Almost certain | Occurring now, or certain to occur unless treated |

**Impact — the worst credible consequence if it materialises:**

| Level | Name | Meaning for EchoMind Enterprise |
|---|---|---|
| 1 | Negligible | Internal inconvenience; no customer visibility |
| 2 | Minor | Degraded feature; workaround exists; no data affected |
| 3 | Moderate | Feature unavailable to a customer, or a published claim shown to be unsupported |
| 4 | Major | Ungrounded or fabricated content reaches a user in a regulated sector; or a deployment is unavailable for a working day; or recoverable data loss |
| 5 | Severe | Customer data leaves the customer's perimeter, or crosses a tenant boundary, or is irrecoverably lost |

**Bands and required response:**

| Score | Band | Required response |
|---|---|---|
| 1–4 | Low | Accept and record. Review at management review. |
| 5–9 | Medium | Treatment planned with a named owning role and a target date. Review at management review. |
| 10–14 | High | Treatment planned and started before the next release that touches the affected area. Reviewed monthly by the Managing Director. |
| 15–25 | Critical | Work on new functionality in the affected area stops until the risk is reduced below 15, or the Managing Director records an explicit, dated, written acceptance of the residual risk with the reason. |

An impact of 5 may not be accepted without a written acceptance recorded in the register, whatever
the likelihood.

## 8. Risk treatment (6.1.2)

For each risk the register records: the risk, its source, the score before treatment, the treatment
option chosen, the owning role, the target date, the evidence that the treatment is in place, and
the score after treatment.

| Option | When it is chosen | Evidence it leaves |
|---|---|---|
| Avoid | The activity creating the risk is not necessary | A commit removing it, and a register entry stating so |
| Reduce | A control can be added | Code, configuration or procedure, cited by path in the register |
| Transfer / share | The risk properly sits with the customer or a supplier, and they can carry it | A statement in the customer requirements record (`SOP-04`) or in deployment documentation |
| Accept | Cost of treatment exceeds the exposure | Dated written acceptance by the Managing Director in the register |

**The treatment must be verifiable.** A register entry that says "we are careful about X" is not a
treatment. `backend/Dockerfile:32` — re-pinning `setuptools>=70,<82` as the last install step, with
the reason in the comment immediately above it — is a treatment, because an auditor can read the
file and see it.

Effectiveness of a treatment is evaluated at the next management review (`SOP-15`) against the
question: *has the risk recurred, and is the evidence cited in the register still true of the code
today?*

## 9. Planning of changes (6.3)

Changes to the QMS, and changes to the product that affect the QMS, are made deliberately and not
as a side effect of something else. Before a change is made, the person making it considers and
records — in the commit body, which for this organisation is the change record (`SOP-01` §3):

1. the purpose of the change and its potential consequences;
2. the integrity of the QMS or of the running system — what else depends on what is being changed;
3. availability of resources to complete the change, including the ability to finish it in one
   working session where the system would otherwise be left in a partial state;
4. allocation or reallocation of responsibilities, where there is more than one person.

Commit `4e27109` is the pattern to follow: it states the defect, the mechanism, the scope of the
blast radius (golden eval 50/52 → 42/52 as the corpus grew), the fix, and the post-fix verification
(0/359 out-of-namespace hits; golden eval 49/52).

Changes that alter a design output are additionally controlled under `SOP-05` §8.

## 10. Review

| Trigger | What is reviewed |
|---|---|
| Each management review (`SOP-15`) | Context (§4), interested parties (§5), the whole risk register, and the effectiveness of treatments |
| Any incident or nonconformity (`SOP-14`) | Whether the risk was in the register; if not, why it was not identified |
| Any new customer sector, new deployment model or new hardware platform | Context and interested parties, before commitment |
| Any release | Risks scored High or Critical that touch the released area |

## 11. Records

| Record | Location | Retention |
|---|---|---|
| Risk and opportunity register | `registers/REG-03_Risk_Register.md` | Life of the QMS + 3 years (SOP-01 §8) |
| Context and interested-party determination | §4 and §5 of this document; revision history in Git | As above |
| Written acceptance of a residual risk | `registers/REG-03_Risk_Register.md`, with the approver's name and date | As above |
| Review of context, parties and risk | Management review minutes under `SOP-15` | 3 years (SOP-01 §8) |
| Technical rationale and verification for a change | Git commit body (`git log`) | Per SOP-01 §8 |

## 12. Current state and gaps

**Stated plainly: risk management at Ajace AI has been practised informally and is genuinely
evidenced — but no risk register existed before this QMS.** The evidence of practice is real: the
inline rationale at `docker-compose.yml:7-11`, `41-44`, `69-72` and `108-112` records what was
tried, what failed, how the failure presented and how the fix was verified; `backend/Dockerfile:30-32`
records a dependency risk and its mitigation next to the mitigation itself; `eval/paper/results/SUMMARY.json`
records where measurement contradicted an assumption rather than hiding it. What was missing was a
single place where those risks are listed, scored, owned and reviewed. `registers/REG-03_Risk_Register.md`
is seeded from exactly this documented evidence and from nothing else.

| Gap | Current state | Consequence |
|---|---|---|
| No risk register before this QMS | `registers/` contained no register at the time of writing | Risks were treated but never aggregated, scored or reviewed as a set |
| No documented context or interested-party determination before this QMS | §4 and §5 are the first written determination | Requirements of interested parties were understood but not traceable |
| No management review has yet taken place | `SOP-15` is drafted; no minutes exist | No review cycle has run; the register has never been formally reviewed |
| No backup of the only volume holding customer data | `docker-compose.yml:101-102` mounts volume `echomind_data` at `/data`; `scripts/export_offline_bundle.sh` exports the Ollama and TensorRT-LLM model-cache volumes only and does not reference `echomind_data`; no other script in `scripts/` backs it up | Total, unrecoverable loss of customer data on volume loss. Impact 5. |
| Application authentication is off by default and does not cover WebSockets | `backend/app/core/config.py:230` (`AUTH_ENABLED` default `0`), `:235` (`CORS_ALLOW_ORIGINS` default `*`), `:236` (`RATE_LIMIT_PER_MIN` default `0`); `backend/app/main.py:135` states in the code that WebSocket endpoints are not gated | Public exposure depends entirely on Cloudflare Access (`docs/PUBLIC_DEPLOYMENT.md` step 4) |
| No encryption at rest | No encryption-at-rest mechanism is configured in `docker-compose.yml` for `echomind_data` | Physical or volume-level access exposes customer content |
| Unbounded container logs | `docker-compose.yml` sets no `logging:` options, so the default `json-file` driver applies with no `max-size` or `max-file` | Logs grow until the disk fills, taking the deployment down |
| Unbounded knowledge-base growth from auto-stored transcripts | `backend/app/core/config.py:95` sets `AUTO_STORE_INTERVAL_SEC` to 60 by default; `eval/paper/results/SUMMARY.json` (E2) records that at one point only 421 of 17,514 chunks were content, the remainder auto-saved transcripts | Retrieval quality degrades and evaluation becomes unrepresentative |
| Bus factor 1 | One developer (§4) | Every risk above has the same single owner and the same single point of failure |

Each gap above is carried into `registers/REG-03_Risk_Register.md` with a score, and into
`ISO9001_Gap_Analysis.md`.
