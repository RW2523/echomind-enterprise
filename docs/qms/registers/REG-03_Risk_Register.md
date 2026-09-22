# REG-03 — Risk and Opportunity Register

| Field | Value |
|---|---|
| Document ID | REG-03 |
| Revision | 1.0 |
| Status | **LIVE REGISTER — ratings are PROPOSED and require management confirmation** |
| Owner | Managing Director |
| ISO 9001:2015 clauses | 6.1.1, 6.1.2, 9.3.2 e), 10.2.1 e) |
| Governing procedure | SOP-02 |

---

## How this register was populated

Every risk below is **derived from evidence already in the repository** — code, configuration
comments, commit history or evaluation results — not from a generic risk checklist. The evidence
column cites where each came from, so any entry can be checked.

**The likelihood and impact scores are a proposal.** They are one person's reading of the evidence
and they have not been agreed by management. They are recorded so that the first management review
has something concrete to challenge rather than a blank page. Confirm or change them at that review
and set Revision 1.1.

### Scale

| Likelihood | | Impact | |
|---|---|---|---|
| 1 | Rare — no plausible path currently | 1 | Negligible — cosmetic |
| 2 | Unlikely — would need an unusual combination | 2 | Minor — inconvenience, easily corrected |
| 3 | Possible — could happen in normal operation | 3 | Moderate — a module degraded, effort to correct |
| 4 | Likely — expected within a year at current practice | 4 | Major — customer-visible failure, data at risk, or a false assertion reaching a user |
| 5 | Almost certain — has already happened, or will without action | 5 | Severe — customer data lost or exposed, cross-tenant leak, or loss of customer trust in grounding |

**Rating = Likelihood × Impact.** 1–4 Low · 5–9 Medium · 10–14 High · 15–25 Critical.

---

## Risks

### R-01 — Production dependency installed from a moving Git branch

| | |
|---|---|
| Category | External provider / supply chain |
| Description | `nemo_toolkit[asr]` is installed as `git+https://github.com/NVIDIA/NeMo.git@main` in `backend/Dockerfile:22` and `voice/Dockerfile`. A branch is not a release: every rebuild can pull different code, and the resulting image is not reproducible. |
| Evidence | The risk has **already materialised**: commit `724fb98` (2026-09-04) records NeMo@main pulling `setuptools>=82`, which removed `pkg_resources`, breaking librosa at build (backend) and webrtcvad at runtime (voice crash-loop on boot). |
| Likelihood / Impact / Rating | 5 / 4 / **20 — Critical** (likelihood 5 because it has already occurred) — **residual after pinning: 1 / 4 / 4 Low** |
| Existing controls | **Pinned to commit `60ce9407ef60a3327ffaf2c15931b9b3b834afc2` in both Dockerfiles (2026-09-22)** — the exact build verified running in production (NeMo 3.1.0+60ce9407e). `setuptools>=70,<82` re-pin retained as defence in depth. |
| Treatment | **Treated — cause removed.** The build is now reproducible. Residual: review the pin when a NeMo upgrade is wanted, and re-verify before adopting it. |
| Owner / due | `________` / `________` |
| Residual (after treatment) | 2 / 4 / 8 — Medium |
| Linked | NC-2026-006; QO-6; SOP-09 |

### R-02 — No backup of the only volume containing customer data

| | |
|---|---|
| Category | Customer property / data |
| Description | Docker volume `echomind_data` holds the SQLite database, the FAISS indexes, uploaded source files, boardroom audio and the generated auth secret. There is **no backup script and no documented backup procedure**. `scripts/export_offline_bundle.sh` exports the two *reproducible* model volumes and deliberately skips this one. `docs/USER_MANUAL.md:1158` correctly identifies `echomind_data` as the operator's real backup priority but gives no procedure. |
| Evidence | Repository contains no backup tooling; the bundle script's contents; the user manual statement. |
| Likelihood / Impact / Rating | 3 / 5 / **15 — Critical** |
| Existing controls | **`scripts/backup_data.sh` / `restore_data.sh` (2026-09-22).** Consistent SQLite snapshot via the online backup API; reproducible model caches excluded (351 MB rather than 23 GB); checksummed; archive verified readable. **Restore exercised 2026-09-22** into a throwaway volume — `integrity_check` ok, 32 tables. |
| Treatment | **Treated (2026-09-22).** `install_backup_timer.sh` installs a nightly systemd timer; `BACKUP_REMOTE` replicates off-host by rsync/scp or to a mounted share, and a failed replication aborts loudly rather than leaving a single on-host copy. Residual: the operator must set `BACKUP_REMOTE` and confirm the destination. |
| Owner / due | `________` / `________` |
| Residual | 2 / 3 / 6 — Medium |
| Linked | QO-4 (highest-priority objective); SOP-11 §8 |

### R-03 — Single person holds the organisation's operating knowledge

| | |
|---|---|
| Category | Resource / organisational knowledge |
| Description | 202 commits from a single author identity and email between 2026-02-05 and 2026-09-21. Design rationale lives in that person's head, in commit bodies and in `docs/`. Unavailability of that person halts delivery, support and incident response. It also makes impartial internal audit impossible (ISO 9001:2015 9.2.2 c). |
| Evidence | `git shortlog -sne`; SOP-13 §4. |
| Likelihood / Impact / Rating | 3 / 5 / **15 — Critical** |
| Existing controls | Unusually good written rationale in commit bodies and `docs/` — a genuine partial mitigation. |
| Treatment | **Reduce.** Maintain the documentation discipline deliberately rather than incidentally; identify a second competent person, at minimum on contract, for audit independence and continuity. |
| Owner / due | `________` / `________` |
| Residual | 3 / 3 / 9 — Medium |
| Linked | SOP-03 §7; SOP-13 §4; QO-7 |

### R-04 — Application authentication is off by default and does not cover WebSockets

| | |
|---|---|
| Category | Customer data / access control |
| Description | `AUTH_ENABLED` defaults to `0` (`docker-compose.yml:79`, `backend/app/core/config.py:231`). The auth middleware is HTTP-only — the code states it plainly at `backend/app/main.py:135`: *"WebSocket endpoints are not gated here yet."* `CORS_ALLOW_ORIGINS` defaults to `*`; `RATE_LIMIT_PER_MIN` defaults to `0` (off). Critically, the tenant-isolation boundary in `_effective_ns()` (`backend/app/api/routes/chat.py:25-33`) **only applies when auth is enabled** — with auth off, the knowledge-base namespace is whatever the client sends, unverified. |
| Evidence | The files cited; `docs/PUBLIC_DEPLOYMENT.md` step 4 titled "Put a login wall in front (MANDATORY)". |
| Likelihood / Impact / Rating | 3 / 5 / **15 — Critical** (as deployed on a trusted LAN the likelihood is lower; the rating reflects the default configuration) |
| Existing controls | Cloudflare Access in front of the public instance; the auth implementation itself is sound (PBKDF2-HMAC-SHA256, 200,000 iterations, constant-time verify). The deployment documentation is explicit that gating is mandatory. |
| Treatment | **Reduce.** Extend the auth middleware to WebSocket endpoints; decide whether `AUTH_ENABLED=1` should be the default for customer deployments; narrow the default CORS origin. |
| Owner / due | `________` / `________` |
| Residual | 2 / 4 / 8 — Medium |
| Linked | SOP-11 §5; NC-2026-003 |

### R-05 — Tenant isolation has failed twice in the same way

| | |
|---|---|
| Category | Customer data / design |
| Description | Namespace isolation is enforced by a predicate applied inside each retrieval path rather than structurally. Two separate paths have bypassed it. |
| Evidence | NC-2026-001 (`4e27109`, sparse transcript path, 10/10 out-of-namespace chunks returned) and NC-2026-002 (`ebd232f`, keyword-grep fallback path). |
| Likelihood / Impact / Rating | 3 / 5 / **15 — Critical** — a recurrence of a defect class is evidence the cause was not removed |
| Existing controls | Post-fix audit of all five retrieval paths: 0/359 out-of-namespace hits. Candidate-pool widening so namespace filtering does not starve small tenants. |
| Treatment | **Reduce.** Make the filter structural so a newly added retrieval path cannot bypass it (a single enforced entry point rather than a convention). |
| Owner / due | `________` / `________` |
| Residual | 2 / 5 / 10 — High |
| Linked | REG-04 observation O-2 |

### R-06 — Unbounded data growth with no retention policy

| | |
|---|---|
| Category | Customer data / capacity |
| Description | `AUTO_STORE_INTERVAL_SEC=60` auto-stores live transcript content into the knowledge base every 60 seconds while recording. There is no retention policy, no scheduled deletion and no pruning of the `activity_log` table. Beyond storage cost, accumulated transcript dilutes retrieval quality. |
| Evidence | `eval/paper/results/SUMMARY.json` (E2) records that at one point only **421 of 17,514 chunks were content** — roughly 97.6% of the corpus was auto-accumulated transcript. NC-2026-001 became observable *because* of this growth. |
| Likelihood / Impact / Rating | 4 / 3 / **12 — High** |
| Existing controls | Six manual delete endpoints exist. |
| Treatment | **Reduce.** Define a retention policy per data class; implement scheduled deletion; prune `activity_log`. |
| Owner / due | `________` / `________` |
| Residual | 2 / 2 / 4 — Low |
| Linked | SOP-11 §7 |

### R-07 — Container logs grow without limit

| | |
|---|---|
| Category | Operations |
| Description | `docker-compose.yml` sets no `logging:` driver or options on any service, so Docker's default `json-file` driver applies with no `max-size` or `max-file`. Logs grow until the disk fills, which would take down every service including the database. |
| Evidence | Absence of any `logging:` block in `docker-compose.yml`. |
| Likelihood / Impact / Rating | 4 / 4 / **16 — Critical** on a long-running deployment |
| Existing controls | **Bounded logging (50 MB x 5) on all six services (2026-09-22).** Takes effect on the next `docker compose up -d`; running containers keep their original config until recreated. |
| Treatment | **Reduced.** Residual: verify after the next recreate that the running containers carry the new LogConfig. |
| Owner / due | `________` / `________` |
| Residual | 1 / 4 / 4 — Low |
| Linked | SOP-12 §6 |

### R-08 — No verification is automatically enforced

| | |
|---|---|
| Category | Design and development |
| Description | There is no CI of any kind (`.github/` does not exist), no pre-commit, no linting, no type checking in the build (`vite build` runs without `tsc`, and `tsconfig.json` does not set `strict`). Every verification step depends on a person remembering to run it. |
| Evidence | Absence of the above; `frontend/package.json:8`. |
| Likelihood / Impact / Rating | 4 / 3 / **12 — High** |
| Existing controls | A strong manual verification habit evidenced in commit bodies; the golden evaluation and unit suites exist and are run — just not automatically. |
| Treatment | **Reduce.** Add CI running the unit suites at minimum; wire in `scripts/verify_offline_readiness.sh`, which already exists and is currently called by nothing. |
| Owner / due | `________` / `________` |
| Residual | 2 / 3 / 6 — Medium |
| Linked | QO-2; SOP-07 |

### R-09 — A running instance cannot be traced to its source

| | |
|---|---|
| Category | Configuration control |
| Description | No git tags, no changelog, `frontend/package.json` is `"version": "0.0.0"`, backend/voice/frontend images are untagged, and no endpoint reports a build identifier. If a customer reports a defect, the exact code they are running cannot be established from the running system. |
| Evidence | `git tag` is empty; the files cited. |
| Likelihood / Impact / Rating | 4 / 3 / **12 — High** |
| Existing controls | None. |
| Treatment | **Reduce.** Adopt the release-identification scheme in SOP-08 §6. |
| Owner / due | `________` / `________` |
| Residual | 1 / 3 / 3 — Low |
| Linked | QO-3; ISO 9001:2015 8.5.2 |

### R-10 — Model and dependency licence obligations are unknown

| | |
|---|---|
| Category | Legal / external provider |
| Description | There is no `LICENSE` or `NOTICE` file, no model-licence inventory, no SBOM and no third-party attribution document. `README.md` asserts the models are "all open-weight", but no licence evidence supports the claim in-repo, and the Nemotron STT model is an acceptance-gated model. The product is sold commercially into regulated sectors whose customers will ask. |
| Evidence | Absence of the files; `README.md`. |
| Likelihood / Impact / Rating | 3 / 4 / **12 — High** |
| Existing controls | None. |
| Treatment | **Reduce.** Complete `REG-02`, add a `NOTICE` file, generate an SBOM. |
| Owner / due | `________` / `________` |
| Residual | 1 / 4 / 4 — Low |
| Linked | QO-6; SOP-09 §7 |

### R-11 — Reranker model cannot be fetched at run time and fails over silently

| | |
|---|---|
| Category | Design / external provider |
| Description | `cross-encoder/ms-marco-MiniLM-L-6-v2` is documented in `backend/requirements.txt` as needing pre-download into the image, but `backend/Dockerfile` contains no such step. At run time `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so it cannot be fetched, and the code falls back to a slower LLM rerank (`backend/app/rag/reranker.py:6-9`). Retrieval quality and latency then differ from the tested configuration without any signal. |
| Evidence | The files cited. |
| Likelihood / Impact / Rating | 4 / 3 / **12 — High** |
| Existing controls | A functioning fallback path — which is what makes the failure silent. |
| Treatment | **Reduce.** Pre-download the model in the Dockerfile; log a WARNING when the fallback engages. |
| Owner / due | `________` / `________` |
| Residual | 1 / 2 / 2 — Low |
| Linked | REG-04 observation O-3 (silent failure class) |

### R-12 — SQLite single-writer contention is the known scaling limit

| | |
|---|---|
| Category | Architecture / capacity |
| Description | All persistence is a single SQLite database. `docs/POSTGRES_MIGRATION.md` records the analysis and states *"Status: planned, not executed"*, with the driver that the `activity_log` and chat writes now contend with RAG writes. |
| Evidence | `docs/POSTGRES_MIGRATION.md`. |
| Likelihood / Impact / Rating | 3 / 3 / **9 — Medium** (rises with concurrent users per deployment) |
| Existing controls | The migration runbook is written. |
| Treatment | **Accept for now, monitor.** Define the concurrency threshold that triggers execution of the migration. |
| Owner / due | `________` / `________` |
| Residual | 3 / 3 / 9 — Medium |

### R-13 — No encryption at rest

| | |
|---|---|
| Category | Customer data |
| Description | SQLite, uploaded files and FAISS indexes are all stored unencrypted in the `echomind_data` volume. The indexes are derived from, and can reveal, customer source text. |
| Evidence | No encryption anywhere in the storage path. |
| Likelihood / Impact / Rating | 2 / 5 / **10 — High** (the customer controls the host, which is the principal mitigation) |
| Existing controls | On-premises deployment on customer-controlled hardware; the customer's own disk encryption if any. |
| Treatment | **Reduce / transfer.** State the position explicitly to customers so their own controls can cover it; evaluate volume-level encryption as a deployment option. |
| Owner / due | `________` / `________` |
| Residual | 2 / 3 / 6 — Medium |
| Linked | SOP-11 §6 |

### R-14 — Accelerator platform immaturity

| | |
|---|---|
| Category | External / infrastructure |
| Description | The GB10 / Grace-Blackwell software stack is new, and real defects have originated below the application. |
| Evidence | Extensive inline rationale in `docker-compose.yml:7-11` (worker-spawn CUDA bug and rollback), `:41-44` (`gpus: all` versus the `deploy.resources` form breaking NVML — "Verified: same image works with --gpus all, fails via the deploy form"), `:86-89`, `:164-166` (onnxruntime CUDA execution provider unavailable on ARM); plus a CUDA-graph decoder conflict diagnosed and worked around in the voice service. |
| Likelihood / Impact / Rating | 4 / 3 / **12 — High** |
| Existing controls | Health endpoints returning 503 on a fatal CUDA fault, an in-process watchdog (`backend/app/main.py:100-104`) and `restart: unless-stopped` — an effective self-recovery design. Workarounds and their rollbacks are documented inline. |
| Treatment | **Accept with controls.** Keep the inline rationale discipline; move these known issues into this register so they have owners and review dates rather than living only in configuration comments. |
| Owner / due | `________` / `________` |
| Residual | 4 / 2 / 8 — Medium |

---

## Opportunities (6.1.1)

| Ref | Opportunity | Rationale | Action |
|---|---|---|---|
| **O-01** | Turn the existing evaluation harness into a certified-quality differentiator | Few AI vendors can show a version-controlled question set, a binary pass gate and results that contradict their own marketing. In regulated sectors that is a sales asset, not just an engineering one. | Retain evaluation reports as records; publish the methodology to customers |
| **O-02** | Use the QMS as a procurement enabler | Regulated buyers increasingly require supplier quality evidence. An operating ISO 9001 system shortens procurement. | Complete adoption (ADOPTION_GUIDE.md) |
| **O-03** | The honest-abstention behaviour is a compliance feature | "I could not find that" is what a regulated user needs. Market it as such and measure it. | Add abstention correctness to the evaluation set |
| **O-04** | Pinning and SBOM work also unlocks air-gapped defence procurement | Those buyers require a bill of materials. | Bundle with R-10 treatment |

---

## Review

| Review date | Reviewed by | Changes |
|---|---|---|
| `________` | `________` | Initial confirmation of proposed ratings — **due at the first management review** |
