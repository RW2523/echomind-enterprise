# SOP-06 — Configuration and Change Management

| Field | Value |
|---|---|
| Document ID | SOP-06 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Engineering Lead |
| Approved by | `________________________` |
| Approval date | `____ / ____ / ________` |
| ISO 9001:2015 clauses | 8.1, 8.5.2, 8.5.6, 7.5.3 |

---

## 1. Purpose

To define what constitutes a configuration item of EchoMind Enterprise, how those items are
identified and controlled, and how a change to any of them is proposed, assessed, implemented,
verified and reconciled. ISO 9001:2015 8.5.6 requires that changes to production are reviewed and
controlled to the extent necessary to ensure continuing conformity; 8.5.2 requires that outputs can
be identified and traced where traceability is a requirement.

## 2. Scope

All items that determine the behaviour of a running EchoMind Enterprise instance: application
source, container images, model weights, environment configuration, the evaluation corpus, and the
deployment topology itself. It covers both the reference instance operated by Ajace AI and
customer on-premises installations.

It does **not** cover customer data held in the deployment's volumes — that is customer property
and is addressed separately. It does not cover QMS documents, which are controlled under SOP-01.

## 3. Responsibilities

| Role | Responsibility |
|---|---|
| Engineering Lead | Owns this procedure; decides whether a proposed change meets the significance criteria in §6; authorises emergency changes |
| Developer | Proposes and implements changes; records rationale and verification in the commit body; reconciles any container-side hot patch back to source |
| Release Manager | Confirms, before a release, that the deployed images were built from a known source revision (see SOP-08) |
| Managing Director | Approves changes that alter a contractual commitment to a customer |

## 4. Configuration items

A configuration item (CI) is anything whose state must be known for the behaviour of a deployed
instance to be explicable. For EchoMind Enterprise these are:

| # | Configuration item | Where it lives | Controlled by |
|---|---|---|---|
| CI-1 | Application source (backend, voice, frontend, scripts, eval harness) | This Git repository | Git commit history |
| CI-2 | Container image definitions | `backend/Dockerfile`, `voice/Dockerfile`, `frontend/Dockerfile`, `docker/ollama/Dockerfile` | Git; built by `scripts/build.sh` |
| CI-3 | Built container images | The Docker daemon on the host | **Uncontrolled — see §9** |
| CI-4 | Service topology and runtime configuration | `docker-compose.yml`, `docker-compose.prepare.yml` | Git |
| CI-5 | Environment configuration and secrets | `.env` at the repository root | **Not in Git** (`.gitignore:25-28`); no template exists — see §9 |
| CI-6 | Model weights and engines | Named volumes `trtllm_hf_cache`, `ollama_data`, and `/data/hf_cache` inside `echomind_data` | Populated once by `scripts/prepare_offline.sh`; no inventory of what a given volume contains |
| CI-7 | Models baked into images at build time | Nemotron streaming STT (`backend/Dockerfile:42`, `voice/Dockerfile:44`), Parakeet-TDT (`voice/Dockerfile:50`), Piper voice files (`voice/Dockerfile:62-63`), Kokoro caches (`voice/Dockerfile:66-74`) | The image; changes only on rebuild |
| CI-8 | Piper voice assets mounted at runtime | `./voice/voices` bind-mounted to `/voices` (`docker-compose.yml:124`) — the only source bind mount in the stack | Git, for the files that are tracked |
| CI-9 | The golden evaluation set | `eval/golden/*.jsonl` — 52 items across seven files | Git. A change to a golden item changes the acceptance criterion for the product and is itself a significant change (§6) |
| CI-10 | Customer deployment record | Intended: `registers/REG-08_Release_and_Deployment_Register.md` | **Does not exist — see §9** |

**CI-5 note.** Six secret-bearing variable names are consumed by the stack: `HF_TOKEN`,
`TUNNEL_TOKEN`, `AUTH_SECRET`, `AUTH_ADMIN_USER`, `AUTH_ADMIN_PASSWORD`, `VOICE_AUTH_SECRET`.
`HF_TOKEN` is additionally passed as a Docker **build argument** (`docker-compose.yml:53` and
`:120`), so it is present in build layers unless those layers are pruned. There is no secret
manager, no use of Docker secrets, and no encryption at rest for `.env`.

## 5. Branching and commit practice as actually performed

This section describes what the repository shows, not a target.

- Work is committed **directly to `main`**. Sixteen local and eighteen remote branches exist, but
  only two merge commits appear in the whole history (`3aed690`, `6f4d24e`, both dated early in the
  project); recent work does not go through branches or pull requests.
- The history contains **202 commits from two author identities** (`git log --format=%an | sort -u`
  returns `2523` and `RW2523`). Forty-nine of the last fifty commits are by a single author.
- **Conventional-commit prefixes** (`feat(scope):`, `fix(scope):`, `docs(scope):`) are used on 78
  of 202 commits overall — about 38 per cent — but on 38 of the last 40. The practice is recent and
  improving, not historic.
- **Commit bodies carry the engineering record.** The prevailing style is a body that states the
  symptom, the cause, the fix and the verification performed. `724fb98` and `73f0b4f` are
  representative: both name the failing component, the mechanism, and the end-to-end check run
  afterwards. This is the organisation's real design-change record and must be preserved.

**Rule.** Every commit that changes behaviour carries a body stating (a) what changed, (b) why, and
(c) what was run to verify it. A one-line subject with no body is acceptable only for a change that
cannot alter behaviour (formatting, a comment, a document).

Single-developer working means there is no independent review step. This is an accepted constraint
of the current team size, not a control; it is recorded as a gap in §9.

## 6. Significance criteria — when a Change Request is required

Most commits do not need a form. A **Change Request** (`forms/FRM-02_Change_Request.md`) is required
before implementation when a change meets **any** of the following criteria.

| # | Criterion | Why |
|---|---|---|
| S-1 | Alters retrieval, ranking, chunking, prompting or grounding behaviour | Directly changes what the product asserts to a user; the quality policy treats an ungrounded answer as a top-severity defect (QP-01 §2.1) |
| S-2 | Swaps, upgrades or removes a model (chat LLM, embedding, STT, TTS, reranker, image) | Changes output characteristics, licence position and hardware envelope at once |
| S-3 | Changes how customer data is stored, indexed, exported or deleted | Customer property (8.5.3); affects the offline guarantee |
| S-4 | Changes authentication, session handling, namespace isolation or the permission filter | `73f0b4f` shows how a single mis-named environment variable silently disabled an auth gate |
| S-5 | Changes the container topology, base images, volumes or exposed ports | Changes the attack surface and the upgrade path for every existing installation |
| S-6 | Changes a golden evaluation item or the pass criteria in `eval/run_eval.py` | Moves the acceptance bar; must not be done in the same change as the code it judges |
| S-7 | Affects an already-deployed customer instance, or is required by a customer commitment | Triggers customer communication and a release record (SOP-08) |
| S-8 | Introduces, removes or unpins an external dependency | Supplier control (SOP-09) |

Changes not meeting any criterion are implemented directly under §5 and §7, with the commit body as
the record.

## 7. Change procedure

1. **Proposal.** The developer states the problem and the intended change. For a significant change
   (§6), this is written into `forms/FRM-02_Change_Request.md` before work starts.
2. **Impact assessment.** The proposer identifies which configuration items in §4 are affected,
   which of the significance criteria apply, whether any deployed instance is affected, and what
   verification will be needed (SOP-07). For an S-2 or S-8 change the supplier controls in SOP-09
   apply as well.
3. **Implementation.** Code is changed in the working tree and committed to `main` with a body per
   §5.
4. **Verification.** The verification levels defined in SOP-07 are run according to what the change
   touched. A change meeting S-1, S-2 or S-6 requires a golden evaluation run against the running
   stack; the report is retained as a record.
5. **Reconciliation.** Any container that was hot-patched during development is rebuilt from source
   (§8) before the change is considered complete.
6. **Release.** Deployment is governed by SOP-08. A change is not "done" when it is committed; it is
   done when a built image containing it has been released or explicitly deferred.

## 8. Container hot-patching: a controlled exception

Rebuilding the backend or voice image is slow, because the build downloads and warms large model
weights (`backend/Dockerfile:42`, `voice/Dockerfile:44-50`) and `scripts/build.sh:2-4` deliberately
serialises the backend build to avoid a BuildKit "file already closed" failure during that download.
A `docker cp` of changed files into a running container, followed by `docker commit` and a restart,
is therefore used for fast iteration — particularly on the voice service.

**This is permitted for iteration only, and is subject to the following mandatory rules.**

1. A hot-patched container is a **development state**, never a release. It must not be presented to
   a customer, used to produce an evaluation record, or left running on the reference instance
   beyond the working session.
2. The same change **must** be committed to the repository and the image **must** be rebuilt with
   `./scripts/build.sh` (or `docker compose build`) before the change is treated as delivered.
3. Until that rebuild has been done and the stack restarted from the rebuilt images, the running
   system does not correspond to any source revision and no statement may be made about what it
   contains.

**Evidence for this rule.** Commit `724fb98` records exactly this failure mode. The running
containers carried `docker cp`-applied layers; when the images were finally rebuilt from source,
three latent breakages surfaced at once — a heredoc inside a `RUN` continuation that made
`docker compose build voice` fail outright, a `setuptools>=82` upgrade pulled in by
`nemo_toolkit @ git+…@main` that removed `pkg_resources` and broke `librosa` at build and
`webrtcvad` at runtime, and a boardroom JSON-truncation defect. None of these were visible while the
containers were being patched in place. The drift was real, it accumulated silently, and it was only
discovered because someone eventually did a clean rebuild.

## 9. Traceability requirement — source revision to deployed instance

ISO 9001:2015 8.5.2 requires identification of outputs where traceability is a requirement. For a
product installed inside a customer's perimeter and supported remotely or not at all, traceability
from a running instance back to its source revision **is** a requirement: without it, a customer
fault report cannot be tied to code.

**Required chain (to be established):**

```
git commit SHA  →  annotated git tag  →  image tag carrying that version
                →  build identifier baked into the image
                →  the same identifier returned by a backend endpoint
                →  the deployment register entry for that customer instance
```

**Present state: none of this chain exists.**

| Link | Present state | Evidence |
|---|---|---|
| Git tag | No tags exist at all | `git tag` returns nothing |
| Changelog | None | No `CHANGELOG` in the tree |
| Application version | `frontend/package.json:4` is `"version": "0.0.0"`; no `__version__` in backend, voice or the vendored `nemotron_asr` package | — |
| Image tags | `backend`, `voice` and `frontend` are compose-default builds with no `image:` key, so they are effectively `:latest`. Only `echomind-trtllm:1.2.0rc6` (`docker-compose.yml:5`) and `echomind-ollama:setup` (`:235`) carry a tag | `docker-compose.yml:207-212`, `:47-57`, `:115-125` |
| Build identifier at runtime | No endpoint reports a build or commit identifier | — |
| Deployment register | Does not exist | `docs/qms/registers/` is empty |

**Consequence to state plainly to an auditor:** given a running EchoMind Enterprise instance today,
it is not possible to determine which source revision produced it. Closing this is the single most
valuable configuration-management improvement available to the organisation, and it is a
prerequisite for the release identification scheme in SOP-08 §7.

## 10. Configuration of the deployed environment

The deployed configuration is the composition of `docker-compose.yml`, the host's `.env`, the
contents of the three named volumes, and the bind-mounted voice assets.

- **Six services** are defined: `trtllm`, `backend`, `voice`, `frontend`, `cloudflared` (only with
  `--profile public`, `docker-compose.yml:227`) and `ollama`.
- **Offline by default.** `HF_HUB_OFFLINE=1` and `TRTLLM_SKIP_DOWNLOAD=1` are the defaults for
  `trtllm` (`docker-compose.yml:27-28`); `OLLAMA_OFFLINE=1` is set for `ollama` (`:245`). A running
  instance therefore does not fetch models; the volumes must already hold them.
- **Configuration drift risk in `.env`.** Because no `.env.example` is tracked, the required
  variable set is discoverable only by reading `docker-compose.yml`. A missing or misspelled
  variable fails silently rather than loudly: `docker-compose.yml:143-146` carries an inline warning
  that `VOICE_AUTH_SECRET` must be fed from the backend's `AUTH_SECRET`, placed there after commit
  `73f0b4f` fixed a defect in which the voice WebSocket auth gate loaded an empty secret and was
  effectively non-functional.
- **Change to deployed configuration** is a change to CI-4 or CI-5 and is subject to §6; changing a
  port, a volume, an auth variable or a model identifier on a customer instance meets S-3, S-4 or
  S-5 and requires a Change Request.

## 11. Emergency and hotfix changes

An emergency change is one required to restore service or to remove an active exposure of customer
data.

1. The Engineering Lead authorises it verbally or in writing; the authorisation is recorded
   afterwards, not skipped.
2. The fix is implemented by the shortest safe route. Hot-patching a container under §8 is
   acceptable here.
3. Within **one working day**, the change is committed to `main` with a body describing the
   incident, and the affected images are rebuilt from source. This is not optional; §8's evidence
   applies with more force under time pressure, not less.
4. A Change Request is raised retrospectively where the change meets §6, and the event is recorded
   as a nonconformity where it resulted from a defect in our own work.
5. Where a credential was exposed, the credential is rotated before anything else (SOP-01 §9).

## 12. Records

| Record | Where it lives | Retention |
|---|---|---|
| Change history for every configuration item under Git control | Git commit history (`git log --follow -- <path>`) | Life of the product + 3 years (SOP-01 §8) |
| Rationale and verification for each change | The commit body | As above |
| Change Requests for significant changes | `forms/FRM-02_Change_Request.md`, completed and committed under `records/` | As above |
| Release records | `forms/FRM-03_Release_Record.md` (see SOP-08) | Life of the release + 3 years |
| Deployment register (customer instance → release) | `registers/REG-08_Release_and_Deployment_Register.md` | Life of the installation + 3 years |
| Emergency change authorisations | The commit body of the emergency fix, plus the retrospective Change Request | 3 years |

## 13. Current state and gaps

| # | Requirement of this procedure | Present state | Gap |
|---|---|---|---|
| G-1 | Source revision traceable to deployed instance | No git tags, no changelog, no `__version__`, no build identifier endpoint, untagged images | **Open.** Nothing in the chain exists (§9) |
| G-2 | Change Request form for significant changes | `forms/FRM-02_Change_Request.md` does not exist; only `FRM-00` is present in `docs/qms/forms/` | **Open.** Form to be created |
| G-3 | Deployment register | `docs/qms/registers/` is empty | **Open.** No record of which customers run which build |
| G-4 | Independent review of changes | Single-developer, direct-to-main working; two merge commits in 202 | **Open and partly accepted.** A second reviewer is not available at current team size; compensating control is the commit-body verification record (§5) |
| G-5 | Hot-patch reconciliation | The rule in §8 is stated here for the first time; it has not previously been a written control, and `724fb98` shows drift did occur | **Open.** Compliance from the date this procedure is approved |
| G-6 | `.env` template and secret handling | No `.env.example`; `HF_TOKEN` baked into build layers via build args (`docker-compose.yml:53`, `:120`); no secret manager, no Docker secrets, no encryption at rest | **Open** |
| G-7 | Inventory of model-volume contents | No record of which model versions a given `trtllm_hf_cache` or `ollama_data` volume holds | **Open.** A volume populated at two different dates may contain different weights with no way to tell |
| G-8 | Commit message consistency | 78 of 202 commits carry a conventional prefix | **Improving.** 38 of the last 40 conform; no further action beyond continued practice |

All gaps above are to be carried into `ISO9001_Gap_Analysis.md`.
