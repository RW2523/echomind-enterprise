# SOP-08 — Release and Deployment

| Field | Value |
|---|---|
| Document ID | SOP-08 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Release Manager |
| Approved by | `________________________` |
| Approval date | `____ / ____ / ________` |
| ISO 9001:2015 clauses | 8.5.1, 8.5.4, 8.6 |

---

## 1. Purpose

To define how a build of EchoMind Enterprise is prepared, authorised, deployed, verified after
deployment, rolled back if necessary, and identified thereafter. ISO 9001:2015 8.6 requires that
release to the customer does not proceed until planned arrangements are satisfactorily completed;
8.5.4 requires that outputs are preserved during delivery, which for this product means the model
volumes that make an offline installation work.

## 2. Scope

Both delivery modes in use:

- **(a) On-premises customer install** — an air-gapped or offline installation on customer NVIDIA
  hardware (reference platform: DGX Spark GB10), delivered as an offline bundle.
- **(b) The reference / production instance** operated by Ajace AI, including the optional public
  access path via Cloudflare Tunnel.

It covers the software and model artefacts. It does not cover contract, commercial or acceptance
terms, nor customer data migration.

## 3. Responsibilities

| Role | Responsibility |
|---|---|
| Release Manager | Confirms the release-readiness checklist is complete; authorises release; completes the release record |
| Engineering Lead | Confirms the build is reconciled with source (SOP-06 §8) and that verification per SOP-07 §5 was run against the build being released |
| Developer | Performs the build, the bundle export and the deployment steps |
| Managing Director | Authorises release to a customer site |

## 4. Release preparation

1. **Reconcile the build with source.** Every container must be built from committed source. Any
   container that was hot-patched with `docker cp` during development is rebuilt. SOP-06 §8 states
   why, and commit `724fb98` is the evidence: a clean rebuild after a period of in-place patching
   surfaced three latent breakages simultaneously.
2. **Build.** Use `./scripts/build.sh`. It builds `backend` alone first and then the remainder with
   `BUILDX_METADATA_PROVENANCE=disabled` (`scripts/build.sh:9`, `:12`), because building the backend
   in parallel with the other services triggers a BuildKit "file already closed" failure during the
   long Hugging Face download (`scripts/build.sh:2-4`). Do not substitute a bare
   `docker compose build`.
3. **Run the verification levels required by SOP-07 §5** for what the release contains, and retain
   the golden evaluation report (SOP-07 §8.3).
4. **Run the release-readiness checklist** in §5 below.
5. **Complete the release record** `forms/FRM-03_Release_Record.md`, naming the source commit SHA,
   the checks run and their outcomes, and any deferred non-blocking failure with its reason.

## 5. Release-readiness checklist

| # | Check | Command or evidence | Pass criterion |
|---|---|---|---|
| R-1 | Working tree clean; release built from a committed revision | `git status --porcelain` empty; record `git rev-parse HEAD` | No uncommitted change in the build context |
| R-2 | Images rebuilt from source, not hot-patched | `./scripts/build.sh` completed in this session | Clean build, all services |
| R-3 | Offline readiness | `./scripts/verify_offline_readiness.sh` | Exit 0 — but read §5.1 first |
| R-4 | Verification levels per SOP-07 §5 | Backend and voice unit tests, chunk-coverage check, golden evaluation as applicable | Per SOP-07 §7; no release-blocking failure |
| R-5 | Golden evaluation report retained | `eval/reports/eval_<run_id>.json` copied to a tracked location | Report exists and is committed |
| R-6 | All health-checked services healthy from a cold start | `docker compose up -d` then `docker compose ps` | `trtllm`, `backend`, `voice`, `ollama` healthy |
| R-7 | Manual live verification | Chat with citation, Silent Assistant live check, one voice turn, boardroom analyse and export | Recorded in the release record, naming what was exercised |
| R-8 | Secrets present and correctly named on the target | `.env` complete | In particular `VOICE_AUTH_SECRET` must be fed from the backend's `AUTH_SECRET` (`docker-compose.yml:143-146`) |
| R-9 | Model volumes populated | `docker volume ls` shows `trtllm_hf_cache`, `ollama_data`, `echomind_data` | Present and non-empty |
| R-10 | Release identification recorded | Git tag, image tags, build identifier | **Cannot currently be satisfied — see §7** |

### 5.1 About `scripts/verify_offline_readiness.sh`

The script is real, executable and useful. It makes seven checks, sets `ERR=1` on any failure and
exits 1 if `ERR` is non-zero (`scripts/verify_offline_readiness.sh:64-70`): that the Ollama entrypoint
does not pull unconditionally without an `OLLAMA_OFFLINE` guard; that compose sets `OLLAMA_OFFLINE=1`
and `HF_HUB_OFFLINE=1`; that the Ollama volume exists (warning only); that the backend Dockerfile
pre-downloads the ASR weights; and that the voice Dockerfile pre-downloads the TTS and final-STT
models (warning only).

Two facts must be stated to an auditor. **It is not wired into anything:** no CI, no build step and
no compose hook invokes it; it is mentioned in `OFFLINE_DEPLOYMENT.md:122` and printed as advice at
the end of `scripts/prepare_offline.sh:49`, but neither of those runs it, so it executes only when a
person types the command. **Check 7 is stale:** it greps `voice/Dockerfile` for `whisper.load_model`
(`scripts/verify_offline_readiness.sh:57`), but Whisper is no longer in the voice image — the final
STT is Parakeet-TDT (`voice/Dockerfile:50`) — so it emits `WARN: Voice Dockerfile may not
pre-download all models` on a correct tree. Being a warning, it does not stop the script exiting 0;
the assertion nonetheless no longer tests what it claims and should be updated to the current model
set.

## 6. Authorisation to release

Release is authorised by the Release Manager, and additionally by the Managing Director for any
release delivered to a customer site.

Authorisation is recorded on `forms/FRM-03_Release_Record.md` by the person authorising it, at the
time, naming the release identifier, the source commit SHA, the authoriser and the date. It is never
pre-filled or completed on someone's behalf (SOP-01 §6). Release is not authorised while any
release-blocking failure under SOP-07 §7 is open.

## 7. Release identification — the central gap

**Required scheme (to be adopted).**

| Element | Form | Purpose |
|---|---|---|
| Git tag | Annotated tag `vMAJOR.MINOR.PATCH` on the released commit | Fixes the source revision |
| Image tags | `echomind-backend:<version>`, `echomind-voice:<version>`, `echomind-frontend:<version>`, in addition to the moving tag | Ties the artefact to the tag |
| Build identifier | The commit SHA and build date baked into each image as a label and an environment variable | Survives into the running container |
| Runtime endpoint | A backend endpoint returning that identifier | Lets anyone with access to a running instance state what it is |
| Deployment register entry | `registers/REG-08_Release_and_Deployment_Register.md`: customer, instance, release identifier, deployment date | Ties the customer to the build |

**Present state: none of this exists.**

- `git tag` returns nothing. The repository has **no tags at all** across 202 commits.
- There is no `CHANGELOG`.
- `frontend/package.json:4` is `"version": "0.0.0"`. There is no `__version__` in the backend, the
  voice service or the vendored `nemotron_asr` package.
- `backend`, `voice` and `frontend` are built by compose with no `image:` key
  (`docker-compose.yml:47`, `:115`, `:207`), so they carry compose-default names and are effectively
  `:latest`. Only two images in the stack are versioned at all: `echomind-trtllm:1.2.0rc6`
  (`docker-compose.yml:5`) and `echomind-ollama:setup` (`:235`).
- No endpoint reports a build identifier.
- `docs/qms/registers/` is empty; no deployment register exists.

**Consequence.** A deployed EchoMind Enterprise instance cannot be traced back to a source revision
from the running system. If a customer reports a defect, there is no reliable way to establish which
code they are running other than asking when they installed it. Every other control in this
procedure is weakened by that, and closing it is the prerequisite for R-10.

## 8. Deployment procedure (a) — on-premises customer install

The product is designed to run fully offline. The install therefore separates a single online
preparation step from an offline run.

### 8.1 One-time online preparation (on a machine with internet)

1. `./scripts/prepare_offline.sh`. This builds all images — backend first, then the rest with
   `BUILDX_METADATA_PROVENANCE=disabled` (`scripts/prepare_offline.sh:15-16`) — then starts Ollama
   with the `docker-compose.prepare.yml` overlay so it may pull, waits up to 300 seconds for
   `nomic-embed-text` to appear in the volume (`:21-36`), and stops Ollama so the volume persists
   (`:38-39`).
2. Note the script's own caveat (`:46-47`): the first normal start still fills `trtllm_hf_cache` if
   it is empty, which needs the GPU and can take a long time, and `HF_TOKEN` must be in `.env` for
   the gated model.
3. `./scripts/verify_offline_readiness.sh` (see §5.1).

### 8.2 Export the bundle

`./scripts/export_offline_bundle.sh` produces a directory containing:

| Artefact | Contents |
|---|---|
| `image-backend.tar`, `image-voice.tar`, `image-frontend.tar`, `image-ollama.tar`, `image-trtllm.tar` | The five service images, via `docker save` (`scripts/export_offline_bundle.sh:41`) |
| `ollama_data.tar` | The Ollama model store, i.e. the embedding model (`:50`) |
| `trtllm_hf_cache.tar` | The TensorRT-LLM Hugging Face cache — weights and engines; large (`:58`) |
| `voice-assets/` | Piper voice files from `voice/voices` |
| `OFFLINE_DEPLOYMENT.md` | Deployment instructions, carried with the bundle |
| `MANIFEST.txt` | Export timestamp and contents list (`:70-83`) |

The `MANIFEST.txt` is the delivery record for the bundle and is retained with the release record.
Note that the bundle does **not** carry a release identifier, because none exists (§7); the manifest
records only an export timestamp.

### 8.3 Install on the target (no internet required)

1. Transfer the bundle by the customer's approved medium.
2. `./scripts/import_offline_bundle.sh <bundle>` — loads the images and restores the volumes.
3. Place the customer's `.env` on the host. Confirm `AUTH_SECRET` and `VOICE_AUTH_SECRET` are
   consistent (`docker-compose.yml:143-146`).
4. `docker compose up -d`.
5. Access on `http://<HOST>:3000` or `https://<HOST>:3443` (`docker-compose.yml:214`). The HTTPS
   certificate is self-signed at image build unless the customer supplies one, so a browser warning
   on first access is expected.
6. Run the post-deployment verification in §10.

## 9. Deployment procedure (b) — reference / production instance

1. Build per §4.2 and complete §5.
2. `docker compose up -d` to bring up the five default services. All services carry
   `restart: unless-stopped`.
3. To expose the instance publicly, start the tunnel explicitly:
   `docker compose --profile public up -d cloudflared`. The `cloudflared` service is behind the
   `public` profile (`docker-compose.yml:227`) precisely so that public exposure is an opt-in act,
   and `docker-compose.yml:221` carries the inline warning that it must be gated with Cloudflare
   Access first because EchoMind has no built-in authentication of its own.
4. Run the post-deployment verification in §10.

## 10. Post-deployment verification

| # | Check | Evidence |
|---|---|---|
| P-1 | All health-checked services reach healthy | `docker compose ps`; `trtllm` may take up to its 1200-second start period (`docker-compose.yml:40`) |
| P-2 | Chat returns a grounded answer with a citation from the customer's own corpus | Manual, recorded |
| P-3 | Live transcription with live fact-checking produces a check with quoted proof | Manual, recorded |
| P-4 | One full voice turn completes | Manual, or `eval/voice_e2e_test.py` output read by a person |
| P-5 | Boardroom analyse and export produce a clean report and PDF | Manual, recorded |
| P-6 | Document generation produces a document | Manual, recorded |
| P-7 | No outbound network activity is required in steady state | Observed on the host |
| P-8 | Where the deployment is gated (Cloudflare Access, or the WebSocket auth gate), both a positive and a negative case are exercised | As in commit `73f0b4f`, which recorded an unauthenticated handshake returning HTTP 403 and an authenticated one establishing a session |

The results of P-1 to P-8 are recorded on the release record. For a customer install they form the
installation acceptance evidence.

## 11. Robustness of a released instance

Two design features are evidence that a release degrades safely rather than silently.

**11.1 Healthchecks.** Four of the six services define one:

| Service | Healthcheck | Lines |
|---|---|---|
| `trtllm` | HTTP GET `/v1/models`, 15 s interval, 120 retries, **1200 s start period** to allow engine build | `docker-compose.yml:31-40` |
| `backend` | HTTP GET `/health`, 30 s interval, 3 retries, 180 s start period | `:60-67` |
| `voice` | HTTP GET `/health`, 30 s interval, 3 retries, 120 s start period | `:130-137` |
| `ollama` | `ollama list \| grep -q nomic-embed-text` | `:248-254` |
| `frontend` | **None** | `:207-215` |
| `cloudflared` | **None** | `:223-229` |

**11.2 Self-recovery from a fatal GPU fault.** The backend and voice `/health` endpoints return 503
once a fatal GPU or CUDA fault has poisoned the shared context. An in-process watchdog waits on the
fatal event and calls `os._exit(1)` after a short grace period (`backend/app/main.py:104`, guarded by
`ECHOMIND_EXIT_ON_FATAL_CUDA`), and `restart: unless-stopped` brings the container back with a fresh
CUDA context. The inline comments at `docker-compose.yml:61-62` and `:131-132` record the intent.
This matters in an offline deployment where no one may be watching: a class of fault that would
otherwise require a manual container restart resolves itself.

**Limits to state.** `frontend` and `cloudflared` have no healthcheck, so an nginx or tunnel failure
is not detected by Docker and does not trigger a restart on health grounds — only on process exit.
For a publicly exposed instance, `cloudflared` is the component whose silent failure removes all
access, and it is the one with no health signal.

## 12. Rollback

1. **Decide.** The Release Manager decides to roll back when post-deployment verification fails in a
   way that cannot be corrected in place within the agreed window.
2. **Restore the previous images** from the retained earlier bundle:
   `./scripts/import_offline_bundle.sh <previous bundle>`, then `docker compose up -d`.
3. **Do not touch the data volumes** (§13). **Verify** per §10.
4. **Record** the rollback, its trigger and its outcome on the release record, and raise the cause
   for correction.

**Constraint to state plainly.** Rollback depends on the previous artefact still existing. Because
the service images are untagged and effectively `:latest` (§7), rebuilding or re-importing overwrites
the previous image in place unless a bundle was retained beforehand. **Retaining the previous
bundle, or a tagged copy of the previous images, before deploying a new one is therefore mandatory
for any customer instance** — it is the only rollback path that currently exists.

## 13. Preservation of data and model volumes (8.5.4)

Three named volumes carry everything that must survive an upgrade.

| Volume | Contents | On upgrade |
|---|---|---|
| `echomind_data` | **All customer data**: the SQLite database, FAISS indexes, `/data/uploads`, boardroom audio, `/data/hf_cache` (including the boardroom diarised-ASR model), the Document Studio model cache, and `auth_secret.key` | Never removed. Never recreated |
| `trtllm_hf_cache` | LLM weights and built TensorRT engines | Never removed. Rebuilding it requires internet and GPU time, which an air-gapped site does not have |
| `ollama_data` | The embedding model | Never removed |

Rules:

1. Upgrade with `docker compose up -d` after loading new images. **Never** use
   `docker compose down -v`, and never `docker volume rm` any of the three, on any instance holding
   customer data. `-v` destroys the customer's corpus, their indexes and their auth secret in one
   command, and on an offline site the model volumes cannot be refetched.
2. Confirm volume preservation as part of post-deployment verification. Commit `724fb98` records
   this being checked explicitly after a from-scratch rebuild: *"Data/model volumes preserved."*
3. `./voice/voices` is a host bind mount (`docker-compose.yml:124`), not a volume; its contents live
   on the host filesystem and are preserved by not deleting them.
4. Backup of `echomind_data` is the customer's responsibility under their own retention policy
   unless contracted otherwise. Ajace AI holds no copy — the data never leaves the customer
   perimeter (QP-01 §2.2).

## 14. Records

| Record | Where it lives | Retention |
|---|---|---|
| Release record, including authorisation and the §5 checklist | `forms/FRM-03_Release_Record.md`, completed and committed | Life of the release + 3 years (SOP-01 §8) |
| Source commit SHA for the release | The release record; ideally an annotated git tag once §7 is implemented | As above |
| Golden evaluation report supporting the release | `eval/reports/eval_<run_id>.json`, committed | As above |
| Offline bundle manifest | `MANIFEST.txt` inside the delivered bundle; a copy retained with the release record | As above |
| Post-deployment verification results (P-1 to P-8) | The release record | As above |
| Deployment register: customer instance → release identifier → date | `registers/REG-08_Release_and_Deployment_Register.md` | Life of the installation + 3 years |
| Rollback events | The release record, plus a nonconformity entry where caused by our own defect | 3 years |

## 15. Current state and gaps

| # | Requirement of this procedure | Present state | Gap |
|---|---|---|---|
| G-1 | Releases are identifiable | No git tags, no changelog, no `__version__`, untagged service images, no build-identifier endpoint | **Open.** §7. Nothing in the scheme exists |
| G-2 | Deployed instances are traceable to a release | No deployment register; `docs/qms/registers/` is empty | **Open** |
| G-3 | Release record form | `forms/FRM-03_Release_Record.md` does not exist; only `FRM-00` is present | **Open.** Form to be created |
| G-4 | Release-readiness check is enforced | `scripts/verify_offline_readiness.sh` is not invoked by any CI, build step or hook; it is only mentioned in `OFFLINE_DEPLOYMENT.md:122` and printed as advice by `scripts/prepare_offline.sh:49` | **Open** |
| G-5 | Release-readiness checks are current | Check 7 greps for `whisper.load_model`, which the voice image no longer contains, so it warns on a correct tree | **Open.** Update the assertion to Parakeet-TDT and Piper |
| G-6 | Rollback is reliably possible | Depends entirely on a previous bundle having been retained, because images are untagged and overwritten in place | **Open and partly mitigated** by the mandatory retention rule in §12 |
| G-7 | All services are health-monitored | `frontend` and `cloudflared` have no healthcheck; `cloudflared` is the single point of failure for public access | **Open** |
| G-8 | Release authorisation is recorded | No release has been recorded under this procedure; no prior release records exist | **Open.** Applies from approval |
| G-9 | Customer acceptance evidence | No completed installation record exists in the repository | **Open** |

All gaps above are to be carried into `ISO9001_Gap_Analysis.md`.
