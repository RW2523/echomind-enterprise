# SOP-09 — External Providers and Open Source

| Field | Value |
|---|---|
| Document ID | SOP-09 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Engineering Lead |
| Approved by | `________________________` |
| Approval date | `____ / ____ / ________` |
| ISO 9001:2015 clauses | 8.4.1, 8.4.2, 8.4.3, 8.1(e) |

---

## 1. Purpose

To define how Ajace AI controls externally provided processes, products and services that become
part of, or affect, EchoMind Enterprise. ISO 9001:2015 8.4.1 requires that externally provided items
conform to requirements; 8.4.2 that controls are applied and their effect on the ability to meet
customer requirements is considered; 8.4.3 that requirements are communicated to the provider before
they are relied on. Clause 8.1(e) requires that outsourced processes are controlled.

## 2. Scope

Every external input that reaches a customer deployment:

| Type | Examples in this product |
|---|---|
| Model weights | Chat LLM, embeddings, streaming and final STT, diarised ASR, TTS, reranker, image generation |
| Base container images | `nvcr.io/nvidia/pytorch:25.01-py3`, `ollama/ollama:latest`, `nginx:1.27-alpine`, `node:20-alpine`, `cloudflare/cloudflared:latest` |
| Python packages | `backend/requirements.txt`, `voice/requirements.txt`, plus direct `pip install` lines in both Dockerfiles |
| npm packages | `frontend/package.json` with `package-lock.json` |
| An APT mirror | `mirrors.mit.edu`, substituted for `ports.ubuntu.com` at image build |
| Infrastructure services | Cloudflare (Tunnel and Access), Hugging Face Hub, GitHub, the Ollama registry, NVIDIA NGC, Docker Hub, Let's Encrypt (optional TLS) |

It does not cover commercial procurement of hardware. Ajace AI does not sell hardware; it advises
customers on hardware and hosting choices, and that advice is an engagement activity rather than an
external provision to this product.

## 3. Responsibilities

| Role | Responsibility |
|---|---|
| Engineering Lead | Selects and evaluates providers; owns the pinning and offline-caching controls; approves any new external dependency |
| Developer | Applies the controls in §5 when adding or upgrading a dependency; records the reason in the commit body |
| Managing Director | Accountable for licence compliance and for any commitment made to a customer about model provenance |

## 4. Why supplier control looks different for an offline product

EchoMind Enterprise runs with no outbound network access. Every model weight, package and binary
that the running system needs must already be inside an image or a named volume before the customer
starts it. That inverts the usual supply-chain picture in two ways.

1. **Pre-caching is the control.** Baking model weights into images at build
   (`backend/Dockerfile:42`, `voice/Dockerfile:44`, `:50`, `:62-63`, `:66-74`) and populating the
   `trtllm_hf_cache` and `ollama_data` volumes once via `scripts/prepare_offline.sh` means the
   deployed system has no runtime dependency on any provider. A provider that deletes a model,
   changes its licence or goes offline cannot break a running customer instance. `HF_HUB_OFFLINE=1`
   and `OLLAMA_OFFLINE=1` are set in `docker-compose.yml:27` and `:245` to guarantee this.
2. **All the supplier risk moves to build time.** The moment of exposure is the build, not the run.
   Anything unpinned at build time is a live dependency on whatever the provider is serving that
   day. An unpinned `:latest` base image therefore undermines the whole arrangement: it means two
   builds of the "same" commit can produce two different products, and neither can be reproduced.

## 5. Controls by provision type

| Provision type | Control required | Present state |
|---|---|---|
| Model weights | Pinned model identifier; pre-downloaded into an image or volume; licence reviewed and recorded; model card retained | Identifiers are pinned and pre-caching is in place, except the reranker (§6). **Licence review is not performed** (§8) |
| Base container images | Pinned to an immutable tag or digest | Partial. `nvcr.io/nvidia/pytorch:25.01-py3`, `nginx:1.27-alpine` and `node:20-alpine` are pinned; `ollama/ollama:latest` (`docker/ollama/Dockerfile:2`) and `cloudflare/cloudflared:latest` (`docker-compose.yml:224`) are not |
| Python packages | Exact version pins; upper bounds on any range | Partial — see §7 |
| npm packages | Exact versions, or caret ranges with a committed lockfile and `npm ci` | All dependencies in `frontend/package.json` are caret-ranged; reproducibility rests entirely on `package-lock.json` and the `npm ci` in `frontend/Dockerfile:4` |
| APT mirror | Identified as a supplier; documented; fallback available | **Undocumented before this procedure** — see §9 |
| Infrastructure services | Understood scope; failure mode known; no customer data exposure | Cloudflare Access gates the public reference instance; note `docker-compose.yml:221` records that EchoMind itself has no built-in authentication |
| Verification before use | A new or upgraded dependency is verified per SOP-07 §5 before it is relied on, and a from-scratch build must succeed | Practised but not enforced (no CI, SOP-07 §9.1) |

## 6. Current supplier inventory — models

This table is the model portion of `registers/REG-02_External_Documents_and_Providers.md` and is
maintained with it.

| Role | Model identifier | Provider | How it reaches the product |
|---|---|---|---|
| Chat LLM (production) | `nvidia/Qwen3-30B-A3B-FP4` | NVIDIA (Qwen base, NVFP4 quantisation) | Hugging Face Hub into the `trtllm_hf_cache` volume, once; runtime `HF_HUB_OFFLINE=1` |
| Chat LLM (fallback) | `qwen2.5:7b-instruct-q4_K_M` | Alibaba Qwen, via Ollama | Ollama registry |
| Embeddings | `nomic-embed-text` | Nomic AI, via Ollama | Ollama registry into `ollama_data` |
| Streaming STT | `nvidia/nemotron-speech-streaming-en-0.6b` | NVIDIA — **gated model, requires `HF_TOKEN`** | Baked into both images at build (`backend/Dockerfile:42`, `voice/Dockerfile:44`) |
| Final STT (voice) | `nvidia/parakeet-tdt-0.6b-v2` | NVIDIA | Baked into the voice image (`voice/Dockerfile:47-50`) |
| Boardroom diarised ASR | `microsoft/VibeVoice-ASR-HF` (~18 GB) | Microsoft | Hugging Face Hub into `echomind_data:/data/hf_cache` during prepare |
| TTS (default) | Piper `en_US-lessac-medium` | Rhasspy | `wget` from `huggingface.co/rhasspy/piper-voices` at build (`voice/Dockerfile:62-63`) |
| TTS (option) | Kokoro-82M | hexgrad | `pip install kokoro`, caches baked at build (`voice/Dockerfile:66-74`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Microsoft / sentence-transformers | Hugging Face Hub — **not pre-downloaded; see below** |
| Image generation | `stabilityai/sdxl-turbo`, via diffusers | Stability AI | Hugging Face Hub into `/data/docgen_models` |

**Reranker defect.** `backend/requirements.txt` carries the comment *"Cross-encoder reranker (RAG
step 2) + CE relevance gate — without this the reranker silently falls back to a slow LLM rerank.
Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (pre-download into the image/HF cache for offline
runtime.)"* — but `backend/Dockerfile` contains no `snapshot_download` for that model; the only
model downloads are the ASR weights at `:42`. With `HF_HUB_OFFLINE=1` at runtime the weights cannot
be fetched, so `backend/app/rag/reranker.py:4-7` takes priority 2, the LLM fallback. The product
therefore runs a slower and different reranking path from the one the requirements file describes.
This is a live pre-caching gap with a quality and latency effect, and it is exactly the failure mode
§4.1 exists to prevent.

## 7. Dependency pinning — present state

| File | Entries | Exact | Ranged |
|---|---|---|---|
| `backend/requirements.txt` | 20 | 14 | 6: `scipy>=1.11,<2`; `huggingface-hub>=0.34,<2`; **`accelerate>=0.26.0` (no upper bound)**; `reportlab>=4.0,<5`; `diffusers>=0.38,<0.40`; `sentence-transformers>=5,<6` |
| `voice/requirements.txt` | 9 | 7 | 2: `piper-tts>=1.4.2,<2`; **`huggingface_hub>=0.20.0` (no upper bound)** |
| `frontend/package.json` | all dependencies | 0 | all caret-ranged; reproducibility depends on `package-lock.json` + `npm ci` |

Installs made directly in the Dockerfiles, outside the requirements files, are additional unpinned
surface:

| Install | Location | Risk |
|---|---|---|
| `nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main` | `backend/Dockerfile:19`, `voice/Dockerfile:18` | **A moving branch, not a release.** See §8 |
| `pip install --upgrade torchvision` | `backend/Dockerfile:25`, `voice/Dockerfile:21` | Unpinned upgrade, required because NeMo upgrades torch and the NGC torchvision targets the old one |
| `pip install kokoro soundfile` | `voice/Dockerfile:66` | Unpinned; reinstalls over the pinned `soundfile==0.12.1` from `voice/requirements.txt` |
| `ollama/ollama:latest` | `docker/ollama/Dockerfile:2` | Unpinned base image |
| `cloudflare/cloudflared:latest` | `docker-compose.yml:224` | Unpinned base image on the component that fronts public access |

**Rule going forward.** A new dependency is added with an exact version, or with a range that has
both bounds and a stated reason. A `git+…@main` reference, an `--upgrade` with no version, and a
`:latest` base image each require the Engineering Lead's explicit approval and a note in the commit
body explaining why the pinned alternative was not viable.

## 8. Flagship supplier-risk case: NeMo on `main`

Both the backend and voice images install NVIDIA NeMo from the tip of its `main` branch:

```
pip install --no-cache-dir "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"
```
(`backend/Dockerfile:19`, `voice/Dockerfile:18`)

This is the single largest supplier risk in the product. The dependency is not a release; it is
whatever NVIDIA last pushed. Two builds of the same EchoMind commit, minutes apart, can install
different code.

**Evidenced failure.** Commit `724fb98` records the outage this caused. NeMo on `main` began pulling
`setuptools>=82`, which removed `pkg_resources`. `librosa` imports `pkg_resources` at build time, so
the backend warmup crashed during the image build; `webrtcvad` imports it at runtime, so the voice
container went into a crash loop on boot. Neither change originated in Ajace AI's own code. The
failure surfaced only when the images were finally rebuilt from source after a period of in-place
`docker cp` patching, so the cause and the drift compounded each other.

**Mitigation in place.** Both Dockerfiles now re-pin setuptools as the **last** install step, after
everything that could raise it:

- `backend/Dockerfile:30-32` — with the inline comment recording the cause: *"librosa (via NeMo)
  imports pkg_resources, which setuptools >=82 removed — NeMo installs from git@main, so rebuilds
  drift and dropped it."*
- `voice/Dockerfile:77-79` — *"Re-pin setuptools LAST: kokoro/nemo/torchvision installs above pull
  in setuptools>=82."*

**Assessment.** The mitigation addresses the symptom that was observed. It does not address the
cause, which is that the dependency has no fixed version. The residual risk is that the next
incompatible change from `main` will be a different one, and will be discovered the same way — by a
build or a runtime crash. The controls that would close it are, in ascending order of effort:
pinning to a NeMo release tag; pinning to a specific commit SHA; or vendoring the subset actually
used (a `nemotron_asr` package is already vendored in the tree, so the pattern exists).

Until one of those is adopted, **every image build is a live, unversioned dependency on a
third-party branch**, and this must be stated as such to any customer who asks how the product is
built.

## 9. `mirrors.mit.edu` — an undocumented supplier

Both the backend and voice image builds, and the Ollama image build, rewrite the Ubuntu APT sources
to point at MIT's mirror:

- `backend/Dockerfile:7` and `voice/Dockerfile:5` rewrite `ports.ubuntu.com` →
  `http://mirrors.mit.edu/ubuntu-ports/`
- `docker/ollama/Dockerfile:6` does the same

`mirrors.mit.edu` is therefore a supplier of operating-system packages to every EchoMind image.
Nothing in the repository says why it was chosen, who owns the relationship, or what to do if it is
unavailable. The backend build has a crude resilience measure — a retry after a 20-second sleep
(`backend/Dockerfile:9-10`) — but no alternative mirror.

Recorded here so that it is at least identified. It is entered in
`registers/REG-02_External_Documents_and_Providers.md` as a supplier, with the rationale to be
completed by the Engineering Lead: `________________________`.

## 10. Selection, evaluation and re-evaluation (8.4.1)

### 10.1 Criteria for selection

A provider or component is selected against the following criteria. They are stated here for the
first time; existing choices were made before they were written down, and are assessed against them
retrospectively at the next review.

| # | Criterion | Why it matters for this product |
|---|---|---|
| C-1 | **Weights or source can be obtained once and cached locally** | Without this the component cannot be used offline at all |
| C-2 | **Licence permits commercial redistribution inside a customer deployment** | We ship the weights to defence, government and regulated customers |
| C-3 | **Version can be pinned to something immutable** | Reproducibility of the build (§4.2) |
| C-4 | **No telemetry, no call-home, no runtime service dependency** | QP-01 §2.2 — data does not leave the customer perimeter |
| C-5 | **Runs within the reference hardware envelope** (DGX Spark GB10) | A model that does not fit is not a candidate however good it is |
| C-6 | **Measurable on the golden evaluation** | A model swap is an S-2 change under SOP-06 and must be evidenced, not asserted |
| C-7 | **Acquisition does not require accepting terms we cannot pass to a customer** | Applies directly to the gated Nemotron model (§11) |

### 10.2 Evaluation before adoption

1. The component is added in a branch or working tree and a **from-scratch image build** is run. A
   dependency that only works incrementally is not adopted.
2. Verification per SOP-07 §5 is run. For a model swap this includes the golden evaluation, and the
   report is retained.
3. Licence and provenance are recorded in `registers/REG-02_External_Documents_and_Providers.md`
   **before** the component is merged.
4. The Engineering Lead approves. For a model, the Managing Director confirms the licence position.

### 10.3 Re-evaluation

| Trigger | Action |
|---|---|
| Annually | Review the whole of REG-02: still maintained? licence unchanged? still the best option? |
| A build failure or runtime defect attributable to a provider | Record it against that provider in REG-02; reassess the pinning |
| A provider changes licensing or access terms | Immediate reassessment; a licence change on a shipped model affects every existing customer |
| A model is swapped or upgraded | Full re-evaluation per §10.2 |
| Before each release to a customer | Confirm no dependency has silently changed since the last release (which today is only checkable by rebuilding, because nothing is version-stamped — SOP-08 §7) |

Provider performance is recorded as it is observed, not invented. The NeMo incident in §8 is the
first such record and is the template: what happened, what it cost, what was changed.

**No formal supplier assessment has been carried out for any provider.** Scores, ratings and
approved-supplier status: `________________________`.

## 11. Licence obligations — an unmet control

For a product whose entire value proposition is that it runs open-weight models inside the
customer's perimeter, licence position is a customer-facing requirement, not an administrative
detail. Defence, government and regulated customers will ask what they are being given the right to
run.

The following is the state of the repository.

| Obligation | Present state |
|---|---|
| A licence for Ajace AI's own code | **No `LICENSE` file exists** in the repository |
| Attribution for third-party components | **No `NOTICE` file and no third-party attribution document exists** |
| An inventory of model licences | **None.** No model card, licence text or licence reference is held in-repo for any of the ten models in §6 |
| A software bill of materials | **None.** No SBOM is generated at build or shipped with the offline bundle |
| Evidence for public claims | `README.md` asserts *"Models (all local, all open-weight)"*. The statement may well be correct, but **no licence evidence supporting it exists in the repository**, so it cannot presently be substantiated on request |
| Gated-model terms | `nvidia/nemotron-speech-streaming-en-0.6b` is an acceptance-gated model requiring `HF_TOKEN` (`docker-compose.yml:20-21`, `:148`). The weights are then **baked into images** (`backend/Dockerfile:42`, `voice/Dockerfile:44`) and **shipped in the offline bundle** (`scripts/export_offline_bundle.sh:41`) to customers who never accepted those terms. Whether that redistribution is permitted has not been assessed, and the assessment is: `________________________` |

QP-01 §2.3 commits the organisation to claims that are reproducible from recorded evidence. The
open-weight claim currently is not. Closing this needs four artefacts: a `LICENSE`, a `NOTICE`, a
model-licence inventory in REG-02, and an SBOM generated at build and included in the bundle.

## 12. Communicating requirements to providers (8.4.3)

Clause 8.4.3 requires requirements to be communicated to the provider before reliance. For this
product the position is honest and simple: **Ajace AI has no contractual relationship with any of
the providers in §2 or §6.** Every one of them is consumed under its public terms — a model licence,
an open-source licence, or a public service's terms of use. There is nothing to negotiate and no
provider to communicate a specification to.

The obligation therefore inverts. What must be communicated is what the organisation **accepts**:
the terms attached to each component. That acceptance is recorded in
`registers/REG-02_External_Documents_and_Providers.md`, which is the 8.4.3 record for this product.

The one relationship with a commercial account behind it is **Cloudflare** (Tunnel and Access on the
public reference instance). Its terms and the account under which it operates are recorded in
REG-02.

## 13. Records

| Record | Where it lives | Retention |
|---|---|---|
| External providers and documents register, including model licences and versions in use | `registers/REG-02_External_Documents_and_Providers.md` | Life of the product + 3 years (SOP-01 §8) |
| Pinned dependency versions at a point in time | `backend/requirements.txt`, `voice/requirements.txt`, `frontend/package-lock.json`, the Dockerfiles — all in Git history | As above |
| Rationale for adding, upgrading or unpinning a dependency | The commit body (SOP-06 §5) | As above |
| Provider incidents and their resolution | The commit body of the fix (e.g. `724fb98`), summarised against the provider in REG-02 | 3 years |
| Verification evidence for a model swap | Golden evaluation report retained per SOP-07 §8.3 | Life of the release + 3 years |
| Offline bundle contents as delivered | `MANIFEST.txt` in the bundle; copy retained with the release record | Life of the release + 3 years |
| Licence inventory, `LICENSE`, `NOTICE`, SBOM | To be created — see §14 | Life of the product + 3 years |

## 14. Current state and gaps

| # | Requirement of this procedure | Present state | Gap |
|---|---|---|---|
| G-1 | Build-time dependencies are pinned to something immutable | `nemo_toolkit @ git+…@main` in both images; `torchvision --upgrade`; `kokoro`/`soundfile` reinstall; `ollama/ollama:latest`; `cloudflare/cloudflared:latest` | **Open.** NeMo on `main` is the highest-severity item and has already caused one evidenced outage (§8) |
| G-2 | Unbounded version ranges are avoided | `accelerate>=0.26.0` and `huggingface_hub>=0.20.0` have no upper bound | **Open** |
| G-3 | Every model used at runtime is pre-cached | The reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` is not pre-downloaded despite the requirements comment saying it should be; with `HF_HUB_OFFLINE=1` the code silently falls back to the LLM rerank | **Open.** Functional and performance impact today |
| G-4 | Repository carries a licence | No `LICENSE` file | **Open** |
| G-5 | Third-party attribution | No `NOTICE` file, no attribution document | **Open** |
| G-6 | Model-licence inventory | None; no model cards or licence texts held in-repo | **Open.** Blocks substantiating the README's open-weight claim |
| G-7 | Software bill of materials | None generated or shipped | **Open** |
| G-8 | Gated-model redistribution assessed | Nemotron STT is acceptance-gated, baked into images and shipped in the offline bundle; permissibility unassessed | **Open.** Assess before the next customer delivery |
| G-9 | Suppliers are identified | `mirrors.mit.edu` was an undocumented dependency of every image build until this procedure | **Partly closed** by §9; rationale and fallback still to be recorded |
| G-10 | Formal provider evaluation and re-evaluation | No supplier has been formally assessed; no re-evaluation cycle has run | **Open.** First review due on approval of this procedure |
| G-11 | External providers register exists | `docs/qms/registers/` is empty; `REG-02` does not yet exist | **Open.** REG-02 is a prerequisite for most of the controls above |
| G-12 | Verification of external items before use is enforced | Practised manually; no CI, so a dependency change can reach `main` unverified (SOP-07 §9.1) | **Open** |

All gaps above are to be carried into `ISO9001_Gap_Analysis.md`.
