# REG-02 — External Documents and Providers Register

| Field | Value |
|---|---|
| Document ID | REG-02 |
| Revision | 1.0 |
| Status | **LIVE REGISTER — licence column is INCOMPLETE and must be completed (see §5)** |
| Owner | Lead Engineer |
| ISO 9001:2015 clauses | 7.5.3.2 (documents of external origin), 8.4.1, 8.4.2 |
| Governing procedure | SOP-09 |

---

## 1. Purpose

A single inventory of everything Ajace AI depends on but does not produce: model weights, container
base images, software packages, infrastructure services and external standards. ISO 9001:2015
requires that documents of external origin are identified and controlled (7.5.3.2) and that
externally provided processes, products and services are controlled (8.4).

For an offline product this register has a second purpose that matters more than the clause: **the
supply chain is the attack surface and the reproducibility surface.** Every item below is baked into
an image or a volume at build time and then runs, unchanged and unreachable, inside a customer's
perimeter for months.

---

## 2. Models

| # | Role | Model identifier | Provider | Sourced from | Pinned? | Licence | Licence verified |
|---|---|---|---|---|---|---|---|
| M-01 | Chat LLM (production) | `nvidia/Qwen3-30B-A3B-FP4` | NVIDIA (Qwen base, NVFP4 quantisation) | Hugging Face Hub → `trtllm_hf_cache` volume, one-time prepare | Yes — by name | `________` | ☐ |
| M-02 | Chat LLM (fallback) | `qwen2.5:7b-instruct-q4_K_M` | Alibaba Qwen, via Ollama | Ollama registry → `ollama_data` | Yes — by tag | `________` | ☐ |
| M-03 | Embeddings | `nomic-embed-text` | Nomic AI, via Ollama | Ollama registry → `ollama_data` | Tag only, no digest | `________` | ☐ |
| M-04 | Streaming STT | `nvidia/nemotron-speech-streaming-en-0.6b` | NVIDIA | HF Hub, baked into images at build (`backend/Dockerfile:42`, `voice/Dockerfile:44`) | Yes — by name | **Gated model — acceptance-gated terms; requires `HF_TOKEN`** | ☐ **priority** |
| M-05 | Final STT (voice) | `nvidia/parakeet-tdt-0.6b-v2` | NVIDIA | HF Hub, baked into voice image (`voice/Dockerfile:47-51`) | Yes — by name | `________` | ☐ |
| M-06 | Diarised ASR (boardroom) | `microsoft/VibeVoice-ASR-HF` (~18 GB) | Microsoft | HF Hub → `echomind_data:/data/hf_cache` via prepare | Yes — by name | `________` | ☐ |
| M-07 | TTS (default) | Piper `en_US-lessac-medium` | Rhasspy | `wget` from `huggingface.co/rhasspy/piper-voices` at build (`voice/Dockerfile:62-63`) | By URL | `________` | ☐ |
| M-08 | TTS (additional voices) | `en_US-{john,kusal,joe,amy}-medium` | Rhasspy | Tracked in-repo at `voice/voices/` (~302 MB), bind-mounted at `docker-compose.yml:124` | In-repo | `________` | ☐ |
| M-09 | TTS (option) | Kokoro-82M | hexgrad | `pip install kokoro` (`voice/Dockerfile:66-74`) | **No — unpinned** | `________` | ☐ |
| M-10 | Cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Microsoft / sentence-transformers | HF Hub — **not pre-downloaded; see R-11** | By name | `________` | ☐ |
| M-11 | Image generation | `stabilityai/sdxl-turbo` | Stability AI | HF Hub → `/data/docgen_models` | Yes — by name | `________` — note SDXL-Turbo has usage conditions that must be checked against commercial use | ☐ **priority** |

## 3. Base images and infrastructure

| # | Item | Provider | Pin | Risk note |
|---|---|---|---|---|
| I-01 | `nvcr.io/nvidia/pytorch:25.01-py3` | NVIDIA NGC | Pinned tag | Backend and voice base |
| I-02 | `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6` | NVIDIA NGC | Pinned tag | A release *candidate* in production use |
| I-03 | `ollama/ollama:latest` | Docker Hub | **Unpinned** | `docker/ollama/Dockerfile:2` — rebuild is non-reproducible |
| I-04 | `cloudflare/cloudflared:latest` | Docker Hub | **Unpinned** | `docker-compose.yml:224` |
| I-05 | `node:20-alpine`, `nginx:1.27-alpine` | Docker Hub | Pinned tags | Frontend build and serve |
| I-06 | Ubuntu APT packages via **`mirrors.mit.edu`** | MIT | n/a | `backend/Dockerfile:7` and `docker/ollama/Dockerfile:6` rewrite `ports.ubuntu.com` → the MIT mirror to work around transient mirror-sync failures. An **undeclared third-party dependency in the build path** — record it or remove it. |

## 4. Services

| # | Provider | Used for | Dependency type | Notes |
|---|---|---|---|---|
| S-01 | **Cloudflare** | Zero Trust Tunnel and Access for the public reference instance `echomind-ajace.com` | Run-time, public instance only | Access is the **only** login wall on the public instance (`docs/PUBLIC_DEPLOYMENT.md` step 4, "MANDATORY"). Not used in customer on-premises deployments. |
| S-02 | **Hugging Face Hub** | All model weights | Build-time / prepare-time only | Offline at run time (`HF_HUB_OFFLINE=1`) |
| S-03 | **GitHub** | Source hosting **and** a live build-time dependency on `NVIDIA/NeMo@main` | Both | See risk R-01 — the `@main` dependency is the highest-rated supplier risk |
| S-04 | **Ollama registry** | Embedding and fallback chat models | Prepare-time only | |
| S-05 | **NVIDIA NGC** | Base images | Build-time | |
| S-06 | **Docker Hub** | Base images | Build-time | Two images unpinned (I-03, I-04) |
| S-07 | **Let's Encrypt** | Optional trusted TLS certificate | Optional | `docs/HTTPS_TRUSTED_CERTIFICATE.md` |

> **No cloud AI inference provider is used.** The only cloud-inference path in the codebase is
> `DOCGEN_IMAGE_BACKEND=nim`, which is **not enabled**; production uses on-device `diffusers`. This
> is a material claim in customer conversations and it is worth keeping true and re-verifying at
> each management review.

## 5. Software dependency pinning summary

| Manifest | Entries | Exact | Ranged | Unpinned |
|---|---|---|---|---|
| `backend/requirements.txt` | 20 | 14 | 6 (`scipy`, `huggingface-hub`, `reportlab`, `diffusers`, `sentence-transformers`, and `accelerate` **with no upper bound**) | — |
| `voice/requirements.txt` | 9 | 7 | 2 (`piper-tts`; `huggingface_hub` **no upper bound**) | — |
| `frontend/package.json` | all | 0 | all caret-ranged | Reproducibility rests on `package-lock.json` + `npm ci` |
| Dockerfile direct installs | — | — | — | `nemo_toolkit[asr] @ git+…@main` (**moving branch**), `torchvision` (`--upgrade`), `kokoro`, `soundfile` |

## 6. Documents of external origin (7.5.3.2)

| Document | Version / date in use | Held where | Notes |
|---|---|---|---|
| ISO 9001:2015 *Quality management systems — Requirements* | 2015 | Not held in-repo (copyright) | The QMS is written against it; a licensed copy must be available to the auditor and the document owner |
| Model cards and licences for M-01 … M-11 | Per §2 | Not held | **Gap** — see §7 |
| Third-party package licences | Per §5 | Not held | **Gap** — no SBOM |

## 7. Open actions on this register

| Ref | Action | Priority | Owner | Due |
|---|---|---|---|---|
| A-1 | Complete the licence column for all eleven models; record each licence's commercial-use and attribution obligations | **High** — the product is sold commercially into regulated sectors | `________` | `________` |
| A-2 | Confirm the Nemotron gated-model terms (M-04) permit redistribution baked into a customer-delivered image | **High** — the model is baked into images shipped to customers | `________` | `________` |
| A-3 | Confirm SDXL-Turbo (M-11) licence terms against commercial use | **High** | `________` | `________` |
| A-4 | Add a `LICENSE` and a `NOTICE` file to the repository | High | `________` | `________` |
| A-5 | Generate an SBOM for backend, voice and frontend images | Medium | `________` | `________` |
| A-6 | Pin `nemo_toolkit` to a release or commit sha (risk R-01) | **Critical** | `________` | `________` |
| A-7 | Pin `ollama/ollama` and `cloudflare/cloudflared` to digests | Medium | `________` | `________` |
| A-8 | Decide whether the MIT mirror substitution (I-06) is retained and declared, or removed | Medium | `________` | `________` |

## 8. Re-evaluation

External providers are re-evaluated at each management review (SOP-15) and whenever an image is
rebuilt. Performance evidence to bring to that review: build failures attributable to a provider,
model behaviour changes, licence changes, and availability incidents.

| Review date | Reviewed by | Outcome |
|---|---|---|
| `________` | `________` | |
