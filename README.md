<h1 align="center">EchoMind Enterprise</h1>

<p align="center"><strong>The private, on-premises AI workspace.</strong><br/>
Chat, live meeting intelligence, voice, and document generation — running entirely on your own GPUs, fully offline.</p>

<p align="center">
  <img alt="Deployment" src="https://img.shields.io/badge/Deployment-100%25%20On--Prem-0e7490?style=flat-square">
  <img alt="Offline" src="https://img.shields.io/badge/Runtime-Offline%20%2F%20Air--gapped-0891b2?style=flat-square">
  <img alt="GPU" src="https://img.shields.io/badge/Accelerator-NVIDIA%20GPU-76b900?style=flat-square&logo=nvidia&logoColor=white">
  <img alt="Backend" src="https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white">
  <img alt="Frontend" src="https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61dafb?style=flat-square&logo=react&logoColor=black">
  <img alt="Orchestration" src="https://img.shields.io/badge/Runs%20on-Docker%20Compose-2496ed?style=flat-square&logo=docker&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-Proprietary-555?style=flat-square">
</p>

<p align="center">
  <a href="https://github.com/RW2523/echomind-enterprise/wiki"><b>📖 Wiki</b></a> ·
  <a href="docs/USER_MANUAL.md"><b>📘 User Manual</b></a> ·
  <a href="docs/CAPABILITIES.md"><b>🧠 Capabilities</b></a> ·
  <a href="docs/EchoMind_Marketing_Book.pdf"><b>📕 Marketing Book</b></a>
</p>

---

> Most AI assistants quietly send your prompts, files, and recordings to someone else's cloud. **EchoMind doesn't.** Every model — the LLM, embeddings, speech-to-text, text-to-speech, and image generation — is pre-cached and served locally on your own NVIDIA GPU hardware. No third-party AI APIs, no telemetry, no internet dependency at runtime. You can unplug the network cable and it keeps working.

Built for **defense, government, legal, finance, healthcare**, and any team for whom data sovereignty is non-negotiable.

## ✨ The Platform — Five Modules, One Web App

| Module | What it does |
|---|---|
| 💬 **Knowledge Chat** | Ask plain-language questions over your documents and transcripts, with **citations** (document · section · page). Hybrid retrieval (vector + BM25) with cross-encoder reranking. |
| 🎙️ **Live Transcription + Silent Assistant** | Real-time speech-to-text that **fact-checks each statement live** against your knowledge base — labelled *Supported / Contradicted / Unverified / Violating / Risky*. |
| 🧑‍💼 **Boardroom** | Capture a whole meeting → **speaker-diarized transcript** → AI meeting report (summary, decisions, contradictions, recommendations) → export PDF/PPTX. |
| 🔊 **Voice Conversation** | Natural, **full-duplex** speech-to-speech (barge-in, backchannels, memory), wired to the same RAG so spoken answers come from your corpus. |
| 📄 **Document Studio** | Turn a topic, a chat, or your sources into a polished document using **18 templates**, with **on-device AI images**, exported to PDF & PPTX. |

## 🗺️ Product Vision

EchoMind is a **building block**, not a one-size product. Ajace AI runs N customizations per customer — **we don't sell hardware; we fix the problem statement and advise the hardware + cloud that fit your security posture.**

- 🛰️ **Edge devices** — connect to the host's knowledge base while staying **fully offline**; intelligence at the edge, data kept home.
- 🔐 **Secure offline→online content export** — when content must go online, an **offline risk/sensitivity evaluation** flags every risk and can **redact** before anything crosses the gateway.
- 🧱 **Offline & online layers** — sensitive work stays isolated; online capability is reached only through a controlled, audited gateway.

## 🏗️ Architecture

```
                         ┌──────────────────────────────────────────────┐
   Browser (SPA) ──────▶ │  frontend (nginx)   :3000 http / :3443 https  │
   HTTPS / WSS           │  React + Vite + Tailwind                       │
                         └───────┬───────────────────────────┬──────────┘
                          /api   │                    /voice  │
                                 ▼                            ▼
                   ┌──────────────────────────┐   ┌────────────────────────┐
                   │  backend (FastAPI)  :8000 │   │  voice (FastAPI)  :8000 │
                   │  RAG · Chat · Transcribe  │   │  STT + LLM + TTS loop   │
                   │  Boardroom · Document Std │   │  (speech-to-speech /ws) │
                   └───┬───────────┬───────┬───┘   └───────┬─────────┬──────┘
            embeddings │       LLM │       │ STT/TTS    LLM │     STT │ TTS
                       ▼           ▼       ▼ (GPU)          ▼         ▼ (GPU)
              ┌──────────────┐ ┌────────────────────────┐  │   Nemotron   Piper
              │ ollama :11434│ │ trtllm  :8355 (OpenAI)  │◀─┘   (in-proc)  (ONNX)
              │ nomic-embed  │ │ Llama-3.1-8B-Instruct-FP4│
              └──────────────┘ └────────────────────────┘
```

| Service | Role | GPU |
|---|---|---|
| `frontend` | nginx serving the React SPA; proxies `/api` & `/voice` | – |
| `backend` | FastAPI: ingestion, RAG/chat, transcription, boardroom, Document Studio | 1 |
| `voice` | FastAPI WebSocket speech-to-speech loop | 1 |
| `trtllm` | OpenAI-compatible chat LLM — **Llama-3.1-8B-Instruct-FP4** | all |
| `ollama` | **Embeddings only** — `nomic-embed-text` | all |

## 🚀 Quick Start

**Prerequisites:** Linux host with NVIDIA GPU(s) + drivers, Docker + Docker Compose + NVIDIA Container Toolkit, and an `.env` with `HF_TOKEN` (to fetch the gated Nemotron model at build time).

```bash
# One-time, WITH internet: build images + populate model volumes
./scripts/prepare_offline.sh

# Thereafter: fully offline
docker compose up -d
```

Then open **`http://<HOST_IP>:3000`** (or **`https://<HOST_IP>:3443`**). Microphone features (voice, transcription) require HTTPS or `localhost`.

<details>
<summary>Build inline, air-gapped transfer & FAISS-GPU</summary>

- **Build inline:** `docker compose up --build` (or `./scripts/build.sh` if BuildKit chokes on the parallel model download).
- **Air-gapped:** export/import the `trtllm_hf_cache`, `ollama_data`, and `echomind_data` volumes — see [OFFLINE_DEPLOYMENT.md](OFFLINE_DEPLOYMENT.md).
- **Faster RAG:** set the backend build arg `USE_FAISS_GPU: "1"` in `docker-compose.yml`, then rebuild the backend.
</details>

## 🔐 Privacy & Security

- **All inference is local** — no outbound model/API calls, no telemetry; data never leaves the host.
- **Injection-resistant** — retrieved/recorded content is fenced as *untrusted data* in every prompt.
- **Trusted-network design** — this build ships without authentication and with wildcard CORS. Deploy on an **isolated network**; if exposed more broadly, put it behind your own SSO / VPN / authenticating reverse proxy and lock down CORS first. See [Security & Privacy](https://github.com/RW2523/echomind-enterprise/wiki/Security-and-Privacy).

## 📚 Documentation

| | |
|---|---|
| 📖 **[Project Wiki](https://github.com/RW2523/echomind-enterprise/wiki)** | Full handbook: architecture, every module, config, deployment |
| 📘 **User Manual** | [Markdown](docs/USER_MANUAL.md) · [PDF](docs/USER_MANUAL.pdf) — complete, step-by-step, 16 chapters |
| 🧠 **[Capabilities & How It Works](docs/CAPABILITIES.md)** | What EchoMind is and how each part works |
| 📕 **Generated showcase** | [Marketing Book](docs/EchoMind_Marketing_Book.pdf) · [User Guide](docs/EchoMind_User_Guide.pdf) — produced *in* Document Studio |
| 🔒 **HTTPS** | [Trusted cert (prod)](docs/HTTPS_TRUSTED_CERTIFICATE.md) · [Local (mkcert)](docs/HTTPS_LOCAL_TRUSTED.md) |
| 🔎 **Internals** | [RAG & chunking](docs/RAG_AND_CHUNKING_EXPLAINED.md) · [RAG flow](docs/RAG_FLOW.md) · [Chat flow](docs/CHAT_AND_RAG_FLOW.md) · [Voice flow](docs/CONVERSATION_AI_AND_WAKE_WORD_FLOW.md) · [Transcript storage](docs/TRANSCRIPT_STORAGE_FLOW.md) |

## 🛠️ Tech Stack

**Frontend:** React 19 · Vite 6 · TypeScript · Tailwind · nginx
**Backend / Voice:** Python · FastAPI · Uvicorn (NVIDIA PyTorch base image)
**Retrieval:** FAISS (dense) · rank-bm25 (sparse) · cross-encoder reranker · SQLite
**Models (all local):** Llama-3.1-8B-Instruct-FP4 (TensorRT-LLM) · `nomic-embed-text` (Ollama) · NVIDIA Nemotron streaming ASR · VibeVoice diarization · Piper TTS · SDXL-Turbo
**Infra:** Docker Compose · NVIDIA Container Toolkit · offline-first model caches in named volumes

## 🖥️ Requirements

- Linux + NVIDIA GPU(s) with recent drivers (DGX / DGX Spark supported; x86_64 and ARM64)
- Docker, Docker Compose, NVIDIA Container Toolkit
- `HF_TOKEN` for the gated Nemotron model (build time only)

---

<p align="center"><sub>© Ajace AI · EchoMind Enterprise — a private AI platform. Proprietary.</sub></p>
