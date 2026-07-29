<h1 align="center">EchoMind Enterprise</h1>

<p align="center"><strong>The private, on-premises AI workspace.</strong><br/>
Chat, live meeting intelligence, voice, and document generation — running entirely on your own GPU hardware, fully offline.</p>

<p align="center">
  <img alt="Deployment" src="https://img.shields.io/badge/Deployment-100%25%20On--Prem-0e7490?style=flat-square">
  <img alt="Offline" src="https://img.shields.io/badge/Runtime-Offline%20%2F%20Air--gapped-0891b2?style=flat-square">
  <img alt="GPU" src="https://img.shields.io/badge/Accelerator-NVIDIA%20DGX%20Spark%20(GB10)-76b900?style=flat-square&logo=nvidia&logoColor=white">
  <img alt="Models" src="https://img.shields.io/badge/Models-100%25%20Open--Weight-16a34a?style=flat-square">
  <img alt="Backend" src="https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white">
  <img alt="Frontend" src="https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61dafb?style=flat-square&logo=react&logoColor=black">
  <img alt="Orchestration" src="https://img.shields.io/badge/Runs%20on-Docker%20Compose-2496ed?style=flat-square&logo=docker&logoColor=white">
  <img alt="Eval" src="https://img.shields.io/badge/Golden%20Eval-48%2F52-8b5cf6?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/License-Proprietary-555?style=flat-square">
</p>

<p align="center">
  <a href="https://github.com/RW2523/echomind-enterprise/wiki"><b>📖 Wiki</b></a> ·
  <a href="docs/USER_MANUAL.md"><b>📘 User Manual</b></a> ·
  <a href="docs/CAPABILITIES.md"><b>🧠 Capabilities</b></a> ·
  <a href="docs/EchoMind_Marketing_Book.pdf"><b>📕 Marketing Book</b></a>
</p>

---

> Most AI assistants quietly send your prompts, files, and recordings to someone else's cloud. **EchoMind doesn't.** Every model — the LLM, embeddings, speech-to-text, text-to-speech, and image generation — is open-weight, pre-cached, and served locally on your own NVIDIA hardware. No third-party AI APIs, no telemetry, no internet dependency at runtime. You can unplug the network cable and it keeps working.

Built for **defense, government, legal, finance, healthcare**, and any team for whom data sovereignty is non-negotiable. The reference deployment runs the *entire* platform — LLM, two speech models, retrieval, and image generation — on a **single NVIDIA DGX Spark (GB10 Grace-Blackwell, 128 GB unified memory)**.

## ✨ The Platform — Five Modules, One Web App

| Module | What it does |
|---|---|
| 💬 **Knowledge Chat** | Ask plain-language questions over your documents and transcripts, with **precise citations** (document · section · page). Hybrid retrieval (FAISS + BM25, weighted RRF) with a cross-encoder reranker and a relevance gate — junk context is never force-fit into an answer. |
| 🎙️ **Live Transcription + Silent Assistant** | Real-time speech-to-text that **fact-checks each spoken paragraph live** against your knowledge base — labelled *Supported / Contradicted / Unverified / Violating / Risky*, with per-industry rule packs. |
| 🧑‍💼 **Boardroom** | Capture a whole meeting → **speaker-diarized transcript** → AI meeting report (summary, decisions, contradictions, recommendations) → export PDF/PPTX. |
| 🔊 **Voice Conversation** | Natural, **full-duplex** speech-to-speech (sub-second replies, barge-in, backchannels, semantic endpointing, memory), wired to the same RAG so spoken answers come from your corpus. |
| 📄 **Document Studio** | Turn a topic, a chat, or your sources into a polished document using **18 templates**, with **on-device AI images** (SDXL-Turbo), exported to PDF & PPTX. |

### 🧠 Conversational intelligence

Chat and voice share a **three-layer routing stack** so the assistant behaves like a person, not a search box:

1. **Rules** (0 ms) — greetings, thanks, and topic refusals are recognized instantly and never trigger retrieval.
2. **Semantic intent** (~15 ms, embedding-based) — novel phrasings ("how's life treating you?") are classified by *meaning*; the query embedding is reused by retrieval, so real questions pay nothing.
3. **Cross-encoder relevance gate** — if even the best retrieved passage doesn't relate to the question, the context is discarded and the answer is honest instead of force-fit. Citations are filtered the same way: only passages the model scored as genuinely relevant are cited.

Ten personas (Financial Advisor, Lawyer, Clinical Assistant, Teacher, and more) each answer in their own voice — with universal guards: never infer the user's situation from stored documents, always drop a declined topic, never fabricate citations.

### 🏷️ Vertical packs — one app, five industry editions

The same deployment serves isolated industry experiences by subdomain: **Health · Law · Meeting Rooms · Retail · Bank**. Each pack gets its own knowledge-base **namespace** (enforced inside the search index — tenants cannot leak), persona, theme, and UI copy. Adding a vertical is configuration, not a rebuild.

## 🗺️ Product Vision

EchoMind is a **building block**, not a one-size product. Ajace AI runs N customizations per customer — **we don't sell hardware; we fix the problem statement and advise the hardware + cloud that fit your security posture.**

- 🛰️ **Edge devices** — connect to the host's knowledge base while staying **fully offline**; intelligence at the edge, data kept home.
- 🔐 **Secure offline→online content export** — when content must go online, an **offline risk/sensitivity evaluation** flags every risk and can **redact** before anything crosses the gateway.
- 🧱 **Offline & online layers** — sensitive work stays isolated; online capability is reached only through a controlled, audited gateway.

## 🏗️ Architecture

```
                         ┌──────────────────────────────────────────────┐
   Browser (SPA) ──────▶ │  frontend (nginx)   :3000 http / :3443 https │
   HTTPS / WSS           │  React + Vite + Tailwind                     │
                         └───────┬───────────────────────────┬──────────┘
                          /api   │                    /voice │
                                 ▼                           ▼
                   ┌───────────────────────────┐  ┌─────────────────────────┐
                   │  backend (FastAPI)  :8000 │  │  voice (FastAPI)  :8000 │
                   │  RAG · Chat · Transcribe  │  │  STT → LLM → TTS loop   │
                   │  Boardroom · Document Std │  │  (speech-to-speech /ws) │
                   └───┬───────────────┬───────┘  └───────┬─────────┬───────┘
             LLM +     │      FAISS +  │ STT (GPU)   LLM  │     STT │ TTS (CPU)
             embeddings│      BM25 + CE│ Nemotron         │  Nemotron (GPU)
                       ▼      reranker ▼ (in-proc)        ▼  + Parakeet (CPU)
              ┌──────────────────────┐            ┌──────────────┐   Piper /
              │  ollama  :11434      │◀───────────│  same ollama │   Kokoro
              │  Qwen2.5-7B (chat)   │            └──────────────┘
              │  nomic-embed (RAG)   │
              └──────────────────────┘
```

| Service | Role | GPU |
|---|---|---|
| `frontend` | nginx serving the React SPA; proxies `/api` & `/voice` (WS-aware, unbuffered streaming) | – |
| `backend` | FastAPI: ingestion, RAG/chat, live transcription + Silent Assistant, Boardroom, Document Studio | ✅ |
| `voice` | FastAPI WebSocket speech-to-speech loop (VAD, semantic endpointing, barge-in) | ✅ |
| `ollama` | Chat LLM (**Qwen2.5-7B-Instruct, Q4**) + embeddings (`nomic-embed-text`) | ✅ |
| `trtllm` | *Optional, profile-gated:* TensorRT-LLM serving **Qwen3-30B-A3B-FP4** — enabled once GB10 FP4 kernels mature (`--profile trtllm`) | ✅ |
| `cloudflared` | *Optional, profile-gated:* public access via Cloudflare Tunnel (`--profile public`) | – |

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
- **Public demo:** `docker compose --profile public up -d cloudflared` — see [docs/PUBLIC_DEPLOYMENT.md](docs/PUBLIC_DEPLOYMENT.md) and gate it with Cloudflare Access.
</details>

## ✅ Quality — the golden-question eval

Retrieval and conversation quality are measured, not eyeballed. A **52-question golden suite** (facts mined and verified from the actual corpus, per namespace, plus routing regressions) runs against the live stack:

```bash
python3 eval/run_eval.py          # exit code 0 = all pass; JSON report in eval/reports/
```

Scored: routing (small talk / refusals / off-corpus must never cite), expected-document hit-rate, **citation precision (0.98)**, answer facts, hallucination canaries, latency. Current: **48/52** — the remainder are deliberate hard sentinels (cross-volume enumeration, cross-document comparison). Run it after any retrieval/prompt/model change. See [eval/README.md](eval/README.md).

## 🔐 Privacy & Security

- **All inference is local** — no outbound model/API calls, no telemetry; data never leaves the host.
- **Injection-resistant** — retrieved/recorded content is fenced as *untrusted data* in every prompt; documents cannot steer the model.
- **Tenant isolation** — knowledge-base namespaces are enforced inside every index search path, with a second post-retrieval filter.
- **Authentication is opt-in** — local accounts + JWT ship in the box (`AUTH_ENABLED=1`); the default build runs open for trusted networks with wildcard CORS. If exposed publicly, enable auth and/or front it with Cloudflare Access / SSO / VPN and lock down CORS. See [Security & Privacy](https://github.com/RW2523/echomind-enterprise/wiki/Security-and-Privacy).

## 📚 Documentation

| | |
|---|---|
| 📖 **[Project Wiki](https://github.com/RW2523/echomind-enterprise/wiki)** | Full handbook: architecture, every module, config, deployment |
| 📘 **User Manual** | [Markdown](docs/USER_MANUAL.md) · [PDF](docs/USER_MANUAL.pdf) — complete, step-by-step, 16 chapters |
| 🧠 **[Capabilities & How It Works](docs/CAPABILITIES.md)** | What EchoMind is and how each part works |
| 🌐 **[Public deployment](docs/PUBLIC_DEPLOYMENT.md)** | Cloudflare Tunnel + Access runbook |
| 📕 **Generated showcase** | [Marketing Book](docs/EchoMind_Marketing_Book.pdf) · [User Guide](docs/EchoMind_User_Guide.pdf) — produced *in* Document Studio |
| 🔒 **HTTPS** | [Trusted cert (prod)](docs/HTTPS_TRUSTED_CERTIFICATE.md) · [Local (mkcert)](docs/HTTPS_LOCAL_TRUSTED.md) |
| 🔎 **Internals** | [RAG & chunking](docs/RAG_AND_CHUNKING_EXPLAINED.md) · [RAG flow](docs/RAG_FLOW.md) · [Chat flow](docs/CHAT_AND_RAG_FLOW.md) · [Voice flow](docs/CONVERSATION_AI_AND_WAKE_WORD_FLOW.md) · [Transcript storage](docs/TRANSCRIPT_STORAGE_FLOW.md) |

## 🛠️ Tech Stack

**Frontend:** React 19 · Vite 6 · TypeScript · Tailwind · nginx
**Backend / Voice:** Python · FastAPI · Uvicorn (NVIDIA PyTorch base image)
**Retrieval:** FAISS (dense) · rank-bm25 (sparse) · weighted RRF fusion · `ms-marco-MiniLM-L-6-v2` cross-encoder reranker + relevance gate · Anthropic-style contextual chunk headers · SQLite
**Models (all local, all open-weight):** Qwen2.5-7B-Instruct Q4 via Ollama (chat) · Qwen3-30B-A3B-FP4 via TensorRT-LLM (design target, profile-gated) · `nomic-embed-text` (embeddings) · NVIDIA Nemotron streaming ASR (live partials, GPU) · NVIDIA Parakeet-TDT (accurate final STT) · VibeVoice diarization · Piper TTS (+ Kokoro-82M option) · SDXL-Turbo (images)
**Infra:** Docker Compose · NVIDIA Container Toolkit · offline-first model caches in named volumes · optional Cloudflare Tunnel

## 🖥️ Requirements

- Linux + NVIDIA GPU(s) with recent drivers — reference: **DGX Spark GB10** (driver ≥ 580.159), x86_64 and ARM64 supported
- Docker, Docker Compose, NVIDIA Container Toolkit
- `HF_TOKEN` for the gated Nemotron model (build time only)

---

<p align="center"><sub>© Ajace AI · EchoMind Enterprise — a private AI platform. Proprietary.</sub></p>
