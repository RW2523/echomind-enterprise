# EchoMind Enterprise — Capabilities & How It Works

EchoMind Enterprise is a **fully on‑premises, offline‑first AI knowledge platform**. It runs entirely on your own NVIDIA GPU hardware (DGX / DGX Spark) with **no external API calls, no telemetry, and no internet dependency at runtime** — every model (LLM, embeddings, speech‑to‑text, text‑to‑speech, image generation) is pre‑cached and served locally inside Docker.

It bundles five products behind one web UI:

| Module | What it does |
|---|---|
| **Knowledge Chat** | Ask questions over your uploaded documents and transcripts with citations (advanced RAG). |
| **Live Transcription + Silent Assistant** | Real‑time speech‑to‑text that fact‑checks what's said against your knowledge base, live. |
| **Boardroom** | Record a whole meeting → speaker‑diarized transcript → AI meeting report (PDF/PPTX). |
| **Voice Conversation** | Full‑duplex speech‑to‑speech assistant (barge‑in, backchannels, memory), RAG‑connected. |
| **Document Studio** | Generate a polished, templated document from a chat, uploaded sources, or a brief → export to PDF & PPTX, with on‑device AI images. |

> **Security posture:** This build ships **without authentication** and with wildcard CORS — it is designed for a **trusted, isolated network**. See [Security & Privacy](#security--privacy) before exposing it.

---

## Table of contents

- [Architecture overview](#architecture-overview)
- [Module 1 — Knowledge Chat (RAG)](#module-1--knowledge-chat-rag)
- [Module 2 — Live Transcription + Silent Assistant](#module-2--live-transcription--silent-assistant)
- [Module 3 — Boardroom](#module-3--boardroom)
- [Module 4 — Voice Conversation](#module-4--voice-conversation)
- [Module 5 — Document Studio](#module-5--document-studio)
- [Data & storage](#data--storage)
- [Security & privacy](#security--privacy)
- [Configuration highlights](#configuration-highlights)
- [Deployment](#deployment)
- [Technology stack](#technology-stack)

---

## Architecture overview

EchoMind is a set of containerized microservices orchestrated by Docker Compose. The browser talks only to nginx; nginx proxies API and WebSocket traffic to the backend and voice services. All model inference happens on local GPUs.

```
                         ┌──────────────────────────────────────────────┐
   Browser (SPA) ──────▶ │  frontend (nginx)   :3000 http / :3443 https  │
   HTTPS/WSS             │  React + Vite + Tailwind, serves static build  │
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

### Services (docker-compose)

| Service | Container | Port | GPU | Role |
|---|---|---|---|---|
| `frontend` | echomind-frontend | 3000 (http), 3443 (https) | – | nginx serving the React SPA; reverse-proxies `/api`→backend and `/voice`→voice |
| `backend` | echomind-backend | 8000 (internal) | 1 | FastAPI: document ingestion, RAG/chat, live transcription, boardroom, Document Studio |
| `voice` | echomind-voice | 8002 (host) → 8000 | 1 | FastAPI WebSocket speech‑to‑speech loop (STT → LLM → TTS) |
| `trtllm` | TensorRT‑LLM | 8355 (internal) | all | OpenAI‑compatible chat LLM server — **Llama‑3.1‑8B‑Instruct‑FP4** |
| `ollama` | Ollama | 11434 | all | **Embeddings only** — `nomic-embed-text` (chat does *not* use Ollama) |

### Models

| Purpose | Model | Served by |
|---|---|---|
| Chat / generation LLM | `nvidia/Llama-3.1-8B-Instruct-FP4` (4‑bit FP4) | trtllm (`LLM_BASE_URL=http://trtllm:8355/v1`) |
| Embeddings (RAG) | `nomic-embed-text` | ollama (`OLLAMA_EMBED_URL`) |
| Speech‑to‑text (streaming) | `nvidia/nemotron-speech-streaming-en-0.6b` (NeMo) | in‑process in backend (transcribe) and voice |
| Meeting diarization | VibeVoice‑ASR (subprocess) with a Nemotron clustering fallback | backend (boardroom) |
| Text‑to‑speech | Piper (ONNX, e.g. `en_US-lessac-medium`) | voice |
| Document images | `stabilityai/sdxl-turbo` (diffusers) | backend (Document Studio) |

### Reranking
A cross‑encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks RAG candidates when enabled.

### Offline‑first design
- Runtime flags `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `TRTLLM_SKIP_DOWNLOAD=1`, `OLLAMA_OFFLINE=1`, `VOICE_OFFLINE=1` block all network model fetches.
- Weights are populated once (`scripts/prepare_offline.sh` / `docker-compose.prepare.yml`) into Docker volumes (`trtllm_hf_cache`, `ollama_data`, `echomind_data`), then the stack runs air‑gapped.
- **GPU fault recovery:** a fatal CUDA fault (which poisons the shared GPU context) is detected by a watchdog; the affected service's `/health` returns 503 and the process exits, and Docker's `restart: unless-stopped` recreates it with a fresh context.

---

## Module 1 — Knowledge Chat (RAG)

**What it does.** Upload PDFs / DOCX / PPTX (and capture transcripts), then ask natural‑language questions. Answers are grounded in your corpus, stream token‑by‑token, and come with source citations (document, section path, page). It is tuned for very large, structured regulatory documents (e.g. a 7,000‑page DoD FMR) as well as everyday docs and meeting transcripts.

**Key capabilities**
- Hybrid retrieval (dense vector + sparse BM25) with weighted Reciprocal Rank Fusion.
- Hierarchical, "book‑aware" retrieval: section index, table‑of‑contents routing, parent/child chunks, cross‑reference graph expansion, dedicated glossary index for definition queries.
- Cross‑encoder reranking + optional MMR diversity.
- Query understanding: type classification (definition / citation / procedural / narrative), deterministic query variants (typo fixes, quoted phrases), optional LLM intent rewrite.
- Evidence extraction, optional evidence/answer "gates" that refuse to answer on weak context, and prompt‑injection fencing of all retrieved text as untrusted data.
- Personas (Teacher, Financial Advisor, Lawyer, AI Expert & Manager, General Assistant, Funny & Calming, EchoMind Guide) and time‑window filters (24h / 48h / 1w / all).
- Multi‑turn memory via a rolling conversation summary.

**How it works**

1. **Ingestion** (`POST /api/docs/upload` → `rag/parse.py` → `rag/chunking/pipeline.py` → `rag/index.py`)
   - Extract text (PyMuPDF/pypdf, python‑docx, python‑pptx), strip repeated headers/footers, track page offsets.
   - Detect document type (BOOK / SENSITIVE / FAQ / USER / UNSTRUCTURED) and chunk accordingly (book = parent 2k–3.5k tok + child 500–700 tok, sentence‑aware).
   - For books, generate **contextual headers** per chunk (LLM section summary + chunk role) for better embeddings.
   - Embed via Ollama `nomic-embed-text`; index into **FAISS** (dense), **BM25** (sparse), plus **section**, **glossary**, **TOC**, and **cross‑reference** indexes. Metadata + chunks persist in SQLite.

2. **Retrieval** (`rag/advanced.py::retrieve_semantic_first`)
   - Classify query → resolve explicit section refs / TOC routing → restrict to candidate sections → dense+sparse hybrid → RRF fusion → time‑decay/tag boost → rerank → graph expansion → glossary priority → dedupe, context‑window filter, authoritative‑source preference, context‑budget trim.

3. **Answer** (`api/routes/chat.py`)
   - Build evidence block, assemble persona system prompt + history + fenced context, then stream from TensorRT‑LLM (`/v1/chat/completions`). Post‑process to attach citations; persist messages and update the conversation summary in the background.

**Main endpoints:** `POST /api/chat/ask`, `POST /api/chat/ask-stream`, `POST /api/chat/ask-voice[-stream]`, `POST /api/docs/upload`, `GET /api/docs/list`, `POST /api/chat/debug-retrieval`.

---

## Module 2 — Live Transcription + Silent Assistant

**What it does.** Streams microphone audio to a real‑time transcript. As complete sentences/paragraphs form, the **Silent Assistant** quietly checks each statement against your knowledge base and surfaces a card labelled **Supported / Contradicted / Unverified / Violating / Risky Statement** with a short explanation and source references — useful for compliance, fact‑checking, and live note‑taking. Transcripts are auto‑saved into the knowledge base so they become searchable in Knowledge Chat.

**How it works**
- The browser captures 16 kHz PCM and streams it over a WebSocket (`/api/transcribe/ws`). The backend runs the **Nemotron streaming ASR** (GPU, serialized via a semaphore) and emits live partials.
- `session_state.py` stabilizes text (anti‑duplication, overlap removal) and segments it into paragraphs using punctuation + silence/length rules.
- When a paragraph finalizes, `analyzer.py::analyze_segment` runs a **document‑only RAG lookup** + an LLM fact‑check (only emits if confidence ≥ 60), stores it in `transcript_analysis`, and pushes an `analysis` card to the UI.
- Auto‑store (default every 60 s and on stop) writes the transcript to SQLite and embeds it into the KB (tagging + conversation‑type inference). Manual refine/store and editing are also supported.
- Robustness: bounded PCM queue with backpressure, per‑session concurrency caps, max concurrent sessions, and client auto‑reconnect that preserves the visible transcript.

**Main endpoints:** `WS /api/transcribe/ws`, `GET /api/transcribe/list`, `GET/DELETE /api/transcribe/transcripts/{id}`, `GET .../analysis`, `POST /api/transcribe/refine`, `POST /api/transcribe/store`.

---

## Module 3 — Boardroom

**What it does.** Captures an entire meeting, produces a **speaker‑diarized transcript** ("Speaker 1/2/…"), then generates an **AI meeting report** — executive summary, per‑speaker summaries & key points, key topics, RAG‑verified facts, contradictions against your documents, recommendations, and overall sentiment — exportable as **PDF or PPTX**.

**How it works**
- The frontend records audio (MediaRecorder, WebM) and uploads it in chunks (`POST /api/boardroom/sessions/{id}/chunks`) with format/size/path‑traversal validation.
- **Finalize** concatenates chunks → 16 kHz WAV (ffmpeg) → diarization. Primary path is **VibeVoice‑ASR in an isolated subprocess** (separate CUDA context so it can't poison the shared model); fallback is a Nemotron pipeline with energy VAD + spectral speaker clustering + segment merging. GPU work is serialized by a semaphore.
- **Analyse** builds the full transcript, runs a document‑only RAG lookup, and asks the LLM for a structured JSON report (`analyse_meeting`), stored as `report_json`.
- **Export** renders the report to PDF (reportlab) or PPTX (python‑pptx). State transitions (`recording → processing → transcribed → analysing → analysed`) are enforced atomically; the UI polls every 3 s while processing.

**Main endpoints:** `POST /api/boardroom/sessions`, `POST .../chunks`, `POST .../finalize`, `POST .../analyse`, `GET .../report`, `GET .../export?format=pdf|pptx`.

---

## Module 4 — Voice Conversation

**What it does.** A natural, full‑duplex **speech‑to‑speech** assistant. You talk, it listens, thinks, and replies in a synthesized voice — and it's connected to your knowledge base, so questions about your documents/transcripts are answered from RAG. It supports personas and a configurable assistant/user name, timezone, and location.

**Conversational features**
- **Barge‑in:** start talking over the assistant and it stops instantly and listens.
- **Lead phrases:** an instant, no‑LLM filler ("Let me check that…") removes dead‑air while the LLM/RAG runs.
- **Backchannels:** brief "mm‑hmm / I see" during long user turns so it never feels frozen.
- **Semantic endpointing:** adaptive silence thresholds detect end‑of‑turn faster when the sentence sounds complete.
- **Listen‑only mode** and **memory queries** ("what did I say in the last 5 minutes", "summarize the last 10 min") over a rolling 30‑minute window, plus voice commands (set name, fact‑check, recap, clear memory).

**How it works**
- Browser captures mic audio, resamples to 16 kHz in an AudioWorklet, and streams 20 ms PCM frames over a WebSocket (`/voice/ws`). Playback uses a separate gapless‑scheduled audio clock.
- The voice service runs **WebRTC VAD + endpointing → Nemotron STT (streaming partial + final) → LLM (streaming) → Piper TTS (phrase‑committed, chunked)**. The LLM is the same TensorRT‑LLM server (`LLM_URL=http://trtllm:8355`), tuned for brevity.
- Knowledge questions are routed to the backend (`/api/chat/ask-voice-stream`) so spoken answers are RAG‑grounded; transcripts injected as context are fenced as untrusted data.
- CUDA fault auto‑recovery: a watchdog exits the process on a fatal GPU error so Docker restarts it with a clean context; `/health` returns 503 in the meantime. *(Because STT holds the GPU, a fatal CUDA fault is recovered only by a container restart — there is no in‑process recovery.)*

---

## Module 5 — Document Studio

**What it does.** Turns a topic, a chat, or uploaded reference documents into a **finished, professionally formatted document** and exports it as **PDF and PPTX** — with optional **AI‑generated images** rendered on‑device. It ships **18 built‑in templates** and supports uploading your own template to match a house style.

**Built‑in templates (18):** Technical Document · Business / Executive Report · SOP / Process Document · Training / Learning Guide · Legal / Compliance Brief · Meeting / Conversation Summary · Pitch Deck · Whitepaper / Research Report · Product Requirements (PRD) · Project Proposal · Case Study · System Architecture Document · Marketing Book / Go‑to‑Market Playbook · Promotional Flyer · Brand Book / Brand Guidelines · Marketing Campaign Plan · Social Media Playbook · Product Launch / Go‑to‑Market Plan.

**Key capabilities**
- Three sources: **from a chat**, **from uploaded documents**, or **from a brief/topic**.
- A flat, renderer‑agnostic block model (heading, paragraph, bullets, table, flow diagram, callout, image, divider, page break) drives both renderers.
- Professional **PDF** (reportlab: cover hero, live table of contents, themed sections, banded tables, page‑fitted flow diagrams, callouts, embedded images) and **PPTX** (python‑pptx 16:9: themed slides, native hanging‑indent bullets, flow chains, image slides).
- **On‑device images** via SDXL‑Turbo (diffusers); an "art‑director" sub‑step decides placement and rewrites prompts to be diffusion‑safe. Offline synthetic (`local`) and cloud (`nim`/`comfyui`) backends are also supported.
- **Custom template upload** (pptx / docx / pdf / md / txt) so generated content can be rendered into your own deck.
- Quality guards baked into the pipeline: strict on‑topic enforcement, refusal/scaffolding/markdown scrubbing, placeholder bans, heading de‑duplication, empty‑section pruning, and word‑boundary truncation.

**How it works** (`backend/app/docgen/`)
1. **Plan** — a planner produces a section blueprint from the chosen template + subject/sources.
2. **Write** — sections are written in parallel by the LLM, each strictly grounded in the document subject; output is normalized into the block model (`models.py`).
3. **Illustrate** — an image planner + art‑director decide per‑section prompts and generate images with SDXL‑Turbo (`images.py`); images are placed at the end of their section.
4. **Render & export** — `render_pdf.py` / `render_pptx.py` produce the files. Generation runs as a background **job** you poll, and exports are served with `Cache-Control: no-store` so re‑exports are never stale.

**Main endpoints:** `GET /api/docgen/templates`, `GET /api/docgen/image-status`, `POST /api/docgen/templates/upload`, `POST /api/docgen/generate`, `GET /api/docgen/jobs[/{id}]`, `GET /api/docgen/jobs/{id}/export`, `DELETE /api/docgen/jobs/{id}`.

> **Hardware note:** With images enabled, SDXL‑Turbo is loaded as a singleton holding ~18 GB of GPU memory; budget for this alongside the LLM and STT models.

---

## Data & storage

Everything persists under the `echomind_data` volume (`/data`):

```
/data
├── echomind.sqlite         # chats, messages, documents, chunks, book_sections,
│                           # transcripts, transcript_analysis, boardroom_sessions,
│                           # docgen_jobs, docgen_templates
├── faiss.index             # dense vectors (documents + transcripts)
├── faiss_meta.json         # chunk metadata
├── sparse_meta.json        # BM25 term index
├── faiss_transcript.index  # transcript-only dense index
├── faiss_section.index     # hierarchical section index (+ section_meta.json)
├── faiss_glossary.index    # glossary/definitions index (+ glossary_meta.json)
├── cross_ref_graph.json    # section cross-reference graph
├── docgen_models/          # SDXL-Turbo (diffusers) cache
├── boardroom/<id>/         # uploaded meeting audio chunks
└── uploads/                # uploaded source files
```

Model weights live in separate volumes: `trtllm_hf_cache` (LLM) and `ollama_data` (embeddings). For air‑gapped installs, these volumes can be exported/imported as tarballs.

---

## Security & privacy

**Privacy by design.** All inference is local. There are no outbound model/API calls and no telemetry; data never leaves the host. All retrieved/recorded content is fenced as *untrusted data* in LLM prompts to resist prompt‑injection from the corpus.

**Deliberate gaps (deploy on a trusted network).**
- **No authentication.** There is no login, API key, or session — anyone who can reach the backend port can call every endpoint.
- **Wildcard CORS** (`allow_origins=["*"]`).

If you expose EchoMind beyond an isolated LAN, put it behind an authenticating reverse proxy / VPN / mTLS and lock down CORS first. Uploads are validated for type and size, and boardroom chunk paths are checked against traversal.

---

## Configuration highlights

All settings are environment variables (see `backend/app/core/config.py`, `docker-compose.yml`, and the voice service config). A few of the most important:

| Variable | Purpose | Typical value |
|---|---|---|
| `LLM_BASE_URL` / `LLM_MODEL` | Chat LLM endpoint & model | `http://trtllm:8355/v1` · `nvidia/Llama-3.1-8B-Instruct-FP4` |
| `OLLAMA_EMBED_URL` / `OLLAMA_EMBED_MODEL` | Embeddings endpoint & model | `http://ollama:11434/api/embeddings` · `nomic-embed-text` |
| `ECHOMIND_ASR_MODEL_NAME` / `ECHOMIND_ASR_DEVICE` | STT model & device | `nvidia/nemotron-speech-streaming-en-0.6b` · `cuda` |
| `RAG_USE_SECTION_RETRIEVAL`, `RAG_USE_RERANKER`, `RAG_USE_GRAPH_EXPANSION`, `RAG_CONTEXT_MAX_CHARS` | RAG behaviour (130+ knobs) | enabled · `24000` |
| `DOCGEN_IMAGE_BACKEND` / `DOCGEN_DIFFUSERS_MODEL` | Document Studio images | `diffusers` · `stabilityai/sdxl-turbo` |
| `TRANSCRIPT_WS_MAX_SESSIONS`, `TRANSCRIPT_GPU_CONCURRENCY` | Live transcription limits | `16` · `1` |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `TRTLLM_SKIP_DOWNLOAD`, `OLLAMA_OFFLINE`, `VOICE_OFFLINE` | Offline mode | `1` |
| `HF_TOKEN` | Build‑time download of gated models (Nemotron) | (from `.env`) |

---

## Deployment

**Prerequisites:** Linux host with NVIDIA GPU(s) + drivers, Docker + Docker Compose + NVIDIA Container Toolkit, and an `.env` with `HF_TOKEN` (for the gated Nemotron model at build time).

```bash
# One-time, with internet: build images and populate model volumes
./scripts/prepare_offline.sh

# Thereafter: fully offline
docker compose up -d

# Open the app
#   http://<HOST_IP>:3000   (or  https://<HOST_IP>:3443)
```

For trusted HTTPS (no browser warning) see `docs/HTTPS_TRUSTED_CERTIFICATE.md` (production, Let's Encrypt) or `docs/HTTPS_LOCAL_TRUSTED.md` (local, mkcert). nginx is tuned for streaming (`proxy_buffering off`) and long‑lived WebSockets (`proxy_read_timeout 86400`).

---

## Technology stack

- **Frontend:** React 19, Vite 6, TypeScript, Tailwind, served by nginx (HTTP/HTTPS, SPA + `/api` & `/voice` proxy).
- **Backend / Voice:** Python, FastAPI, Uvicorn (base image `nvcr.io/nvidia/pytorch`).
- **Retrieval:** FAISS (dense), rank‑bm25 (sparse), cross‑encoder reranker; SQLite for metadata.
- **Models:** TensorRT‑LLM (Llama‑3.1‑8B‑FP4), Ollama `nomic-embed-text`, NVIDIA Nemotron streaming ASR, VibeVoice diarization, Piper TTS, SDXL‑Turbo (diffusers).
- **Docs:** pypdf, PyMuPDF, python‑docx, python‑pptx, reportlab, Pillow.
- **Infra:** Docker Compose, NVIDIA Container Toolkit; offline‑first with model caches in named volumes.

---

### Related docs
- `README.md` — quick start, HTTPS, offline run
- `docs/RAG_AND_CHUNKING_EXPLAINED.md`, `docs/RAG_FLOW.md`, `docs/BOOKRAG_CHECKLIST.md` — RAG internals
- `docs/CHAT_AND_RAG_FLOW.md`, `docs/CHAT_SESSION_AND_SUMMARY_FLOW.md` — chat pipeline
- `docs/CONVERSATION_AI_AND_WAKE_WORD_FLOW.md` — voice assistant
- `docs/TRANSCRIPT_STORAGE_FLOW.md` — transcription storage
