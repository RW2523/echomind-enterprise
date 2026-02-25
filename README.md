# EchoMind — Enterprise Edition (Clean v2)

This build removes ALL Gemini code and connects the UI to your backend APIs.

## Services

| Service        | Port (host)     | Role |
|----------------|------------------|------|
| Frontend       | 3000 (HTTP), 3443 (HTTPS) | UI; proxies `/api` → backend, `/voice` → voice |
| Backend        | (internal 8000)  | EchoMind API; uses RAG Platform for docs/query/transcript ingest |
| RAG Platform   | (internal 8000)  | Main RAG: Qwen embedder + generator, Qdrant |
| Voice          | 8001             | Voice bot + Whisper; calls backend for RAG answers |
| Qdrant         | 6333             | Vector DB for RAG Platform |
| Ollama         | 11434            | LLM + embeddings (used by backend/voice when not using RAG platform for generation) |

- **Open the app:** http://\<host\>:3000 or https://\<host\>:3443
- **Backend API** is reached via the frontend at `/api` (no need to expose backend port)
- **Voice** is at `/voice` through the frontend, or directly at http://\<host\>:8001
- **Voice AI is connected to RAG** (via `BACKEND_CHAT_URL`): questions about transcripts or uploaded PDFs are answered from the knowledge base (RAG Platform when `RAG_PLATFORM_URL` is set)

## HTTPS (no browser warning)

**Production (server with a hostname)**  
Trusted HTTPS with a free domain and Let's Encrypt:
- Free subdomain: [DuckDNS](https://www.duckdns.org/) → e.g. **echomind.duckdns.org**
- Trusted cert: **`sudo certbot --nginx -d echomind.duckdns.org`**
- **HTTPS:** https://echomind.duckdns.org — no warning. Full steps: **[docs/HTTPS_TRUSTED_CERTIFICATE.md](docs/HTTPS_TRUSTED_CERTIFICATE.md)**

**Local development**  
Trusted HTTPS on localhost with [mkcert](https://github.com/FiloSottile/mkcert):
- Install mkcert, run `mkcert -install`, create certs for localhost, then run the frontend with `VITE_DEV_HTTPS=1` and `VITE_SSL_CERT` / `VITE_SSL_KEY`.
- **HTTPS:** https://localhost:3000 — no warning. Steps: **[docs/HTTPS_LOCAL_TRUSTED.md](docs/HTTPS_LOCAL_TRUSTED.md)**

**Fallback – self-signed**  
The image also serves HTTPS with a self-signed cert (browser will show a warning; use **Advanced** → **Proceed**).

## Build and run the entire application

### Prerequisites

- **Docker** and **Docker Compose** (v2)
- **NVIDIA GPU** and **NVIDIA Container Toolkit** (`nvidia-docker2` or Docker with `nvidia` runtime) for:
  - Ollama (LLM + embeddings)
  - Backend (optional FAISS GPU)
  - RAG Platform (embedder + generator)
  - Voice (Whisper)

### One-command run

From the **repository root**:

```bash
docker compose up --build
```

This builds and starts (in order):

1. **Qdrant** – vector DB (port 6333)
2. **Ollama** – LLM + embeddings (port 11434); pulls models on first start (2–5 min)
3. **RAG Platform** – main RAG (embedder + generator, uses Qdrant)
4. **Backend** – EchoMind API (proxies docs/chat/transcript to RAG platform)
5. **Voice** – voice bot + Whisper (port 8001)
6. **Frontend** – UI with nginx (ports 3000 HTTP, 3443 HTTPS)

### Where to open the app

- **HTTP:** http://\<host\>:3000  
- **HTTPS (self-signed):** https://\<host\>:3443 (accept browser warning or use **Advanced** → **Proceed**)

Replace \<host\> with your machine’s IP or `localhost` if running on the same machine.

The frontend is served by nginx, which proxies:

- `/api/*` → backend (Knowledge Chat, docs, transcripts, Live Transcript WebSocket)
- `/voice/*` → voice service (Voice Conversation WebSocket)

So you only need to open the frontend URL; no need to expose the backend port.

### Run in the background

```bash
docker compose up --build -d
```

Logs: `docker compose logs -f` (or `docker compose logs -f backend rag-platform` for RAG/backend only).

### RAG Platform: "no kernel image is available for execution on the device"

The RAG platform defaults to **CPU** (`DEVICE=cpu`) so it runs on all hosts. If you see a 500 error and `RuntimeError: CUDA error: no kernel image is available for execution on the device`, the PyTorch build does not support your GPU (common on some ARM/NVIDIA setups). Keep `DEVICE=cpu` or fix your PyTorch/CUDA stack.

For a **supported** NVIDIA GPU (e.g. many x86_64 datacenter GPUs), you can speed up the RAG platform by setting in `docker-compose.yml` under `rag-platform` → `environment`: `DEVICE=cuda`.

### Optional: run without GPU

For a **CPU-only** run (slower, for testing):

1. In `docker-compose.yml`, remove or comment out the `deploy.resources.reservations` block under `ollama`, `rag-platform`, `backend`, and `voice`.
2. RAG Platform already uses `DEVICE=cpu` by default.
3. Run: `docker compose up --build`.

Ollama and the backend can still run; the RAG platform will be slower on CPU.

### Build fails with "failed to execute bake: read |0: file already closed"
This can happen at the end of a Buildx build when writing provenance metadata. Disable provenance and rebuild:

```bash
BUILDX_METADATA_PROVENANCE=disabled docker compose build
docker compose up -d
```

Or in one go: `BUILDX_METADATA_PROVENANCE=disabled docker compose up --build`.  
If you use `docker buildx bake` instead of `docker compose build`, run it with the same env var: `BUILDX_METADATA_PROVENANCE=disabled docker buildx bake`.

## Model setup (included in build/start)

- **Kyutai STT** (Live Transcript): Pre-downloaded during backend Docker build.
- **Ollama** (LLM + embeddings): Models (`qwen2.5:7b-instruct`, `nomic-embed-text`) are pulled automatically when Ollama starts.
- **Whisper** (Voice): Base model pre-downloaded during voice Docker build.

On first `docker compose up --build`, Ollama will pull its models (2–5 min). Backend and voice wait until models are ready. No manual `ollama pull` needed.

If you still see Gemini calls in the browser console:
1) Hard refresh (Ctrl+Shift+R) / clear site data
2) Ensure you rebuilt images: `docker compose up --build`

## FAISS GPU (faster RAG search)

By default the backend uses **faiss-cpu**. For faster vector search you can use **faiss-gpu** (requires an NVIDIA GPU and CUDA).

1. In `docker-compose.yml`, set the backend build arg: `USE_FAISS_GPU: "1"`.
2. Rebuild: `docker compose build --no-cache backend && docker compose up -d backend`.

The backend service already has GPU access in `docker-compose.yml`. No code changes are needed—the same `faiss` API is used; the GPU build just runs the index on the GPU.

**Note:** The PyPI `faiss-gpu` package (1.7.2) is archived and only provides wheels for Python ≤3.10. If the backend image uses Python 3.11+, the GPU build may fail; in that case keep `faiss-cpu` or use a conda base image with `faiss-gpu`.

## Live Transcript (Kyutai STT)

The **Real-Time Transcription** tab uses **Kyutai STT** (`kyutai/stt-1b-en_fr`) for streaming speech-to-text. No Whisper—Kyutai only.

- **Sample rate:** 24 kHz (Kyutai)
- **Works on:** x86_64 and ARM64 (e.g. DGX Spark)
- **Deps:** `moshi`, `huggingface-hub` (included in `backend/requirements.txt`). On ARM64: `libopus-dev` required for sphn.

On first use, the model (~1B params) is downloaded from Hugging Face. Requires PyTorch (provided by the NVIDIA PyTorch base image). For DGX Spark (ARM64), ensure the backend Dockerfile uses an ARM64-compatible base image; the dependencies support both architectures.
