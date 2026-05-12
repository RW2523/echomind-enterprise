# EchoMind — Enterprise Edition (Clean v2)

This build removes ALL Gemini code and connects the UI to your backend APIs.

**Product modes (Transcribe, Conversation, Assistant, Silent Assistant)** are defined in **[docs/ECHOMIND_FOUR_MODE_IMPLEMENTATION_PLAN.md](docs/ECHOMIND_FOUR_MODE_IMPLEMENTATION_PLAN.md)**.

## Services
- Frontend: http://<DGX_IP>:3000 (HTTP) or https://<DGX_IP>:3443 (HTTPS)
- Backend API: proxied under /api
- Voice bot: proxied under /voice (direct: http://<DGX_IP>:8002 by default; set `VOICE_HOST_PORT` in `.env` to change). **Voice AI is connected to RAG** (via `BACKEND_CHAT_URL`): questions about your transcripts or uploaded PDFs are answered from the knowledge base.
- **TensorRT-LLM** (chat): OpenAI-compatible API on port **8355** (`docker compose` service `trtllm`; model weights live in volume `trtllm_hf_cache`). Set `HF_TOKEN` in `.env` if your `MODEL_HANDLE` is gated.
- **Ollama** (embeddings only for RAG): http://<DGX_IP>:11434 — `nomic-embed-text` in volume `ollama_data`

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

## Run

**Offline-first (recommended):** One-time preparation, then run without internet:

```bash
./scripts/prepare_offline.sh   # once, with internet: builds images + populates Ollama volume
docker compose up -d           # thereafter: fully offline
```

**With build (uses internet for build and first Ollama run):**

```bash
docker compose up --build
```

See **[OFFLINE_DEPLOYMENT.md](OFFLINE_DEPLOYMENT.md)** for export/import to air-gapped machines and troubleshooting.

### Build fails with "failed to execute bake: read |0: file already closed"
This Docker BuildKit bug occurs when the backend's long Hugging Face download runs in parallel with other services. Use the build script (builds backend first, then the rest):

```bash
./scripts/build.sh
docker compose up -d
```

Or use offline preparation (builds everything and populates Ollama once): `./scripts/prepare_offline.sh`

**Alternative:** Disable provenance: `BUILDX_METADATA_PROVENANCE=disabled docker compose build`

## Model setup (included in build/start)

- **Nemotron STT** (Transcribe Mode / `backend/app/transcribe/ws.py`): Pre-downloaded during backend Docker build (`ECHOMIND_ASR_MODEL_NAME`, default `nvidia/nemotron-speech-streaming-en-0.6b`). Runtime uses local cache only (`HF_HUB_OFFLINE=1`).
- **TensorRT-LLM** (chat LLM): Service `trtllm` uses `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6` (see `docker/trtllm/`). On first start, `hf download` fills `trtllm_hf_cache` (can take a long time; needs GPU). Override model with `TRTLLM_MODEL_HANDLE` in `.env`. For air-gapped use after prep, export/import includes `trtllm_hf_cache.tar`; set `TRTLLM_SKIP_DOWNLOAD=1` and `TRTLLM_HF_HUB_OFFLINE=1` on the `trtllm` service once the cache is complete.
- **Ollama** (embeddings only): `nomic-embed-text` in `ollama_data`. Run `./scripts/prepare_offline.sh` once to pull it; chat does not use Ollama (`OLLAMA_EMBED_ONLY=1`).
- **Whisper** (Voice): Base model pre-downloaded during voice Docker build.
- **Piper** (Voice): Default voice baked in voice image; runtime download disabled when `VOICE_OFFLINE=1`.

If you still see Gemini calls in the browser console:
1) Hard refresh (Ctrl+Shift+R) / clear site data
2) Ensure you rebuilt images: `docker compose up --build`

## FAISS GPU (faster RAG search)

By default the backend uses **faiss-cpu**. For faster vector search you can use **faiss-gpu** (requires an NVIDIA GPU and CUDA).

1. In `docker-compose.yml`, set the backend build arg: `USE_FAISS_GPU: "1"`.
2. Rebuild: `docker compose build --no-cache backend && docker compose up -d backend`.

The backend service already has GPU access in `docker-compose.yml`. No code changes are needed—the same `faiss` API is used; the GPU build just runs the index on the GPU.

**Note:** The PyPI `faiss-gpu` package (1.7.2) is archived and only provides wheels for Python ≤3.10. If the backend image uses Python 3.11+, the GPU build may fail; in that case keep `faiss-cpu` or use a conda base image with `faiss-gpu`.

## Transcribe (Nemotron STT)

The **Transcribe** tab uses **Nemotron streaming ASR** (NeMo, default `nvidia/nemotron-speech-streaming-en-0.6b`) via `backend/app/transcribe/ws.py`. Voice Conversation uses Nemotron STT inside the `voice` service (separate container).

- **Sample rate:** 16 kHz for live transcribe WebSocket (matches Nemotron pipeline in backend config).
- **Works on:** x86_64 and ARM64 where the backend/voice images and GPU drivers support the stack.

Weights are downloaded at image build (or first run when not skipped). Requires PyTorch / NeMo dependencies in the backend image.
