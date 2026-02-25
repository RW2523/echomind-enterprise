# EchoMind — Enterprise Edition (Clean v2)

This build removes ALL Gemini code and connects the UI to your backend APIs.

## Services
- Frontend: http://<DGX_IP>:3000 (HTTP) or https://<DGX_IP>:3443 (HTTPS)
- Backend API: proxied under /api
- Voice bot: proxied under /voice (direct: http://<DGX_IP>:8001). **Voice AI is connected to RAG** (via `BACKEND_CHAT_URL`): questions about your transcripts or uploaded PDFs are answered from the knowledge base.
- Ollama: http://<DGX_IP>:11434

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
```bash
docker compose up --build
```

### Build fails with "failed to execute bake: read |0: file already closed"
This Docker BuildKit bug occurs when the backend's long Hugging Face download runs in parallel with other services. Use the build script (builds backend first, then the rest):

```bash
./scripts/build.sh
docker compose up -d
```

Or build and start in one go: `./scripts/build.sh --up`.

**Alternative:** Disable provenance: `BUILDX_METADATA_PROVENANCE=disabled docker compose build`

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
