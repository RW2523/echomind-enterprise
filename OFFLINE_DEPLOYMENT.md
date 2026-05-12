# EchoMind Offline Deployment

This document describes the **offline-first** architecture: one-time online preparation, then fully offline runs. No container performs network fetches at runtime after preparation.

---

## Audit: Internet Dependencies (Removed from Runtime)

### Build-time only (internet required once during `docker compose build` or `./scripts/prepare_offline.sh`)

| Dependency | Where | Purpose |
|------------|--------|---------|
| Docker base images | All Dockerfiles | `nvcr.io/nvidia/pytorch`, `nvcr.io/nvidia/tensorrt-llm/release`, `node:20-alpine`, `nginx`, `ollama/ollama` |
| apt packages | Backend, Voice, Ollama | ffmpeg, libopus-dev, wget, curl, etc. |
| pip packages | Backend, Voice | requirements.txt, moshi, whisper, piper-tts |
| npm packages | Frontend | package.json (npm ci) |
| Nemotron STT (live transcribe) | Backend Dockerfile | `snapshot_download` for `ECHOMIND_ASR_MODEL_NAME` (default `nvidia/nemotron-speech-streaming-en-0.6b`) |
| Piper TTS | Voice Dockerfile | wget from Hugging Face for en_US-lessac-medium |
| Whisper | Voice Dockerfile | `whisper.load_model('base')` |
| Ollama embed model | One-time prepare step | `ollama pull` for `nomic-embed-text` (stored in `ollama_data`) |
| TensorRT-LLM weights | First `trtllm` start with internet | `hf download` into volume `trtllm_hf_cache` (or restore from `trtllm_hf_cache.tar`) |

### Runtime (no internet)

- **Ollama**: With `OLLAMA_OFFLINE=1`, entrypoint only checks that required models exist in the volume; never runs `ollama pull`. With `OLLAMA_EMBED_ONLY=1`, only the embedding model is required. If missing, exits with a clear error.
- **TensorRT-LLM**: With `TRTLLM_SKIP_DOWNLOAD=1` and `HF_HUB_OFFLINE=1` on the `trtllm` service, no Hugging Face download at startup; the `trtllm_hf_cache` volume must already contain the model (populate online once or import `trtllm_hf_cache.tar`).
- **Backend**: `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; Nemotron ASR weights load from local Hugging Face cache only when offline. No `snapshot_download` over the network at runtime.
- **Voice**: Piper and Whisper use assets baked in the image (or mounted `./voice/voices`). `VOICE_OFFLINE=1` disables the `/voices/download` API so no runtime Piper download from Hugging Face.
- **Frontend**: Static build in image; nginx serves from disk. CSS (Tailwind) and fonts are bundled at build time; no CDN or Google Fonts at runtime. Favicon is a data URI (no external request).

### Where assets live

| Asset | Location | Persists across restarts |
|-------|----------|---------------------------|
| Ollama embeddings | Docker volume `ollama_data` | Yes (volume) |
| TensorRT-LLM HF cache | Docker volume `trtllm_hf_cache` | Yes (volume) |
| Nemotron STT | Backend image (HF cache under `/root/.cache/huggingface`) | Yes (image) |
| Piper default voice | Voice image `/voices` (or host `./voice/voices` if mounted) | Yes (image or host) |
| Whisper base | Voice image (whisper cache) | Yes (image) |
| Frontend CSS/fonts | Frontend image (Tailwind bundled; system fonts) | Yes (image) |

---

## One-Time Online Preparation

Run **once** on a machine with internet:

```bash
./scripts/prepare_offline.sh
```

This will:

1. Build all images (backend, voice, frontend, ollama, trtllm) — downloads base images, apt, pip, npm, Nemotron ASR (backend), Piper, Whisper.
2. Start Ollama with `OLLAMA_OFFLINE=0` so it pulls `nomic-embed-text` into the `ollama_data` volume (chat LLM is TensorRT-LLM, not Ollama).
3. Wait until the embed model is present.
4. Stop Ollama. The volume keeps the model.

**TensorRT-LLM:** The first time you run `docker compose up -d`, the `trtllm` service downloads the configured `MODEL_HANDLE` into `trtllm_hf_cache` (long-running; GPU required). For a fully offline stack after that, set on the `trtllm` service: `TRTLLM_SKIP_DOWNLOAD=1` and `TRTLLM_HF_HUB_OFFLINE=1`, and keep the populated volume (or import `trtllm_hf_cache.tar` from `./scripts/export_offline_bundle.sh`).

After caches and volumes are populated, **normal startup can avoid the network** (see env vars above for TRT + existing backend/voice/Ollama guards).

---

## Running Offline

Start the stack (no internet required):

```bash
docker compose up -d
```

Stop and start again anytime; assets live in images plus `ollama_data` and `trtllm_hf_cache` (when TRT is in offline mode).

---

## Export Bundle (for air-gapped machine)

On a machine that has run `prepare_offline.sh` and has built images:

```bash
./scripts/export_offline_bundle.sh [OUTPUT_DIR]
# Default: ./offline-bundle-YYYYMMDD
```

This creates:

- `image-backend.tar`, `image-voice.tar`, `image-frontend.tar`, `image-ollama.tar`, `image-trtllm.tar`
- `ollama_data.tar` (Ollama embed store)
- `trtllm_hf_cache.tar` (TensorRT-LLM Hugging Face cache — large)
- `voice-assets/` (optional copy of `voice/voices`)
- `MANIFEST.txt`, `OFFLINE_DEPLOYMENT.md`

Copy the whole bundle (e.g. USB) to the air-gapped machine.

---

## Import on Air-Gapped Machine

On the offline machine, with the repo (or at least `docker-compose.yml`, `Dockerfiles`, and app code) and the bundle:

```bash
./scripts/import_offline_bundle.sh /path/to/offline-bundle-YYYYMMDD /path/to/echomind-enterprise
```

Then start:

```bash
cd /path/to/echomind-enterprise
docker compose up -d
```

No network access is required if Ollama and TensorRT-LLM volumes were restored (and TRT offline env vars are set as above).

---

## Verification

Check that the setup is offline-ready (no runtime pull/download):

```bash
./scripts/verify_offline_readiness.sh
```

Checks include: Ollama entrypoint guarded by `OLLAMA_OFFLINE`, compose sets `OLLAMA_OFFLINE=1` and `HF_HUB_OFFLINE=1`, backend Dockerfile pre-downloads Nemotron ASR weights, voice Dockerfile pre-downloads Piper/Whisper, and Ollama volume exists after prepare. TensorRT-LLM offline mode is opt-in via `TRTLLM_SKIP_DOWNLOAD` / `TRTLLM_HF_HUB_OFFLINE` on the `trtllm` service.

---

## Troubleshooting

### "Required models are missing" (Ollama)

- You started with `OLLAMA_OFFLINE=1` and the `ollama_data` volume was empty (or `nomic-embed-text` was never pulled).
- **Fix**: Run one-time preparation with internet:  
  `OLLAMA_OFFLINE=0 docker compose up -d ollama`  
  Wait for health (`nomic-embed-text` listed), then `docker compose stop ollama`. After that, `docker compose up -d` works offline for embeddings.

### TensorRT-LLM: download fails or health never passes

- **Gated model**: set `HF_TOKEN` in `.env` (compose passes it into `trtllm`).
- **First start slow**: engine build can exceed the healthcheck `start_period`; watch `docker compose logs -f trtllm`.
- **Offline without cache**: populate `trtllm_hf_cache` online once, then set `TRTLLM_SKIP_DOWNLOAD=1` and `TRTLLM_HF_HUB_OFFLINE=1` on the `trtllm` service, or restore `trtllm_hf_cache.tar` from an export bundle.

### "Nemotron STT" / ASR model not found in local cache

- Backend was built with `SKIP_MODEL_DOWNLOAD=1` or the Hugging Face cache is missing.
- **Fix**: Rebuild backend without `SKIP_MODEL_DOWNLOAD`: remove or set to `0` in docker-compose build args, then `docker compose build backend`.

### Voice: "Piper model not found at /voices/..."

- No Piper files in `/voices` (empty bind mount or no default in image).
- **Fix**: Ensure `voice/voices` contains at least `en_US-lessac-medium.onnx` and `.onnx.json`, or do not override `/voices` so the image default is used. For import, copy `voice-assets` from the bundle into `voice/voices`.

### After import: volume name mismatch

- Compose creates a volume named `{project}_ollama_data`. The import script creates/restores using the project directory name. If your project directory name differs from the machine where you exported, create the volume and restore manually, e.g.:  
  `docker volume create <project>_ollama_data`  
  then restore `ollama_data.tar` into it as in `import_offline_bundle.sh`.

---

## Summary of Changes

- **Ollama**: Entrypoint checks for required models; if missing and `OLLAMA_OFFLINE=1`, exits with an error. Pull only when `OLLAMA_OFFLINE=0` (prepare step).
- **Backend**: `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; Nemotron ASR uses the Hugging Face cache offline when weights were baked in at build; clear error if model missing.
- **Voice**: `VOICE_OFFLINE=1` disables the Piper download API; Piper/Whisper use only image or mounted assets.
- **Compose**: `trtllm` service (TensorRT-LLM) with persistent `trtllm_hf_cache`; `OLLAMA_OFFLINE=1` and `OLLAMA_EMBED_ONLY=1` for Ollama; backend/voice point `LLM_*` at `trtllm:8355`.
- **Scripts**: `prepare_offline.sh`, `verify_offline_readiness.sh`, `export_offline_bundle.sh`, `import_offline_bundle.sh` for one-time prep, verification, and air-gapped move (bundle includes TRT image + HF cache tar when present).

After preparation, avoid runtime network by using offline guards on backend/voice/Ollama and TRT (`TRTLLM_SKIP_DOWNLOAD` + `TRTLLM_HF_HUB_OFFLINE` once the HF cache volume is full).
