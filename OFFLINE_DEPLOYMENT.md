# EchoMind Offline Deployment

This document describes the **offline-first** architecture: one-time online preparation, then fully offline runs. No container performs network fetches at runtime after preparation.

---

## Audit: Internet Dependencies (Removed from Runtime)

### Build-time only (internet required once during `docker compose build` or `./scripts/prepare_offline.sh`)

| Dependency | Where | Purpose |
|------------|--------|---------|
| Docker base images | All Dockerfiles | `nvcr.io/nvidia/pytorch`, `node:20-alpine`, `nginx`, `ollama/ollama` |
| apt packages | Backend, Voice, Ollama | ffmpeg, libopus-dev, wget, curl, etc. |
| pip packages | Backend, Voice | requirements.txt, moshi, whisper, piper-tts |
| npm packages | Frontend | package.json (npm ci) |
| Kyutai STT | Backend Dockerfile | `snapshot_download('kyutai/stt-1b-en_fr')` |
| Piper TTS | Voice Dockerfile | wget from Hugging Face for en_US-lessac-medium |
| Whisper | Voice Dockerfile | `whisper.load_model('base')` |
| Ollama models | One-time prepare step | `ollama pull` for LLM + embed (stored in volume) |

### Runtime (no internet)

- **Ollama**: With `OLLAMA_OFFLINE=1`, entrypoint only checks that required models exist in the volume; never runs `ollama pull`. If missing, exits with a clear error.
- **Backend**: `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; Kyutai loads from local Hugging Face cache only (`local_files_only=True`). No `snapshot_download` over the network.
- **Voice**: Piper and Whisper use assets baked in the image (or mounted `./voice/voices`). `VOICE_OFFLINE=1` disables the `/voices/download` API so no runtime Piper download from Hugging Face.
- **Frontend**: Static build in image; nginx serves from disk. CSS (Tailwind) and fonts are bundled at build time; no CDN or Google Fonts at runtime. Favicon is a data URI (no external request).

### Where assets live

| Asset | Location | Persists across restarts |
|-------|----------|---------------------------|
| Ollama LLM + embed | Docker volume `ollama_data` | Yes (volume) |
| Kyutai STT | Backend image (HF cache under `/root/.cache/huggingface`) | Yes (image) |
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

1. Build all images (backend, voice, frontend, ollama) — downloads base images, apt, pip, npm, Kyutai, Piper, Whisper.
2. Start Ollama with `OLLAMA_OFFLINE=0` so it pulls `qwen2.5:7b-instruct-q4_K_M` and `nomic-embed-text` into the `ollama_data` volume.
3. Wait until both models are present.
4. Stop Ollama. The volume keeps the models.

After this, **normal startup never uses the network**.

---

## Running Offline

Start the stack (no internet required):

```bash
docker compose up -d
```

Stop and start again anytime; all assets are in images or the `ollama_data` volume.

---

## Export Bundle (for air-gapped machine)

On a machine that has run `prepare_offline.sh` and has built images:

```bash
./scripts/export_offline_bundle.sh [OUTPUT_DIR]
# Default: ./offline-bundle-YYYYMMDD
```

This creates:

- `image-backend.tar`, `image-voice.tar`, `image-frontend.tar`, `image-ollama.tar`
- `ollama_data.tar` (Ollama model store)
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

No network access is required.

---

## Verification

Check that the setup is offline-ready (no runtime pull/download):

```bash
./scripts/verify_offline_readiness.sh
```

Checks include: Ollama entrypoint guarded by `OLLAMA_OFFLINE`, compose sets `OLLAMA_OFFLINE=1` and `HF_HUB_OFFLINE=1`, backend Dockerfile pre-downloads Kyutai, voice Dockerfile pre-downloads Piper/Whisper, and Ollama volume exists after prepare.

---

## Troubleshooting

### "Required models are missing" (Ollama)

- You started with `OLLAMA_OFFLINE=1` and the `ollama_data` volume was empty.
- **Fix**: Run one-time preparation with internet:  
  `OLLAMA_OFFLINE=0 docker compose up -d ollama`  
  Wait for health (both models present), then `docker compose stop ollama`. After that, `docker compose up -d` works offline.

### "Kyutai STT model not found in local cache"

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
- **Backend**: `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; Kyutai uses `snapshot_download(..., local_files_only=True)` when offline; clear error if model missing.
- **Voice**: `VOICE_OFFLINE=1` disables the Piper download API; Piper/Whisper use only image or mounted assets.
- **Compose**: `OLLAMA_OFFLINE=1` for ollama service; backend and voice get offline env vars.
- **Scripts**: `prepare_offline.sh`, `verify_offline_readiness.sh`, `export_offline_bundle.sh`, `import_offline_bundle.sh` for one-time prep, verification, and air-gapped move.

No runtime `pull`, `wget`, or `snapshot_download` over the network after preparation.
