# EchoMind — Frontend

React + Vite single-page app for EchoMind (Knowledge Chat / RAG, Live Transcription, Voice AI,
Boardroom). It talks to the backend and voice services over HTTP/WebSocket — **there is no Gemini
or other client-side LLM key**; all model calls go through the backend.

## Run locally

**Prerequisites:** Node.js 18+

1. Install dependencies: `npm install`
2. Start the dev server: `npm run dev`

By default the app calls the backend under `/api` and the voice service under `/voice` (same origin,
proxied by nginx in the Docker setup). For local dev against running services, set `VITE_API_BASE`
(empty string means same-origin) and ensure the backend (`:8000`) and voice (`:8002`) are reachable.

### HTTPS (mic access requires a secure context)
- `VITE_DEV_HTTPS=1` with `VITE_SSL_CERT` / `VITE_SSL_KEY` to serve the dev server over HTTPS.
- In Docker, the frontend container serves HTTPS on `:3443` (see root `README.md`).

## Build

`npm run build` → static assets in `dist/` (served by nginx in the container image).

## Notes
- `VOICE_HOST_PORT` (root `.env`) controls the host port for direct voice access (default 8002).
- The whole stack is intended to run via `docker compose` from the repo root.
