# EchoMind — Endpoints & Request Wiring

Quick reference to confirm all endpoints and frontend calls are connected.

## Backend API (prefix `/api`, served at `backend:8000` in Docker; nginx proxies `/api/` → backend)

| Method | Path | Request | Response | Frontend usage |
|--------|------|---------|----------|----------------|
| GET | `/health` | — | `{ok, app}` | (not called by UI) |
| POST | `/api/docs/upload` | `FormData` with `file` | `{ok, doc_id, chunks}` | `uploadDocument()` in Uploader |
| POST | `/api/chat/create` | `{title}` | `{chat_id}` | `createChat()` in KnowledgeChat |
| POST | `/api/chat/ask` | `{chat_id, message}` | `{answer, citations[]}` | `askChat()` in KnowledgeChat |
| WS | `/api/transcribe/ws` | `{type:"audio", pcm16_b64}` / `{type:"stop"}` | `ready` / `partial` / `final` / `error` | `transcribeWsUrl()` + LiveTranscription |
| POST | `/api/transcribe/refine` | `{raw_text}` | `{refined}` | `refineTranscript()` in LiveTranscription |
| POST | `/api/transcribe/store` | `{raw_text, refined_text?}` | `{transcript_id, tags, created_at}` | `storeTranscript()` in LiveTranscription |

## Voice app (served at `voice:8000` in Docker; nginx proxies `/voice/` → voice)

| Method | Path | Request | Response | Frontend usage |
|--------|------|---------|----------|----------------|
| GET | `/` | — | HTML (Unmute UI) | iframe `src="/voice/"` in VoiceConversation |
| WS | `/ws` (proxied as `/voice/ws`) | `{type:"audio_frame", pcm16_b64, ts}` / `{type:"set_context", ...}` | `hello`, `audio_out`, `asr_final`, etc. | voice/static/index.html (uses `/voice/ws` when under `/voice`) |

## Frontend → API base URL

- **Production (Docker):** `API_BASE` is `""`, so requests go to same origin; nginx handles `/api` and `/voice`.
- **Development (Vite):** Proxy in `vite.config.ts` sends `/api` → `http://127.0.0.1:8000`, `/voice` → `http://127.0.0.1:8001`. Run backend on 8000 and voice on 8001 when developing locally.

## WebSocket proxying (nginx)

- `location /api/` and `location /voice/` include `Upgrade` and `Connection "upgrade"` so WebSockets work through nginx.

## Fixes applied in this pass

1. **nginx:** WebSocket support added for `/api/` and `/voice/` (Upgrade, Connection, read timeout).
2. **voice/static/index.html:** WebSocket URL uses `/voice/ws` when the page is loaded under `/voice/`.
3. **frontend:** Added `ICONS.Wave` (VoiceConversation used `ICONS.wave`), aligned `TranscriptEntry` with `rawText` used in LiveTranscription.
4. **vite.config.ts:** Dev proxy for `/api` and `/voice` so local frontend can call backend and voice without CORS.
