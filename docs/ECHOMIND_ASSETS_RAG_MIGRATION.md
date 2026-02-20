# EchoMind Frontend + Assets RAG Backend — Migration Architecture

## Goal

- **Keep:** EchoMind **frontend** (Vite, Knowledge Chat, Live Transcription, Voice Conversation, Settings).
- **Replace:** EchoMind backend RAG and persistence with the **assets/** (Chatbot Spark) stack: Milvus, Postgres, LangGraph agent, MCP RAG tool.
- **Preserve:** EchoMind’s **transcript** concept: live transcripts, saved transcripts, list/store/update, and **transcripts in RAG** (indexed in Milvus and searchable like documents).

Result: one frontend, one powerful backend that uses assets’ fast RAG flow and adds first-class transcript support.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  EchoMind Frontend (unchanged)                                           │
│  - Knowledge Chat  - Live Transcription  - Voice Conversation  - Settings │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    │ /api/*  (REST + WebSocket /api/transcribe/ws)
                    │ /voice/* (Voice service)
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Unified Backend (assets base + EchoMind API + transcripts)              │
│  - Postgres: chats, messages, transcripts, config                        │
│  - Milvus: documents + transcript chunks (same collection, metadata)     │
│  - LangGraph agent + MCP (RAG, code, image)                             │
│  - EchoMind API: /api/docs/*, /api/chat/*, /api/transcribe/*             │
│  - Optional: /api/transcribe/ws (Kyutai) here or separate microservice   │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    │ depends_on
                    ▼
┌──────────────┐  ┌─────────────┐  ┌────────┐  ┌───────┐  ┌──────────────┐
│  Postgres    │  │  Milvus     │  │  etcd  │  │ MinIO │  │ Model svcs   │
│  (chats,     │  │  (vectors)  │  │        │  │       │  │ (LLM, embed) │
│   messages,  │  │             │  │        │  │       │  │ or Ollama    │
│   transcripts)│  │             │  │        │  │       │  │              │
└──────────────┘  └─────────────┘  └────────┘  └───────┘  └──────────────┘
```

- **Voice service** stays separate; it calls the unified backend at `BACKEND_CHAT_URL` for `/api/chat/ask-voice`.
- **Live transcript WebSocket** can run inside the unified backend (Kyutai in same process) or remain a small EchoMind “transcribe” service that only does WS + Kyutai and calls the unified backend for store/list/refine/tags.

---

## API Mapping: EchoMind Frontend → Unified Backend

| EchoMind (current) | Unified backend implementation |
|--------------------|---------------------------------|
| **Docs** | |
| `POST /api/docs/upload` | Ingest file(s) → save to disk → `vector_store._load_documents` + `index_documents`; add source to config; return `{ok, doc_id, chunks}` (map task_id/source to doc_id). |
| `GET /api/docs/list` | Return list from config `sources` (or new `documents` Postgres table) excluding `transcript_%` entries. |
| `DELETE /api/docs/{id}` | Delete source/collection by name; remove from config (or delete from `documents` table + remove from Milvus). |
| `GET /api/docs/usage` | Approximate from Milvus collection size + Postgres; or new usage endpoint. |
| `GET /api/docs/data-preview` | Documents (sources) + transcripts from Postgres; chunk preview from Milvus or stored metadata. |
| `POST /api/docs/delete-all` | Clear config sources; drop/clear Milvus collection; clear Postgres messages/chats; clear transcripts table. |
| **Chat** | |
| `POST /api/chat/create` | Create chat in Postgres (assets `postgres_storage.save_messages_immediate(chat_id, [])` + set_chat_metadata); return `{chat_id}`. |
| `POST /api/chat/ask` | Load history from Postgres; if “last N hours” transcript query → fetch transcripts from DB and answer with LLM; else invoke agent (with RAG tool when question needs KB); persist user/assistant messages; return `{answer, citations}`. |
| `POST /api/chat/ask-stream` | Same as ask but stream agent output (assets agent already supports streaming); return NDJSON `{type: chunk|done|error}`. |
| `POST /api/chat/ask-voice` | Same as ask (no chat_id); used by voice service. |
| **Transcribe** | |
| `GET /api/transcribe/list` | List from Postgres `transcripts` table (id, title, tags, echotag, created_at, name, location). |
| `WebSocket /api/transcribe/ws` | Kyutai STT + SessionState (current EchoMind logic). Run in unified backend or keep separate transcribe service. |
| `POST /api/transcribe/refine` | LLM refine (use backend LLM or Ollama). |
| `POST /api/transcribe/tags` | LLM or rule-based tags (reuse EchoMind tagging). |
| `POST /api/transcribe/store` | Insert into Postgres `transcripts`; index `raw_text` (+ refined) into Milvus with `source=transcript_<id>`, `type=transcript`; return `{transcript_id, title, ...}`. |
| `PATCH /api/transcribe/transcripts/{id}` | Update name, location, tags in Postgres. |
| **Voice** | Unchanged; voice service at `/voice/`; calls backend `BACKEND_CHAT_URL` for ask-voice. |

---

## Transcripts in the Unified Backend

### Postgres: `transcripts` table

Same shape as EchoMind for compatibility:

- `id` (PK), `title`, `raw_text`, `polished_text`, `tags_json`, `echotag`, `echodate`, `created_at`, `updated_at`, `name`, `location`

### Milvus: transcript chunks

- Reuse the same Milvus collection as documents (e.g. `context`).
- When storing a transcript: split `raw_text` (and optionally `polished_text`) with the same text splitter as documents; add chunks with metadata:
  - `source`: `transcript_<id>` or display title so RAG and “sources” can include transcripts.
  - `type`: `"transcript"` for filtering if needed.
  - Optional: `echotag`, `created_at` for time/scoping later.
- RAG (MCP `search_documents` / agent) already uses `vector_store.get_documents(question, sources=...)`; if `sources` is empty or includes transcript sources, transcript chunks are retrieved with document chunks.
- List “sources” for the UI: union of config `sources` (uploaded docs) and transcript titles (or `transcript_<id>`) from Postgres so the user can select documents and/or transcripts for context.

### Auto-store (live transcript)

- When the Live Transcript WebSocket auto-stores or user stores: call the same “store transcript” path: Postgres insert/append + Milvus index (append chunks with `source=transcript_<id>`).
- Optional: append-only transcript row (one row per session) with `append_transcript_chunk` and periodic Milvus add for new chunks.

---

## Implementation Strategy

### Phase 1: Backend foundation (assets + transcripts)

1. **New unified backend** (e.g. `backend/` replaced or `backend-unified/`):
   - Start from **assets/backend**: FastAPI, Postgres, Milvus, `vector_store`, `agent`, MCP client, config.
   - Add **transcripts**:
     - Postgres: create `transcripts` table in `postgres_storage` (or separate module).
     - `transcript_store`: create, list, get, update (name, location, tags), append_chunk.
     - When storing a transcript: save to Postgres; build `Document` list from raw (and polished) text; set metadata `source=transcript_<id>`, `type=transcript`; call `vector_store.index_documents(documents)`.
     - Add “transcript” sources to config (or separate list) so `get_sources` / selected_sources can include transcripts for RAG.
2. **EchoMind API adapter routes** under `/api`:
   - **Docs:** implement `/api/docs/upload`, `list`, `delete`, `usage`, `data-preview`, `delete-all` using assets ingest + config + transcripts table + Milvus.
   - **Chat:** implement `/api/chat/create`, `ask`, `ask-stream`, `ask-voice` using Postgres history + agent (and “last N hours” transcript path when applicable).
   - **Transcribe:** implement `GET /api/transcribe/list`, `POST /api/transcribe/refine`, `tags`, `store`, `PATCH /api/transcribe/transcripts/{id}` using Postgres transcripts + LLM for refine/tags.
3. **WebSocket** `/api/transcribe/ws`:
   - Option A: Run inside unified backend (add Kyutai + SessionState + same WS handler as EchoMind).
   - Option B: Keep EchoMind backend as a small “transcribe” service that only handles WS and calls the unified backend REST for store/list/refine/tags.

### Phase 2: Frontend and deploy

4. **Frontend:** No code changes if API contract is preserved; only ensure `API_BASE` points to the unified backend (e.g. `/api` via nginx).
5. **Docker Compose:**
   - One compose file: frontend, unified backend, Postgres, Milvus, etcd, MinIO, (optional model containers or Ollama), voice service.
   - If transcribe WS is in unified backend: no separate transcribe service. If not: add small transcribe service that uses unified backend for persistence.

### Phase 3: RAG behavior and citations

6. Ensure agent’s RAG tool (MCP `search_documents`) uses the same Milvus collection and can return citations for both document and transcript chunks (metadata `source`, `type`).
7. Map agent stream events to EchoMind’s NDJSON format (`chunk`, `done` with `answer` + `citations`).

---

## File Layout (Proposed)

```
echomind-enterprise/
├── frontend/                    # Unchanged EchoMind Vite app
├── voice/                       # Unchanged; BACKEND_CHAT_URL → unified backend
├── backend/                     # REPLACED by unified backend
│   ├── main.py                  # FastAPI: assets routes + /api/* EchoMind adapter
│   ├── agent.py                 # From assets (LangGraph + MCP)
│   ├── vector_store.py          # From assets (Milvus)
│   ├── postgres_storage.py      # From assets + transcripts table + methods
│   ├── transcript_store.py      # NEW: transcript CRUD + Milvus indexing
│   ├── routes/
│   │   ├── docs.py              # EchoMind /api/docs/* (using vector_store + config + transcripts)
│   │   ├── chat.py              # EchoMind /api/chat/* (using agent + postgres_storage)
│   │   ├── transcribe.py        # EchoMind /api/transcribe/* REST + optional WS
│   ├── tools/mcp_servers/       # From assets (RAG, etc.)
│   └── ...
├── assets/                      # Reference only; code merged into backend/
├── docker-compose.yml           # frontend, backend (unified), voice, postgres, milvus, etcd, minio, ollama or model svcs
└── docs/
    └── ECHOMIND_ASSETS_RAG_MIGRATION.md  # This file
```

---

## Decisions to Confirm

1. **Transcribe WebSocket:** In unified backend (single service) vs separate small transcribe service (current EchoMind backend trimmed to WS only).
2. **LLM/embedding:** Use assets’ model containers (Qwen3-Embedding, gpt-oss/others) vs Ollama for everything (simpler, one less stack). Hybrid: Ollama for refine/tags/voice; assets models for main RAG/chat.
3. **Voice:** Keep as-is; only `BACKEND_CHAT_URL` points to unified backend.

Once these are fixed, implementation can follow the phases above and reuse this doc as the single source of truth for the new structure.

---

## Running EchoMind frontend with assets backend

From the **repo root**:

```bash
# Assets stack (Postgres, Milvus, etcd, MinIO) + assets backend (with EchoMind API) + EchoMind frontend + voice
docker compose -f assets/docker-compose.yml -f docker-compose.echomind-assets.yml up -d --build
```

With **model containers** (LLM + embedding) for chat and RAG:

```bash
docker compose -f assets/docker-compose.yml -f assets/docker-compose-models.yml -f docker-compose.echomind-assets.yml up -d --build
```

- **Frontend:** http://localhost:3000 (proxies `/api` to the assets backend).
- **Voice:** http://localhost:8001 (uses `BACKEND_CHAT_URL=http://backend:8000` for RAG).
- **Backend:** EchoMind REST at `/api/*` (docs, chat, transcribe) and assets routes at `/health`, `/ingest`, `/ws/chat/{chat_id}`, etc.

### Live Transcript WebSocket

The EchoMind frontend expects **WebSocket** `/api/transcribe/ws` for the Live Transcription tab. This is **not** implemented in the assets backend in the first cut. You can:

- **Option A:** Run the **original EchoMind backend** as a second service (e.g. only for `/api/transcribe/ws`), and have nginx route `/api/transcribe/ws` to it and the rest of `/api` to the assets backend.
- **Option B:** Add the full transcribe WebSocket (Kyutai STT + SessionState) into the assets backend and build the backend image with PyTorch/Kyutai (see `backend/` Dockerfile and `backend/app/transcribe/` in the main EchoMind backend).
