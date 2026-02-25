# Backend RAG Flow and RAG Platform Integration

## Overview

When `RAG_PLATFORM_URL` is set (e.g. `http://rag-platform:8000`), the backend **uses the RAG platform** for all document storage, retrieval, and query. When it is not set, the backend uses the **legacy RAG** (FAISS + Ollama embeddings, local SQLite documents/chunks).

## Request Flow (when RAG_PLATFORM_URL is set)

### Docs (`/api/docs/*`)

| Frontend → Backend | Backend → RAG Platform | Response to frontend |
|--------------------|------------------------|----------------------|
| `POST /api/docs/upload` | `POST /docs/upload` (file) | `{ ok, doc_id, chunks }` |
| `GET /api/docs/list` | `GET /docs/list` | `{ documents: [{ id, filename, filetype, created_at }] }` |
| `DELETE /api/docs/:id` | `DELETE /docs/:id` | `{ ok, deleted }` |
| `GET /api/docs/usage` | `GET /docs/usage` | `{ usage_bytes, capacity_bytes }` |
| `GET /api/docs/data-preview` | `GET /docs/data-preview` + backend transcripts | `{ documents, chunks, transcripts }` |
| `POST /api/docs/delete-all` | `POST /docs/delete-all` + clear backend DB | `{ ok, message }` |

Transcript list and metadata always come from the **backend** DB; document list and RAG data come from the RAG platform when configured.

### Chat (`/api/chat/*`)

| Frontend → Backend | Backend → RAG Platform | Response to frontend |
|--------------------|------------------------|----------------------|
| `POST /api/chat/create` | — | Backend DB only: `{ chat_id }` |
| `POST /api/chat/ask` | `POST /query` (user_query, mode=general if no KB) | `{ answer, citations }` (evidence → citations) |
| `POST /api/chat/ask-stream` | `POST /query` (same) | NDJSON: `{ type, text }`, `{ type, answer, citations }`, `{ type, error }` |
| `POST /api/chat/ask-voice` | `POST /query` when not time-window transcript | `{ answer }` |

Conversation history and message storage stay in the **backend** DB. “Recent transcript” time-window queries (e.g. “summarise my transcript last hour”) still use backend DB transcript text + legacy LLM path (no RAG platform).

### Transcribe (`/api/transcribe/*`)

| Frontend → Backend | Backend → RAG Platform | Notes |
|--------------------|------------------------|--------|
| `POST /api/transcribe/store` | `POST /transcripts/ingest_batch` after DB store | Transcript saved in backend DB, then content sent to RAG platform |
| WebSocket live transcript `store` / auto-store | `POST /transcripts/ingest` per chunk (via `kb_add_text`) | Chunks sent under `transcript_id`: `ws_{session_id}` |

List/refine/tags/PATCH/WS stay backend-only; only **storage of transcript content for RAG** is forwarded to the RAG platform.

## Code Paths

- **`app/rag_platform_client.py`**: HTTP client for RAG platform (upload_doc, list_docs, delete_doc, get_usage, get_data_preview, delete_all_docs, query, ingest_transcript_chunk, ingest_transcript_batch).
- **`app/api/routes/docs.py`**: If `rag_platform_configured()`, all doc operations proxy to the client; else use `app/rag/index` + backend DB.
- **`app/api/routes/chat.py`**: If `rag_platform_configured()`, ask/ask-stream/ask-voice call `rag_query()` and map `evidence` → `citations`; else use `app/rag/advanced` (answer_with_citations, answer_stream).
- **`app/transcribe/store_to_db.py`**: After saving to backend DB, if `rag_platform_configured()`, calls `rag_ingest_transcript_batch()`; else calls `index.add_text()`.
- **`app/kb.py`**: If `rag_platform_configured()`, `kb_add_text` calls `rag_ingest_chunk()` (transcript_id=`ws_{session_id}`); else uses `faiss_index.add_text`. `kb_search` returns `[]` when RAG platform is configured.

## API Contract (frontend expectations)

- **Upload**: `{ ok: boolean, doc_id?: string, chunks?: number }`
- **List docs**: `{ documents: { id, filename, filetype, created_at }[] }`
- **Delete doc**: `{ ok: boolean, deleted: string }`
- **Usage**: `{ usage_bytes: number, capacity_bytes: number | null }`
- **Data preview**: `{ documents, chunks, transcripts }` (transcripts from backend)
- **Delete all**: `{ ok: boolean, message: string }`
- **Ask**: `{ answer: string, citations: any[] }`
- **Ask stream**: NDJSON lines `{ type: "chunk", text }`, `{ type: "done", answer, citations }`, `{ type: "error", message }`

All of these are preserved when using the RAG platform.
