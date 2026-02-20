# Packaging & Integration Guide

Use this multi-agent RAG backend as a **packagable service** inside your own application: run it in Docker next to your existing frontend/backend and **replace your old RAG** with this stack.

---

## 1. What You Get

- **Backend API** (FastAPI): chat (WebSocket), document ingest, RAG retrieval, sources, models, chat history.
- **Infra**: Postgres, Milvus, etcd, MinIO (used by Milvus). Optionally **model containers** (LLM + embedding) from this repo.

Your app keeps its **frontend and backend**. You either:

- **Option A – Direct:** Your frontend calls this backend at `http://chatbot-backend:8000` (or the host/port you expose). No proxy.
- **Option B – Proxy:** Your backend proxies requests to this backend (e.g. `/api/chatbot/*` → `http://chatbot-backend:8000/*`). Frontend only talks to your backend.

---

## 2. Running the Stack (No Frontend)

From the `assets` directory (or wherever you copied this package):

```bash
# Copy env and set at least CORS_ORIGINS to your frontend origin(s)
cp .env.example .env
# Edit .env: CORS_ORIGINS=http://localhost:4000 (or your app URL)

# Stack (backend + Postgres + Milvus + etcd + MinIO) + model containers
docker compose -f docker-compose.stack.yml -f docker-compose-models.yml up -d --build
```

- **Backend** will be on **port 8000** (or `BACKEND_PORT` from `.env`).
- **Model containers** (gpt-oss-20b, qwen3-embedding, deepseek-coder, etc.) must run for chat and RAG; they use the same Docker network (`chatbot-net`) so backend can reach them.

To run **only infra + backend** (e.g. you attach your own embedding/LLM later):

```bash
docker compose -f docker-compose.stack.yml up -d
```

Then configure `MILVUS_ADDRESS` and `EMBEDDING_SERVICE_URL` so the backend can reach your services.

---

## 3. Composing With Your Application

### 3.1 Same host / same compose

Include this stack and your app in one compose project:

```yaml
# your-app/docker-compose.yml
services:
  your-frontend:
    build: ./frontend
    ports: ["4000:4000"]
    environment:
      - CHATBOT_API_URL=http://chatbot-backend:8000
    depends_on: [chatbot-backend]

  your-backend:
    build: ./backend
    ports: ["5000:5000"]
    environment:
      - CHATBOT_API_URL=http://chatbot-backend:8000
    depends_on: [chatbot-backend]

  # Include the stack (paste or use extends/include)
  chatbot-backend:
    build: ./path/to/multi-agent-chatbot/assets/backend
    ...
  postgres:
    ...
  milvus:
    ...
  # etc.
```

Or use **Compose includes** (Compose v2.24+):

```yaml
# your-app/docker-compose.yml
include:
  - path: /path/to/multi-agent-chatbot/assets/docker-compose.stack.yml
  - path: /path/to/multi-agent-chatbot/assets/docker-compose-models.yml
services:
  your-frontend:
    ...
  your-backend:
    ...
```

Set **CORS** so the browser allows your frontend origin:

```bash
CORS_ORIGINS=http://localhost:4000,https://myapp.example.com
```

### 3.2 Separate host / separate stack

Run the stack on another host or cluster:

```bash
# On the “chatbot” host
cd /path/to/multi-agent-chatbot/assets
docker compose -f docker-compose.stack.yml -f docker-compose-models.yml up -d
```

Your app (frontend/backend) then calls `http://<chatbot-host>:8000`. Configure CORS for your frontend origin and ensure firewall allows 8000.

---

## 4. API Contract (Replace Your Old RAG)

Base URL: `http://<backend-host>:8000`.

### 4.1 RAG & documents

| Method | Path | Description |
|--------|------|-------------|
| POST   | `/ingest` | Ingest documents (multipart: files). Returns `task_id`, status `queued`. |
| GET    | `/ingest/status/{task_id}` | Status of ingestion task. |
| GET    | `/sources` | List available document sources (e.g. filenames). |
| GET    | `/selected_sources` | Currently selected sources for RAG. |
| POST   | `/selected_sources` | Set selected sources (JSON body, see below). |
| DELETE | `/collections/{collection_name}` | Delete a collection (e.g. `context`). |

**Replace old RAG flow:**

1. **Upload:** `POST /ingest` with your PDFs/docs (same as before, but to this backend).
2. **Sources:** `GET /sources` and `POST /selected_sources` so the supervisor only searches the sources you want.
3. **Ask:** Use the **chat** API below; the supervisor will call the internal `search_documents` tool when the user asks document questions. You do **not** call a separate “RAG query” endpoint; RAG is invoked by the agent via tools.

So: replace “call my old RAG API” with “call this chat API”; document Q&A goes through chat.

### 4.2 Chat (streaming)

| Method | Path | Description |
|--------|------|-------------|
| WebSocket | `/ws/chat/{chat_id}` | Real-time chat: send messages, receive streamed tokens and history. |

**WebSocket messages**

- **Client → server (JSON):**
  - `{ "message": "user text" }`
  - Optional: `{ "message": "...", "image_id": "uuid" }` (use image from `POST /upload-image`).

- **Server → client (JSON):**
  - `{ "type": "history", "messages": [...] }` – chat history.
  - `{ "type": "token", "data": "..." }` – streamed token.
  - `{ "type": "tool_start" | "tool_end" | "node_start" | "node_end", "data": "..." }` – optional UX.
  - Final content may also be sent as a string (see existing frontend).
  - `{ "type": "error", "data": "..." }` or `"content"` – error.

### 4.3 Chat lifecycle (session)

| Method | Path | Description |
|--------|------|-------------|
| GET    | `/chat_id` | Current chat ID (creates one if needed). |
| POST   | `/chat_id` | Set current chat ID (body: `{ "chat_id": "uuid" }`). |
| GET    | `/chats` | List all chat IDs. |
| GET    | `/chat/{chat_id}/metadata` | Chat metadata (e.g. name). |
| POST   | `/chat/rename` | Rename chat (body: `{ "chat_id", "new_name" }`). |
| POST   | `/chat/new` | Create new chat and set as current. |
| DELETE | `/chat/{chat_id}` | Delete one chat. |
| DELETE | `/chats/clear` | Clear all chats. |

### 4.4 Models & images

| Method | Path | Description |
|--------|------|-------------|
| GET    | `/available_models` | List model names (from `MODELS` env). |
| GET    | `/selected_model` | Current model. |
| POST   | `/selected_model` | Set model (body: `{ "model": "gpt-oss-20b" }`). |
| POST   | `/upload-image` | Upload image (form: `image`, `chat_id`). Returns `{ "image_id": "uuid" }`. |

### 4.5 Health

- **GET** `/health` – readiness/health check. Returns `{"status": "ok"}`.

---

## 5. Environment Variables (Backend)

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG_PATH` | `./config.json` | Config file path (backend and RAG MCP server). |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins. |
| `MILVUS_ADDRESS` | `milvus:19530` | Milvus host:port (backend and RAG). |
| `EMBEDDING_SERVICE_URL` | `http://qwen3-embedding:8000` | Embedding API base URL. |
| `EMBEDDING_MODEL` | `qwen3-embedding-custom` | Embedding model name. |
| `MODELS` | (required) | Comma-separated model container names (e.g. `gpt-oss-20b`). |
| `POSTGRES_*` | (see .env.example) | Postgres connection. |

---

## 6. Flow Summary for “Replace Old RAG”

1. **Start stack** (and models) so the backend and Milvus/embedding are up.
2. **Ingest docs:** `POST /ingest` with files; poll `GET /ingest/status/{task_id}` until done.
3. **Optional:** `GET /sources` → `POST /selected_sources` with the source names you want.
4. **Chat:** Open WebSocket ` /ws/chat/{chat_id}` (get `chat_id` from `GET /chat_id` or `POST /chat/new`). Send `{ "message": "Summarize the key points from my documents" }`. The supervisor will call `search_documents` and stream the answer.
5. Your **frontend** either calls this backend directly (set `CHATBOT_API_URL` and CORS) or your **backend** proxies to it.

You now use this backend as the single place for RAG + multi-agent chat instead of your old RAG service.
