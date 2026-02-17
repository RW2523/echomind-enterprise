# End-to-End Chat + RAG Flow

This document describes how a single user chat message flows from the frontend to the backend, how RAG (retrieval-augmented generation) is applied, and **how many LLM requests** are made for each path.

---

## 1. Frontend → Backend (Entry Point)

### 1.1 User types a message

- **Component:** `frontend/components/KnowledgeChat.tsx`
- User types in the input and submits (e.g. Send).
- `send()` runs:
  - Appends a **user** message to local state.
  - Appends a placeholder **assistant** message (empty content).
  - Calls `askChatStream(chatId, message, callbacks, options)`.

### 1.2 API call

- **Service:** `frontend/services/backend.ts` → `askChatStream()`
- **HTTP:** `POST /api/chat/ask-stream`
- **Body:** `{ chat_id, message, persona?, context_window?, advanced_rag?, use_knowledge_base? }`
- **Response:** NDJSON stream: lines of `{"type":"chunk","text":"..."}` then one `{"type":"done","answer":"...","citations":[...]}` (or `{"type":"error","message":"..."}`).

### 1.3 Frontend stream handling

- Reads the response body with a `ReadableStream` reader.
- For each line: if `type === "chunk"` → `onChunk(text)` (append to assistant message); if `type === "done"` → `onDone({ answer, citations })`; if `type === "error"` → `onError`.
- UI updates the assistant message incrementally, then sets final content and citations on `done`.

---

## 2. Backend: Chat Route

- **Route:** `backend/app/api/routes/chat.py` → `POST /ask-stream` (`ask_stream`).

### 2.1 Request handling

1. **Load chat history** from DB: `SELECT role, content FROM messages WHERE chat_id=? ORDER BY created_at ASC`.
2. **Load conversation summary** (if any): `_get_conversation_summary(chat_id)` from `chats.conversation_summary`.
3. **Persist user message:** `INSERT INTO messages (id, chat_id, role='user', content, created_at)`.
4. **Stream response:** generator `gen()` that calls `answer_stream(...)` and:
   - For each `("chunk", text, None)` → yields `{"type":"chunk","text":text}`.
   - On `("done", full_answer, citations)` → inserts assistant message into DB, yields `{"type":"done","answer":...,"citations":...}`, and schedules **background** `_update_summary_background(conversation_summary, user_msg, assistant_msg, chat_id)`.
5. **Background task:** `update_conversation_summary(prev_summary, user_msg, assistant_msg)` → new summary stored in `chats.conversation_summary` (does not block the HTTP response).

So: **one user message** triggers **one** call to `answer_stream(...)`, and after the stream completes, **one** background LLM call to update the conversation summary.

---

## 3. RAG Flow: `answer_stream` / `answer`

- **Module:** `backend/app/rag/advanced.py`
- **Streaming entry:** `answer_stream(...)` (same logic as `answer()`, but streams the final LLM reply).

High-level branches:

- **A. No knowledge base** (`use_knowledge_base=False`)  
  → **General-only:** no retrieval, one LLM call for reply (+ 1 for summary in background).

- **B. General/small talk** (`_is_general_conversation(question)` is True)  
  → **Fast path:** no retrieval, one LLM for reply (+ 1 for summary).

- **C. Advanced RAG** (`advanced_rag=True`)  
  → Single-query retrieval (embedding only), no intent/rewrite; one LLM for answer (+ 1 for summary).

- **D. Standard RAG** (`use_knowledge_base=True`, `advanced_rag=False`, not general)  
  → Intent classification → retrieval (with query expansion) → build context → one LLM for answer (+ 1 for summary). Optional: rerank (extra LLM).

Below we walk **D** in order, then summarize **LLM request counts** for all paths.

---

## 4. Standard RAG Path (D) – Step by Step

### 4.1 General-conversation check (no LLM)

- **Function:** `_is_general_conversation(question)`
- **Logic:** Heuristic: empty string; exact phrases (e.g. “hi”, “thanks”); or very short non-questions (no “what/which/when/…” and no `?`). Short “real” queries (e.g. “pricing”, “setup”) are **not** treated as general.
- **LLM calls:** **0**

### 4.2 Document/metadata for intent (no LLM)

- **Function:** `_get_document_titles()`
- **Logic:** Reads from DB: doc titles, whether any transcripts exist, transcript echotags. Used only to build the intent-classification prompt.
- **LLM calls:** **0**

### 4.3 Intent classification (1 LLM)

- **Function:** `_classify_intent(question, doc_titles, has_transcripts, transcript_echotags)`
- **Logic:** One LLM call with a system prompt that describes “general / document / transcript” and includes doc/transcript hints. User message = question (truncated). Model returns one word: `general` | `document` | `transcript`.
- **Override:** If the user clearly asks for transcript/recording/time range (e.g. “last N transcripts”), intent is forced to `"transcript"` even if the LLM said something else.
- **LLM calls:** **1** (`chat.chat` in `backend/app/rag/advanced.py`)

### 4.4 If intent is `"general"`

- **Action:** Call `_answer_general_stream(...)` (or `_answer_general` for non-stream).
- **LLM:** One chat (or stream) with system + history + user message; no RAG context.
- **Then:** Background conversation summary update.
- So for this branch: **1 (intent) + 1 (answer) + 1 (summary) = 3 LLM requests** (summary is after response).

### 4.5 Retrieval (when intent is `document` or `transcript`)

- **Function:** `retrieve(question, k, context_window, intent, document_titles, has_transcripts, transcript_echotags)`

Steps inside `retrieve()`:

1. **Query expansion (0 or 1 LLM)**  
   - **Function:** `generate_queries(...)` — **skipped when `RAG_QUERY_REWRITE_ENABLED` is False (default).**  
   - **Logic:** When enabled, one LLM call with intent-specific system prompt to produce 1–3 search queries. When disabled, only the original question is used (plus deterministic variants).  
   - **LLM calls:** **0** (default) or **1**

2. **Deterministic query variants (no LLM)**  
   - **Function:** `get_deterministic_query_variants(question)`  
   - **Logic:** Rule-based variants (e.g. normalized terms). Merged with LLM queries.

3. **Vector + sparse search (no LLM)**  
   - **Index:** `FaissIndex` in `backend/app/rag/index.py`  
   - For each final query: **dense** search (FAISS / embeddings) and **sparse** search (BM25).  
   - **Embeddings:** Via `OllamaEmbeddings` (separate from the chat LLM; not counted as “LLM request” in the list below).  
   - Results are merged with weighted RRF (reciprocal rank fusion).

4. **Optional rerank (0 or 1 LLM)**  
   - **Function:** `_rerank_hits(question, hits, top_n)`  
   - **When:** Only if `RAG_RERANK_ENABLED` is True (**default: False** for faster response).  
   - **Logic:** One LLM call: send question + excerpts, model returns relevance scores; hits reordered by score.  
   - **LLM calls:** **0** (default) or **1** (if enabled).

5. **Filtering**  
   - By `context_window`, optional time decay, tag boost, “last N transcripts”, time window, location, etc. No LLM.

### 4.6 Build RAG context (0 LLM in current code)

- **Function:** `_build_rag_context(question, hits)` (standard) or `_build_rag_context_fast(question, hits)` (advanced RAG).
- **Logic:** For each hit (and optional parent chunk), format a context block. In the current code, **`use_compression` is set to `False`** inside `_build_rag_context`, so **no per-chunk `compress()` LLM calls** are made; chunks are only truncated.  
  (Config has `RAG_COMPRESS_CONTEXT`; if the code were to use it and set `use_compression = True`, each chunk could trigger one `compress()` LLM call.)
- **LLM calls:** **0**

### 4.7 Final answer (1 LLM, streamed)

- **Function:** `chat.chat_stream(msgs, ...)` (from `backend/app/rag/llm.py` → `OpenAICompatChat`).
- **Logic:** Messages = RAG system prompt (+ optional persona) + (optional conversation summary + question + context block) or last 10 history turns + current question + context. Single **streaming** chat/completions request to the configured LLM (e.g. Ollama).
- **LLM calls:** **1** (one request; tokens streamed back).

### 4.8 Conversation summary (1 LLM, background)

- **Function:** `update_conversation_summary(previous_summary, user_message, assistant_message)`  
- **When:** Called from the chat route **after** the stream is done, in a background task.  
- **Logic:** One LLM call to produce an updated short summary (goals, constraints, decisions, key facts). Result is stored in `chats.conversation_summary` for the next turn.
- **LLM calls:** **1** (does not block the HTTP response).

---

## 5. Other Paths (A, B, C)

### A. `use_knowledge_base=False`

- **Flow:** `_answer_general_stream` (or `_answer_general`) only. No intent, no retrieval, no RAG context.
- **LLM:** 1 (answer) + 1 (summary in background) = **2 LLM requests**.

### B. General/small talk (`_is_general_conversation(question) == True`)

- **Flow:** Same as A: general answer + background summary.
- **LLM:** **2 LLM requests**.

### C. `advanced_rag=True`

- **Flow:**  
  - No intent classification.  
  - `retrieve_single_query(question, TOP_K)` → embedding-only search (no LLM).  
  - `_build_rag_context_fast(question, hits)` → truncation only, no compress.  
  - One streamed answer LLM call.  
  - Background summary.
- **LLM:** 1 (answer) + 1 (summary) = **2 LLM requests**.

### D. Standard RAG (document/transcript, full path)

- **LLM:** 1 (intent) + 0 or 1 (query rewrite, only if `RAG_QUERY_REWRITE_ENABLED`) + 0 or 1 (rerank, only if `RAG_RERANK_ENABLED`) + 1 (answer) + 1 (summary).  
- **Default (both optional features off):** **3 LLM requests** (intent + answer + summary). With query rewrite: 4; with rerank: +1.

---

## 6. LLM Request Count Summary

| Path | Condition | LLM requests (sync) | + background summary |
|------|-----------|----------------------|------------------------|
| No KB | `use_knowledge_base=False` | 1 (answer) | +1 → **2 total** |
| General | `_is_general_conversation(question)` | 1 (answer) | +1 → **2 total** |
| General after intent | Intent = `general` | 1 (intent) + 1 (answer) | +1 → **3 total** |
| Advanced RAG | `advanced_rag=True` | 1 (answer) | +1 → **2 total** |
| Standard RAG | document/transcript, default (no rewrite, no rerank) | 1 (intent) + 1 (answer) | +1 → **3 total** |
| Standard RAG + query rewrite | `RAG_QUERY_REWRITE_ENABLED=True` | +1 (query rewrite) | **4 total** |
| Standard RAG + rerank | `RAG_RERANK_ENABLED=True` | +1 (rerank) | **4 or 5 total** |

So:

- **Minimum** for a single user message (general or advanced RAG): **2 LLM requests** (answer + summary).
- **Typical RAG** (standard, no rerank): **4 LLM requests** (intent, query rewrite, answer, summary).
- **With rerank:** **5 LLM requests**.

All “answer” requests in streaming mode are **one** HTTP request to the LLM that streams tokens; we count that as one LLM request.

---

## 7. Data Flow Diagram (Standard RAG)

```
[User types message]
        │
        ▼
[KnowledgeChat.send] ──► askChatStream(chatId, message, callbacks, options)
        │
        ▼
[POST /api/chat/ask-stream]
        │
        ├─ Load history (DB)
        ├─ Load conversation_summary (DB)
        ├─ INSERT user message (DB)
        │
        ▼
[answer_stream(message, history, persona, context_window, conversation_summary, use_knowledge_base, advanced_rag)]
        │
        ├─ use_knowledge_base=False? ──► _answer_general_stream ──► [1 LLM] ──► stream chunks + done
        │
        ├─ _is_general_conversation? ──► _answer_general_stream ──► [1 LLM] ──► stream chunks + done
        │
        ├─ advanced_rag=True?
        │       │
        │       ├─ retrieve_single_query (embedding only, no LLM)
        │       ├─ _build_rag_context_fast (truncate only)
        │       └─ [1 LLM] chat_stream ──► stream chunks + done
        │
        └─ Standard RAG:
                │
                ├─ _get_document_titles() (DB)
                ├─ _classify_intent() ──► [LLM 1]
                ├─ intent=general? ──► _answer_general_stream ──► [LLM 2] ──► done
                │
                ├─ retrieve():
                │       ├─ generate_queries() ──► [LLM 2]
                │       ├─ Dense + sparse search (embeddings, no LLM)
                │       └─ optional _rerank_hits() ──► [LLM 3 if RAG_RERANK_ENABLED]
                │
                ├─ _build_rag_context() (no compress in current code)
                ├─ [LLM 3 or 4] chat_stream(question + context) ──► stream chunks
                └─ yield ("done", answer, citations)
        │
        ▼
[StreamingResponse: chunk lines + done line]
        │
        ├─ INSERT assistant message (DB)
        └─ background_tasks.add_task(_update_summary_background)
                │
                └─ update_conversation_summary() ──► [LLM 4 or 5]
```

---

## 8. Key Files

| Layer | File | Role |
|-------|------|------|
| Frontend | `frontend/components/KnowledgeChat.tsx` | UI, send(), calls askChatStream |
| Frontend | `frontend/services/backend.ts` | askChatStream, createChat, API_BASE |
| Backend API | `backend/app/api/routes/chat.py` | /create, /ask, /ask-stream, history & summary DB, background summary |
| RAG orchestration | `backend/app/rag/advanced.py` | answer(), answer_stream(), intent, retrieve(), context build, general/RAG branches |
| Retrieval | `backend/app/rag/index.py` | FaissIndex, dense + transcript index, sparse (BM25) |
| LLM client | `backend/app/rag/llm.py` | OpenAICompatChat.chat(), chat_stream() → /chat/completions |

This is the full end-to-end chat and RAG flow and the exact LLM request counts per path.
