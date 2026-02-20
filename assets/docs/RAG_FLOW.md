# End-to-End RAG Flow: Chunking, Embedding & Retrieval

This document explains the full **Retrieval-Augmented Generation (RAG)** pipeline in the multi-agent chatbot: from document upload through chunking, embedding, storage in Milvus, and query-time retrieval and answer generation.

---

## 1. High-Level RAG Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           INGESTION (Indexing) Pipeline                                   │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│   User uploads files  →  Save to disk  →  Load & parse  →  Chunk  →  Embed  →  Milvus   │
│        (PDF, etc.)         (uploads/)    (Unstructured)   (split)   (Qwen3)   (vector DB)│
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           QUERY (Retrieval) Pipeline                                     │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│   User question  →  Supervisor decides  →  RAG tool  →  Embed query  →  Similarity     │
│   in chat             to use RAG            invoked      (Qwen3)         search in      │
│                                                                  Milvus  →  Top-k chunks │
│                                                                                          │
│   Top-k chunks  →  Build context string  →  LLM (gpt-oss)  →  Grounded answer to user    │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Ingestion Flow (Step by Step)

### 2.1 User uploads documents

- **Where:** Frontend → “Upload Documents” in sidebar → `DocumentIngestion` component.
- **API:** `POST /api/ingest` with multipart form data (files).
- **Backend:** `main.py` → `ingest_files()` creates a `task_id`, stores `filename` + raw `content` per file, and queues **background** work via `process_and_ingest_files_background()` in `utils.py`.

So ingestion is **asynchronous**: the API returns immediately with `task_id` and status `"queued"`; the real work runs in a background task.

### 2.2 Saving files to disk

- **Where:** `utils.py` → `process_and_ingest_files_background()`.
- **Steps:**
  1. Create directory `uploads/<task_id>/`.
  2. For each file in `file_info`: write binary `content` to `uploads/<task_id>/<filename>`.
  3. Collect `file_paths` and `file_names` for the next stage.

Files are stored under `uploads/` so the loader can open them by path (required by Unstructured and PyPDF).

### 2.3 Loading and parsing documents

- **Where:** `vector_store.py` → `_load_documents(file_paths=...)`.
- **Input:** List of absolute paths to the saved files (e.g. PDFs, text files).

For **each file**:

1. **Source name:**  
   `source_name = os.path.basename(file_path)` (e.g. `"whitepaper.pdf"`).  
   This becomes the `source` in metadata and is used later for **source filtering** in the UI (e.g. “Select Sources”).

2. **Loading by file type:**
   - **Primary:** `UnstructuredLoader(file_path)` from `langchain_unstructured`.  
     Uses the Unstructured library to parse PDFs, Word, HTML, etc. into one or more LangChain `Document` objects (each has `page_content` and `metadata`).
   - **Fallback (e.g. if Unstructured fails for a PDF):**  
     - Try **PyPDF**: `PdfReader` → extract text per page → join with `"\n\n"` into one string → wrap in a single `Document`.
     - If that fails or yields no text: **raw read** of the file as UTF-8 text (with `errors="ignore"`).
     - If still no content: create a minimal `Document` with `page_content = "Document: <filename>"`.

3. **Metadata normalization:**  
   For every `Document`:
   - Set/overwrite: `source`, `file_path`, `filename` (all stored for filtering and display).
   - Other keys from the loader are kept only if they are simple types (lists/dicts/sets are stringified) so Milvus/metadata stays serializable.

4. **Aggregation:**  
   All `Document`s from all files are extended into a single list: `documents.extend(docs)`.

**Output:** One list of LangChain `Document` objects (one or more per file), each with `page_content` (text) and cleaned `metadata` (including `source`).

---

## 3. Chunking (Splitting) in Detail

### 3.1 Why chunking?

- Models and embedding APIs have **maximum input length**; long documents must be split.
- **Retrieval** works on **chunks**, not whole documents: we search for the most relevant *pieces* of text.
- Smaller chunks give more precise retrieval; too small can lose context. **Overlap** between chunks helps avoid cutting sentences or ideas in the middle.

### 3.2 What this project uses

- **Splitter:** `RecursiveCharacterTextSplitter` from `langchain_text_splitters` (in `vector_store.py`).
- **Parameters:**
  - **`chunk_size=1000`**  
    Target size of each chunk in **characters** (not tokens). The splitter tries to keep chunks around this length.
  - **`chunk_overlap=200`**  
    Number of characters shared between two consecutive chunks. So chunk 2 starts 200 characters before the “logical” end of chunk 1, reducing the chance that a sentence or key phrase is split across a boundary.

### 3.3 How RecursiveCharacterTextSplitter works

1. **Separators (in order of preference):**  
   It tries to split on natural boundaries, typically in this order:  
   `"\n\n"` → `"\n"` → `" "` → `""` (character-by-character as last resort).  
   So it prefers paragraph breaks, then line breaks, then spaces, then any character.

2. **Algorithm (conceptually):**
   - Take the next piece of text (initially the full `page_content`).
   - If it’s already ≤ `chunk_size`, emit it as one chunk.
   - Otherwise, split on the first separator that appears (e.g. `"\n\n"`), then recursively split the resulting parts until each part is ≤ `chunk_size`, respecting `chunk_overlap` between adjacent chunks.

3. **Result:**  
   A list of **smaller** `Document` objects. Each keeps the same `metadata` as the original (including `source`, `file_path`, `filename`). Only `page_content` is split.

### 3.4 Where chunking is invoked

- In `vector_store.py` → `index_documents(documents)`:
  - `splits = self.text_splitter.split_documents(documents)`  
  - Then `self._store.add_documents(splits)`  
So **every** piece of text that goes into the vector store is first chunked, then embedded chunk-by-chunk.

**Summary:**  
Documents are loaded per file (Unstructured/PyPDF/raw), normalized, then split into ~1000-character chunks with 200-character overlap using `RecursiveCharacterTextSplitter`. Those chunks are what get embedded and stored in Milvus.

---

## 4. Embedding in Detail

### 4.1 Role of embeddings

- **Embedding** = a fixed-size vector (list of floats) that represents the *semantic* meaning of a text.
- Similar meanings → vectors that are “close” in distance (e.g. cosine or L2).  
- So we can **embed the user query** and **embed each stored chunk**, then **search** for the chunks whose vectors are closest to the query vector.

### 4.2 Model and service

- **Model:** Qwen3-Embedding-4B (e.g. `Qwen3-Embedding-4B-Q8_0.gguf`).
- **Runtime:** Runs in a separate container **qwen3-embedding** (see `docker-compose-models.yml`), exposing an **OpenAI-compatible HTTP API** on port 8000 (e.g. `/v1/embeddings`).
- **Usage:** The backend does **not** call the model directly; it calls this HTTP API.

### 4.3 CustomEmbeddings class (vector_store.py)

- **Purpose:** Adapt the Qwen3 embedding service (OpenAI-style API) to the interface expected by LangChain’s Milvus integration: `embed_documents(texts)` and `embed_query(text)`.
- **Endpoint:** `POST {host}/v1/embeddings` with JSON body:  
  `{"input": text, "model": self.model}`  
  (For multiple texts, the current implementation calls the API **once per text** in a loop.)
- **Response:** Expects `data[0]["embedding"]` to be the vector (list of floats).  
- **Methods:**
  - `embed_documents(texts: list[str])` → `list[list[float]]` (one vector per text). Used when **indexing** chunks.
  - `embed_query(text: str)` → `list[float]` (single vector). Used when **searching** with the user query.

So:
- **Indexing:** Each chunk’s `page_content` is embedded via `embed_documents`; Milvus stores those vectors (plus metadata).
- **Query:** The user question is embedded via `embed_query`; that vector is used for similarity search in Milvus.

### 4.4 Dimensions and performance

- The embedding dimension is whatever Qwen3-Embedding-4B returns (e.g. 4096 or similar; exact value is in the model card). Milvus creates the collection with that dimension.
- Embedding is the **GPU-heavy** part of ingestion (and of query); the qwen3-embedding container uses the GPU for fast inference.

---

## 5. Vector Store (Milvus)

### 5.1 Role

- **Milvus** is a vector database: it stores (vector, metadata) and supports **similarity search** (e.g. by cosine or L2).
- **Collection name:** `"context"` (single collection for all RAG chunks).
- **Backend:** LangChain’s `Milvus` wrapper (`langchain_milvus`) uses `CustomEmbeddings` as `embedding_function` and connects to Milvus at `uri` (e.g. `http://milvus:19530`).

### 5.2 Indexing (add_documents)

- `_store.add_documents(splits)` (in `index_documents`):
  - For each chunk `Document`, LangChain/Milvus:
    - Calls `embed_documents([chunk.page_content])` (or equivalent) to get one vector per chunk.
    - Inserts (vector, metadata) into the `context` collection.
- After all inserts, `flush_store()` is called so that data is persisted and searchable immediately.

### 5.3 Retrieval (get_documents)

- **Method:** `vector_store.get_documents(query, k=8, sources=None)`.
- **Steps:**
  1. **Embed the query:** `embed_query(query)` → one vector.
  2. **Build search options:**  
     - `search_type="similarity"` (e.g. cosine or L2, depending on how the collection is configured).  
     - `search_kwargs = {"k": 8}` (top 8 chunks).  
     - If `sources` is provided (list of source names from config):  
       - Build a filter expression, e.g. `source == "file1.pdf" || source == "file2.pdf"`.  
       - `search_kwargs["expr"] = filter_expr` so Milvus only returns chunks from those sources.
  3. **Run search:** `retriever = self._store.as_retriever(...); docs = retriever.invoke(query)`.
  4. **Return:** List of LangChain `Document` objects (chunks) with `page_content` and `metadata` (including `source`).

So at query time we get the **top-k most similar chunks** (optionally restricted by selected sources), not the full documents.

---

## 6. Query-Time RAG Flow (When the User Asks a Question)

### 6.1 Chat path

1. User sends a message in the UI.
2. Frontend sends it over WebSocket to the backend.
3. Backend pushes a `HumanMessage` into the conversation and runs the **supervisor agent** (LangGraph) with tool-calling enabled.
4. The **supervisor** (e.g. gpt-oss-20b) can decide to call tools. One of the tools is **`search_documents`**, provided by the **RAG MCP server**.

### 6.2 RAG MCP server and tool

- **Implementation:** `backend/tools/mcp_servers/rag.py`.
- **Tool:** `search_documents(query: str)`.
- When the supervisor calls this tool with the user’s question (or a refined query), the RAG server runs a **two-step graph**: **retrieve** → **generate**.

### 6.3 Retrieve node

- **Input:** `state["question"]` (the query), `state.get("sources")` (optional list of source names from config, e.g. selected in the UI).
- **Action:**  
  - `vector_store.get_documents(question, k=8, sources=sources)`  
  - Same vector store and Milvus as in ingestion; uses the same embeddings for the query.
- **Output:** `state["context"]` = list of retrieved chunk `Document`s.

### 6.4 Generate node

- **Input:** `state["context"]` (retrieved chunks).
- **Action:**
  - **Hydrate context:** `_hydrate_context(context)` builds one string:  
    `"\n\n".join([doc.page_content for doc in context])`.
  - **Prompt:** System prompt says: “Use the following retrieved context to answer the question. If no context, say so. Don’t make up information not in the context. Keep it concise.”  
    The placeholder `{context}` is filled with that string.
  - **LLM call:** Same supervisor-style model (e.g. gpt-oss-20b) via `model_client.chat.completions.create(...)` with that system prompt and the user question.
- **Output:** `state["messages"]` with the final answer (e.g. `AIMessage`).

### 6.5 Back to the user

- The RAG tool returns the **text** of that answer to the supervisor.
- The supervisor can use it as a **ToolMessage** and optionally add its own wording or combine with other tools; the final assistant reply is streamed back over the WebSocket to the UI.

So the **end-to-end RAG path** is:  
User question → Supervisor → `search_documents(query)` → Embed query → Milvus similarity search → Top-k chunks → Concatenate as context → LLM with context → Answer → Back to user.

---

## 7. End-to-End Flow Summary (Reference)

| Phase        | Component              | What happens |
|-------------|------------------------|--------------|
| **Upload**  | Frontend, `POST /ingest` | Files sent to backend; background task queued. |
| **Save**    | `utils.process_and_ingest_files_background` | Files written under `uploads/<task_id>/`. |
| **Load**    | `vector_store._load_documents` | UnstructuredLoader (or PyPDF/raw) → list of `Document`s per file, metadata normalized. |
| **Chunk**   | `RecursiveCharacterTextSplitter` | Split into ~1000-char chunks, 200-char overlap; same metadata. |
| **Embed**   | `CustomEmbeddings` → qwen3-embedding API | Each chunk → one vector; stored with chunk in Milvus. |
| **Store**   | Milvus (`context` collection) | Vectors + metadata (e.g. `source`, `file_path`, `filename`) persisted; flush. |
| **Config**  | `config_manager` | New filenames added to `config.sources`; UI “Select Sources” reads `selected_sources`. |
| **Query**   | User message → Supervisor | Supervisor may call tool `search_documents(query)`. |
| **Retrieve**| RAG agent `retrieve` node | `get_documents(query, k=8, sources=selected_sources)` → embed query, similarity search, top-k chunks. |
| **Generate**| RAG agent `generate` node | Chunks → context string → LLM with “answer from context” prompt → final answer. |
| **Respond** | Supervisor → WebSocket | Answer (as tool result and/or assistant message) streamed to user. |

---

## 8. Key Files (Quick Reference)

- **Ingestion API:** `backend/main.py` (`/ingest`, background task).
- **Background ingestion:** `backend/utils.py` (`process_and_ingest_files_background`).
- **Load / chunk / embed / store:** `backend/vector_store.py` (`_load_documents`, `index_documents`, `CustomEmbeddings`, `get_documents`).
- **Chunking:** `RecursiveCharacterTextSplitter` in `vector_store.py` (chunk_size=1000, chunk_overlap=200).
- **Embedding service:** Qwen3-Embedding container; called via `CustomEmbeddings` in `vector_store.py`.
- **RAG at query time:** `backend/tools/mcp_servers/rag.py` (RAGAgent graph: retrieve → generate; `search_documents` tool).
- **Config/sources:** `backend/config.py` (sources, selected_sources); updated in `utils.py` after successful indexing.

This is the complete RAG flow from document upload through chunking, embedding, storage, and retrieval-backed answer generation.
