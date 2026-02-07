# EchoMind RAG Bot & Chunking — Full Explanation

This document explains how the EchoMind RAG (Retrieval-Augmented Generation) system works end-to-end and how the chunking pipeline processes documents and transcripts.

---

## Part 1: How the Entire RAG Bot Works

The RAG bot has three main phases: **ingestion** (getting content into the knowledge base), **retrieval** (finding relevant chunks for a question), and **generation** (answering using those chunks plus the LLM). The chat API ties them together and adds conversation summary and persona.

---

### 1.1 Ingestion: How Content Enters the Knowledge Base

Content gets into the system in two ways. Both end up in the **same** vector index (FAISS + BM25) and SQLite DB.

#### A. Document upload (`POST /api/docs/upload`)

**Flow:**

1. **Route** (`backend/app/api/routes/docs.py`): Receives an uploaded file (PDF, DOCX, PPTX, or plain text).

2. **Parse** (`backend/app/rag/parse.py`):
   - **PDF**: `pypdf.PdfReader` → extract text from each page, joined by newlines.
   - **DOCX**: `python-docx` → one string from all paragraph texts.
   - **PPTX**: `python-pptx` → text from each shape on each slide.
   - **Other**: Treated as UTF-8 text.

3. **Index** (`backend/app/rag/index.py` → `add_document`):
   - Generates a new `doc_id`.
   - Calls **`chunk_document(text, doc_id)`** (see Part 2 for chunking details).
   - From the returned chunks, takes only **embed chunks** (all chunks that are **not** parent chunks: `embed_chunks = [c for c in all_chunks if not c.is_parent]`).
   - **Embed**: Each embed chunk’s text is sent to the **Ollama embeddings API** (`OLLAMA_EMBED_URL` + `OLLAMA_EMBED_MODEL`, e.g. `nomic-embed-text`). Text is truncated to `EMBED_MAX_CHARS` at a word boundary so the embed model never overflows.
   - **Store**:
     - **FAISS**: Vectors are L2-normalized and added to a single `faiss.IndexFlatIP` (inner product = cosine similarity). The index and a meta file (`chunk_ids`, `source_by_chunk`) are saved to disk.
     - **SQLite**: One row in `documents` (id, filename, filetype, created_at, meta_json). One row per chunk in `chunks` (id, doc_id, chunk_index, text, source_json).
     - **BM25 (sparse)**: Chunk texts are tokenized (lowercase, regex `[a-z0-9]{2,}`) and added to a BM25Okapi index; `chunk_ids` and `corpus_tokens` are persisted.

So: **file → parse → chunk_document → embed (embed chunks only) → FAISS + SQLite + BM25**.

#### B. Transcript store (`POST /api/transcribe/store`)

**Flow:**

1. **Route** (`backend/app/api/routes/transcribe.py`): Receives `raw_text`, optional `refined_text` (or legacy `polished_text`), optional `echotag`.

2. **Tags**: Optional LLM call to get 3–6 comma-separated topic tags from `raw_text`. `echotag` is set from request or from these tags.

3. **DB**: One row in `transcripts` (id, raw_text, polished_text, tags_json, echotag, echodate, created_at). The refined/structured notes are stored in the `polished_text` column; API accepts `refined_text`. `echodate` = current time (ISO).

4. **Index**: `index.add_text("transcript_{tid}", raw_text + "\n\n" + refined_text, meta)` with `meta = { type: "transcript", tags, echotag, echodate, created_at }` (refined text is the value stored in `polished_text` column).  
   **`add_text`** simply calls **`add_document(title, "text", text, meta)`**. So the transcript is treated as one “document” with:
   - `filename` = `"transcript_{tid}"`
   - `filetype` = `"text"`
   - Same pipeline: **chunk_document** → embed (embed chunks only) → FAISS + SQLite + BM25.

So: **transcript text + meta → add_document (chunking + embed + store)**. Documents and transcripts live in the same index; metadata (e.g. `type: "transcript"`, tags, echodate) is in `documents.meta_json` and in chunk `source` for filtering/boosting later.

---

### 1.2 Retrieval: How Relevant Chunks Are Found for a Question

Entry point is **`retrieve(question, k, context_window)`** in `backend/app/rag/advanced.py`. It returns a list of top‑k hits (each with `chunk_id`, `score`, `text`, `source`).

**Step 1 — Query expansion**

- **`generate_queries(question)`**:
  - If **intent-aware rewriting** is on: the question is **classified** (factual / procedural / exploratory / temporal) via a short LLM call. Then it is **rewritten** with an intent-specific prompt into **2 alternative search queries** (e.g. same fact focus, or how-to focus). Original question + up to 3 variants → **up to 4 queries**.
  - If off: one generic “2 alternative queries” LLM call; again original + variants, capped at 4.
- Purpose: different phrasings improve recall (e.g. “cost” vs “pricing”) while intent-specific prompts reduce noisy expansions.

**Step 2 — Hybrid search (dense + sparse) per query**

- For **each** of the 1–4 queries:
  - **Dense** (`index.search(query, k_per_query)` in `rag/index.py`):
    - Query is embedded with the same Ollama embed model.
    - FAISS `IndexFlatIP` search returns top‑k_per_query vectors by inner product (cosine on L2-normalized vectors).
    - For each result, chunk `text` and `source_json` are read from SQLite; returned as `{ chunk_id, score, text, source }`.
  - **Sparse** (`index.sparse.search(query, k_per_query)` in `rag/sparse.py`):
    - Query is tokenized (same regex as index time). BM25 scores all chunks; top‑k_per_query by score.
    - Same shape: `{ chunk_id, score, text, source }`.

**Step 3 — Merge: weighted RRF**

- **`_weighted_rrf(dense_hits_per_query, sparse_hits_per_query, k, dense_weight, sparse_weight)`**:
  - For each chunk that appears in any list, a **reciprocal rank fusion** score is computed:  
    `dense_weight * (1/(RRF_K+rank))` for each dense list and `sparse_weight * (1/(RRF_K+rank))` for each sparse list (RRF_K = 60).
  - Chunks are sorted by this combined RRF score; top‑k (or more if rerank is on) are kept. Final score is either the dense score (when the chunk came from dense) or a normalized value derived from RRF.
  - Default weights (e.g. 0.6 dense, 0.4 sparse) favor semantic similarity slightly over keyword match.

**Step 4 — Hard time filter (optional)**

- **`_filter_hits_by_context_window(hits, context_window)`**:  
  If `context_window` is `"24h"`, `"48h"`, or `"1w"`, each hit’s **document** is looked up in `documents` by `doc_id` in `source`; only chunks whose document `created_at` is within that window are kept. `"all"` = no filter.

**Step 5 — Time-decay scoring (optional)**

- **`_apply_time_decay(hits, doc_created, halflife_days)`**:  
  For each hit, the document’s `created_at` is parsed; **score** is multiplied by `exp(-age_days / halflife_days)` (with a floor). So recent documents rank higher without dropping old ones entirely.

**Step 6 — Tag boost (optional)**

- **`_apply_tag_boost(hits, question, doc_meta, factor)`**:  
  For chunks whose document has `meta.type == "transcript"` and `meta.tags`, the query is tokenized and compared to tag tokens. If there is overlap, score is boosted by `(1 + factor * overlap)` (capped at 1.0). This helps when transcript tags describe the content and match the question.

**Step 7 — Authoritative preference (optional)**

- **`_prefer_authoritative_sort(hits)`**:  
  Chunks are sorted by score descending; then, when scores are equal or very close, chunks from **authoritative** documents (filetype pdf/docx/pptx/txt and **not** transcript) come before transcript chunks. So uploaded docs win over transcripts in ties.

**Step 8 — Optional rerank**

- If **RAG_RERANK_ENABLED**: top `RAG_RERANK_CANDIDATES` hits are sent to the LLM in one batch; the model scores each excerpt 0–10 for relevance to the question. Hits are reordered by that score and only **RAG_RERANK_TOP_N** are returned. Otherwise, the list is just sliced to the first **k**.

So retrieval is: **question → 1–4 queries → dense + sparse per query → weighted RRF → optional time filter → optional time decay → optional tag boost → authoritative sort → optional rerank → top k**.

---

### 1.3 Context Building: From Hits to Prompt Blocks

**`_build_rag_context(question, hits)`** in `advanced.py` turns the list of hits into **numbered context blocks** that will be pasted into the LLM prompt.

- For each hit in order:
  - If the chunk has a **parent_chunk_id** (from long-form chunking) and that parent hasn’t been added yet:
    - The parent chunk’s full text is loaded from `chunks` and truncated to **`_parent_context_max_chars()`** (configurable, e.g. 1600) at a word boundary.
    - This truncated parent text is appended as one block. (So one parent can appear once and give broader context for its children.)
  - The **child** (or non–parent) chunk text is then **compressed** by an LLM call: **`compress(question, chunk_text, source)`**.
    - The compression prompt asks to keep only sentences **directly needed to answer the question**, copy verbatim, and to prefix **"[Partial]: "** or **"[Conflicting]: "** when relevant.
  - The compressed text is appended as another block. The same hit is also stored in **enriched** (for citations) and **chunk_ids_used** (for logging).

- **Optional sentence dedupe**: If `RAG_DEDUPE_SENTENCES` is on, **`_dedupe_overlapping_sentences(blocks, overlap_ratio)`** runs:
  - Each block is split into sentences (by `.!?`).
  - For each sentence, a normalized word set is computed. If its Jaccard similarity with any previously kept sentence is ≥ `overlap_ratio`, the sentence is dropped. So later blocks don’t repeat the same facts and token use goes down.

- **Format**: Blocks are formatted as **`[1] ... [2] ...`** by **`_rag_context_block(blocks)`** and passed to the LLM as the “Context” part of the user message (no visible “source” or filename in the prompt unless you enable citation exposure).

---

### 1.4 Generation: How the Answer Is Produced

**Entry:** Chat API (`POST /api/chat/ask` or `POST /api/chat/ask-stream`) receives `chat_id`, `message`, optional `persona`, optional `context_window`.

- **Load**: Messages for the chat and **conversation_summary** (from `chats.conversation_summary`) are read.

- **Decide path** (`answer` / `answer_stream` in `advanced.py`):
  - If **`_is_general_conversation(message)`** (greetings, “thanks”, very short non‑question phrases): **No retrieval.** The LLM is called with a friendly “EchoMind” system prompt (plus optional persona) and **conversation summary + current question** (or raw last 10 messages if no summary). No RAG context, no citations.
  - Otherwise:
    - **Retrieve**: `retrieve(message, TOP_K, context_window)` (as above).
    - If **no hits** or **best score < RAG_RELEVANCE_THRESHOLD**: Same as general — no RAG, answer from conversation summary (or history) + question.
    - Else **RAG path**:
      - **Context**: `_build_rag_context(message, hits)` → blocks → `_rag_context_block(blocks)` → one string `ctx_block`.
      - **User message**: Either **conversation summary + current question + ctx_block** (if summary exists) or **last 10 messages +** “Question: … Context: …” (legacy).
      - **System prompt**: **`_rag_system_prompt(persona)`** — “EchoMind, use ONLY the provided context; do not add facts; if parts contradict say so; if context is insufficient say clearly ‘The provided context does not contain enough information to answer this.’”
      - **LLM**: Ollama-compatible `chat` or `chat_stream` with that system prompt and user content. Temperature and max_tokens from config.
      - **Citations**: Only if `RAG_EXPOSE_SOURCES` is True: build citation list from enriched chunks (filename, chunk_index, score, snippet).

- **After the turn**: The new user message and assistant reply are used to **update the conversation summary** via an LLM call (goals, constraints, decisions, key facts). The new summary is stored in `chats.conversation_summary`.

So the “entire RAG bot” is: **ingest (docs + transcripts) → chunk → embed → FAISS + BM25 + DB**; at answer time **expand query → hybrid search → RRF → filters/boosts → (optional rerank) → build context (parent + compress + dedupe) → LLM with summary + question + context → update summary**.

---

## Part 2: The Entire Chunking Process

Chunking is the pipeline that turns a single **document text** (or transcript text) into a list of **Chunk** objects. Only **non‑parent** chunks are embedded and stored in FAISS/BM25; parent chunks are stored in DB and used only at **retrieval time** to expand context.

**Entry point:** **`chunk_document(text, doc_id)`** in `backend/app/rag/chunking/pipeline.py`.

---

### 2.1 Pipeline Overview

```text
text
  → detect_document_type(text)     → DocType (FAQ | BOOK | SENSITIVE | USER)
  → sanitize_text(text)            → (clean_text, redacted, sensitivity_level)
  → chunk by strategy (FAQ / long-form / sensitive / unstructured)
  → assign doc_id, chunk_id, chunk_index
  → return List[Chunk]
```

---

### 2.2 Step 1: Document Type Detection

**`detect_document_type(text)`** in `detect.py` uses **heuristics only** (no LLM):

1. **SENSITIVE**  
   If **`_pii_density(text) >= 0.015`** (fraction of characters that are part of a PII regex match), the doc is treated as sensitive. PII patterns include: email, phone, SSN, card numbers, long numeric IDs.

2. **FAQ**  
   If **`_looks_like_faq(text, lines)`**:
   - At least 2 lines that look like questions (e.g. “Q: …?”, “Question 2: …”).
   - Either many such lines (e.g. ≥ 3 and ≥ 10% of lines) or the first 2000 chars contain “FAQ” / “frequently asked”.

3. **BOOK (long-form)**  
   If **`_looks_like_long_form(text, lines, length)`**:
   - Length ≥ 8000 chars, at least 5 paragraph breaks (`\n\n`).
   - Either ≥ 2 “heading-like” lines (e.g. “Chapter …”, “Section …”) or (≥ 20 paragraph breaks and length > 20k) or (average line length > 80 and length > 15k).

4. **USER (default)**  
   If none of the above: treated as unstructured / user-style content.

Order of checks: **SENSITIVE → FAQ → BOOK → USER**.

---

### 2.3 Step 2: Sanitization (PII Redaction)

**`sanitize_text(text)`** in `sanitize.py`:

- Applies a list of **regex patterns** and replaces matches with labels like `[REDACTED_EMAIL]`, `[REDACTED_PHONE]`, `[REDACTED_SSN]`, etc.
- Returns:
  - **clean_text**: redacted string.
  - **redacted**: True if any replacement was made.
  - **sensitivity_level**:  
    - **HIGH** if any redaction happened.  
    - **MEDIUM** if PII density > 0.005 but no redaction.  
    - **LOW** otherwise.

This **clean_text** and **sensitivity_level** are what the chunkers see; chunk metadata (e.g. `redacted`, `sensitivity_level`) is stored in the Chunk model and in `source_json` for audit.

---

### 2.4 Step 3: Chunk by Document Type

The pipeline branches on **DocType** and calls the corresponding chunker with `(clean_text, sensitivity_level, redacted)`.

---

#### 2.4.1 FAQ chunker (`chunk_faq`)

- **Goal**: One chunk per **Q&A pair**; never split a question from its answer.
- **Split**: Regex on “start of next question” (e.g. newline + optional number + “Q.” / “Question:”). The first segment may be intro; the rest are Q&A blocks.
- **Output**: One **Chunk** per non-empty block; `text` = block (capped at **MAX_FAQ_CHARS** = 8000). All chunks are **non‑parent** (`is_parent=False`), so all are embedded.
- **DocType**: FAQ.

---

#### 2.4.2 Long-form / “book” chunker (`chunk_long_form`)

- **Goal**: Long documents get **parent** chunks (large context) and **child** chunks (smaller, used for retrieval). At **retrieval** time, when a child is in the hit list, its parent text can be included once to give the LLM more context.

**Sizes (constants in chunkers.py):**

- Parent: **2,000–3,500 chars** (`_PARENT_MIN`, `_PARENT_MAX`).
- Child: **400–700 chars** (`_CHILD_MIN`, `_CHILD_MAX`), with **2-sentence overlap** between consecutive children (`_CHILD_OVERLAP = 80` is used as overlap in sentence count).

**Process:**

1. Split text into **paragraphs** (`\n\n`).
2. Group paragraphs into **parent chunks**: add paragraphs until adding the next would exceed `_PARENT_MAX`; then start a new parent, optionally keeping the last paragraph as overlap. Each parent is 2k–3.5k chars (except possibly the last).
3. For each parent:
   - Split parent text into **sentences** (regex on `.!?` and newlines).
   - **`_group_sentences_to_size(sentences, _CHILD_MIN, _CHILD_MAX, overlap_sentences=2)`**: Group sentences into segments of 400–700 chars, never splitting a sentence; keep 2 sentences overlap between consecutive segments. These segments are the **child** chunks.
   - Build one **Chunk** with `is_parent=True`, `text=parent_text` (no embedding).
   - Build one **Chunk** per child with `is_parent=False`, `parent_chunk_id=<parent’s chunk_id>` (set in pipeline). Only these children are embedded.

**Pipeline** then assigns `chunk_id` and `doc_id` to parent and children and appends: **parent first, then its children**, then next parent, etc. **`_assign_indices`** sets `chunk_index` in order. So the flat list is [P1, C1, C2, …, P2, C3, …]. Only the **C** entries go to FAISS/BM25.

---

#### 2.4.3 Sensitive chunker (`chunk_sensitive`)

- **Goal**: Small, low-overlap chunks for PII-heavy content so that retrieval returns minimal necessary context.
- **Sizes**: **~450 chars** per chunk (`_SENSITIVE_SIZE`), **0** sentence overlap.
- **Process**: Split clean text into **sentences**, then **`_group_sentences_to_size(..., target_min=225, target_max=450, overlap_sentences=0)`**. One **Chunk** per resulting segment; all non‑parent, all embedded.
- **DocType**: SENSITIVE.

---

#### 2.4.4 Unstructured / “user” chunker (`chunk_unstructured`)

- **Goal**: Robust chunking for mixed or poor formatting (e.g. pasted text, transcripts).
- **Sizes**: **~800 chars** target (`_USER_SIZE`), **1-sentence overlap** between consecutive chunks.
- **Process**: Same as sensitive: **sentences** → **`_group_sentences_to_size(..., target_min=400, target_max=800, overlap_sentences=1)`**. One **Chunk** per segment; all non‑parent, all embedded.
- **DocType**: USER.

**Transcripts** stored via `add_text` use the same **chunk_document** path; they usually have no FAQ structure and are not long enough for BOOK, so they typically fall into **USER** (or SENSITIVE if PII is high) and get 400–800 char sentence-boundary chunks with 1-sentence overlap.

---

### 2.5 Sentence Splitting and Grouping (Shared)

- **`_sentences(text)`**: Splits on regex `(?<=[.!?])\s+|\n\n+` so boundaries are `. ! ?` followed by space or double newline. No mid-sentence splits.
- **`_group_sentences_to_size(sentences, target_min, target_max, overlap_sentences)`**:
  - Adds sentences to a current buffer until adding the next would exceed **target_max**; then emits a chunk (joined by space), and either keeps the last **overlap_sentences** sentences in the buffer or starts fresh.
  - So chunk sizes stay within target range and consecutive chunks can overlap by a few sentences for continuity.

---

### 2.6 Chunk Model and Source Metadata

**`Chunk`** (in `models.py`) has:

- **doc_id**, **chunk_id**, **text**, **doc_type**, **sensitivity_level**, **redacted**, **section**, **parent_chunk_id**, **is_parent**, **chunk_index**, etc.

**`to_source_dict(filename, filetype)`** serializes for DB and for retrieval:

- **doc_id**, **filename**, **chunk_index**, **filetype**, **doc_type**, **section**, **sensitivity_level**, **redacted**, **is_parent**, and **parent_chunk_id** (if set). This is stored in **chunks.source_json** and in **index.meta["source_by_chunk"]**. So at retrieval time every hit has **source** with filetype, filename, doc_id, parent_chunk_id (for long-form), etc., which drives parent expansion, authoritative vs transcript, and tag boost.

---

### 2.7 Summary: Chunking End-to-End

| Step | What happens |
|------|-------------------------------|
| 1 | **Detect type**: SENSITIVE (PII) → FAQ (Q&A) → BOOK (long) → USER (default). |
| 2 | **Sanitize**: Redact PII, get sensitivity level. |
| 3 | **Chunk by type**: FAQ = one chunk per Q&A; BOOK = parents (2k–3.5k) + children (400–700, overlap 2); SENSITIVE = ~450 char, 0 overlap; USER = ~800 char, 1-sentence overlap. |
| 4 | **Sentence-aware**: All non-FAQ chunkers split on sentence boundaries and group with optional overlap. |
| 5 | **IDs**: Pipeline assigns doc_id, chunk_id, chunk_index; for BOOK, parent_chunk_id on children. |
| 6 | **Index**: Only chunks with **is_parent=False** are embedded and added to FAISS and BM25; parents are stored in DB and used only when building RAG context. |

That is how the entire RAG bot works and how the full chunking process runs from raw text to stored, retrievable chunks.
