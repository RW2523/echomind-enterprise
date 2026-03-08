# EchoMind RAG Flow (Knowledge Chat)

End-to-end flow from user question to streamed answer with citations.

---

## 1. Entry point

**API:** `POST /chat/ask-stream`  
**Handler:** `ask_stream()` in `backend/app/api/routes/chat.py`

- Reads `message`, `chat_id`, `use_knowledge_base`, `source_options` (Transcript / Document / General).
- Loads conversation history and optional summary from DB.
- Calls `answer_stream(question, history, …)` and streams NDJSON: `{"type":"chunk","text":"…"}` then `{"type":"done","answer":"…","citations":[…]}`.

---

## 2. Fast paths (no retrieval)

Inside `answer_stream()` / `answer()`:

| Condition | Result |
|-----------|--------|
| `use_knowledge_base == False` | Answer via **general** LLM only (`_answer_general_stream`). |
| Source options: neither Transcript nor Document | Same: general only. |
| `_is_general_conversation(question)` (greeting, thanks, “how are you”) | General reply, no RAG. |

Otherwise → **retrieval + RAG pipeline**.

---

## 3. Retrieval: `retrieve_semantic_first(question, k, …)`

**Returns:** `(source_type, hits)` where `source_type ∈ {"transcript", "document", "general", "insufficient"}`.

### 3.1 Query embedding (once)

- Embed the question with `index.emb.embed([q])` (Ollama `nomic-embed-text`).
- Same vector is reused for section index, transcript search, and document search.

### 3.2 Query classification (optional)

- `classify_query_type(question)` → e.g. conceptual, procedural, definition, citation.
- `get_rrf_weights(q)` → dynamic dense/sparse weights for RRF.
- For **definition** queries, optional **glossary** search first; if hits found → return `("document", glossary_hits)`.

### 3.3 Section restriction (documents only)

- **Section resolver:** `SectionResolver.resolve(question)` parses explicit refs (e.g. “paragraph 030201”, “Volume 5 Chapter 3”).
- **TOC / section index:** If `index.section_index` and/or `index.toc_index` exist, run section-level search to get `allowed_section_paths`.
- Document search is then **restricted** to those sections (or global if no restriction / fallback allowed).

### 3.4 Parallel search

- **Transcript:**  
  - Dense: `index.search_transcript_only(q, k, query_vector=qv)`  
  - Sparse: `index.transcript_sparse.search(q, k)`  
  - Optional: time range, location, “last N transcripts” filters.
- **Document:**  
  - If `allowed_section_paths`:  
    - `index.search_document_restricted(q, k, allowed_section_paths, …)`  
    - `index.search_document_sparse_restricted(q, k, allowed_section_paths, …)`  
  - Else:  
    - `index.search_document_only(q, k, …)`  
    - `index.search_document_only_sparse(q, k)`  

### 3.5 Merge and source selection

- **Transcript:** Dense + sparse merged with **weighted RRF** (e.g. 0.6 / 0.4), then time/location/last-N filters → `transcript_hits`.
- **Document:** Dense + sparse merged with **dynamic RRF** weights → `document_hits`.
- Optional **keyword/grep fallback** if best document score &lt; threshold (e.g. for acronyms).
- **Pick source:**  
  - If transcript best score ≥ threshold and (no document or transcript ≥ document) → `("transcript", transcript_hits)`.  
  - Else if document best score ≥ threshold → `("document", document_hits)`.  
  - Else if general allowed → `("general", [])`.  
  - Else → `("insufficient", [])`.

---

## 4. Post-retrieval routing

| source_type   | What happens |
|---------------|--------------|
| `insufficient` | Return `INSUFFICIENT_CONTEXT_MSG` (+ optional `handle_missing_information`), no LLM. |
| `general`     | Call `_answer_general_stream` (no RAG). |
| `transcript` or `document` | Call `_run_rag_pipeline(question, hits, source_type)` then generate answer. |

---

## 5. RAG pipeline: `_run_rag_pipeline(question, hits, source_type)`

Used only for **document** (and conceptually similar for transcript). Produces context and metadata for the LLM.

### 5.1 Section resolution (documents)

- `SectionResolver.resolve(question)` → explicit section refs, resolved paths (once per request).

### 5.2 Rerank (documents)

- **Cross-encoder reranker** over top-K hits → keep top-N (`RAG_RERANK_TOP_K` / `RAG_RERANK_FINAL_N`).

### 5.3 Graph expansion (documents, optional)

- If top hit score &lt; confidence threshold: expand via parent/child/sibling chunks, then rescore.

### 5.4 Sort, dedupe, section limit, trim

- Sort by score; optional dedupe by section; limit sections per answer; trim to context budget (`RAG_CONTEXT_MAX_CHARS`).

### 5.5 Evidence extraction and evidence gate (documents)

- **Evidence sentences:** `extract_evidence_sentences(question, hits, …)` from chunks.
- **Evidence gate:** `gate_evidence(question, evidence_sentences, …)` → pass/fail + fallback message.
- If **evidence gate enabled** and **failed** → early exit with fallback message and citations (no LLM).

### 5.6 Build context

- **Context blocks:** `_build_rag_context(question, hits)` → list of text blocks (with optional compression, parent expansion, dedupe).
- Optional **evidence-only** block prepended for documents.
- For **explicit section** or **comparison** queries: prepend `[CITATION REFERENCE]` or `[COMPARISON REFERENCE]` blocks.
- **TOC guardrail:** If question is “chapters/contents” and context has no TOC signals → early exit with message.
- **Answer gate:** `gate_context(question, ctx_block, hits, …)`; if fail → early exit with fallback.
- Final `ctx_block` = concatenation of blocks (up to context budget).

### 5.7 Output of pipeline

- `_RagPipelineResult`: `ctx_block`, `enriched` hits, `doc_ids`, `resolved`, `evidence_sentences`, `timing`, and optional `early_exit` + `early_exit_msg` + `early_exit_citations`.

---

## 6. Answer generation (stream path)

- If **early exit** from pipeline → stream `early_exit_msg` and `early_exit_citations`, then `done`.
- If **strict-citations mode** (`RAG_STRICT_CITATIONS`): call `_answer_with_strict_citations(...)` (non-streaming), then stream full answer and citations, then `done`.
- Else:
  - **Messages:** `_build_llm_messages(question, ctx_block, history, persona, conversation_summary)`  
    - System: `_rag_system_prompt(persona)` (DoD financial advisor, citation rules, conciseness).  
    - User: question + optional format hint + “Document excerpts:” + `ctx_block`.
  - **Stream:** `chat.chat_stream(msgs, …)` → yield `("chunk", delta, None)` for each token.
  - **Postprocess:** `_postprocess_answer_text(ans, question, enriched, resolved, doc_ids, source_type)`:
    - If `information_missing(ans)` → append `handle_missing_information(question)`.
    - Else if `is_inferred(ans)` → `improve_inference_transparency_async` (e.g. append “For more detail, see: …”).
    - Else if no citation in text → optionally `improve_citation_accuracy(ans, section_ref, …)`.
  - **Citations:** Built from `enriched` via `_build_citation(x)` when `RAG_EXPOSE_SOURCES` is True.
  - Yield `("done", ans, citations)`.

---

## 7. Data flow summary

```
User question
    → Fast path? (no KB / general only / greeting) → General LLM → Stream + done
    → retrieve_semantic_first
        → Embed query
        → (Optional) glossary for definition
        → Section restriction (resolver + TOC/section index)
        → Parallel: transcript (dense + sparse), document (dense + sparse) [section-restricted or global]
        → RRF merge, filters, source selection → (source_type, hits)
    → If insufficient → INSUFFICIENT_CONTEXT_MSG
    → If general → _answer_general_stream
    → If transcript/document → _run_rag_pipeline
        → Resolve sections, rerank, graph expansion, dedupe, trim
        → Evidence extraction + evidence gate (optional early exit)
        → _build_rag_context → ctx_block
        → TOC guardrail, answer gate (optional early exits)
    → Build LLM messages (system + history + user with ctx_block)
    → chat_stream → postprocess (missing/inferred/citation)
    → Stream chunks then ("done", answer, citations)
```

---

## 8. Key files

| Layer        | File / symbol |
|-------------|----------------|
| API         | `backend/app/api/routes/chat.py` → `ask_stream`, `answer_stream` |
| Retrieval   | `backend/app/rag/advanced.py` → `retrieve_semantic_first` |
| Section/TOC | `advanced.py` → `_get_section_restricted_paths`; `index.section_index`, `index.toc_index` |
| RAG pipeline| `advanced.py` → `_run_rag_pipeline` |
| Context     | `advanced.py` → `_build_rag_context`, `_build_rag_context_fast` |
| Evidence    | `evidence_gate.py` → `gate_evidence`; `advanced.py` → `extract_evidence_sentences` |
| Answer      | `advanced.py` → `_build_llm_messages`, `_rag_system_prompt`, `_postprocess_answer_text` |
| Citations   | `citation_utils.py` → `_build_citation`, `is_inferred`, `improve_inference_transparency_async` |
| Index       | `backend/app/rag/index.py` (FAISS + BM25, document/transcript, section, TOC) |
