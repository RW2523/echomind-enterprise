# EchoMind RAG Flow — Complete Explanation

This document describes the **end-to-end RAG (Retrieval-Augmented Generation)** pipeline: from document ingestion and indexing to query classification, retrieval, context building, and answer generation with citations.

---

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           INGESTION (one-time per document)                       │
│  Upload → Parse → Detect DocType → Chunk → Embed → FAISS + BM25 + Section/Glossary │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           QUERY (per user message)                                │
│  Question → Classify → Embed → Retrieve (dense + sparse, transcript vs doc)     │
│  → Pick source (transcript | document | general | insufficient)                  │
│  → RAG pipeline (rerank, evidence, context, gate) → LLM → Postprocess → Citations│
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Ingestion (Indexing)

**Entry points:** `POST /api/docs/upload` (file upload) or `add_text()` for transcripts (e.g. Live Transcript saves, sample transcripts).

### 1.1 Parse and detect type

- **Parse** (`rag/parse.py`): Extracts text from PDF, DOCX, PPTX, or plain text. Returns `(filetype, text, estimated_pages, page_offsets)`.
- **Detect DocType** (`rag/chunking/detect.py`): Classifies the document as one of:
  - **BOOK** — long, structured (e.g. DoD FMR): hierarchical chunking with parent/child, section paths, page numbers.
  - **FAQ** — Q&A pairs.
  - **GOVERNMENT** / **RECORDS** — other structured or semi-structured.
  - **UNSTRUCTURED** — generic text; simple semantic chunking.

### 1.2 Chunking (`rag/chunking/`)

- **chunk_document()** (pipeline): Dispatches to the right chunker by `DocType`.
- **BOOK**: Parent chunks (section-level) + child chunks (smaller segments). Section paths (e.g. `Volume 5 > Chapter 3 > 030201`), page numbers, and metadata are attached. Chunk size/overlap and parent/child limits are configurable (`BOOK_PARENT_*`, `BOOK_CHILD_*`, `CHUNK_SIZE`, `CHUNK_OVERLAP`).
- **Transcripts**: Treated as unstructured or a dedicated path; chunks get metadata: `transcript_id`, `echodate`, `epoch`, `tags`, `location`, `name`.
- **Other types**: FAQ chunker, sensitive chunker, or generic long-form/semantic chunking.

Output: list of **Chunk** objects (each has `chunk_id`, `text`, `doc_id`, `section_path`, `page_number`, `is_parent`, etc.).

### 1.3 Embedding and indexing (`rag/index.py` — `FaissIndex`)

- **Embeddings** (`rag/embeddings.py`): **OllamaEmbeddings** calls `OLLAMA_EMBED_URL` (e.g. `nomic-embed-text`) to get a vector per chunk. Text is truncated to `EMBED_MAX_CHARS` to avoid context overflow.
- **Contextual retrieval (BOOK only):** For BOOK docs, before embedding, the pipeline can:
  - Build **section summaries** (LLM) per section.
  - Assign **chunk roles** (LLM) per child chunk.
  - Build a **contextualized text** per chunk: header (doc title, section summary, chunk role) + raw chunk text. That string is what gets embedded — so retrieval is “section-aware”.
- **FAISS**: Vectors are L2-normalized; index is **inner-product** (equivalent to cosine similarity). Child chunks are embedded; parent chunks are stored in DB for later expansion. Optional IVF index when vector count exceeds a threshold.
- **BM25 (sparse)** (`rag/sparse.py`): Same chunk texts are added to a BM25 index (keyword/Lucene-style). Stored under `META_PATH` and `SPARSE_TRANSCRIPT_META_PATH` for transcript-only sparse.
- **Transcript vs document split**: If the doc is a transcript (`filename` starts with `transcript_` or `meta.type == "transcript"`), chunks are added to both the **main** index and the **transcript-only** index (separate FAISS + sparse). This allows the query path to search “transcripts only” or “documents only”.
- **Section index** (`rag/section_index.py`): For BOOK docs, section-level embeddings (from section summaries or full section text) are stored. Used at query time to **restrict** which sections are searched (BookRAG).
- **Glossary index** (`rag/glossary_index.py`): Glossary sections from BOOK docs are indexed separately for **definition** queries.
- **TOC index** (`rag/indexes/toc_index.py`): Table-of-contents structure (if uploaded) for routing to Volume/Chapter/Section.
- **Cross-reference graph** (`rag/cross_ref_graph.py`): References between sections (e.g. “see paragraph 030205”) are extracted and stored for graph expansion during retrieval.
- **DB**: Rows in `documents` and `chunks`; `chunks` store `text`, `source_json` (metadata for citation), and optionally `contextualized_text`. **book_sections** stores section metadata and full section text for section-level resolution.

Result: **Dense (FAISS) + Sparse (BM25) + Section + Glossary + TOC** are populated. Transcripts live in both global and transcript-only indexes.

---

## Part 2: Query (Retrieval + Answer)

**Entry point:** Chat stream — `POST /api/chat/ask-stream` calls `answer_stream()` in `rag/advanced.py`.

### 2.1 Deciding whether to use RAG

- If **use_knowledge_base** is False → answer with LLM only (general stream), no retrieval.
- **Source options** (`transcript`, `document`, `general`): If both transcript and document are disabled → general only.
- **General conversation** (`_is_general_conversation()`): Greetings, thanks, small talk → skip RAG, answer with LLM only.
- Otherwise → run **retrieval** and then the RAG pipeline.

### 2.2 Retrieval: `retrieve_semantic_first()`

Goal: **One embedding of the query**, then parallel searches; merge with **dynamic dense/sparse weights**; choose **source type** (transcript vs document vs general vs insufficient).

#### Step 1: Query embedding (once)

- Query is embedded with **OllamaEmbeddings**; vector is L2-normalized. Same vector is reused for all downstream searches.

#### Step 2: Query classification (`rag/query_classifier.py`)

- **classify_query_type()**: Returns one of `citation` | `definition` | `procedural` | `conceptual`.
  - **citation**: Paragraph/section/code numbers, quoted phrases → prefer sparse (BM25).
  - **definition**: “What is X”, “define” → prefer dense; also triggers **glossary-first** for documents.
  - **procedural**: “How to”, “steps”, “requirements” → BM25-leaning.
  - **conceptual**: General question → strongly dense.
- **get_rrf_weights()**: Returns `(dense_weight, sparse_weight)` for **document** retrieval (e.g. definition → 0.7/0.3, citation → 0.3/0.7). Transcript retrieval uses fixed weights (e.g. 0.6/0.4).

#### Step 3: Glossary (definition + document only)

- If query type is **definition** and document search is on, **glossary index** is searched first. If there are enough glossary hits, return `("document", glossary_hits)` and skip main document search.

#### Step 4: Section restriction (BookRAG, document only)

- **SectionResolver** (`rag/book/section_resolver.py`): Parses the query for explicit refs (e.g. “paragraph 030201”, “Volume 5 Chapter 3”). Produces `explicit_section_ids`, volumes, chapters, comparison pairs.
- **_get_section_restricted_paths()**: Uses **section index** (and optionally **TOC index**) to resolve the query to a set of **allowed section paths**. So retrieval can be limited to “only these sections” instead of the whole corpus. If the user says “what does paragraph 030201 say?”, the system first finds which section path that code maps to, then searches only chunks in that path. **no_global_fallback**: when the user gave an explicit section ref, we may disable fallback to the rest of the doc if that section isn’t found.

#### Step 5: Parallel search

- **Transcript path** (if `source_options.transcript`):
  - Dense: `index.search_transcript_only(query, k, query_vector=qv)` (FAISS on transcript index).
  - Sparse: `index.transcript_sparse.search(query, k)` (BM25 on transcript chunks).
- **Document path** (if `source_options.document`):
  - If there are **allowed_section_paths**: section-restricted search (dense + sparse only within those sections; optional no_global_fallback).
  - Else: global document search (dense + sparse over all non-transcript chunks).

All of these can run in parallel (e.g. `asyncio.gather`).

#### Step 6: Merge dense + sparse (RRF)

- **Reciprocal Rank Fusion (RRF)** with **dynamic weights** from the query classifier:
  - Transcript: fixed weights (e.g. 0.6 dense, 0.4 sparse).
  - Document: `dense_w`, `sparse_w` from `get_rrf_weights()`.
- Result: one ranked list for **transcript** and one for **document**.

#### Step 7: Transcript-specific filters

- **Context window**: Optional filter by document `created_at` (24h, 48h, 1w, or “all”).
- **Time decay**: Optional score decay by age (`RAG_TIME_DECAY_HALFLIFE_DAYS`).
- **Tag boost**: Optional score boost when transcript tags overlap query terms.
- **Time range / location / “last N”**: Parsed from the question (e.g. “last 2 hours”, “in office”) and applied to transcript hits.

#### Step 8: Pick source type

- Compare best **transcript** score vs best **document** score to **RAG_RELEVANCE_THRESHOLD**.
- If transcript wins and above threshold → `("transcript", transcript_hits)`.
- If document wins and above threshold → `("document", document_hits)`.
- If both below threshold and **general** allowed → `("general", [])` (no RAG; answer from LLM only).
- If both below and general not allowed → `("insufficient", [])` (show “couldn’t find confident answer” style message).

So: **retrieve_semantic_first()** returns `(source_type, hits)` where `source_type` is one of `transcript` | `document` | `general` | `insufficient`.

### 2.3 Post-retrieval: `_run_rag_pipeline()`

Only when `source_type` is **transcript** or **document** (and we have hits). For **general** or **insufficient**, the stream path returns a direct or “missing info” message without this pipeline.

#### 2.3.1 Section resolution (document only)

- **SectionResolver.resolve(question)**: Extracts explicit section refs and resolves them to concrete `section_path`s (and comparison pairs if “0301 vs 0402”). Stored in `resolved` and reused for evidence, citation, and gating.

#### 2.3.2 Rerank (document only)

- **Cross-encoder reranker** (`rag/reranker.py`): Optional LLM-based rerank of the top-K hits. Keeps top `RAG_RERANK_FINAL_N` to improve precision.

#### 2.3.3 Graph expansion (document only)

- If top hit score is below **RAG_GRAPH_EXPANSION_CONFIDENCE_THRESHOLD**, **cross-reference graph** is used to add related sections (e.g. “see also 030205”). Added chunks are rescored so they don’t dominate.

#### 2.3.4 Sort, dedupe, limit sections, trim

- Sort by score; optional **dedupe by section** (e.g. max 2 chunks per section); **_limit_sections_in_context** (e.g. top 2 sections by score, with bonus for explicit section refs); **_trim_hits_to_context_budget** so total context size stays under `RAG_CONTEXT_MAX_CHARS`.

#### 2.3.5 Evidence extraction (document only)

- **extract_evidence_sentences()** (`rag/evidence_extractor.py`): From the reranked hits, extracts sentences that best support the query (keyword overlap, policy-style wording, section match). Produces a list of **EvidenceSentence** (sentence, score, section, keyword_hits, has_policy_word).
- **gate_evidence()** (`rag/evidence_gate.py`): Optional **EvidenceGate**: checks concept coverage (query terms in evidence), section match, rerank score, policy-word ratio. If **RAG_USE_EVIDENCE_GATE** is True and the gate fails, the pipeline returns an early exit (“Unable to find strong supporting evidence”) with optional citations, and does **not** call the LLM.

#### 2.3.6 Build context block

- **_build_rag_context()**:
  - For each hit: optionally attach **parent chunk** text (for child hits) up to `RAG_PARENT_CONTEXT_MAX_CHARS`.
  - Per hit: either use chunk text truncated to `RAG_VERBATIM_MAX_CHARS`, or (if **RAG_COMPRESS_CONTEXT**) use an LLM **compress()** step to keep only answer-critical sentences (with “[Partial]” / “[Conflicting]” labels). Chunks that contain key query terms can bypass compression (verbatim).
  - **Metadata** (doc_type, section_path, page) is prepended to each block for the LLM and for citation.
  - Blocks are formatted as `[1] ... [2] ...` and concatenated into **ctx_block**.
- For **document**, an **evidence-only** block can be prepended (strongest evidence sentences) so the model grounds on them first.
- **Comparison queries** (“0301 vs 0402”): Content for both sections is fetched and optionally summarized by LLM (**_extract_key_differences_async**), then a “[COMPARISON REFERENCE]” block is prepended.
- **Explicit section citation**: If the user asked for a specific section and it was resolved, a “[CITATION REFERENCE – Section X]” block with that section’s content can be prepended.

#### 2.3.7 TOC guardrail (document only)

- If the question is a “TOC/chapter list” style query but the built context doesn’t contain TOC-like content, an early exit message is returned (“couldn’t find table of contents in the excerpts”) to avoid hallucinated structure.

#### 2.3.8 Answer gating (document only)

- **gate_context()** (`rag/answer_gating.py`): Optional check that the context actually contains enough signal to answer (e.g. explicit section refs must appear in context). If not, return a fallback message and optional citations without calling the LLM.

#### 2.3.9 LLM messages and streaming

- **System prompt**: RAG system prompt (DoD FMR advisor, cite only from context, lead with answer, guardrail for off-topic). Persona and conversation summary can be included.
- **User message**: Built by **_build_user_content_with_summary()**: current question, optional response-format hint, then **document excerpts** (the ctx_block), then optional **conversation summary** (for follow-up questions).
- **History**: Last N turns of chat history are included.
- **Streaming**: **OpenAICompatChat.chat_stream()** streams tokens from the LLM (Ollama). The stream yields `("chunk", delta, None)`; at the end, **citations** are built from **enriched** hits and the final event is `("done", full_answer, citations)`.

#### 2.3.10 Citations

- **_build_citation()** for each **enriched** hit:
  - **Metadata validation** (_is_citation_metadata_valid): For BOOK, requires section_path, page_number, and doc identity. Invalid chunks are not exposed as citations.
  - **Citation dict**: filename, doc_id, chunk_id, snippet, score, doc_type; for BOOK also volume, chapter, section, section_path, page_number. Transcripts get a humanized name (e.g. “Transcript”) when filename is `transcript_*`.
- Only built when **RAG_EXPOSE_SOURCES** is True; otherwise citations can be used internally only (audit).

#### 2.3.11 Postprocess

- **_postprocess_answer_text()**: Optional cleanup of the model output (e.g. strip spurious citations, normalize section refs). Uses enriched hits, resolved sections, and doc_ids.

---

## Part 3: Data Flow Summary

| Stage | Where | What |
|-------|--------|------|
| **Upload** | `docs/upload`, `index.add_document()` | Parse → chunk → embed → FAISS + BM25 + section/glossary/TOC/transcript indexes + DB |
| **Transcript add** | `index.add_text()` | Same as add_document; stored in main + transcript index |
| **Query** | `answer_stream` | use_knowledge_base, source_options, general check |
| **Retrieve** | `retrieve_semantic_first` | Embed query → classify → glossary? → section paths? → parallel transcript/doc search → RRF → filters → pick source_type |
| **Pipeline** | `_run_rag_pipeline` | Resolve sections → rerank → graph expand → sort/dedupe/trim → evidence extract → evidence gate? → build context → TOC guard? → answer gate? → ctx_block |
| **LLM** | `chat.chat_stream()` | System + history + user (question + ctx_block + summary) → stream tokens |
| **Citations** | `_build_citation()` | From enriched hits; only if RAG_EXPOSE_SOURCES and valid metadata |
| **Response** | Chat API | NDJSON stream: chunk events then done(answer, citations) |

---

## Anti-hallucination measures

The pipeline reduces hallucination in several ways:

1. **System prompt rules** (`_rag_system_prompt`, `_strict_citation_system_prompt` in `advanced.py`):
   - "Answer ONLY from the provided document excerpts. Never invent facts, section numbers, or page references."
   - "Cite every factual claim inline … Only cite sections that appear in the context."
   - "Do NOT fabricate an answer from general knowledge." If the context lacks the answer, say so and suggest where to look.
   - **Guardrail**: Refuse off-topic questions with a fixed reply (DoD FMR / regulatory advisor only).

2. **Evidence gate** (optional, `RAG_USE_EVIDENCE_GATE`): Before calling the LLM, evidence sentences are extracted and scored (concept coverage, section match, policy-word ratio). If confidence is below threshold, the pipeline **does not call the LLM** and returns a fallback message (e.g. "Unable to find strong supporting evidence in the regulation") with optional top sections. This stops the model from answering when evidence is weak.

3. **Answer gate** (optional, `RAG_USE_ANSWER_GATING`): Verifies that the built context actually contains the requested section IDs and enough query terms. If the user asked for "paragraph 030201" but that section is not in the retrieved context, the pipeline returns a structured fallback ("Relevant section not confidently identified") instead of letting the LLM infer.

4. **TOC guardrail**: For "table of contents" style questions, if the context has no TOC-like content, an early exit message is returned so the model does not invent a fake structure.

5. **Strict citation mode** (`RAG_STRICT_CITATIONS`): For document answers, the system can use a non-streaming path that requires inline citations in every response. If the first attempt has no citations, it retries with a stronger instruction; if still no citations, it returns "Insufficient context to answer with citation" and does **not** show an uncited answer.

6. **Insufficient source type**: When retrieval scores are below `RAG_RELEVANCE_THRESHOLD` and the user has not allowed "general" fallback, `source_type` is set to `"insufficient"`. The API then returns a deterministic message (or `handle_missing_information` suggestion) and **never** calls the LLM with weak or empty context — so the model cannot hallucinate from thin air.

7. **Citation metadata validation** (`_build_citation`, `metadata_validation.py`): Only chunks with valid BOOK metadata (section_path, page_number, volume, chapter) are exposed as citations. Malformed or "Segment N" fallbacks are rejected so the UI does not show fake section references.

8. **Evidence-first context**: For document queries, the strongest evidence sentences are prepended to the context block so the LLM is nudged to ground on them first.

Together, these measures keep answers tied to retrieved context, refuse when evidence is weak or missing, and avoid inventing section numbers or off-topic content.

---

## Part 4: Key Files Reference

| Component | File(s) |
|-----------|---------|
| Ingestion entry | `api/routes/docs.py` (upload), `api/routes/transcribe.py` (transcript save) |
| Chunking | `rag/chunking/pipeline.py`, `chunkers.py`, `models.py`, `detect.py` |
| Embeddings | `rag/embeddings.py` (OllamaEmbeddings) |
| Index (FAISS, BM25, section, glossary, TOC) | `rag/index.py`, `rag/section_index.py`, `rag/glossary_index.py`, `rag/indexes/toc_index.py` |
| Sparse | `rag/sparse.py` (Bm25Index) |
| Query classification | `rag/query_classifier.py` |
| Section resolution | `rag/book/section_resolver.py`, `section_id.py` |
| Retrieval + source pick | `rag/advanced.py` (`retrieve_semantic_first`, `_get_section_restricted_paths`) |
| RAG pipeline | `rag/advanced.py` (`_run_rag_pipeline`, `_build_rag_context`, `_build_citation`) |
| Evidence | `rag/evidence_extractor.py`, `rag/evidence_gate.py` |
| Reranker | `rag/reranker.py` |
| Answer gating | `rag/answer_gating.py` |
| LLM | `rag/llm.py` (OpenAICompatChat) |
| Chat API | `api/routes/chat.py` (ask_stream → answer_stream) |

---

## Part 5: Configuration (env/settings)

Relevant `core/config.py` and env vars:

- **TOP_K**, **RAG_RELEVANCE_THRESHOLD**: retrieval size and minimum score to use RAG.
- **RAG_USE_QUERY_CLASSIFIER**, **RAG_DENSE_RRF_WEIGHT**, **RAG_SPARSE_RRF_WEIGHT**: query-type and RRF weights.
- **RAG_RERANK_TOP_K**, **RAG_RERANK_FINAL_N**: rerank size.
- **RAG_USE_EVIDENCE_GATE**, **RAG_USE_ANSWER_GATING**: gates that can skip the LLM.
- **RAG_CONTEXT_MAX_CHARS**, **RAG_VERBATIM_MAX_CHARS**, **RAG_PARENT_CONTEXT_MAX_CHARS**: context sizing.
- **RAG_COMPRESS_CONTEXT**, **RAG_VERBATIM_QUERY_TERMS**: compression and verbatim bypass.
- **RAG_EXPOSE_SOURCES**: whether to return citations to the client.
- **RAG_STRICT_CITATIONS**: use strict citation generation (non-streaming) for document answers.

This is the full RAG flow from document/transcript ingestion through to streamed answer and citations.
