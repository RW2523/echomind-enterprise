# RAG Generic Improvements – Implementation Summary

## 1. Summary of changes

- **Intent detection + retrieval profiles**  
  - Added `classify_query_intent(question)` (deterministic heuristics) returning `structure` | `factual` | `procedural` | `exploratory`.  
  - Per-intent retrieval profiles: `k_per_query`, `top_k`, `dense_weight`, `sparse_weight`, `rerank_*`, `relevance_threshold`.  
  - `retrieve()` now returns `(hits, intent, expanded_queries)` and uses the profile when `RAG_USE_INTENT_PROFILES` is on.

- **Retrieval-failure safe fallback**  
  - When retrieval is empty or below threshold, the pipeline **never** falls back to general knowledge silently.  
  - Returns a refusal message: *"I couldn't find this in the retrieved excerpts. Try rephrasing or ask for a different detail."*  
  - For **STRUCTURE** intent, tries deterministic structure fallback (headings from `doc_headings`) before refusing.  
  - Explicit **answer_mode**: `RAG_GROUNDED` | `RAG_INSUFFICIENT_EVIDENCE` (and `general` / `general_fallback` for non-RAG).  
  - Logged for debugging.

- **Structure index (deterministic)**  
  - New table **doc_headings(doc_id, idx, heading_text, chunk_id, chunk_index)**.  
  - Headings extracted at ingestion via regex (CHAPTER/PART/SECTION, ALL CAPS lines, numbered 1. / I. / A), stored per document.  
  - On STRUCTURE questions: prefer verbatim list from `doc_headings`; if none and no TOC signals in context, respond *"I can't find headings or a table of contents in the extracted text."*  
  - No inferred or hallucinated lists.

- **Evidence gating**  
  - RAG system prompt strengthened: *"Only use provided context. Do not add names/examples/facts not present in context. If context is insufficient, explicitly say so."*  
  - Post-check: for STRUCTURE, list items in the answer must appear verbatim in context or structure index; for number/date questions, numbers must appear in context.  
  - If check fails, response is replaced with a safe refusal or the no-headings message.

- **PDF/text normalization**  
  - Existing `normalize_extracted_text()` kept; added optional **remove_repeated_headers_footers()** (lines repeating ≥3 times, configurable).  
  - Pipeline uses `RAG_NORMALIZE_STRIP_HEADERS_FOOTERS` to optionally strip before chunking.

- **Chunking/compression for tables and TOC**  
  - **Skip compression** when a chunk is table-like (many columns, repeated spaces, numbers) or heading/TOC-like (Contents, CHAPTER N, section list).  
  - **Adjacent chunk expansion**: when a hit contains "contents", "chapter", "section", "table of contents", also fetch `chunk_index ± 1` for the same document.

- **Observability**  
  - Logs per query: **intent**, **expanded_queries**, **rrf_candidates**, **top_k**; and per answer: **answer_mode**, **chunk_ids**, **doc_ids**, **structure_fallback_used**, **evidence_postcheck**.

---

## 2. Code edits by file

| File | Change |
|------|--------|
| **backend/app/rag/intent.py** | **New.** `classify_query_intent()`, `get_retrieval_profile()`, intent constants. |
| **backend/app/rag/structure.py** | **New.** `_extract_headings_from_text()`, `extract_headings_from_chunks()`, `store_doc_headings()`, `get_doc_headings_for_docs()`, `get_headings_verbatim_list()`, `backfill_doc_headings()`. |
| **backend/app/core/db.py** | **Add** `CREATE TABLE IF NOT EXISTS doc_headings(...)`. |
| **backend/app/core/config.py** | **Add** `RAG_USE_INTENT_PROFILES`, `RAG_REFUSAL_ON_NO_EVIDENCE`, `RAG_STRUCTURE_FALLBACK`, `RAG_NORMALIZE_STRIP_HEADERS_FOOTERS`, `RAG_SKIP_COMPRESS_TABLES_HEADINGS`, `RAG_ADJACENT_CHUNK_EXPANSION`, `RAG_EVIDENCE_POSTCHECK`. |
| **backend/app/rag/normalize.py** | **Add** `remove_repeated_headers_footers()`; **extend** `normalize_extracted_text(..., remove_headers_footers=False)`. |
| **backend/app/rag/chunking/pipeline.py** | **Use** `normalize_extracted_text(..., remove_headers_footers=settings.RAG_NORMALIZE_STRIP_HEADERS_FOOTERS)`. |
| **backend/app/rag/chunking/models.py** | **Ensure** `to_source_dict()` includes `doc_id` and `chunk_index` (for adjacent expansion and structure). |
| **backend/app/rag/index.py** | **After** inserting chunks, call `extract_headings_from_chunks()` and `store_doc_headings()`. **On** `delete_document`, **delete** from `doc_headings` first. |
| **backend/app/rag/advanced.py** | **Import** intent + structure; **add** `RAG_GROUNDED` / `RAG_INSUFFICIENT_EVIDENCE`, `REFUSAL_MESSAGE`, `NO_HEADINGS_MESSAGE`; **add** `_get_doc_ids_in_context_window()`, `_get_chunk_by_doc_index()`, `_is_table_like()`, `_is_heading_toc_chunk()`, `_expand_hits_with_adjacent()`, `_answer_evidence_check()`, `_format_headings_response()`. **Change** `retrieve()` to return `(hits, intent, expanded_queries)` and use intent profiles. **Change** `_build_rag_context()` to expand adjacent chunks and skip compression for table/heading chunks. **Change** `answer()` / `answer_stream()`: refusal on no evidence, structure fallback, structure index first for STRUCTURE, evidence post-check, **answer_mode** in return and logs. |
| **backend/scripts/backfill_doc_headings.py** | **New.** Calls `init_db()` and `backfill_doc_headings()` for existing DBs. |

---

## 3. SQL migration and backfill

- **Migration**: Table is created in `app.core.db.init_db()`:

```sql
CREATE TABLE IF NOT EXISTS doc_headings(
  doc_id TEXT,
  idx INTEGER,
  heading_text TEXT,
  chunk_id TEXT,
  chunk_index INTEGER,
  PRIMARY KEY (doc_id, idx)
);
```

- **Backfill** for existing documents (no re-upload):

```bash
cd backend
PYTHONPATH=. python -m scripts.backfill_doc_headings
```

Or from repo root:

```bash
cd backend && PYTHONPATH=. python scripts/backfill_doc_headings.py
```

---

## 4. Config defaults (intent profiles and feature flags)

- **Intent profiles** (in code, `intent.get_retrieval_profile()`):

| Intent    | k_per_query | top_k | dense_weight | sparse_weight | relevance_threshold |
|----------|-------------|-------|--------------|---------------|----------------------|
| structure | 24          | 12    | 0.45         | 0.55          | 0.35                 |
| factual  | 12          | 8     | 0.55         | 0.45          | 0.45                 |
| procedural | 14        | 8     | 0.6          | 0.4           | 0.45                 |
| exploratory | base_k     | base_k| 0.6          | 0.4           | 0.45                 |

- **Feature flags** (env / `config.py`):

| Setting | Default | Description |
|---------|---------|-------------|
| RAG_USE_INTENT_PROFILES | 1 | Use intent-based retrieval profiles. |
| RAG_REFUSAL_ON_NO_EVIDENCE | 1 | Return refusal instead of general answer when retrieval fails. |
| RAG_STRUCTURE_FALLBACK | 1 | For STRUCTURE, try doc_headings before refusing. |
| RAG_NORMALIZE_STRIP_HEADERS_FOOTERS | 0 | Strip repeated header/footer lines during normalize. |
| RAG_SKIP_COMPRESS_TABLES_HEADINGS | 1 | Don’t compress table-like or TOC/heading chunks. |
| RAG_ADJACENT_CHUNK_EXPANSION | 1 | Add chunk_index±1 for hits that look like TOC. |
| RAG_EVIDENCE_POSTCHECK | 1 | Post-check answer for ungrounded list items/numbers. |

---

## 5. Test plan (generic queries)

Use any uploaded PDF/document set; no reference to a specific book.

| Query | Expected behavior |
|-------|-------------------|
| *"List the sections/chapters"* | If headings exist in structure index or in retrieved context: verbatim list only. If none: *"I can't find headings or a table of contents in the extracted text."* Never an invented list. |
| *"How many parts are there?"* | If count is in context: answer with that number. If not: refusal or *"context does not contain enough information"*. No guessed number. |
| *"What does the document say about &lt;term&gt;?"* | Answer only from retrieved context; no extra names/examples. If nothing found: refusal. |
| *"Give the exact date/number mentioned for &lt;topic&gt;"* | Only if present in context; otherwise refusal. Evidence post-check can replace ungrounded numbers. |
| *"Summarize the introduction"* | Exploratory; answer from context only. |

**Sanity checks**

- Ask *"List the sections"* on a doc with no TOC → no-headings or refusal.  
- Ask *"List the sections"* on a doc with a real TOC → only verbatim headings from index or context.  
- Empty retrieval or very low scores → refusal (or structure list if STRUCTURE and headings exist).  
- Logs show `answer_mode`, `intent`, and `structure_fallback_used` where applicable.

---

## 6. Guarantee: no invented structure lists

- Structure lists are **only** from: (1) `doc_headings` (extracted at ingest), or (2) retrieved context that contains TOC-like signals.  
- If neither is available, the answer is the fixed no-headings message or the generic refusal.  
- The model is not asked to infer or invent a table of contents or chapter list; the evidence post-check can replace list-style answers that contain items not present in context or structure index.
