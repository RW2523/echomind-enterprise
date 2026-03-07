# BookRAG-lite++ Improvement Plan
## EchoMind Regulatory QA — DoD FMR 7000.14-R

---

## Overview

This document describes the design rationale, implementation status, and future roadmap for the
**BookRAG-lite++** upgrade to the EchoMind RAG pipeline. The goal is to make the system
significantly more accurate for:

- Exact section and clause lookups in the DoD Financial Management Regulation (~7300 pages)
- Threshold / exception / deadline queries ("maximum 30 days", "not to exceed 5%")
- Table-based evidence retrieval
- Comparison queries across sections or volumes
- "Where does it say X?" exact provenance questions
- Multi-file corpus retrieval across volume-split PDFs

The upgrade preserves all existing behavior for general enterprise documents, FAQ docs,
transcripts, and normal chat. New functionality is **gated by BOOK doc-type detection and
config flags** so it only activates for regulatory documents.

---

## Phase 1 — Explicit Section Lookup (Direct Lookup)

### 1.0 Direct Section Lookup (Config: `BOOK_DIRECT_SECTION_LOOKUP`)
**Status: ✅ Implemented**

- When query contains explicit refs (030201, 030201.A, Volume 5 Chapter 3), bypass vector-first retrieval
- `index.get_chunks_by_section_ids(section_ids)` fetches chunks by section_id/clause_id from DB
- If >= `BOOK_DIRECT_SECTION_MIN_CHUNKS` (default 2) found → use as primary; else fall back to hybrid
- Section resolver now extracts clause-level refs (030201.A) via `detect_clause_ids` from clause_parser
- Debug info: `explicit_section_ids`, `resolution_success` in debug endpoint

---

## Phase 1 — Parsing and Structural Fidelity

### 1.1 OCR Fallback (Config: `BOOK_OCR_FALLBACK`)
**Status: Scaffolded (disabled by default)**

- Detect pages with fewer than `BOOK_OCR_MIN_TEXT_CHARS` (default 80) characters
- Run pytesseract OCR on those pages only — not the full PDF
- Store `extraction_source: "ocr"` in the page_index row
- Required dependency: `pytesseract`, `Pillow`, `pdf2image`
- Enable only when corpus includes scanned pages

### 1.2 Page Index (Config: `BOOK_PAGE_INDEX_ENABLED`)
**Status: ✅ Implemented**

New `page_index` table populated during ingestion:
```
doc_id, page_number_pdf, page_number_logical, page_text (first 4000 chars),
page_char_start, page_char_end, section_path, has_table, has_low_text, extraction_source
```

- Used for per-page citation previews
- `has_table` and `has_low_text` flags enable quality diagnostics
- Indexed by `(doc_id, page_number_pdf)` for fast lookup

### 1.3 Clause-Level Structure (Config: `BOOK_CLAUSE_CHUNKING_ENABLED`)
**Status: ✅ Implemented**

- `clause_parser.py` detects DoD-style codes: `030201`, `030201.A`, `030201.A.1`
- `dominant_clause_id()` returns the most prominent clause in a chunk
- `clause_id` stored in Chunk model, source_json, and chunks DB column
- Enables clause-level citation (`clause_id: "030201.A"`) in citation objects

### 1.4 Table Heuristics (Config: `BOOK_TABLE_EXTRACTION`)
**Status: ✅ Implemented**

- `has_table_heuristic()` detects tab-separated rows, pipe chars, table headers
- `has_table` stored in Chunk model and chunks DB column
- Used for table score boosting during threshold/table queries

### 1.5 Canonical IDs
**Status: ✅ Implemented**

- `build_canonical_id()` generates: `vol_05_ch_03_sec_030201_page_0142_chunk_02`
- Stored in `canonical_id` field on Chunk and citation object
- Deterministic across re-ingestion of the same document

---

## Phase 2 — Chunking Upgrade

### 2.1 Retrieval Text with Heading Path (Config: `BOOK_HEADING_PATH_IN_EMBED`)
**Status: ✅ Implemented**

For BOOK chunks when `RAG_USE_CONTEXTUAL_RETRIEVAL=0`:
```
[Volume 5 > Chapter 3 > Section 030201 PURPOSE]
[Clause 030201.A]
<raw chunk text>
```

- `retrieval_text` stored in Chunk and chunks DB column
- Used for both dense (FAISS) and sparse (BM25) embedding
- When `RAG_USE_CONTEXTUAL_RETRIEVAL=1`, the LLM-generated context header already 
  includes heading path — no redundancy

### 2.2 Prev/Next Chunk Links
**Status: ✅ Implemented**

- `prev_chunk_id` / `next_chunk_id` assigned to sibling children within the same section
- Stored in Chunk model, source_json, and chunks DB columns
- Used by adjacency expansion in `advanced.py`

### 2.3 Parent/Child Evidence Types
**Status: ✅ Implemented**

- `evidence_type`: `"parent"` | `"child"` | `"clause"` | `"table"` | `"page"` | `"section_summary"`
- Stored in Chunk and citation objects
- Enables frontend to show evidence type in citation modal

### 2.4 Chapter/Volume Summaries (Config: `BOOK_CHAPTER_SUMMARIES`)
**Status: Scaffolded (disabled by default)**

- When enabled, generate LLM summaries at chapter level during ingestion
- Used as routing nodes for "what is in Chapter 3?" queries
- Builds on existing `section_summaries` infrastructure

---

## Phase 3 — Retrieval Upgrade

### 3.1 New Query Types: Threshold and Table
**Status: ✅ Implemented**

Added to `query_classifier.py`:

| Type | Trigger Patterns | Dense/Sparse Weights |
|------|-----------------|----------------------|
| `threshold` | minimum, maximum, limit, deadline, exception, waiver, penalty | 0.25 / 0.75 (very BM25-heavy) |
| `table` | table, schedule, matrix, exhibit, rate table | 0.40 / 0.60 |
| `comparison` | compare, difference, versus, distinguish | 0.70 / 0.30 |

### 3.2 Adjacency Expansion (Config: `BOOK_ADJACENCY_EXPANSION`)
**Status: ✅ Implemented**

After reranking top BOOK hits:
1. Fetch `prev_chunk_id` and `next_chunk_id` for each top-8 hit
2. Load adjacent chunks from DB
3. Assign score = parent_score × 0.75
4. Add to hit list, re-sort by score
5. Prevents missed answers when the relevant clause is split at a chunk boundary

Max adjacent chunks per side: `BOOK_ADJACENCY_MAX_CHUNKS` (default 1).

### 3.3 Table/Threshold Score Boost (Config: `BOOK_TABLE_SCORE_BOOST`)
**Status: ✅ Implemented**

When query type is `threshold` or `table`:
- Chunks with `has_table=True` get `score += BOOK_TABLE_SCORE_BOOST` (default 0.15)
- Applied after reranking, before graph expansion
- Improves recall for schedule/matrix/rate questions

### 3.4 Comparison Query Section Limit Relaxation
**Status: ✅ Implemented**

When `is_comparison_query()` is True:
- `MAX_SECTIONS_PER_ANSWER` raised to at least 4 (normally 2)
- `RAG_MAX_CHUNKS_PER_SECTION` raised to at least 4
- Allows comparing two sections with 2 chunks each

### 3.5 Strict Regulatory Grounding (Config: `BOOK_STRICT_GROUNDED`)
**Status: ✅ Implemented**

When enabled (default True for BOOK docs):
- System prompt injected with strict grounding notice:
  - "Answer ONLY from evidence in context blocks"
  - "Label inferences as [Inferred from context]"
  - "For numeric thresholds: quote the exact value"

---

## Phase 4 — Citation and Page Fidelity

### 4.1 Per-Chunk Page Attribution
**Status: ✅ Improved**

- Child chunks now use per-chunk `char_start` + `page_offsets` for accurate page attribution
- Falls back to parent page only when `page_offsets` is not available
- `page_start` and `page_end` fields support multi-page chunk spans

### 4.2 Citation Object Extended
**Status: ✅ Implemented**

New fields added to citation objects:
- `clause_id`: e.g., `"030201.A"`
- `canonical_id`: e.g., `"vol_05_ch_03_sec_030201_page_0142_chunk_02"`
- `page_start` / `page_end`: for multi-page chunks
- `evidence_type`: `"child"` | `"parent"` | `"table"` | etc.
- `has_table`: True when chunk contains table content

### 4.3 Page Index for Preview
**Status: ✅ Scaffolded**

- `page_index` table allows looking up the exact PDF page for a given char offset
- `GET /docs/{doc_id}/file#page=N` works when `page_number` is accurate

---

## Phase 5 — Answer Generation and Verification

### 5.1 Strict Regulatory System Prompt
**Status: ✅ Implemented**

See Phase 3.5 above. The `_rag_system_prompt()` now conditionally injects a strict grounding
notice for BOOK/regulatory documents.

### 5.2 Post-Generation Verifier (Config: `BOOK_VERIFIER_ENABLED`)
**Status: Planned (disabled by default)**

Design:
1. After answer generation, extract all factual claims from the answer
2. For each claim, check it has supporting text in the evidence context
3. If any claim has no support → flag as "overreach" and either:
   - Replace with a cautious fallback sentence
   - Return a verification warning in the citation metadata
4. Implementation: new `backend/app/rag/book/verifier.py`
5. Adds ~1 LLM call; only for BOOK docs with `BOOK_VERIFIER_ENABLED=1`

---

## Phase 6 — Evaluation Framework

### 6.1 Eval Sample Cases
**Status: ✅ Implemented**

See `backend/app/rag/eval/regulatory_eval_sample.json` (30 cases across 10 buckets).

### 6.2 Evaluation Metrics
Implemented in existing `backend/app/rag/eval/book_eval.py`:
- `retrieval_section_recall_at_k`
- `citation_correctness`
- `answer_keyword_coverage`
- `refusal_correctness`

**Remaining to implement:**
- `clause_recall_at_k` — check `expected_clauses` against retrieved `clause_id` values
- `page_correctness` — validate page_number matches expected_pages
- `adjacency_expansion_recall` — check whether adjacency chunks were needed and added
- `latency_breakdown` — rerank_ms, adjacency_ms, graph_ms, context_build_ms

### 6.3 Root-Cause Diagnostics
See taxonomy in `regulatory_eval_sample.json`:
`parse_failure` | `hierarchy_failure` | `table_failure` | `retrieval_miss` |
`reranker_miss` | `adjacency_miss` | `citation_mapping_failure` | `generation_overreach`

---

## Configuration Reference

All new flags with defaults (all can be overridden via environment variables):

| Config Key | Default | Description |
|------------|---------|-------------|
| `BOOK_HEADING_PATH_IN_EMBED` | `1` | Prepend heading path to BOOK chunk embed text |
| `BOOK_CLAUSE_CHUNKING_ENABLED` | `1` | Detect and store clause IDs |
| `BOOK_ADJACENCY_EXPANSION` | `1` | Fetch prev/next sibling chunks after reranking |
| `BOOK_ADJACENCY_MAX_CHUNKS` | `1` | Max adjacent chunks per side |
| `BOOK_PAGE_INDEX_ENABLED` | `1` | Populate page_index table |
| `BOOK_OCR_FALLBACK` | `0` | OCR for low-text pages (requires pytesseract) |
| `BOOK_OCR_MIN_TEXT_CHARS` | `80` | Min chars/page to skip OCR |
| `BOOK_STRICT_GROUNDED` | `1` | Inject strict grounding notice in LLM prompt |
| `BOOK_VERIFIER_ENABLED` | `0` | Post-generation answer verifier |
| `BOOK_CHAPTER_SUMMARIES` | `0` | Generate chapter-level summaries |
| `BOOK_TABLE_EXTRACTION` | `1` | Detect table heuristics |
| `BOOK_TABLE_SCORE_BOOST` | `0.15` | Score boost for table chunks on threshold/table queries |
| `BOOK_CORPUS_MULTI_FILE` | `1` | Allow retrieval across multi-file DoD FMR volumes |

---

## Migration / Index Rebuild

### After this upgrade, existing indexed documents need to be re-ingested for:
1. `retrieval_text` (heading path in embed) to take effect on existing chunks
2. `clause_id`, `canonical_id`, `prev_chunk_id`, `next_chunk_id` to be populated
3. `page_index` table to be populated for existing docs
4. New BM25/FAISS indexes to use enriched retrieval_text

### To rebuild:
```bash
# 1. Via API: delete all and re-upload documents
POST /docs/delete-all
# Then re-upload all volume PDFs

# 2. Via Docker: mount volumes and reset data directory
docker compose down
rm -rf data/  # clears FAISS, BM25, SQLite
docker compose up -d
# Re-upload all documents via UI or API
```

### New DB columns (added automatically at startup via ALTER TABLE):
- `chunks.clause_id`
- `chunks.prev_chunk_id`
- `chunks.next_chunk_id`
- `chunks.retrieval_text`
- `chunks.canonical_id`
- `chunks.page_start`
- `chunks.page_end`
- `chunks.evidence_type`
- `chunks.has_table`
- New table: `page_index`
- New table: `doc_tables`

These columns are NULL for existing chunks — no breaking change.
