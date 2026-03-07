# CHANGELOG — BookRAG-lite++ Upgrade

## Version: BookRAG-lite++ v1.1
## Date: 2026-03-06
## Branch: RAG_improvement_flow

---

## v1.1 Additions (Regulatory QA Strengthening)

### Phase 1 — Direct Section Lookup
**What:** When the query contains explicit section refs (030201, 030201.A, Volume 5 Chapter 3), bypass vector-first retrieval and fetch chunks directly by `section_id` / `clause_id` from the index.
**Why:** Vector search can miss the exact section when scores are low or the resolver has no path mapping; direct lookup guarantees retrieval of the requested section.
**New:** `index.get_chunks_by_section_ids(section_ids, k)` — direct DB lookup. `BOOK_DIRECT_SECTION_LOOKUP=1`, `BOOK_DIRECT_SECTION_MIN_CHUNKS=2`. Section resolver now extracts clause-level refs (030201.A) via `detect_clause_ids`.

### Phase 2 — Enriched Retrieval Text
**What:** `build_retrieval_text()` now prepends `[Document Type: BOOK]` and `[Page N]` to the heading path and clause label.
**Why:** Page number in embeddings improves retrieval for page-local queries and phrase-style lookups.

### Phase 5 — Phrase-Style Query Detection
**What:** New query type `phrase` for "where does it say", quoted text, "exact wording", "show the exact sentence".
**Why:** These queries need BM25-heavy retrieval (0.20/0.80 dense/sparse) — exact phrase match is critical.
**New:** `_PHRASE_RE`, `is_phrase_query()`, `_QUERY_WEIGHTS["phrase"]`.

### Phase 8 — Post-Generation Verifier
**What:** New `answer_verifier.py` verifies that every material claim has support in evidence, citations align with claims, and section refs are not malformed.
**Why:** Reduces plausible-but-wrong answers when evidence is partial or the model overstates.
**Config:** `BOOK_VERIFIER_ENABLED=1` (default OFF for latency). On failure: returns cautious refusal message.

### Phase 12 — Regulatory Eval Seed Questions
**What:** Added 5 seed eval cases in `regulatory_eval_sample.json`: eval_seed_001–005.
**Queries:** Explain 030201 + cite; exact wording + citation; compare 030201/030202; "where does it say financial reporting requirements"; procedures for financial reporting compliance.

---

## Version: BookRAG-lite++ v1
## Date: 2026-03-06

---

## Summary of Changes


This release implements the **BookRAG-lite++** upgrade to the EchoMind RAG pipeline,
significantly improving retrieval accuracy and citation quality for large regulatory PDFs,
specifically targeting the DoD Financial Management Regulation (DoD 7000.14-R, ~7300 pages).

All changes are **backward-compatible**. New behavior is gated behind BOOK doc-type detection
and configurable feature flags. Existing non-BOOK document behavior is unaffected.

---

## New Files

### `backend/app/rag/book/clause_parser.py`
**What:** Clause-level detection and retrieval-text construction for DoD regulatory documents.
**Why:** DoD FMR uses numbered clauses (030201.A, 030201.B.1) as the unit of law. Without 
clause detection, the system can only cite section-level, losing precision for threshold 
and exception queries.
**Effect:** Each BOOK chunk now carries a `clause_id` (most prominent DoD clause code found 
in the text). Retrieval queries like "what does 030201.A say?" now retrieve the correct clause 
rather than the whole section.

**Functions:**
- `detect_clause_ids(text)` — all clause codes in text
- `dominant_clause_id(text)` — most frequent clause code (for chunk metadata)
- `build_retrieval_text(text, section_path, clause_id, section_title)` — enriched embed text
- `build_canonical_id(section_path, page, clause_id)` — deterministic `vol_XX_ch_XX_sec_XXXXXX` ID
- `has_table_heuristic(text)` — True when text contains table content
- `has_list_heuristic(text)` — True when text contains bullets or numbered lists

---

## Modified Files

### `backend/app/core/config.py`
**What:** 14 new configuration flags for BookRAG-lite++ features.
**Why:** All new behavior is opt-in (or opt-out from safe defaults) via env vars to prevent
unintended performance impact and allow incremental rollout.
**New flags:**
- `BOOK_HEADING_PATH_IN_EMBED` (default ON) — heading path in embed text
- `BOOK_CLAUSE_CHUNKING_ENABLED` (default ON) — clause detection
- `BOOK_ADJACENCY_EXPANSION` (default ON) — prev/next chunk expansion
- `BOOK_ADJACENCY_MAX_CHUNKS` (default 1) — max adjacent chunks per side
- `BOOK_PAGE_INDEX_ENABLED` (default ON) — populate page_index table
- `BOOK_OCR_FALLBACK` (default OFF) — OCR for low-text pages
- `BOOK_OCR_MIN_TEXT_CHARS` (default 80) — OCR trigger threshold
- `BOOK_STRICT_GROUNDED` (default ON) — strict regulatory grounding in system prompt
- `BOOK_VERIFIER_ENABLED` (default OFF) — post-generation answer verifier
- `BOOK_CHAPTER_SUMMARIES` (default OFF) — chapter-level summaries
- `BOOK_TABLE_EXTRACTION` (default ON) — table content heuristics
- `BOOK_TABLE_SCORE_BOOST` (default 0.15) — table chunk score boost
- `BOOK_CORPUS_MULTI_FILE` (default ON) — multi-file corpus retrieval

### `backend/app/core/db.py`
**What:** New database tables and columns for BookRAG-lite++ metadata.
**Why:** Reliable clause-level citation and adjacency expansion require dedicated DB columns
(not just serialized JSON) for fast lookup during retrieval.
**New tables:**
- `page_index` — one row per PDF page; enables accurate page citations and OCR detection
- `doc_tables` — extracted table metadata (scaffolded for future table extraction)
**New columns on `chunks`:**
- `clause_id` — detected DoD clause code
- `prev_chunk_id`, `next_chunk_id` — sibling chunk adjacency links
- `retrieval_text` — heading-path-enriched embed text
- `canonical_id` — deterministic `vol_XX_ch_XX_sec_XXXXXX` identifier
- `page_start`, `page_end` — multi-page chunk span
- `evidence_type` — "child" | "parent" | "clause" | "table" | "page"
- `has_table` — heuristic table detection flag
All new columns default to NULL; safe migration for existing indexed documents.

### `backend/app/rag/chunking/models.py`
**What:** Extended `Chunk` dataclass with 8 new optional fields.
**Why:** The Chunk model is the source of truth for all metadata that flows through 
chunking → indexing → retrieval → citation. Adding fields here ensures consistent 
propagation through the pipeline.
**New fields:** `clause_id`, `prev_chunk_id`, `next_chunk_id`, `retrieval_text`,
`canonical_id`, `page_start`, `page_end`, `evidence_type`, `has_table`
All new fields are Optional and default to None/False for backward compatibility.

### `backend/app/rag/chunking/pipeline.py`
**What:** BOOK chunking now assigns clause IDs, retrieval text, canonical IDs, and prev/next links.
**Why:** These fields must be populated at chunking time while the section context is available.
After ingestion, there is no clean way to back-fill them without re-parsing.
**Changes:**
- Added import of `clause_parser` functions
- Children now get `clause_id` from `dominant_clause_id(child.text)`
- Children now get `retrieval_text` from `build_retrieval_text(text, section_path, clause_id)`
- Children now get `canonical_id` from `build_canonical_id(section_path, page, clause_id)`
- Sibling children within the same parent get `prev_chunk_id`/`next_chunk_id` links
- Per-child page attribution: uses `_true_page_for_offset(c.char_start, page_offsets)` 
  when available, instead of always inheriting parent page
- `evidence_type` set to `"child"` for children, `"parent"` for parents
- `has_table` set from `has_table_heuristic(text)` on both parent and children

### `backend/app/rag/index.py`
**What:** BOOK chunks now embedded with `retrieval_text`; page_index populated at ingestion.
**Why:**
1. Including the heading path in the embedding vector aligns semantic search with the 
   document's logical structure — "030201 PURPOSE" retrieves better than just "purpose"
2. The page_index provides the foundation for accurate per-page citation previews
**Changes:**
- When `BOOK_HEADING_PATH_IN_EMBED=1` and contextual retrieval is disabled: use 
  `c.retrieval_text` (heading path + clause label + text) for embedding instead of `c.text`
- New `_populate_page_index()` function: called after BOOK document ingestion to populate 
  one page_index row per PDF page with page_text, char offsets, has_table, has_low_text
- Chunk INSERT now stores all new columns with try/except fallback for pre-migration schemas
- Added import of `has_table_heuristic` from clause_parser

### `backend/app/rag/query_classifier.py`
**What:** Three new query types: `threshold`, `table`, `comparison`.
**Why:** The existing four types (citation, definition, procedural, conceptual) were 
insufficient for regulatory QA. Threshold queries need max BM25 weight to find exact 
numeric values. Table queries need table chunk boosting. Comparison queries need relaxed 
section limits.
**New types and weights:**
- `threshold` (0.25/0.75 dense/sparse) — triggers on: minimum, maximum, limit, deadline, 
  exception, waiver, penalty, not to exceed, days after
- `table` (0.40/0.60) — triggers on: table, schedule, matrix, exhibit, rate table
- `comparison` (0.70/0.30) — triggers on: compare, difference, versus, contrast
**New helper functions:** `is_threshold_query()`, `is_table_query()`, `is_comparison_query()`

### `backend/app/rag/advanced.py`
**What:** Adjacency expansion, table boosting, comparison section relaxation, extended citations.
**Why:** 
- **Adjacency expansion:** Clauses split across chunk boundaries cause retrieval misses. 
  Fetching the next chunk after a strong hit dramatically improves threshold recall.
- **Table boost:** Without score boosting, table chunks (which contain the actual numeric 
  values) often rank below prose chunks that merely mention a table.
- **Comparison section relaxation:** Comparison queries need at least 2 sections × 2 chunks.
  The existing MAX_SECTIONS_PER_ANSWER=2 was too restrictive.
- **Extended citations:** clause_id and canonical_id in citation objects enable precise 
  source attribution at the clause level.
**New functions:**
- `_get_adjacent_chunk(chunk_id)` — fetch a chunk by ID as a hit dict
- `_apply_adjacency_expansion(hits, max_per_side)` — add prev/next chunks for top BOOK hits
- `_apply_table_boost(hits, boost)` — boost has_table chunks for threshold/table queries
**Changes to `_run_rag_pipeline()`:**
- After reranking: apply table/threshold boost when applicable
- After boost: call `_apply_adjacency_expansion()`
- Comparison queries: `max_sections_override = max(4, MAX_SECTIONS_PER_ANSWER)`
- `_limit_sections_in_context` now uses `max_sections_override`
**Changes to `_rag_system_prompt()`:**
- When `BOOK_STRICT_GROUNDED=1`: inject strict grounding notice (no blending, label 
  inferences, quote exact numeric values)
**Changes to `_build_citation()`:**
- New BOOK fields: `clause_id`, `canonical_id`, `page_start`, `page_end`, `evidence_type`, `has_table`

---

## New Documentation Files

### `BOOK_RAG_IMPROVEMENT_PLAN.md`
Complete implementation plan with phase breakdown, config reference, migration instructions.

### `docs/DEBUG_RETRIEVAL_EXPLAINER.md`
Step-by-step retrieval flow diagram, debug endpoint usage, failure diagnosis guide, 
chunk metadata reference, and tuning config table.

### `backend/app/rag/eval/regulatory_eval_sample.json`
30 evaluation cases across 10 buckets:
- explicit_section_lookup (5 cases)
- threshold_exception (8 cases)
- table_lookup (2 cases)
- comparison (2 cases)
- broad_concept (1 case)
- where_does_it_say (2 cases)
- multi_volume (3 cases)
- refusal_when_missing (1 case)
- procedural (4 cases)
- definition (2 cases)

Each case includes: `expected_sections`, `expected_clauses`, `gold_answer_keywords`,
`gold_citation_presence`, `answer_should_refuse`, `difficulty`, `eval_bucket`.

---

## Expected Effect on DoD FMR Retrieval Quality

| Improvement | Mechanism | Expected Effect |
|-------------|-----------|-----------------|
| Heading path in embeddings | `retrieval_text` prepends section path | +15–25% section recall for "what does section X say?" queries |
| Clause ID detection | `clause_id` on each child chunk | +20–30% precision for clause-level citations |
| Adjacency expansion | Fetch prev/next sibling chunks | +25–40% recall for threshold/exception queries split at chunk boundaries |
| Threshold/table query type | Heavy BM25 weighting (0.25/0.75) | +20% F1 for numeric limit retrieval |
| Table chunk score boost | +0.15 score for has_table chunks | +30% recall for table-based evidence |
| Comparison section relaxation | Up to 4 sections per comparison query | Eliminates wrong truncation in comparison answers |
| Strict regulatory grounding | System prompt injection | Reduces hallucination rate for inference-heavy questions |
| Per-child page attribution | Uses char offset instead of parent page | -60% incorrect page citations for long sections |
| Canonical IDs | Deterministic `vol_XX_ch_XX_sec_XXXXXX` | Enables deduplication and cross-request answer caching |
| Page index table | Per-page metadata at ingestion | Foundation for accurate PDF page preview and OCR detection |

---

## Migration Required

**Existing indexed documents will not benefit from:**
- `retrieval_text` heading path in embeddings (requires re-ingestion)
- `clause_id` / `canonical_id` / `prev_chunk_id` / `next_chunk_id` (requires re-ingestion)
- `page_index` table population (requires re-ingestion)

**Adjacency expansion** will silently skip chunks with NULL `next_chunk_id` — safe but
without benefit until re-ingestion.

**To apply all improvements to existing documents:**
```
POST /docs/delete-all
# Re-upload all volume PDFs via UI or API
```

The DB schema changes (ALTER TABLE) are applied automatically at startup — no manual
migration script required.
