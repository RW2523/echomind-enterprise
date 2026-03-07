# Debug Retrieval Explainer
## EchoMind BookRAG-lite++ Pipeline

This document describes the complete retrieval and answer generation pipeline
as implemented after the BookRAG-lite++ upgrade. Use it for diagnosing retrieval
failures, understanding citation accuracy, and verifying correct behavior.

---

## End-to-End Flow: Question → Answer

```
User question
     │
     ▼
[Step 1] Query Classification
     │  classify_query_type() → citation | threshold | table | comparison |
     │                           definition | procedural | conceptual
     │  classify_query_types() → all applicable types (multi-label)
     │  get_rrf_weights() → (dense_weight, sparse_weight) for hybrid RRF
     ▼
[Step 2] Query Expansion
     │  Deterministic variants: typo fixes, quoted phrases, TOC terms
     │  Optional LLM rewrite (RAG_QUERY_REWRITE=1, disabled by default)
     ▼
[Step 3] Glossary Priority (definition queries)
     │  If query type = "definition": search glossary FAISS index first
     │  If high-scoring glossary hit found → use as primary evidence
     ▼
[Step 4] Section Routing (BOOK documents)
     │  RAG_USE_SECTION_RETRIEVAL=1:
     │  - Search section_index (section-level FAISS) → top RAG_SECTION_TOP_K sections
     │  - Restrict child chunk search to matching sections
     │  - Falls back to global search if no sections score above RAG_SECTION_SCORE_THRESHOLD
     ▼
[Step 4b] Direct Section Lookup (Phase 1 — explicit refs)
     │  When query contains 030201, 030201.A, "Volume 5 Chapter 3":
     │  - index.get_chunks_by_section_ids(explicit_section_ids) → direct DB lookup
     │  - If >= BOOK_DIRECT_SECTION_MIN_CHUNKS (default 2) found → use as primary
     │  - Bypasses vector search for explicit section queries
     ▼
[Step 5] Hybrid Retrieval
     │  Dense search: FAISS inner-product on chunk embeddings
     │    - BOOK chunks: uses retrieval_text (heading path + clause label prepended)
     │      when BOOK_HEADING_PATH_IN_EMBED=1 and RAG_USE_CONTEXTUAL_RETRIEVAL=0
     │    - Or: contextualized_text (LLM-generated header) when RAG_USE_CONTEXTUAL_RETRIEVAL=1
     │  Sparse search: BM25 on same retrieval_text corpus
     │  Merge: Weighted RRF (dense_weight, sparse_weight from Step 1)
     ▼
[Step 6] Initial Top-K
     │  TOP_K (default 15) hits returned per query
     │  Multiple query variants run in parallel; results merged by RRF
     ▼
[Step 7] Cross-encoder Re-rank
     │  RAG_USE_RERANKER=1:
     │  - cross-encoder/ms-marco-MiniLM-L-6-v2 scores top RAG_RERANK_TOP_K=25 hits
     │  - Keeps top RAG_RERANK_FINAL_N=15 after re-ranking
     │  RAG_USE_LLM_RERANKER=1 (fallback): LLM scores each chunk 0-10
     ▼
[Step 8] Table/Threshold Score Boost (BookRAG-lite++)
     │  If query type is "threshold" or "table":
     │  - Chunks with has_table=True get +BOOK_TABLE_SCORE_BOOST (default 0.15)
     │  - Re-sort by score
     ▼
[Step 9] Adjacency Expansion (BookRAG-lite++)
     │  BOOK_ADJACENCY_EXPANSION=1:
     │  - For top 8 BOOK hits, fetch prev_chunk_id and next_chunk_id from DB
     │  - Add adjacent chunks with score = parent_score × 0.75
     │  - Prevents missed answers when answer is in the next clause after the retrieved one
     │  - Max BOOK_ADJACENCY_MAX_CHUNKS (default 1) per side
     ▼
[Step 10] Graph Expansion (cross-references)
     │  RAG_USE_GRAPH_EXPANSION=1 (when top score < RAG_GRAPH_EXPANSION_CONFIDENCE_THRESHOLD):
     │  - Follow "See paragraph X / Refer to Volume Y" links extracted during ingestion
     │  - Add up to RAG_GRAPH_MAX_ADDITIONS=3 new sections
     │  - Re-score additions vs. query
     ▼
[Step 11] Sort, Dedupe, Section Limit, Context Budget
     │  Sort by score descending
     │  RAG_DEDUPE_BY_SECTION: max RAG_MAX_CHUNKS_PER_SECTION (default 2) per section
     │    - Comparison queries: relaxed to ≥4 chunks per section
     │  _limit_sections_in_context: keep top MAX_SECTIONS_PER_ANSWER (default 2) sections
     │    - Comparison queries: relaxed to ≥4 sections
     │  _trim_hits_to_context_budget: drop lowest-scoring hits if total > RAG_CONTEXT_MAX_CHARS
     ▼
[Step 12] Evidence Extraction
     │  extract_evidence_sentences(): extract sentence-level evidence from top chunks
     │  EvidenceGate: composite confidence score
     │    - keyword_coverage: how many query keywords appear in evidence
     │    - section_match_score: does retrieved section match explicit reference in query
     │    - policy_word_ratio: density of regulatory language in evidence
     │    - Rejects and returns fallback if confidence too low
     ▼
[Step 13] Context Building
     │  For each hit:
     │  - Optionally add parent chunk text (up to RAG_PARENT_CONTEXT_MAX_CHARS=2400)
     │  - Add chunk text (verbatim up to RAG_VERBATIM_MAX_CHARS=1600)
     │  - Format: (doc_type, path, page) + text
     │  - Optionally LLM-compress (RAG_COMPRESS_CONTEXT, disabled by default for BOOK)
     │  For citation references: prepend [CITATION REFERENCE – Section X] block
     │  For comparison references: prepend [COMPARISON REFERENCE] block
     │  TOC guardrail: if query asks for chapters but no TOC in context → refuse
     ▼
[Step 14] Answer Generation
     │  LLM: qwen2.5:7b-instruct-q4_K_M (local Ollama)
     │  System prompt: _rag_system_prompt()
     │    - BOOK_STRICT_GROUNDED=1: adds strict regulatory grounding notice
     │    - Instructs: answer from evidence only, cite section/page, label inferences
     │  RAG_STRICT_CITATIONS=1: enforces (section_path, page N) format inline
     │  Response format hint: procedural → numbered steps; comparison → side-by-side
     ▼
[Step 15] Citation Building
     │  For each enriched hit with valid metadata:
     │  - BOOK: section_path, page_number, clause_id, canonical_id, page_start/end
     │  - All: filename, snippet, score, doc_id, doc_type, evidence_type
     │  - Filter: only chunks with metadata_valid=True exposed to client
     ▼
[Step 16] Answer Gate (fallback)
     │  RAG_USE_ANSWER_GATING=1:
     │  - Verify context has explicit section_id + ≥2 key tokens
     │  - If gate fails → return "The provided documents do not contain this information."
     ▼
Final Answer + Citations → Client
```

---

## Retrieval Debug Endpoint

```bash
POST /chat/debug-retrieval
{
  "question": "What is the liability of a certifying officer?",
  "k": 15
}
```

Response includes:
```json
{
  "source_type": "document",
  "resolver": { "explicit_section_ids": ["030201"], "has_explicit_refs": true },
  "toc_hits": [...],
  "section_summary_hits": [...],
  "selected_sections": ["Volume 5 > Chapter 3 > Section 030201"],
  "rejected_sections": [],
  "hits": [
    {
      "rank": 1,
      "score": 0.8432,
      "section_path": "Volume 5 > Chapter 3 > Section 030201",
      "clause_id": "030201.A",
      "canonical_id": "vol_05_ch_03_sec_030201_page_0142_chunk_02",
      "page": 142,
      "has_table": false,
      "evidence_type": "child",
      "text_preview": "030201.A. A certifying officer..."
    }
  ],
  "evidence": [...],
  "gate": { "passed": true, "confidence_score": 0.87 },
  "citations": [...],
  "refused": false
}
```

---

## Diagnosing Common Failure Patterns

### 1. Wrong section retrieved (retrieval_miss)
**Symptoms:** Answer references wrong Volume/Chapter or nearby section.

**Debug steps:**
1. Run `/debug-retrieval` and inspect `selected_sections`
2. Check if expected section appears in `hits` at rank > 5 (below reranker cutoff)
3. Check `toc_hits` — if TOC routing selected wrong section, `section_path` is wrong
4. Check `clause_id` on top hits — if populated, the correct clause may be adjacent

**Fixes:**
- Re-ingest with `BOOK_HEADING_PATH_IN_EMBED=1` (heading path improves recall)
- Check `section_path` in DB: `SELECT source_json FROM chunks WHERE ... LIKE '%030201%'`
- Verify TOC PDF was uploaded correctly

### 2. Answer truncated (adjacency_miss)
**Symptoms:** Answer is partially correct but misses the second half of a rule or threshold.

**Debug steps:**
1. Find the returned chunk and check `next_chunk_id` in chunks table
2. Verify `BOOK_ADJACENCY_EXPANSION=1`
3. Inspect `hits` in debug output: does adjacent chunk appear with `_adjacency: true`?

**Fixes:**
- Ensure `BOOK_ADJACENCY_EXPANSION=1` and `BOOK_ADJACENCY_MAX_CHUNKS=1` or higher
- Re-ingest to populate `prev_chunk_id`/`next_chunk_id` columns

### 3. Table not retrieved (table_failure)
**Symptoms:** Question asks about a rate table or schedule; answer doesn't include table values.

**Debug steps:**
1. Check `has_table` on retrieved chunks in debug output
2. Verify `BOOK_TABLE_EXTRACTION=1` and `BOOK_TABLE_SCORE_BOOST > 0`
3. Inspect raw chunk text for the expected table — were rows collapsed into prose?

**Fixes:**
- Enable `BOOK_TABLE_SCORE_BOOST=0.25` for higher table priority
- Ensure re-ingestion after enabling `BOOK_TABLE_EXTRACTION=1`

### 4. Wrong page number in citation (citation_mapping_failure)
**Symptoms:** PDF preview opens at wrong page; page_number in citation doesn't match actual.

**Debug steps:**
1. Check `page_offsets` in parse.py: were they generated for this document?
2. Inspect chunks table: `SELECT page_start, page_end, source_json FROM chunks WHERE id=?`
3. Verify page_index table has entries for this doc_id

**Fixes:**
- Re-ingest the document (page_offsets may not have been passed to index correctly)
- Check if `parse_any()` is returning `page_offsets` from fitz extraction

### 5. Hallucinated answer (generation_overreach)
**Symptoms:** Answer contains numeric values or policy statements not in retrieved evidence.

**Debug steps:**
1. Check `BOOK_STRICT_GROUNDED=1` is active
2. Inspect `ctx_block` (context blocks) — is the numeric value present in evidence?
3. Check `ev_gate_result.passed` and `confidence_score`

**Fixes:**
- Enable `BOOK_VERIFIER_ENABLED=1` (adds 1 LLM call to verify each claim)
- Increase `RAG_RELEVANCE_THRESHOLD` to raise the bar for weak evidence acceptance
- Enable `RAG_STRICT_CITATIONS=1` for inline (section_path, page N) enforcement

---

## Chunk Metadata Reference

Each chunk's `source_json` contains:

```json
{
  "doc_id": "doc_abc123",
  "filename": "DoD_FMR_Vol5.pdf",
  "filetype": "pdf",
  "doc_type": "book",
  "section_path": "Volume 5 > Chapter 3 > Section 030201 PURPOSE",
  "section_id": "030201",
  "section_title": "030201 PURPOSE",
  "page_number": 142,
  "is_parent": false,
  "parent_chunk_id": "chk_parent_xyz",
  "clause_id": "030201.A",
  "prev_chunk_id": "chk_prev_abc",
  "next_chunk_id": "chk_next_def",
  "canonical_id": "vol_05_ch_03_sec_030201_page_0142_chunk_02",
  "page_start": 142,
  "page_end": 143,
  "evidence_type": "child",
  "has_table": false,
  "metadata_valid": true
}
```

---

## Key Config Variables for Tuning

| Variable | Effect on Retrieval |
|----------|-------------------|
| `TOP_K` | Initial pool size; increase for better recall at cost of latency |
| `RAG_RERANK_TOP_K` | How many hits the cross-encoder scores; higher = better but slower |
| `BOOK_ADJACENCY_MAX_CHUNKS` | More adjacent chunks = better recall for split clauses |
| `BOOK_TABLE_SCORE_BOOST` | Higher = more table-heavy answers for threshold/rate queries |
| `RAG_BOOK_K_PER_QUERY` | Per-query top-K for BOOK section routing |
| `BOOK_HEADING_PATH_IN_EMBED` | ON: heading path boosts semantic accuracy; requires re-ingestion |
| `RAG_SECTION_TOP_K` | How many sections the section router selects before chunk search |
| `RAG_DEDUPE_SECTION_MAX_CHUNKS` | 2 for precision; 4+ for comparison queries |
| `MAX_SECTIONS_PER_ANSWER` | 2 for precision; auto-raised for comparison queries |
| `RAG_CONTEXT_MAX_CHARS` | Total context budget; reduce for faster generation |
| `BOOK_STRICT_GROUNDED` | Stronger grounding notice in system prompt |

---

## Eval Regression Testing

Run the built-in regression test suite:

```bash
POST /chat/eval-contextual-retrieval
```

For the full 30-case benchmark:

```python
# backend/app/rag/eval/book_eval.py
from backend.app.rag.eval.book_eval import run_eval
results = await run_eval(sample_json_path="backend/app/rag/eval/regulatory_eval_sample.json")
```

Expected metrics for a well-configured system (after re-ingestion with BookRAG-lite++):
- `retrieval_section_recall@5` ≥ 0.75 for `explicit_section_lookup` bucket
- `citation_correctness` ≥ 0.70 for BOOK documents
- `refusal_correctness` = 1.0 for `refusal_when_missing` bucket
- `answer_keyword_coverage` ≥ 0.60 for `threshold_exception` bucket
