# BookRAG Upgrade Checklist

## Summary of Changes

The EchoMind RAG system has been upgraded to a BookRAG-style hierarchical retriever for large regulatory books (e.g. DoD FMR 7300 pages).

### New Modules
- `backend/app/rag/book/section_id.py` – Canonical section-id extraction and normalization
- `backend/app/rag/book/section_resolver.py` – Deterministic navigation for explicit refs
- `backend/app/rag/book/toc_builder.py` – TOC from book_sections
- `backend/app/rag/indexes/toc_index.py` – TOC FAISS index for routing
- `backend/app/rag/eval/book_eval.py` – Evaluation harness

### Data Model
- Chunk: added `section_id` (canonical DoD code)
- Book sections: added `canonical_section_id` in metadata
- Source JSON: `section_id` persisted for chunks

### Citation & Grounding (DoD FMR accuracy)
- **SectionResolver**: Indexes DoD codes from `full_section_text` so "Segment N" docs that mention codes (e.g. 030201) in body map correctly
- **Direct citations**: For explicit refs (e.g. "paragraph 030201"), prepends `[CITATION REFERENCE]` with `get_section_content` so the LLM quotes directly
- **Inference transparency**: `improve_inference_transparency_async` and `handle_missing_information` suggest related sections when info is inferred or missing
- **Procedural steps**: `structure_procedural_steps` supports per-step section refs; prompt instructs "Step N: … (see Section X for details)"
- **Dynamic weights**: `RAG_CITATION_PROCEDURAL_DENSE_WEIGHT=0.45`, `RAG_CITATION_PROCEDURAL_SPARSE_WEIGHT=0.55` for citation+procedural queries

### Config (env vars)
- `RAG_SECTION_TOP_K=10` (big book default)
- `RAG_SECTION_SCORE_THRESHOLD=0.35`, `RAG_SECTION_RELAX_THRESHOLD=0.25`
- `RAG_CE_MAX_CHARS=1024` (reranker input 768–1024)
- `RAG_USE_TOC_ROUTING`, `RAG_TOC_TOP_K`, `RAG_TOC_THRESHOLD`
- `RAG_MAX_CHUNKS_PER_SECTION=2` (4 for comparison queries)

---

## Checklist to Run

### 1. Rebuild indexes

After pulling these changes, rebuild indexes so new BOOK documents get `section_id`, TOC, and canonical metadata:

```bash
# Option A: Delete all data and re-upload documents
# (via API or UI: delete all, then upload your regulatory PDFs again)

# Option B: If you have a script to re-index
# Run your existing re-index procedure for BOOK documents
```

The TOC index is rebuilt automatically when:
- A new BOOK document is added
- A document is deleted
- `clear_all()` is called

### 2. Run eval script

**Local run (outside Docker):** Set `ECHOMIND_DATA_DIR` to a writable path (default `/data` needs root):

```bash
cd backend
ECHOMIND_DATA_DIR=./data PYTHONPATH=. python -m app.rag.eval.book_eval
```

Or from project root:
```bash
ECHOMIND_DATA_DIR=./data PYTHONPATH=backend python -m app.rag.eval.book_eval
```

**Inside Docker:** The container uses `/data` by default (mounted from your compose volume).

Report: `$ECHOMIND_DATA_DIR/book_eval_report.json` (section-hit, citation coverage, refusal rate, inferred rate).

### 3. Run server and test 5 sample queries

```bash
docker compose up -d
```

**5 sample queries** (replace codes with ones in your DoD FMR):
1. **Citation**: "What does paragraph 030201 say about purpose?"
2. **Comparison**: "Compare section 0301 and section 0402"
3. **Procedural**: "How do I submit a payment request?"
4. **Definition**: "What is audit readiness?"
5. **Explicit ref**: "Section 0705 requirements"

---

## Acceptance Criteria

- [ ] Queries with explicit section codes (e.g. 030201): top-5 chunks contain requested section_id
- [ ] Comparison queries: both sections retrieved
- [ ] Answers quote/paraphrase from evidence and cite page_number + section_path
- [ ] If grounding fails: refusal + top 3 closest sections (no hallucinated answers)
- [ ] Inferred rate drops; refusal rate increases when retrieval misses

---

## Logging

Each stage logs to INFO:
- `SectionResolver: explicit refs → N path(s)`
- `RAG: TOC routing → N path(s) from M nodes`
- `RAG: dynamic relax threshold`
- `RAG: section restriction applied → N section(s)`
