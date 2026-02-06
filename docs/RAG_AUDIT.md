# RAG Pipeline Audit — EchoMind

**Role:** Principal AI Engineer / RAG Systems Auditor  
**Scope:** Document ingestion → chunking → embedding & indexing → retrieval → generation → conversational consistency & hallucination resistance.

---

## 1. High-level diagnosis

| Area | Status | Summary |
|------|--------|--------|
| **Ingestion & chunking** | ✅ Strong | Adaptive pipeline (book/FAQ/user/sensitive), sentence-boundary chunking, PII sanitization, parent–child for long-form. |
| **Embedding & indexing** | ✅ Adequate | Hybrid FAISS + BM25, RRF merge; embed truncation to avoid context overflow. Parent chunks stored but not used at retrieval. |
| **Retrieval** | ⚠️ Needs tuning | k=8, 4 query variants, RRF; no MMR, no re-ranking, no doc-type-specific k. Redundancy and relevance dilution possible. |
| **Generation & grounding** | ❌ Critical issues | Prompt asks for "Sources list [1],[2]…" and context includes SOURCE: filename#chunk — **sources are exposed** and output is citation-style, not conversational. |
| **Source handling** | ❌ Policy violation | Citations (filenames, chunk refs) are in the prompt and in API response; UI shows "Resources" with file names. **Internal grounding ≠ user-facing output** is not enforced. |
| **Hallucination resistance** | ⚠️ Partial | "Use ONLY the provided context" and "say what's missing" are good; no explicit ban on "likely/might/inferred" for factual claims; compress step can subtly shift meaning. |

**Verdict:** Chunking and retrieval are in good shape; **generation and source-exposure policy are the main risks**. The system is currently tuned for a "search engine with citations" rather than a "knowledgeable assistant that never shows its sources."

---

## 2. Retrieval analysis (k=7–8 impact)

### Current setup

- **TOP_K = 8** (config). So effective k is already 8.
- **Per-query fetch:** `k_per_query = max(k, 4)` → 8 hits per query; 4 query variants → up to 32 candidates before RRF.
- **RRF** fuses dense + sparse per variant, then takes top-k=8. No MMR, no re-ranking.

### Relevance dilution

- **Risk:** With 4 query variants, a chunk that matches only one variant (e.g. BM25 keyword match) can still get enough RRF score to land in top-8. That can pull in **tangential or weakly relevant** chunks.
- **Effect at k=8:** More coverage, but the 6th–8th slots are more likely to be marginal. If the model tries to use them, it may drift or hedge.

### Redundancy between chunks

- **Long-form (book):** Children from the same parent are adjacent in embedding space. Retrieval can return **several children from the same parent** (overlapping content). They add little new information and consume context.
- **FAQ:** One chunk per Q&A — less redundancy, but long FAQs (truncated at 8k chars) can still produce a few very long chunks.
- **No dedupe by parent:** Siblings are not collapsed; we do dedupe by `chunk_id` in RRF but not by `parent_chunk_id`.

### Competing semantic signals

- **Contradiction risk:** If top-8 contains chunks that disagree (e.g. different sections of a policy), the model may blend them or pick one without saying "it depends on which part." No instruction to resolve or flag conflict.
- **Authority blur:** All chunks are presented as numbered context [1]–[8]; the model is not told to prefer higher-ranked (earlier) chunks when they conflict.

### Irrelevant / overlapping / missing

- **Irrelevant:** Possible when RRF promotes a chunk that matched only on a generic term (e.g. "policy") or a variant query that rephrased the question loosely.
- **Overlapping:** Likely when multiple child chunks from the same long-form parent are in the top-8; same narrative, repeated in context.
- **Missing:** For book-like docs, the **parent chunk is never fetched**. If retrieval returns only one child of a section, the model never sees the full 2k–3k parent context — only the 400–700 char child. Important nuance can live in the parent.

### Recommendations (retrieval)

- **k by document type (conceptual):**
  - **Book / long-form:** k=8–10, and **expand with parent**: when a child is in the top-k, fetch its `parent_chunk_id` from DB and include parent text once per parent (so context is richer without multiplying redundant children).
  - **FAQ:** k=4–6; each chunk is self-contained, so lower k reduces noise.
  - **Sensitive / user:** k=6–8; keep current or slightly lower if re-ranking is added.
- **MMR:** Consider Maximal Marginal Relevance (lambda ~0.7) **after** RRF to trade a bit of relevance for diversity and reduce sibling/duplicate-heavy sets.
- **Re-ranking:** If latency allows, a lightweight cross-encoder or a second-stage "relevance to question" pass on top-12 → top-8 would improve precision and reduce irrelevant chunks in the final context.

---

## 3. Generation issues found

### 3.1 Citation-style output (not conversational)

- **Current:** System prompt: *"Provide the answer, then a Sources list referencing [1],[2]…."*  
- **Context block:** Each snippet is followed by `SOURCE: filename#chunkN`.  
- **Effect:** The model is **instructed** to produce academic-style references. Output looks like a report, not a natural assistant.

### 3.2 Source exposure

- **Backend:** Returns `citations` (filename, chunk_index, score, snippet) to the client.  
- **Frontend:** Shows a "Resources" button and lists file names for the message.  
- **Policy breach:** "Sources MUST NOT be exposed to the user" and "No citations, file names, chunk IDs" are violated by design.

### 3.3 Grounding and hedging

- **Good:** "Use ONLY the provided context" and "If context is insufficient, say what's missing" support faithfulness.
- **Gaps:**
  - No explicit rule: "Do not use 'likely', 'might', 'inferred' for factual claims; either state what the context says or say it's missing."
  - No instruction to avoid inventing section titles, bullet names, or "the document explains…" when the context does not explicitly support it.
- **Compress step:** "Extract only sentences that directly help answer" is an extra LLM pass. It can slightly rephrase or drop nuance; if the compressor hallucinates or over-interprets, the main answer is grounded to the **compressed** text, not the original chunk — a form of **citation laundering** (claim grounded in a transformed source).

### 3.4 Partial-context and contradiction

- No instruction to **prefer earlier (higher-ranked) context** when two chunks conflict.
- No instruction to say "Different sections say different things" or "This isn’t clearly stated in your materials" when context is conflicting or vague.

---

## 4. Concrete fixes (bullet points)

### Retrieval

- **Keep k=8** as default; consider making k configurable per doc_type if you later tag queries by doc type (e.g. from recent uploads).
- **Parent expansion:** When building context for the LLM, for each hit with `parent_chunk_id` in `source`, load the parent chunk text from DB and prepend or append it once per unique parent (so the model sees full section context for retrieved children).
- **Optional MMR:** After RRF, run MMR on the top-2k list (e.g. 16) before taking top-k=8 to reduce duplicate/sibling chunks.
- **Optional re-ranking:** Second-stage model or similarity threshold on "question vs chunk" to drop clearly irrelevant chunks before context assembly.

### Grounding & faithfulness

- **Strict system rule:** "Do not add facts not stated in the context. Do not use 'likely', 'might', or 'inferred' for factual claims — either state what the context says or say the information is not in the materials."
- **Conflict rule:** "If different parts of the context contradict each other, say so instead of picking one silently."
- **Compress:** Either (a) keep compress but instruct the compressor to "copy sentences verbatim where possible" and never add interpretation, or (b) skip compress and pass chunk text directly to reduce transformation risk (at the cost of more tokens).

### Conversational generation (no citations to user)

- **Remove from system prompt:** "Provide the answer, then a Sources list referencing [1],[2]…." and any instruction to output [1], [2], or source names.
- **Replace with:** "Answer in a clear, confident, human way. Do not cite sources, mention file names, chunk numbers, or 'according to the document'. If the information is not in the context, say so plainly; do not guess or infer."
- **Context block:** Remove the line `SOURCE: filename#chunkN` from the text the model sees. Keep only numbered content, e.g. `[1] …`, `[2] …`, for internal grounding. Optionally keep source metadata in a separate structured field for logging/audit only, not in the user-facing prompt.

### Source handling policy (internal only)

- **Internal:** Continue to retrieve chunks, optionally compress, and build context with stable identifiers (e.g. chunk_id, doc_id) for logging and debugging.
- **User-facing:** Do **not** return citations (filenames, chunk IDs, snippets) in the API response for the assistant message; or return them only for an internal/admin view. If the product requirement is "user never sees sources," remove or hide the "Resources" list in the UI and never show file names or chunk refs in the answer text.
- **Separation:** Internal grounding = context + chunk metadata in backend only. User-facing output = natural-language answer only, with optional generic phrase like "Based on your resources" (no file names) if you want to signal RAG was used.

### Failure-mode mitigations

- **Semantic drift (multi-turn):** Keep last N turns in history; avoid "summarize previous context" steps that could drift from original sources.
- **Authority hallucination:** System rule: "Do not say 'the document says' or 'Section X states' unless the context explicitly contains that wording or a clear equivalent."
- **Citation laundering:** Minimize compress or make it verbatim-only; do not ask the model to output [1]/[2] so it cannot point to the wrong chunk in user-visible text.
- **Partial-context answers:** Prefer parent expansion so the model sees fuller context for each retrieved child; add "If you are unsure, say so" for low-confidence retrieval (e.g. best_score just above threshold).
- **Concept invention:** "Do not introduce concepts, terms, or section names that do not appear in the context."

---

## 5. Final "golden rules" for this RAG system

1. **Retrieval**
   - Use hybrid (dense + sparse) and RRF; default k=8; consider parent expansion for long-form and optional MMR/re-ranking.
   - When best hit score &lt; RAG_RELEVANCE_THRESHOLD, do not use RAG context; answer as general assistant or refuse.

2. **Grounding**
   - Every factual claim must be supportable from the retrieved context; no guessing; no "likely/might/inferred" for facts.
   - If context is missing or conflicting, say so plainly. Do not silently pick one side or invent.

3. **Conversational output**
   - Answer like a knowledgeable human: clear, confident, no academic citations, no [1]/[2], no "according to the document," no file names or chunk IDs in the answer.

4. **Source handling**
   - Use sources only for internal grounding. Do not expose citations, file names, or chunk references to the user unless product explicitly requires it (then treat as separate feature with clear policy).

5. **Refusal**
   - When retrieval is weak (below threshold) or context is insufficient, respond as a general assistant or say "I don’t have enough in your materials to answer that" — do not fabricate.

6. **Chunking**
   - Keep adaptive chunking and sentence-boundary rules; do not split FAQ Q from A; for long-form, keep parent–child and consider using parent at retrieval time.

7. **Auditability**
   - Log which chunks were retrieved and used (chunk_id, doc_id, score) for debugging and audit; keep this server-side only, not in user-facing response.

---

## 6. Recommended configuration (summary)

| Parameter | Recommendation |
|-----------|----------------|
| **TOP_K** | 8 (default); consider 6 for FAQ-heavy workloads, 8–10 for book-heavy with parent expansion. |
| **Re-ranking** | Optional: re-rank top-12 → top-8 by question–chunk relevance; or MMR on top-16 → 8. |
| **Chunk size / overlap** | Leave to adaptive chunking (child 400–700, parent 2k–3.5k, FAQ one Q&A, etc.); ensure no chunk exceeds EMBED_MAX_CHARS before embed. |
| **Max context** | Reserve space for 8 chunks + compress (or 8 × ~300 tokens) + history; keep LLM_MAX_TOKENS (512) so answers stay concise. |
| **Refusal** | When best_score &lt; RAG_RELEVANCE_THRESHOLD (0.45), do not inject RAG context; answer generally or say you don’t have that in the materials. |
| **Generation** | Remove all citation and source instructions from the prompt; instruct natural, grounded, no sources in output. |

Implementing the **generation and source-handling** changes will have the largest impact on trustworthiness and conversational behavior; retrieval and parent-expansion improvements will then improve answer quality without exposing internals.

---

## Implementation status (post-audit)

| Audit item | Status | Location |
|------------|--------|----------|
| Parent expansion at retrieval | Done | `advanced.py`: `_get_chunk_text`, `_build_rag_context` — fetches parent chunk when `parent_chunk_id` in source, adds once per parent (truncated to 2400 chars). |
| Context block without SOURCE lines | Done | `_rag_context_block` builds `[1] …`, `[2] …` only; no filename/chunk in prompt. |
| Conversational + grounding system prompt | Done | `_RAG_SYSTEM_PROMPT`: no citations, no "likely/might/inferred", conflict rule, no concept invention, refuse when missing. |
| Compress verbatim-focused | Done | `compress()`: "Copy sentences verbatim where possible; do not paraphrase or add interpretation." |
| Sources not exposed to user | Done | `RAG_EXPOSE_SOURCES` (default False) in config; citations returned only when True. Frontend shows Resources only when `citations.length > 0`. |
| Audit logging | Done | `logger.info("RAG answer", extra={"chunk_ids": ..., "doc_ids": ...})` in `answer()` and `answer_stream()`. |
| Refusal when below threshold | Already present | `best_score < RAG_RELEVANCE_THRESHOLD` → general assistant, no RAG. |
| Optional MMR / re-ranking | Not implemented | Can be added later; RRF + parent expansion and prompt changes address main issues. |
