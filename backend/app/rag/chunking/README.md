# Chunking Pipeline (Architecture & Examples)

## Pipeline Overview

1. **Input**: Raw document text (from PDF/DOCX/PPTX or paste).
2. **Detect type** (`detect_document_type`): Heuristics, no model calls.
   - **SENSITIVE**: PII density ≥ 1.5% → small chunks, redaction.
   - **FAQ**: Many Q/A patterns, or "FAQ" in first 2k chars → one chunk per Q+A.
   - **BOOK**: Long (≥8k chars), many paragraphs, optional headings → parent-child chunking.
   - **USER**: Default for mixed/poor formatting → sentence-aware medium chunks.
3. **Sanitize** (`sanitize_text`): Regex-based PII detection; redact (email, phone, SSN, card, IDs). Sets `redacted` and `sensitivity_level`.
4. **Chunk by type**:
   - **FAQ**: Split on next-question boundary; one chunk per Q+A (never split question from answer).
   - **Long-form**: Build parent chunks (2k–3.5k chars) at paragraph boundaries; split each parent into child chunks (400–700 chars) at sentence boundaries with small overlap; children reference `parent_chunk_id`.
   - **Sensitive**: Small chunks (~450 chars), no overlap, sentence-boundary only.
   - **User**: Medium chunks (~800 chars), sentence-boundary, 1-sentence overlap.
5. **Assign IDs**: `chunk_document` assigns `doc_id`, `chunk_id` (and `parent_chunk_id` for children).
6. **Index**: Only non-parent chunks are embedded and stored in FAISS/sparse; all chunks (including parents) are stored in DB for possible context expansion.

## Chunk Schema

- **Chunk**: `doc_id`, `chunk_id`, `text`, `doc_type`, `sensitivity_level`, `redacted`, `section`, `parent_chunk_id`, `is_parent`, `chunk_index`, optional `char_start`/`char_end`.
- **ParentChildChunk**: `parent` (Chunk with `is_parent=True`), `children` (list of Chunk with `parent_chunk_id` set).

## Examples

### Long-form paragraph → chunks

Input (excerpt):
```
Chapter 1. Introduction. The study of rapid cognition has shown that
experts often make better decisions in the blink of an eye than after
lengthy analysis. This book explores when thin-slicing works and when
it fails. [ ... 2500 chars ... ]
```

Output: One parent chunk (full 2k–3k segment); 3–5 child chunks (400–700 chars each) at sentence boundaries, each with `parent_chunk_id` pointing to the parent.

### FAQ → chunks

Input:
```
Q: What is Blink about?
A: Blink is about the power of thinking without thinking.

Q: Who wrote it?
A: Malcolm Gladwell.
```

Output: Two chunks. Chunk 1 = "Q: What is Blink about?\nA: Blink is about..."; Chunk 2 = "Q: Who wrote it?\nA: Malcolm Gladwell."

### Sensitive customer record → chunks

Input:
```
Customer John Doe, johndoe@example.com, 555-123-4567. Account ID: AB12345678.
```

After sanitization:
```
Customer John Doe, [REDACTED_EMAIL], [REDACTED_PHONE]. Account ID: [REDACTED_ID].
```

Output: One or two small chunks (~450 chars), `redacted=True`, `sensitivity_level=HIGH`; only redacted text is embedded.
