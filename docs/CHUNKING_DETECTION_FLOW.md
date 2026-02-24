# Document type detection flow (chunking)

This doc explains how the chunking pipeline decides **document type** (FAQ, Book, Sensitive, User) and why a book can be mistaken for FAQ.

---

## 1. Where detection runs

- **Entry:** `chunk_document(text, doc_id)` in `backend/app/rag/chunking/pipeline.py`
- **Steps:** normalize text → **detect_document_type(text)** → sanitize (PII) → dispatch to the right chunker (FAQ, long-form/book, sensitive, unstructured).

---

## 2. Detection order (important)

In `backend/app/rag/chunking/detect.py`, `detect_document_type()` applies checks **in this order**:

| Order | Type       | Condition |
|-------|------------|-----------|
| 1     | **SENSITIVE** | `_pii_density(text) >= 0.015` (PII-heavy content) |
| 2     | **FAQ**    | `_looks_like_faq(text, lines)` returns True |
| 3     | **BOOK**   | `_looks_like_long_form(text, lines, length)` returns True |
| 4     | **USER**   | Default (everything else) |

So **FAQ is checked before BOOK**. If the text looks like FAQ, it is never tested for long-form. That is why a book can be misclassified as FAQ.

---

## 3. FAQ detection (`_looks_like_faq`)

- **Minimum:** at least 4 non-empty lines.
- **Question pattern:** A line counts as a “question” if it matches:
  - Optional leading number (e.g. `1.` or `2)`)
  - Optional prefix `Q`, `Question`, `Q.`, `Question:`, etc.
  - Rest of the line **ends with `?`**
- **Rules:**
  - If **q_count ≥ max(3, len(lines) // 10)** → **FAQ**
  - **Or** if the string `"FAQ"` or `"frequently asked"` appears in the **first 2000 characters** → **FAQ**

So any document with many lines ending in `?` (e.g. section titles like “What is X?”, “How does Y work?”) can hit the FAQ threshold and be classified as FAQ even when it’s a long-form book.

---

## 4. Long-form / book detection (`_looks_like_long_form`)

- **Minimum length:** 8,000 characters.
- **Paragraph structure:** at least 5 `\n\n` (paragraph breaks).
- **Then** one of:
  - **Heading-like lines:** at least 2 lines that look like “Chapter …”, “Part …”, “Section …”, or “1. …” (and length &lt; 120), **or**
  - **Long and structured:** at least 20 paragraph breaks **and** length &gt; 20,000, **or**
  - **Long paragraphs:** average line length &gt; 80 **and** length &gt; 15,000.

So books are only considered if they pass these length/structure checks, but **only after** the FAQ check. If FAQ wins first, BOOK is never evaluated.

---

## 5. Why a book was mistaken for FAQ

1. The book has **many lines ending with `?`** (e.g. headings or rhetorical questions).
2. **FAQ is evaluated before BOOK**, so once `_looks_like_faq` returns True, the type is fixed as FAQ.
3. Long-form checks (length, paragraph breaks, headings) are never run for that document.

---

## 6. Fix (recommended)

**Check long-form before FAQ** when the text is long enough to be a book (e.g. length ≥ 8,000). Then:

- Long, structured documents are classified as BOOK even if they contain many question-shaped lines.
- Short FAQ-like documents still become FAQ.

Implementation: in `detect_document_type()`, if `length >= 8000` (or the same minimum used in `_looks_like_long_form`), call `_looks_like_long_form` first; if True, return BOOK. Otherwise keep the current order (sensitive → FAQ → long-form → user). See `detect.py` change below.
