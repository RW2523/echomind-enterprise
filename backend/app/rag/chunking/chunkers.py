"""
Type-specific chunkers. Sentence-aware; no mid-sentence splits.
Uses settings.CHUNK_SIZE / CHUNK_OVERLAP when available for faster, focused retrieval.
Sizing is token-aware; structure-aware for BOOK (section/chapter metadata).
"""
from __future__ import annotations
import re
from typing import List, Optional, Tuple

from .models import Chunk, DocType, ParentChildChunk, SensitivityLevel


# --- 1. Token-aware sizing (no hard dependency on OpenAI/HF tokenizers) ---
def token_len(text: str) -> int:
    """
    Approximate token count for chunk size limits. Lightweight: ~4 chars per token for English,
    with a word-based upper bound to avoid undercounting for dense text. Use this for all
    size checks so embedding/context limits are respected.
    """
    if not (text or "").strip():
        return 0
    t = text.strip()
    char_based = (len(t) + 3) // 4
    word_based = len(t.split()) * 2  # conservative: ~0.5 tokens per word lower bound
    return max(char_based, word_based // 2)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Return prefix of text with at most max_tokens (approximate). Keeps sentence boundaries when possible."""
    if not text or max_tokens <= 0:
        return ""
    if token_len(text) <= max_tokens:
        return text
    # Approximate char boundary: ~4 chars per token
    max_chars = max(1, max_tokens * 4)
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0] if max_chars < len(text) else text
    return truncated or text[:max_chars]


def _get_chunk_size_overlap() -> tuple[int, int]:
    """Use config for chunk size/overlap when available (faster retrieval, good quality)."""
    try:
        from ...core.config import settings
        size = getattr(settings, "CHUNK_SIZE", 600) or 600
        overlap = getattr(settings, "CHUNK_OVERLAP", 80) or 80
        return (max(200, min(1200, size)), max(40, min(200, overlap)))
    except Exception:
        return (600, 80)

# Sentence boundary (naive); safer split uses _sentences() with abbrev/citation heuristics
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n\n+")
# Min/max child size for long-form (child size follows config when used in unstructured path)
_PARENT_MIN, _PARENT_MAX = 2000, 3500
_CHILD_MIN, _CHILD_MAX = 400, 700
_CHILD_OVERLAP = 80
# Sensitive: small, minimal overlap
_SENSITIVE_SIZE = 450
_SENSITIVE_OVERLAP = 40
# Unstructured (user): use config defaults (600/80) for snappier retrieval
_USER_SIZE = 600
_USER_OVERLAP = 80
# FAQ: whole Q+A per chunk; cap by tokens so embedding API context is not exceeded (constants = tokens)
MAX_FAQ_TOKENS = 8000


# --- 2. Safer sentence segmentation (no mid-sentence splits; avoid abbrev/citation false splits) ---
def _ends_with_abbrev(s: str) -> bool:
    """True if s ends with common abbreviation (Dr., e.g., Fig. 1, etc.) so we should not split after it."""
    if not s or len(s) < 3:
        return False
    s = s.rstrip()
    if not s:
        return False
    # Trailing single capital + period (e.g. "Fig. 1" already captured by next)
    if re.search(r"\s(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|e\.g|i\.e|al|Fig|Vol|No|approx)\.\s*$", s, re.I):
        return True
    if re.search(r"[A-Z]\.\s*$", s):  # single letter abbreviation
        return True
    if re.search(r"\[\d+\]\s*$|\d+\)\s*$", s):  # citation-like end
        return True
    return False


def _sentences(text: str) -> List[str]:
    """
    Split into sentences; never split mid-sentence. Uses regex boundaries but merges
    segments that end with abbreviations (Dr., e.g., Fig.), bullet points, or citation
    patterns to avoid false splits.
    """
    if not text.strip():
        return []
    parts = _SENTENCE_RE.split(text)
    trimmed = [p.strip() for p in parts if p.strip()]
    if not trimmed:
        return []
    merged: List[str] = []
    buf = trimmed[0]
    for i in range(1, len(trimmed)):
        next_seg = trimmed[i]
        if _ends_with_abbrev(buf):
            buf = buf + " " + next_seg
        else:
            merged.append(buf)
            buf = next_seg
    if buf:
        merged.append(buf)
    return merged


# --- 3. Structure awareness for BOOK: chapter/section detection ---
# Patterns for section headings (capture title, then text until next heading or end)
_CHAPTER_RE = re.compile(r"(?m)^\s*(?:Chapter\s+\d+(?:\s*[-:]\s*)?|Part\s+[IVXLCDM]+\s*[-:]\s*)(.+?)\s*$", re.IGNORECASE)
_SECTION_MARKDOWN_RE = re.compile(r"(?m)^\s*#{1,4}\s+(.+?)\s*$")
_SECTION_NUM_RE = re.compile(r"(?m)^\s*(\d+(?:\.\d+)*\s*[.\s]\s*.+?)\s*$")


def _split_book_into_sections(text: str) -> List[Tuple[Optional[str], str]]:
    """
    Detect chapter/section headings and split book into (section_title, section_text) pairs.
    If no structure is detected, returns [(None, text)] so caller can fall back to single-section behavior.
    """
    text = (text or "").strip()
    if not text:
        return []

    # Try chapter-style first (Chapter 1, Part I, etc.)
    chapters = list(_CHAPTER_RE.finditer(text))
    if chapters:
        sections = []
        for i, m in enumerate(chapters):
            title = m.group(1).strip()
            start = m.end()
            end = chapters[i + 1].start() if i + 1 < len(chapters) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append((title, body))
        if sections:
            return sections

    # Try markdown-style ## Heading
    md_heads = list(_SECTION_MARKDOWN_RE.finditer(text))
    if md_heads:
        sections = []
        for i, m in enumerate(md_heads):
            title = m.group(1).strip()
            start = m.end()
            end = md_heads[i + 1].start() if i + 1 < len(md_heads) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append((title, body))
        if sections:
            return sections

    # No structure detected: single section
    return [(None, text)]


def _group_sentences_to_size(
    sentences: List[str],
    target_min: int,
    target_max: int,
    overlap_sentences: int = 0,
) -> List[str]:
    """
    Group sentences into chunks by token count (target_min/target_max in tokens).
    Never splits a sentence; uses token_len() for all size checks.
    """
    if not sentences:
        return []
    chunks = []
    current: List[str] = []
    current_tokens = 0
    for s in sentences:
        s_tokens = token_len(s)
        space_tokens = token_len(" ") if current else 0
        if current_tokens + space_tokens + s_tokens > target_max and current:
            chunk_text = " ".join(current)
            chunks.append(chunk_text)
            if overlap_sentences > 0 and len(current) > overlap_sentences:
                current = current[-overlap_sentences:]
                current_tokens = token_len(" ".join(current))
            else:
                current = []
                current_tokens = 0
        current.append(s)
        current_tokens += (space_tokens + s_tokens) if current_tokens else s_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_faq(
    text: str,
    sensitivity_level: SensitivityLevel,
    redacted: bool,
) -> List[Chunk]:
    """
    One chunk per Q&A pair. Never split question from answer.
    Splits on start of next question (e.g. "Q:", "1. Q", "Question 2:").
    """
    chunks_out: List[Chunk] = []
    text = (text or "").strip()
    if not text:
        return []

    next_q = re.compile(r"\n\s*(?:\d+[.)]\s*)?(?:Q(?:uestion)?\s*[:.]?\s*)", re.IGNORECASE)
    parts = next_q.split(text)
    if len(parts) <= 1:
        parts = [text]
    idx = 0
    for block in parts:
        block = block.strip()
        if not block or token_len(block) < 5:
            continue
        if token_len(block) > MAX_FAQ_TOKENS:
            block = _truncate_to_tokens(block, MAX_FAQ_TOKENS)
        chunks_out.append(
            Chunk(
                doc_id="",
                chunk_id="",
                text=block,
                doc_type=DocType.FAQ,
                sensitivity_level=sensitivity_level,
                redacted=redacted,
                section=None,
                parent_chunk_id=None,
                is_parent=False,
                chunk_index=idx,
            )
        )
        idx += 1
    if not chunks_out:
        faq_text = _truncate_to_tokens(text, MAX_FAQ_TOKENS) if token_len(text) > MAX_FAQ_TOKENS else text
        chunks_out.append(
            Chunk(
                doc_id="",
                chunk_id="",
                text=faq_text,
                doc_type=DocType.FAQ,
                sensitivity_level=sensitivity_level,
                redacted=redacted,
                section=None,
                parent_chunk_id=None,
                is_parent=False,
                chunk_index=0,
            )
        )
    return chunks_out


def _adaptive_overlap_sentences(chunk_max_tokens: int) -> int:
    """
    Overlap in number of sentences: scale with chunk size for better continuity.
    Larger chunks get more overlap; keep sentence-boundary safety. Range 1–4.
    """
    return max(1, min(4, (chunk_max_tokens // 150)))


def chunk_long_form(
    text: str,
    sensitivity_level: SensitivityLevel,
    redacted: bool,
    section: Optional[str] = None,
) -> List[ParentChildChunk]:
    """
    Parent chunks (token-sized _PARENT_MIN–_PARENT_MAX) for context; child chunks for retrieval.
    Children reference parent_chunk_id. Section metadata set when provided (from structure detection).
    Overlap is adaptive by chunk size.
    """
    text = (text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    # Parent grouping by token count (constants = tokens)
    parent_chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for p in paragraphs:
        p_tokens = token_len(p)
        sep_tokens = token_len("\n\n") if current else 0
        if current_tokens + sep_tokens + p_tokens > _PARENT_MAX and current:
            parent_chunks.append("\n\n".join(current))
            overlap_paras = [current[-1]] if len(current) >= 1 else []
            current = overlap_paras + [p] if p_tokens <= _PARENT_MAX else [p]
            current_tokens = token_len("\n\n".join(current))
        else:
            current.append(p)
            current_tokens += (sep_tokens + p_tokens) if current_tokens else p_tokens
    if current:
        parent_chunks.append("\n\n".join(current))

    csize, _ = _get_chunk_size_overlap()
    child_min = min(_CHILD_MIN, csize // 2)
    child_max = min(_CHILD_MAX, csize + 100)
    overlap_sentences = _adaptive_overlap_sentences(child_max)

    out: List[ParentChildChunk] = []
    for pi, parent_text in enumerate(parent_chunks):
        if token_len(parent_text) < _PARENT_MIN and pi < len(parent_chunks) - 1:
            continue
        sentences = _sentences(parent_text)
        child_texts = _group_sentences_to_size(
            sentences,
            child_min,
            child_max,
            overlap_sentences=overlap_sentences,
        )
        if not child_texts:
            child_texts = [_truncate_to_tokens(parent_text, child_max)] if parent_text else []

        parent_chunk = Chunk(
            doc_id="",
            chunk_id="",
            text=parent_text,
            doc_type=DocType.BOOK,
            sensitivity_level=sensitivity_level,
            redacted=redacted,
            section=section,
            parent_chunk_id=None,
            is_parent=True,
            chunk_index=pi,
        )
        children = [
            Chunk(
                doc_id="",
                chunk_id="",
                text=ct,
                doc_type=DocType.BOOK,
                sensitivity_level=sensitivity_level,
                redacted=redacted,
                section=section,
                parent_chunk_id="",
                is_parent=False,
                chunk_index=pi * 100 + ji,
            )
            for ji, ct in enumerate(child_texts)
        ]
        out.append(ParentChildChunk(parent=parent_chunk, children=children))
    return out


def chunk_sensitive(
    text: str,
    sensitivity_level: SensitivityLevel,
    redacted: bool,
) -> List[Chunk]:
    """Small chunks, minimal overlap; sentence-boundary only."""
    text = (text or "").strip()
    if not text:
        return []
    sentences = _sentences(text)
    chunk_texts = _group_sentences_to_size(
        sentences,
        target_min=_SENSITIVE_SIZE // 2,
        target_max=_SENSITIVE_SIZE,
        overlap_sentences=0,
    )
    if not chunk_texts:
        chunk_texts = [_truncate_to_tokens(text, _SENSITIVE_SIZE)]
    return [
        Chunk(
            doc_id="",
            chunk_id="",
            text=ct,
            doc_type=DocType.SENSITIVE,
            sensitivity_level=sensitivity_level,
            redacted=redacted,
            section=None,
            parent_chunk_id=None,
            is_parent=False,
            chunk_index=i,
        )
        for i, ct in enumerate(chunk_texts)
    ]


def chunk_unstructured(
    text: str,
    sensitivity_level: SensitivityLevel,
    redacted: bool,
) -> List[Chunk]:
    """Robust chunking for poor/mixed formatting; sentence-aware. Uses CHUNK_SIZE/CHUNK_OVERLAP from config."""
    text = (text or "").strip()
    if not text:
        return []
    csize, coverlap = _get_chunk_size_overlap()
    sentences = _sentences(text)
    overlap_sentences = 1 if coverlap < 120 else 2
    chunk_texts = _group_sentences_to_size(
        sentences,
        target_min=csize // 2,
        target_max=csize,
        overlap_sentences=overlap_sentences,
    )
    if not chunk_texts:
        chunk_texts = [_truncate_to_tokens(text, csize * 2)]
    return [
        Chunk(
            doc_id="",
            chunk_id="",
            text=ct,
            doc_type=DocType.USER,
            sensitivity_level=sensitivity_level,
            redacted=redacted,
            section=None,
            parent_chunk_id=None,
            is_parent=False,
            chunk_index=i,
        )
        for i, ct in enumerate(chunk_texts)
    ]
