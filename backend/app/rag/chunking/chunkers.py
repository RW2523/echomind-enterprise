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
        size = getattr(settings, "CHUNK_SIZE", 650) or 650
        overlap = getattr(settings, "CHUNK_OVERLAP", 120) or 120
        return (max(200, min(1200, size)), max(40, min(200, overlap)))
    except Exception:
        return (650, 120)


def _get_book_limits() -> tuple[int, int, int, int]:
    """Return (parent_min, parent_max, child_min, child_max) from config or module defaults."""
    try:
        from ...core.config import settings
        return (
            getattr(settings, "BOOK_PARENT_MIN_TOKENS", _PARENT_MIN) or _PARENT_MIN,
            getattr(settings, "BOOK_PARENT_MAX_TOKENS", _PARENT_MAX) or _PARENT_MAX,
            getattr(settings, "BOOK_CHILD_MIN_TOKENS", _CHILD_MIN) or _CHILD_MIN,
            getattr(settings, "BOOK_CHILD_MAX_TOKENS", _CHILD_MAX) or _CHILD_MAX,
        )
    except Exception:
        return (_PARENT_MIN, _PARENT_MAX, _CHILD_MIN, _CHILD_MAX)

# Sentence boundary (naive); safer split uses _sentences() with abbrev/citation heuristics
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n\n+")
# Min/max child size for long-form (child size follows config when used in unstructured path)
_PARENT_MIN, _PARENT_MAX = 2000, 3500
_CHILD_MIN, _CHILD_MAX = 500, 700
_CHILD_OVERLAP_SENTENCES = 3  # 3 sentences ≈ 80-150 tokens depending on sentence length
# Sensitive: small, minimal overlap
_SENSITIVE_SIZE = 450
_SENSITIVE_OVERLAP = 40
# Unstructured (user): use config defaults (600/80) for snappier retrieval
_USER_SIZE = 600
_USER_OVERLAP = 80
# FAQ: whole Q+A per chunk; cap by tokens so embedding API context is not exceeded (constants = tokens)
MAX_FAQ_TOKENS = 8000


def _estimate_content_density(text: str) -> float:
    """
    Estimate content density (0.0–1.0) for dynamic chunk sizing.
    Dense technical content (numbers, abbreviations, legal references) → smaller chunks.
    Narrative/discursive text → larger chunks.
    """
    if not (text or "").strip():
        return 0.5
    sample = text[:4000]
    chars = len(sample)
    if chars == 0:
        return 0.5
    num_count = len(re.findall(r"\d+(?:\.\d+)?", sample))
    abbrev_count = len(re.findall(r"\b[A-Z]{2,6}\b", sample))
    ref_count = len(re.findall(r"(?:§|section|article|paragraph|table|figure|appendix)\s*\d", sample, re.I))
    paren_count = sample.count("(") + sample.count(")")
    density_signal = (num_count * 2 + abbrev_count * 3 + ref_count * 5 + paren_count) / max(1, chars)
    return min(1.0, density_signal * 80)


def _dynamic_child_range(text: str) -> Tuple[int, int]:
    """
    Return (child_min, child_max) adjusted by content density.
    Dense technical content → smaller chunks (500–600 tokens).
    Narrative content → larger chunks (550–700 tokens).
    """
    density = _estimate_content_density(text)
    if density > 0.6:
        return (500, 600)
    if density > 0.3:
        return (500, 650)
    return (550, 700)


# --- 2. Safer sentence segmentation (no mid-sentence splits; avoid abbrev/citation false splits) ---
def _ends_with_abbrev(s: str) -> bool:
    """True if s ends with common abbreviation (Dr., e.g., Fig. 1, Sec., DoD, etc.) so we should not split after it."""
    if not s or len(s) < 3:
        return False
    s = s.rstrip()
    if not s:
        return False
    if re.search(
        r"\s(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|e\.g|i\.e|al|Fig|Vol|No|approx|Sec|Art|Par|Dept|Div|Reg|Rev|Chap|App|Ref|DoD|U\.S)\.\s*$",
        s,
        re.I,
    ):
        return True
    if re.search(r"[A-Z]\.\s*$", s):
        return True
    if re.search(r"\[\d+\]\s*$|\d+\)\s*$", s):
        return True
    # DoD-style: ends with a section/paragraph reference number like "0101." or "2.3.4."
    if re.search(r"\d{2,6}\.\s*$", s):
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
# Patterns for section headings (capture title, then text until next heading or end).
# Priority order: DoD numbered paragraphs → Chapter/Part/Volume → deep numbered → markdown.
_CHAPTER_RE = re.compile(
    r"(?m)^\s*(?:"
    r"Chapter\s+\d+(?:\s*[-:]\s*)?|"
    r"Part\s+[IVXLCDM\d]+\s*[-:]\s*|"
    r"Volume\s+\d+\s*[-:]\s*|"
    r"Appendix\s+[A-Z\d]+\s*[-:]\s*|"
    r"Article\s+\d+\s*[-:]\s*"
    r")(.+?)\s*$",
    re.IGNORECASE,
)
_DOD_NUMBERED_RE = re.compile(
    r"(?m)^\s*(\d{4,6})\s+([A-Z][A-Z\s,/&-]{2,80})\s*$"
)
_SECTION_LABEL_RE = re.compile(
    r"(?m)^\s*(?:Section|Sec\.?)\s+([\d.]+)\s*[-:]?\s*(.+?)\s*$",
    re.IGNORECASE,
)
_SECTION_MARKDOWN_RE = re.compile(r"(?m)^\s*#{1,4}\s+(.+?)\s*$")
_DEEP_NUM_RE = re.compile(
    r"(?m)^\s*(\d+(?:\.\d+){1,4})\s+(.+?)\s*$"
)


def _split_sections_from_matches(
    text: str,
    matches: list,
    title_group: int = 1,
) -> List[Tuple[str, str]]:
    """Build (title, body) pairs from regex match objects."""
    sections: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        title = m.group(title_group).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


def _split_book_into_sections(text: str) -> List[Tuple[Optional[str], str]]:
    """
    Detect chapter/section headings and split book into (section_title, section_text) pairs.
    Supports DoD-style numbered paragraphs (0101 GENERAL), Chapter/Part/Volume/Appendix/Article,
    Section X.Y, deep numbered sections (1.2.3.4), and markdown headings.
    If no structure is detected, falls back to logical paragraph segmentation for very large texts.
    """
    text = (text or "").strip()
    if not text:
        return []

    # 1. DoD-style: "0101 GENERAL", "010201 PURPOSE" (4–6 digit code + ALL-CAPS title)
    dod_matches = list(_DOD_NUMBERED_RE.finditer(text))
    if len(dod_matches) >= 3:
        sections = _split_sections_from_matches(text, dod_matches, title_group=0)
        if sections:
            return sections

    # 2. Chapter/Part/Volume/Appendix/Article
    # title_group=0 keeps the heading LABEL ("Chapter 4 Travel Reimbursement"). _CHAPTER_RE
    # puts the label in a non-capturing group, so the default title_group=1 yields only the
    # trailing title ("Travel Reimbursement") — which _is_valid_book_section_path rejects,
    # so every chapter-structured book silently fell through to flat chunk_unstructured.
    chapters = list(_CHAPTER_RE.finditer(text))
    if chapters:
        sections = _split_sections_from_matches(text, chapters, title_group=0)
        if sections:
            return sections

    # 3. "Section 0301" / "Sec. 2.3"
    sec_labels = list(_SECTION_LABEL_RE.finditer(text))
    if len(sec_labels) >= 2:
        sections = _split_sections_from_matches(text, sec_labels, title_group=0)
        if sections:
            return sections

    # 4. Deep numbered: "1.2.3 Title"
    deep_nums = list(_DEEP_NUM_RE.finditer(text))
    if len(deep_nums) >= 3:
        sections = _split_sections_from_matches(text, deep_nums, title_group=0)
        if sections:
            return sections

    # 5. Markdown ## Heading
    md_heads = list(_SECTION_MARKDOWN_RE.finditer(text))
    if md_heads:
        sections = _split_sections_from_matches(text, md_heads)
        if sections:
            return sections

    # 6. Fallback for very large unstructured text: split into ~10k-char logical segments at paragraph boundaries
    if len(text) > 50_000:
        return _fallback_paragraph_segments(text, target_segment_chars=10_000)

    return [(None, text)]


def _fallback_paragraph_segments(text: str, target_segment_chars: int = 10_000) -> List[Tuple[Optional[str], str]]:
    """Split large text without headings into segments at paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [(None, text)]
    segments: List[Tuple[Optional[str], str]] = []
    current: List[str] = []
    current_len = 0
    seg_idx = 1
    for p in paragraphs:
        plen = len(p)
        if current_len + plen > target_segment_chars and current:
            segments.append((f"Segment {seg_idx}", "\n\n".join(current)))
            seg_idx += 1
            current = []
            current_len = 0
        current.append(p)
        current_len += plen
    if current:
        segments.append((f"Segment {seg_idx}", "\n\n".join(current)))
    return segments


def _get_large_section_threshold() -> int:
    """Token threshold for splitting large sections. Default 10000."""
    try:
        from ...core.config import settings
        return int(getattr(settings, "LARGE_SECTION_TOKEN_THRESHOLD", 10000))
    except Exception:
        return 10000


def handle_large_sections(
    section_text: str,
    max_tokens_per_chunk: Optional[int] = None,
) -> List[str]:
    """
    Split large sections into multiple chunks for consistent contextual retrieval.

    When section_text exceeds the threshold (~10k tokens by default), splits at
    paragraph boundaries into chunks of max_tokens_per_chunk (default: parent max).
    Ensures each chunk retains coherent context and cross-references stay accurate.
    """
    text = (section_text or "").strip()
    if not text:
        return []
    tokens = token_len(text)
    threshold = _get_large_section_threshold()
    if tokens <= threshold:
        return [text]
    cfg_p_min, cfg_p_max, _, _ = _get_book_limits()
    max_tok = max_tokens_per_chunk or cfg_p_max
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text]
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0
    sep_tokens = token_len("\n\n")
    for p in paragraphs:
        p_tokens = token_len(p)
        if current_tokens + sep_tokens + p_tokens > max_tok and current:
            chunks.append("\n\n".join(current))
            overlap = [current[-1]] if len(current) >= 1 else []
            current = overlap + [p] if p_tokens <= max_tok else [p]
            current_tokens = token_len("\n\n".join(current))
        else:
            current.append(p)
            current_tokens = (current_tokens + sep_tokens + p_tokens) if current_tokens else p_tokens
    if current:
        chunks.append("\n\n".join(current))
    return chunks if chunks else [text]


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
    # SPLIT oversized blocks instead of truncating them. The old code replaced an
    # over-long block with its first MAX_FAQ_TOKENS tokens and dropped the rest —
    # silent data loss: the tail was never chunked, embedded, or indexed, while the
    # upload still reported success. A misdetected 31-page PDF lost 53% of its text.
    parts = [
        b
        for p in parts
        for b in (
            _group_sentences_to_size(_sentences(p), MAX_FAQ_TOKENS // 2, MAX_FAQ_TOKENS)
            if token_len(p) > MAX_FAQ_TOKENS
            else [p]
        )
    ]

    idx = 0
    for block in parts:
        block = block.strip()
        if not block or token_len(block) < 5:
            continue
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
    Larger chunks get more overlap; keep sentence-boundary safety. Range 2–4.
    Targets ~80-150 tokens of overlap (2-4 sentences).
    """
    return max(2, min(4, (chunk_max_tokens // 150)))


def chunk_long_form(
    text: str,
    sensitivity_level: SensitivityLevel,
    redacted: bool,
    section: Optional[str] = None,
    section_path: Optional[str] = None,
) -> List[ParentChildChunk]:
    """
    Parent chunks (token-sized _PARENT_MIN–_PARENT_MAX) for context; child chunks for retrieval.
    Children reference parent_chunk_id. Section/section_path metadata set when provided.
    Child chunk sizes are dynamically adjusted based on content density.
    """
    text = (text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    cfg_p_min, cfg_p_max, cfg_c_min, cfg_c_max = _get_book_limits()

    parent_chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for p in paragraphs:
        p_tokens = token_len(p)
        sep_tokens = token_len("\n\n") if current else 0
        if current_tokens + sep_tokens + p_tokens > cfg_p_max and current:
            parent_chunks.append("\n\n".join(current))
            overlap_paras = [current[-1]] if len(current) >= 1 else []
            current = overlap_paras + [p] if p_tokens <= cfg_p_max else [p]
            current_tokens = token_len("\n\n".join(current))
        else:
            current.append(p)
            current_tokens += (sep_tokens + p_tokens) if current_tokens else p_tokens
    if current:
        parent_chunks.append("\n\n".join(current))

    child_min_dyn, child_max_dyn = _dynamic_child_range(text)
    child_min = max(cfg_c_min, child_min_dyn)
    child_max = min(cfg_c_max, child_max_dyn) if child_max_dyn <= cfg_c_max else cfg_c_max
    overlap_sentences = _adaptive_overlap_sentences(child_max)

    out: List[ParentChildChunk] = []
    for pi, parent_text in enumerate(parent_chunks):
        if token_len(parent_text) < cfg_p_min and pi < len(parent_chunks) - 1:
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
            section_path=section_path,
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
                section_path=section_path,
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
