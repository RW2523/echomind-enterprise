"""
Session state for real-time transcription: stabilized transcript, anti-duplication,
paragraph segmentation. Used by the /ws handler.
"""
from __future__ import annotations
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from ..core.config import settings

# Commit when recent_buffer ends with strong punctuation
PUNCT_END = re.compile(r".*[.?!]\s*$")
# No space before punctuation
NO_SPACE_BEFORE = re.compile(r"\s+([.,?!:;)])\s*")


@dataclass
class Paragraph:
    """One segment of the transcript (for lectures/meetings)."""
    paragraph_id: str
    raw_text: str
    start_ts: float
    end_ts: float
    char_count: int
    polished_text: Optional[str] = None
    tags: List[str] = field(default_factory=list)


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces, no space before punctuation."""
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    t = NO_SPACE_BEFORE.sub(r"\1", t)
    return t


# def _normalize_piece(piece: str) -> str:
#     """Normalize STT piece but preserve leading space (SentencePiece ▁ word boundary)."""
#     if not piece:
#         return ""
#     # Collapse internal multiple spaces, fix no-space-before-punct
#     t = re.sub(r"\s+", " ", piece)
#     t = NO_SPACE_BEFORE.sub(r"\1", t)
#     # Strip trailing but preserve leading (word boundary)
#     return t.rstrip() if t.startswith(" ") else t.strip()

def _normalize_piece(piece: str) -> str:
    if not piece:
        return ""
    # convert SentencePiece marker if it appears
    piece = piece.replace("▁", " ")
    # normalize internal whitespace and punctuation spacing
    t = re.sub(r"\s+", " ", piece)
    t = NO_SPACE_BEFORE.sub(r"\1", t)
    # only remove trailing whitespace; keep leading if present
    return t.rstrip()

def _max_suffix_prefix_overlap(tail: str, incoming: str, k: int) -> int:
    """Return length of maximum overlap (suffix of tail == prefix of incoming) up to k chars."""
    if not tail or not incoming or k <= 0:
        return 0
    n = min(k, len(tail), len(incoming))
    for L in range(n, 0, -1):
        if tail[-L:] == incoming[:L]:
            return L
    return 0


class SessionState:
    """
    Maintains raw transcript, recent buffer, last emit, and segments.
    - append_piece(piece, ts_ms): add STT piece with anti-dup, normalize, maybe commit.
    - get_display_text(): raw_text + recent_buffer for client.
    - maybe_commit(ts_ms): commit buffer into raw_text on punctuation/silence/length.
    - maybe_new_paragraph(ts_ms): close current paragraph and start new one if rules met.
    - finalize(): flush buffer and close last paragraph.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.raw_text = ""
        self.recent_buffer = ""
        self.last_emit_text = ""
        self.last_piece_ts_ms: Optional[int] = None
        self.segments: List[Paragraph] = []
        self._current_paragraph_start_ts: Optional[float] = None
        self._current_paragraph_start_index: int = 0  # index into raw_text where current paragraph started
        self._paragraph_counter = 0
        self._paused = False

        self.silence_commit_ms = settings.TRANSCRIPT_SILENCE_COMMIT_MS
        self.paragraph_silence_ms = settings.TRANSCRIPT_PARAGRAPH_SILENCE_MS
        self.max_paragraph_chars = settings.TRANSCRIPT_MAX_PARAGRAPH_CHARS
        self.buffer_max_chars = settings.TRANSCRIPT_RECENT_BUFFER_MAX_CHARS
        self.overlap_k = settings.TRANSCRIPT_OVERLAP_K
        self._continuation_expected = False
        self._last_piece_norm = ""

    def _next_paragraph_id(self) -> str:
        self._paragraph_counter += 1
        return f"p{self._paragraph_counter}"

    def _close_current_paragraph(self, end_ts: float) -> Optional[Paragraph]:
        """Close current paragraph if any text; return it."""
        sep = " " if self.raw_text and self.recent_buffer and not self.raw_text.endswith(" ") and not self.recent_buffer.startswith(" ") else ""
        current_text = (self.raw_text + sep + self.recent_buffer).strip()
        segment_text = current_text[self._current_paragraph_start_index:].strip()
        if not segment_text:
            return None
        pid = self._next_paragraph_id()
        p = Paragraph(
            paragraph_id=pid,
            raw_text=segment_text,
            start_ts=self._current_paragraph_start_ts or 0.0,
            end_ts=end_ts,
            char_count=len(segment_text),
        )
        self.segments.append(p)
        self._current_paragraph_start_index = len(current_text)
        self._current_paragraph_start_ts = None
        return p

    def append_piece(self, piece: str, ts_ms: int) -> None:
        """Append a text piece from STT with anti-duplication and whitespace normalization."""
        if self._paused or not piece:
            return
        piece = _normalize_piece(piece)
        if not piece:
            return
        #tail = (self.raw_text + self.recent_buffer).strip()
        tail = (self.raw_text + self.recent_buffer)
        # Skip exact duplicate: if incoming piece is identical to end of tail, do not append
        if tail and len(piece) <= len(tail) and tail.endswith(piece):
            self.last_piece_ts_ms = ts_ms
            return
        # Anti-duplication: max suffix/prefix overlap between existing tail and incoming
        overlap = _max_suffix_prefix_overlap(tail, piece, self.overlap_k)
        if overlap > 0:
            piece = piece[overlap:]
        if not piece:
            self.last_piece_ts_ms = ts_ms
            return
        # Pieces from STT already include word-boundary spaces (▁→" "); concatenate directly.
        # self.recent_buffer += piece
        # self.last_piece_ts_ms = ts_ms
        
        # --- boundary-aware merge for Kyutai subword streaming ---

        prev_text = self.recent_buffer if self.recent_buffer else self.raw_text
        prev_last = prev_text[-1] if prev_text else ""

        starts_new_word = piece.startswith(" ")
        is_punct = bool(re.match(r"^[.,?!:;)]+", piece.strip()))

        # Heuristic: very short alphabetic tokens often indicate we are mid-word ("te"+"rrible", "Ja"+"ke", "Le"+"a"+"ve")
        # This helps us avoid inserting spaces inside words.
        def looks_like_prefix(tok: str) -> bool:
            t = tok.strip()
            return t.isalpha() and len(t) <= 2  # keep conservative: 1–2 letters

        # Decide whether to insert a missing space:
        # Insert space when:
        # - previous ends with alnum
        # - incoming starts with alnum
        # - incoming does NOT start with space
        # - we are NOT expecting a continuation
        should_insert_space = (
            prev_text
            and prev_last.isalnum()
            and piece
            and piece[0].isalnum()
            and not starts_new_word
            and not self._continuation_expected
            and not is_punct
        )

        if should_insert_space:
            self.recent_buffer += " "

        # Append token
        self.recent_buffer += piece

        # Update continuation state for next token
        # If token explicitly starts a new word, next token is NOT automatically a continuation.
        if starts_new_word or is_punct:
            self._continuation_expected = False
        else:
            # If this token looks like a prefix (1–2 letters), expect continuation next.
            # Otherwise, default to not expecting continuation.
            self._continuation_expected = looks_like_prefix(piece)

        self._last_piece_norm = piece
        self.last_piece_ts_ms = ts_ms

    def get_display_text(self) -> str:
        """Full text to send to client (raw + recent buffer)."""
        if not self.recent_buffer:
            return self.raw_text.strip()
        r = self.raw_text.strip()
        if r:
            sep = " " if not r.endswith(" ") and not self.recent_buffer.startswith(" ") else ""
            return r + sep + self.recent_buffer
        return self.recent_buffer

    def maybe_commit(self, ts_ms: int) -> bool:
        """
        Commit recent_buffer into raw_text when: ends with .?! or silence > threshold or buffer too long.
        Returns True if a commit happened.
        """
        if not self.recent_buffer.strip():
            return False
        now = ts_ms
        silence_gap = (now - self.last_piece_ts_ms) if self.last_piece_ts_ms is not None else 0
        should_commit = (
            bool(PUNCT_END.search(self.recent_buffer))
            or silence_gap >= self.silence_commit_ms
            or len(self.recent_buffer) >= self.buffer_max_chars
        )
        if not should_commit:
            return False
        # Commit (recent_buffer may already start with space from ▁)
        sep = ""
        if self.raw_text and not self.raw_text.endswith(" ") and self.recent_buffer and not self.recent_buffer.startswith(" "):
            sep = " "
        self.raw_text += sep + self.recent_buffer
        self.recent_buffer = ""
        return True

    def maybe_new_paragraph(self, ts_ms: int) -> Optional[Paragraph]:
        """
        If we have committed text and (strong punct + silence) or paragraph too long, close paragraph and start new.
        Returns new Paragraph if one was closed.
        """
        sep = " " if self.raw_text and self.recent_buffer and not self.raw_text.endswith(" ") and not self.recent_buffer.startswith(" ") else ""
        current_full = (self.raw_text + sep + self.recent_buffer).strip()
        segment_so_far = current_full[self._current_paragraph_start_index:].strip()
        if not segment_so_far:
            return None
        if self._current_paragraph_start_ts is None:
            self._current_paragraph_start_ts = ts_ms / 1000.0
        now = ts_ms
        silence_gap = (now - self.last_piece_ts_ms) if self.last_piece_ts_ms is not None else 0
        ends_strong = bool(PUNCT_END.search(current_full))
        over_length = len(segment_so_far) >= self.max_paragraph_chars
        if (ends_strong and silence_gap >= self.paragraph_silence_ms) or over_length:
            return self._close_current_paragraph(now / 1000.0)
        return None

    def finalize(self) -> None:
        """Flush buffer into raw_text and close last paragraph."""
        if self.recent_buffer.strip():
            sep = ""
            if self.raw_text and not self.raw_text.endswith(" ") and self.recent_buffer and not self.recent_buffer.startswith(" "):
                sep = " "
            self.raw_text += sep + self.recent_buffer
            self.recent_buffer = ""
        end_ts = time.time()
        sep = " " if self.raw_text and self.recent_buffer and not self.raw_text.endswith(" ") and not self.recent_buffer.startswith(" ") else ""
        current_full = (self.raw_text + sep + self.recent_buffer).strip()
        segment_so_far = current_full[self._current_paragraph_start_index:].strip()
        if segment_so_far:
            self._close_current_paragraph(end_ts)

    def differs_from_last_emit(self) -> bool:
        """True if current display text differs meaningfully from last_emit_text."""
        current = self.get_display_text()
        return current != self.last_emit_text

    def mark_emitted(self) -> None:
        """Call after sending partial to client."""
        self.last_emit_text = self.get_display_text()

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False
