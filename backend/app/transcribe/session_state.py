"""
Session state for real-time transcription: stabilization, concatenation, paragraph segmentation.
Designed for streaming STT pieces (e.g. SentencePiece tokens or Whisper partials).
"""
from __future__ import annotations
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from ..core.config import settings


# -----------------------------------------------------------------------------
# Paragraph (one segment of the transcript)
# -----------------------------------------------------------------------------
@dataclass
class Paragraph:
    paragraph_id: str
    raw_text: str
    start_ts: float
    end_ts: float
    char_count: int
    polished_text: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "paragraph_id": self.paragraph_id,
            "raw_text": self.raw_text,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "char_count": self.char_count,
            "polished_text": self.polished_text,
            "tags": self.tags,
        }


# -----------------------------------------------------------------------------
# Normalize whitespace for display (collapse spaces, no space before punctuation)
# -----------------------------------------------------------------------------
def _normalize_whitespace(text: str) -> str:
    if not text or not text.strip():
        return ""
    # Collapse multiple spaces/newlines to single space
    t = re.sub(r"\s+", " ", text.strip())
    # No space before punctuation
    t = re.sub(r"\s+([.,?!:;)]+)", r"\1", t)
    # Ensure space after sentence-ending when followed by word (optional)
    t = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", t)
    return t


def _ends_with_strong_punctuation(text: str) -> bool:
    return bool(text and text.strip() and re.search(r"[.?!]\s*$", text.strip()))


# -----------------------------------------------------------------------------
# Maximum suffix/prefix overlap to avoid duplicating repeated phrases
# -----------------------------------------------------------------------------
def _max_overlap_len(tail: str, incoming: str, max_k: int) -> int:
    """Return length of overlapping prefix of incoming with suffix of tail (up to max_k chars)."""
    if not tail or not incoming or max_k <= 0:
        return 0
    tail = tail[-max_k:] if len(tail) > max_k else tail
    for n in range(min(len(tail), len(incoming), max_k), 0, -1):
        if tail[-n:] == incoming[:n]:
            return n
    return 0


# -----------------------------------------------------------------------------
# SessionState: stabilized transcript + segments
# -----------------------------------------------------------------------------
class SessionState:
    """
    Maintains:
    - raw_text: committed transcript so far
    - recent_buffer: last N chars not yet committed
    - last_emit_text: last partial we sent (for rate-limit and diff)
    - last_piece_ts: timestamp of last piece (silence detection)
    - segments: list of Paragraph
    """

    def __init__(self, session_id: str, paragraph_id_prefix: str = "p"):
        self.session_id = session_id
        self.paragraph_id_prefix = paragraph_id_prefix
        self._raw_text = ""
        self._recent_buffer = ""
        self._last_emit_text = ""
        self._last_piece_ts: Optional[float] = None
        self._segment_counter = 0
        self.segments: List[Paragraph] = []
        self._current_paragraph_start_ts: Optional[float] = None
        self._current_paragraph_start_index: int = 0  # index into _raw_text where current para started
        self._paused = False

        self.silence_commit_ms = settings.TRANSCRIPT_SILENCE_COMMIT_MS
        self.paragraph_silence_ms = settings.TRANSCRIPT_PARAGRAPH_SILENCE_MS
        self.max_buffer_chars = settings.TRANSCRIPT_MAX_BUFFER_CHARS
        self.max_paragraph_chars = settings.TRANSCRIPT_MAX_PARAGRAPH_CHARS
        self.overlap_guard_chars = settings.TRANSCRIPT_OVERLAP_GUARD_CHARS

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def is_paused(self) -> bool:
        return self._paused

    def append_piece(self, piece: str, ts_ms: Optional[float] = None) -> None:
        """Append a new text piece (from STT). Normalize, anti-dupe, add to recent_buffer."""
        if self._paused or not piece:
            return
        piece = _normalize_whitespace(piece)
        if not piece:
            return
        now = ts_ms / 1000.0 if ts_ms is not None else time.time()
        self._last_piece_ts = now

        # Anti-duplication: only append non-overlapping remainder
        tail = self._raw_text + self._recent_buffer
        overlap = _max_overlap_len(tail, piece, self.overlap_guard_chars)
        if overlap > 0:
            piece = piece[overlap:].lstrip()
        if not piece:
            return

        self._recent_buffer = (self._recent_buffer + " " + piece).strip() if self._recent_buffer else piece

    def get_display_text(self) -> str:
        """Full text to show (raw + recent buffer)."""
        if not self._recent_buffer:
            return self._raw_text
        return (self._raw_text + " " + self._recent_buffer).strip()

    def _commit_buffer_to_raw(self, ts_ms: Optional[float] = None) -> bool:
        """Move recent_buffer into raw_text. Returns True if something was committed."""
        if not self._recent_buffer.strip():
            return False
        self._raw_text = (self._raw_text + " " + self._recent_buffer).strip() if self._raw_text else self._recent_buffer
        self._recent_buffer = ""
        return True

    def maybe_commit(self, ts_ms: Optional[float] = None) -> bool:
        """
        Commit recent_buffer into raw_text when:
        - buffer ends with .?!
        - OR silence gap > SILENCE_COMMIT_MS
        - OR buffer length > threshold
        Returns True if a commit happened.
        """
        if not self._recent_buffer.strip():
            return False
        now_ms = ts_ms if ts_ms is not None else (time.time() * 1000)
        now_sec = now_ms / 1000.0
        silence_gap_ms = (now_sec - self._last_piece_ts) * 1000 if self._last_piece_ts is not None else 0

        should_commit = (
            _ends_with_strong_punctuation(self._recent_buffer)
            or silence_gap_ms >= self.silence_commit_ms
            or len(self._recent_buffer) >= self.max_buffer_chars
        )
        if should_commit:
            return self._commit_buffer_to_raw(ts_ms)
        return False

    def maybe_new_paragraph(self, ts_ms: Optional[float] = None) -> Optional[Paragraph]:
        """
        Start a new paragraph when:
        - committed text ends with strong punctuation AND silence > PARAGRAPH_SILENCE_MS
        - OR current paragraph exceeds MAX_PARAGRAPH_CHARS
        Returns the closed Paragraph if one was created.
        """
        now_ms = ts_ms if ts_ms is not None else (time.time() * 1000)
        now_sec = now_ms / 1000.0
        silence_gap_ms = (now_sec - self._last_piece_ts) * 1000 if self._last_piece_ts is not None else 0

        current_text = self._raw_text[self._current_paragraph_start_index :].strip()

        if not current_text:
            return None

        start_ts = self._current_paragraph_start_ts or now_sec
        should_close = (
            _ends_with_strong_punctuation(self._raw_text) and silence_gap_ms >= self.paragraph_silence_ms
        ) or len(current_text) >= self.max_paragraph_chars

        if should_close:
            self._segment_counter += 1
            pid = f"{self.paragraph_id_prefix}{self._segment_counter}"
            para = Paragraph(
                paragraph_id=pid,
                raw_text=current_text,
                start_ts=start_ts,
                end_ts=now_sec,
                char_count=len(current_text),
            )
            self.segments.append(para)
            self._current_paragraph_start_ts = now_sec
            self._current_paragraph_start_index = len(self._raw_text)
            return para
        return None

    def finalize(self) -> None:
        """Flush buffer into raw_text and close final paragraph."""
        self._commit_buffer_to_raw()
        current_text = self._raw_text[self._current_paragraph_start_index :].strip()
        if current_text:
            now = time.time()
            self._segment_counter += 1
            pid = f"{self.paragraph_id_prefix}{self._segment_counter}"
            start_ts = self._current_paragraph_start_ts or now
            self.segments.append(
                Paragraph(
                    paragraph_id=pid,
                    raw_text=current_text,
                    start_ts=start_ts,
                    end_ts=now,
                    char_count=len(current_text),
                )
            )
        self._current_paragraph_start_ts = None
        self._current_paragraph_start_index = len(self._raw_text)
        self._recent_buffer = ""

    def get_segments_for_emit(self) -> List[dict]:
        """Return segments as list of dicts for WS events."""
        return [s.to_dict() for s in self.segments]

    def should_emit_partial(self, rate_limit_ts: float, min_interval: float) -> bool:
        """True if we should send a partial (differs from last_emit and rate limit allows)."""
        display = self.get_display_text()
        if display == self._last_emit_text:
            return False
        return rate_limit_ts >= min_interval

    def mark_partial_emitted(self) -> None:
        self._last_emit_text = self.get_display_text()

    def get_paragraph_by_id(self, paragraph_id: str) -> Optional[Paragraph]:
        for p in self.segments:
            if p.paragraph_id == paragraph_id:
                return p
        return None

    def get_last_paragraph(self) -> Optional[Paragraph]:
        return self.segments[-1] if self.segments else None
