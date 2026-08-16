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
    role: Optional[str] = None
    sentences: List[dict] = field(default_factory=list)   # [{sentence_id,text,char_start,char_end,role}]


# Sentence splitter for live text: split after . ! ? followed by whitespace, protecting common
# abbreviations and decimals ("3.5 percent", "Mr. Rao", "U.S.C. 402").
_ABBREV = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|No|vs|etc|Inc|Ltd|Co|U\.S|U\.S\.C|e\.g|i\.e)\.", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


# Streaming STT (Nemotron cache-aware) terminates almost every ~560 ms chunk hypothesis with a
# period and capitalizes the next chunk, so naive splitting yields fragments like
# "The early termination fee is one hundred." / "Fifty dollars unless you're within the fourteen."
# A fragment is merged with its neighbour when it obviously cannot stand alone.
_DANGLING_END = {
    "a", "an", "the", "my", "your", "his", "her", "their", "our", "its", "this", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "am", "to", "of", "in", "on", "at", "for", "with",
    "by", "from", "and", "or", "but", "nor", "unless", "until", "even", "after", "before", "than", "that",
    "which", "who", "whom", "whose", "about", "into", "onto", "upon", "over", "under", "within", "without",
    "because", "if", "when", "while", "where", "whether", "should", "could", "would", "can", "will", "may",
    "might", "must", "have", "has", "had", "do", "does", "did", "not", "very", "so", "too", "also", "just",
    "as", "per", "versus", "vs", "between", "through", "during", "against", "toward", "towards", "like",
    "hundred", "thousand", "million", "billion", "point", "double", "triple", "dot", "slash", "dash",
    "number", "id", "mr", "mrs", "ms", "dr",
}
_CONTINUATION_START = {
    "and", "or", "but", "nor", "unless", "until", "even", "than",
    "that", "which", "who", "whom", "whose", "as",
    "with", "without", "for", "of", "to", "in", "on", "at", "by", "from", "into",
    "onto", "over", "under", "within", "about", "between", "through", "during", "against", "per", "id", "number",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero", "oh", "ten", "eleven",
    "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "point", "double", "triple", "dollars", "percent", "days", "weeks", "months", "years", "day", "week",
    "month", "year", "cents", "pounds", "euros", "rupees", "lakh", "crore",
}
_MIN_SENTENCE_WORDS = 5
_MAX_MERGED_WORDS = 45


def _words(t: str):
    return re.findall(r"[A-Za-z0-9']+", t)


def _fragment_incomplete(a: str) -> bool:
    """Would a reader consider `a` an unfinished sentence?"""
    w = _words(a)
    if not w:
        return True
    last = w[-1].lower()
    if last in _DANGLING_END:
        return True
    if len(w) < _MIN_SENTENCE_WORDS and not a.rstrip().endswith("?"):
        return True
    return False


def _fragment_continues(b: str) -> bool:
    """Does `b` look like the continuation of the previous fragment?"""
    w = _words(b)
    if not w:
        return False
    first = w[0]
    if first[0].islower():
        return True
    if first.lower() in _CONTINUATION_START:
        return True
    if first.isdigit():
        return True
    return False


_MID_PERIOD = re.compile(r"(\w)\.\s+([A-Za-z0-9']+)")


def clean_merged_sentence(text: str) -> str:
    """Remove the STT chunk-boundary periods inside a merged sentence:
    'fee is one hundred. Fifty dollars' -> 'fee is one hundred fifty dollars' — only where the
    word before the period is a dangling word or the word after is a continuation word."""
    def _fix(m):
        before = m.group(1); after = m.group(2)
        # find the full word before
        return m.group(0)
    out = text
    # iterate over candidate boundaries
    def repl(m):
        start = m.start()
        # word before the period
        wb = re.search(r"([A-Za-z0-9']+)\.$", out[: m.start(2)].rstrip())
        word_before = wb.group(1).lower() if wb else ""
        word_after = m.group(2)
        if word_before in _DANGLING_END or word_after.lower() in _CONTINUATION_START or word_after.isdigit():
            wa = (word_after[0].lower() + word_after[1:]) if (word_after.lower() in _CONTINUATION_START and not word_after.isupper()) else word_after
            return f"{m.group(1)} {wa}"
        return m.group(0)
    # `sub` consumes the after-word, so a chain like "number. Seventeen. Of" needs another pass
    for _ in range(4):
        new = _MID_PERIOD.sub(repl, out)
        if new == out:
            break
        out = new
    return out


def should_merge_fragments(a: str, b: str) -> bool:
    if len(_words(a)) + len(_words(b)) > _MAX_MERGED_WORDS:
        return False
    return _fragment_incomplete(a) or _fragment_continues(b)


def split_live_sentences(text: str):
    """Yield (start, end) char spans of complete sentences in text (last dangling fragment
    excluded unless it ends with terminal punctuation)."""
    if not text or not text.strip():
        return []
    protected = _ABBREV.sub(lambda m: m.group(0).replace(".", "\x00"), text)
    protected = re.sub(r"(\d)\.(\d)", lambda m: m.group(1) + "\x00" + m.group(2), protected)
    spans = []
    start = 0
    for m in _SENT_SPLIT.finditer(protected):
        end = m.start()
        if end > start:
            spans.append((start, end))
        start = m.end()
    tail = protected[start:]
    if tail.strip() and re.search(r"[.!?][\"')]*\s*$", tail):
        spans.append((start, len(protected)))
    return spans


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces, no space before punctuation."""
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    t = NO_SPACE_BEFORE.sub(r"\1", t)
    return t


def _normalize_piece(piece: str) -> str:
    """Normalize STT piece but preserve leading space (SentencePiece ▁ word boundary)."""
    if not piece:
        return ""
    # Collapse internal multiple spaces, fix no-space-before-punct
    t = re.sub(r"\s+", " ", piece)
    t = NO_SPACE_BEFORE.sub(r"\1", t)
    # Strip trailing but preserve leading (word boundary)
    return t.rstrip() if t.startswith(" ") else t.strip()


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
        self.prev_piece_ts_ms: Optional[int] = None  # ts of the piece before the latest (for real silence gaps) (M12)
        self.segments: List[Paragraph] = []
        self._current_paragraph_start_ts: Optional[float] = None
        self._current_paragraph_start_index: int = 0  # index into raw_text where current paragraph started
        self._paragraph_counter = 0
        self._paused = False
        # Silent Assistant v2: sentence cursor within the CURRENT paragraph (committed text only)
        self._sentence_cursor: int = 0            # char offset (relative to paragraph start) already emitted as sentences
        self._sentence_counter: int = 0
        self._current_paragraph_sentences: List[dict] = []
        self.current_role: Optional[str] = None
        self.last_piece_wall: float = 0.0

        self.silence_commit_ms = settings.TRANSCRIPT_SILENCE_COMMIT_MS
        self.paragraph_silence_ms = settings.TRANSCRIPT_PARAGRAPH_SILENCE_MS
        self.max_paragraph_chars = settings.TRANSCRIPT_MAX_PARAGRAPH_CHARS
        self.buffer_max_chars = settings.TRANSCRIPT_RECENT_BUFFER_MAX_CHARS
        self.overlap_k = settings.TRANSCRIPT_OVERLAP_K

    def _next_paragraph_id(self) -> str:
        self._paragraph_counter += 1
        # Prefix with the (server-generated) session id so ids stay unique across reconnects —
        # otherwise a new connection restarts at p1 and its cards overwrite the previous ones.
        sid = (self.session_id or "")[:8]
        return f"{sid}-p{self._paragraph_counter}" if sid else f"p{self._paragraph_counter}"

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
            role=self.current_role,
            sentences=list(self._current_paragraph_sentences),
        )
        self.segments.append(p)
        self._current_paragraph_start_index = len(current_text)
        self._current_paragraph_start_ts = None
        self._sentence_cursor = 0
        self._current_paragraph_sentences = []
        return p

    def _peek_paragraph_id(self) -> str:
        sid = (self.session_id or "")[:8]
        n = self._paragraph_counter + 1
        return f"{sid}-p{n}" if sid else f"p{n}"

    def pop_completed_sentences(self, force: bool = False) -> List[dict]:
        """Return NEW complete sentences from the committed part of the current paragraph
        (raw_text after paragraph start), advancing the cursor. With force=True the dangling
        fragment is returned too (used on paragraph close / idle flush / EOS).
        Each item: {sentence_id, paragraph_id, text, char_start, char_end, role} with offsets
        relative to the paragraph text (matches Paragraph.raw_text)."""
        committed = self.raw_text.strip()
        para_text = committed[self._current_paragraph_start_index:] if len(committed) > self._current_paragraph_start_index else ""
        # paragraph text as it will appear in Paragraph.raw_text (stripped) — offsets relative to it
        lead = len(para_text) - len(para_text.lstrip())
        para_text = para_text.strip()
        if not para_text or self._sentence_cursor >= len(para_text):
            return []
        pending = para_text[self._sentence_cursor:]
        spans = split_live_sentences(pending)
        pid = self._peek_paragraph_id()
        # Coalesce STT chunk fragments into real sentences. `spans` are punctuation-delimited
        # pieces; merge a piece into its predecessor when the predecessor cannot stand alone or
        # the piece obviously continues it. A candidate is FINALIZED only when the following
        # piece proves it complete (lookahead) — the last candidate stays pending unless force.
        cands: List[tuple] = []          # (start, end)
        for (a, b) in spans:
            if not pending[a:b].strip():
                continue
            if cands:
                pa, pb = cands[-1]
                if should_merge_fragments(pending[pa:pb], pending[a:b]):
                    cands[-1] = (pa, b)
                    continue
            cands.append((a, b))
        # dangling tail (no terminal punctuation yet) may still be a continuation of the last candidate
        tail_start = cands[-1][1] if cands else 0
        tail = pending[tail_start:]
        tail_txt = tail.strip()
        finalize_upto = len(cands)
        if not force:
            # hold the last candidate: it may still be continued by text that hasn't arrived
            if cands and (not tail_txt or _fragment_continues(tail_txt) or _fragment_incomplete(pending[cands[-1][0]:cands[-1][1]])):
                finalize_upto = len(cands) - 1
            elif cands and tail_txt and not _fragment_continues(tail_txt):
                finalize_upto = len(cands)     # tail starts a NEW sentence => last candidate is complete
        out: List[dict] = []
        consumed = 0
        for (a, b) in cands[:finalize_upto]:
            txt = clean_merged_sentence(pending[a:b].strip())
            self._sentence_counter += 1
            out.append({
                "sentence_id": f"{pid}-s{self._sentence_counter}",
                "paragraph_id": pid,
                "text": txt,
                "char_start": self._sentence_cursor + a,
                "char_end": self._sentence_cursor + b,
                "role": self.current_role,
            })
            consumed = b
        if force:
            rest = clean_merged_sentence(pending[consumed:].strip())
            if rest and len(rest.split()) >= 2:
                self._sentence_counter += 1
                out.append({
                    "sentence_id": f"{pid}-s{self._sentence_counter}",
                    "paragraph_id": pid,
                    "text": rest,
                    "char_start": self._sentence_cursor + consumed,
                    "char_end": self._sentence_cursor + len(pending),
                    "role": self.current_role,
                })
                consumed = len(pending)
        if consumed:
            self._sentence_cursor += consumed
        self._current_paragraph_sentences.extend(out)
        _ = lead
        return out

    def force_commit(self) -> bool:
        """Commit the recent buffer regardless of punctuation/silence (idle flush)."""
        if not self.recent_buffer.strip():
            return False
        sep = ""
        if self.raw_text and not self.raw_text.endswith(" ") and not self.recent_buffer.startswith(" "):
            sep = " "
        self.raw_text += sep + self.recent_buffer
        self.recent_buffer = ""
        return True

    def close_paragraph_now(self, ts_ms: int) -> Optional[Paragraph]:
        """Close the current paragraph on wall-clock idle (independent of the STT audio clock)."""
        sep = " " if self.raw_text and self.recent_buffer and not self.raw_text.endswith(" ") and not self.recent_buffer.startswith(" ") else ""
        current_full = (self.raw_text + sep + self.recent_buffer).strip()
        segment_so_far = current_full[self._current_paragraph_start_index:].strip()
        if not segment_so_far:
            return None
        return self._close_current_paragraph(ts_ms / 1000.0)

    def append_piece(self, piece: str, ts_ms: int) -> None:
        """Append a text piece from STT with anti-duplication and whitespace normalization."""
        if self._paused or not piece:
            return
        piece = _normalize_piece(piece)
        if not piece:
            return
        tail = (self.raw_text + self.recent_buffer).strip()
        # Skip exact duplicate: if incoming piece is identical to end of tail, do not append.
        # Do NOT stamp the timestamp on fully-deduped pieces, else the real silence gap is lost. (M12)
        if tail and len(piece) <= len(tail) and tail.endswith(piece):
            return
        # Anti-duplication: max suffix/prefix overlap between existing tail and incoming
        overlap = _max_suffix_prefix_overlap(tail, piece, self.overlap_k)
        if overlap > 0:
            piece = piece[overlap:]
        if not piece:
            return
        # Pieces from STT already include word-boundary spaces (▁→" "); concatenate directly.
        self.recent_buffer += piece
        # Remember the previous piece's ts so silence_gap reflects the real inter-piece pause. (M12)
        self.prev_piece_ts_ms = self.last_piece_ts_ms
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

    def get_live_partial(self) -> str:
        """Text of the current (not yet committed as a segment) paragraph.

        This is the portion of get_display_text() that starts after all
        committed segments, i.e. what is actively being transcribed right now.
        Always non-empty while the user is speaking; resets to '' when a new
        paragraph is closed.
        """
        sep = (
            " "
            if self.raw_text
            and self.recent_buffer
            and not self.raw_text.endswith(" ")
            and not self.recent_buffer.startswith(" ")
            else ""
        )
        current_full = (self.raw_text + sep + self.recent_buffer).strip()
        if self._current_paragraph_start_index < len(current_full):
            return current_full[self._current_paragraph_start_index:]
        # Fallback: if index has drifted past current length (edge case after finalize)
        return self.recent_buffer.strip()

    def maybe_commit(self, ts_ms: int) -> bool:
        """
        Commit recent_buffer into raw_text when: ends with .?! or silence > threshold or buffer too long.
        Returns True if a commit happened.
        """
        if not self.recent_buffer.strip():
            return False
        now = ts_ms
        # Gap between the latest piece and the one before it = actual pause (clamped >= 0). (M12)
        silence_gap = max(0, (now - self.prev_piece_ts_ms)) if self.prev_piece_ts_ms is not None else 0
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
        silence_gap = max(0, (now - self.prev_piece_ts_ms)) if self.prev_piece_ts_ms is not None else 0  # (M12)
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
