"""
EchoMind Conversation Intelligence: rolling buffer for conversation context.
Stores entries with timestamps and optional tags; supports time-window and keyword retrieval.
Designed so storage can be swapped for persistence later.
"""
from __future__ import annotations
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable

# Default rolling window (minutes); entries older than this are dropped
DEFAULT_WINDOW_MINUTES = 30.0


@dataclass
class MemoryEntry:
    """Single entry in the conversation memory."""
    ts_start: float
    ts_end: float
    text: str
    tags: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    speaker: Optional[str] = None  # "user" | "assistant" | None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts_start": self.ts_start,
            "ts_end": self.ts_end,
            "text": self.text,
            "tags": self.tags,
            "entities": self.entities,
            "speaker": self.speaker,
        }


def _heuristic_tags(text: str) -> List[str]:
    """Lightweight topic/keyword heuristics for tagging (no embeddings)."""
    t = (text or "").lower()
    tags = []
    if re.search(r"\b(fact|check|verify|true|false|claim)\b", t):
        tags.append("fact_check")
    if re.search(r"\b(summarize|summary|recap)\b", t):
        tags.append("summary")
    if re.search(r"\b(when|time|minute|hour|last)\b", t):
        tags.append("temporal")
    if re.search(r"\b(what did i say|what did we discuss)\b", t):
        tags.append("recall")
    return tags


class ConversationMemory:
    """
    Rolling buffer of conversation entries. Keeps last N minutes of entries.
    Interface: add_text(), query_last(minutes), query_topic(q), summarize_last(minutes).
    """

    def __init__(self, window_minutes: float = DEFAULT_WINDOW_MINUTES):
        self.window_minutes = max(0.1, float(window_minutes))
        self._entries: List[MemoryEntry] = []
        # Optional: for debug logging
        self._debug_log: Optional[Callable[[str], None]] = None

    def set_debug_log(self, fn: Callable[[str], None]) -> None:
        """Set a callback for debug logging (e.g. lambda msg: logger.info(msg))."""
        self._debug_log = fn

    def _log(self, msg: str) -> None:
        if self._debug_log:
            try:
                self._debug_log(msg)
            except Exception:
                pass

    def _evict_old(self, now: Optional[float] = None) -> None:
        cutoff = (now or time.time()) - (self.window_minutes * 60.0)
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.ts_end >= cutoff]
        if before != len(self._entries):
            self._log(f"ConversationMemory: evicted {before - len(self._entries)} old entries")

    def add_text(
        self,
        text: str,
        speaker: Optional[str] = "user",
        tags: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        ts: Optional[float] = None,
    ) -> MemoryEntry:
        """Append a single utterance. ts is start time; ts_end = ts (same moment)."""
        if not (text or "").strip():
            raise ValueError("add_text requires non-empty text")
        now = ts if ts is not None else time.time()
        entry_tags = list(tags) if tags else _heuristic_tags(text)
        entry = MemoryEntry(
            ts_start=now,
            ts_end=now,
            text=(text or "").strip(),
            tags=entry_tags,
            entities=list(entities) if entities else [],
            speaker=speaker,
        )
        self._evict_old(now)
        self._entries.append(entry)
        self._log(f"ConversationMemory: add_text speaker={speaker} len={len(entry.text)} tags={entry_tags}")
        return entry

    def query_last(self, minutes: float) -> List[MemoryEntry]:
        """Return entries from the last N minutes (by ts_end)."""
        now = time.time()
        self._evict_old(now)
        cutoff = now - (minutes * 60.0)
        out = [e for e in self._entries if e.ts_end >= cutoff]
        self._log(f"ConversationMemory: query_last({minutes}) -> {len(out)} entries")
        return out

    def query_topic(self, q: str) -> List[MemoryEntry]:
        """Keyword fallback: entries whose text contains any word from q (case-insensitive)."""
        if not (q or "").strip():
            return []
        self._evict_old()
        words = set(re.findall(r"\w+", (q or "").lower()))
        if not words:
            return self._entries.copy()
        out = [e for e in self._entries if any(w in (e.text or "").lower() for w in words)]
        self._log(f"ConversationMemory: query_topic({q!r}) -> {len(out)} entries")
        return out

    def summarize_last(self, minutes: float) -> str:
        """Return a single concatenated transcript of the last N minutes (for LLM context)."""
        entries = self.query_last(minutes)
        if not entries:
            return ""
        # Order by ts_start, format with timestamps
        entries = sorted(entries, key=lambda e: e.ts_start)
        parts = []
        for e in entries:
            ts_str = time.strftime("%H:%M", time.localtime(e.ts_start))
            who = (e.speaker or "user").capitalize()
            parts.append(f"[{ts_str}] {who}: {e.text}")
        return "\n".join(parts)

    def get_entries_for_context(self, minutes: float, max_chars: int = 4000) -> str:
        """Same as summarize_last but with a character cap for prompt safety."""
        s = self.summarize_last(minutes)
        if len(s) <= max_chars:
            return s
        return s[-max_chars:].strip()
