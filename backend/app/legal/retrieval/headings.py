"""The section heading a passage sits under (2026-11-12, evaluation fix list item 1).

A heading is written once, at the start of its section ("## 12. Termination"), so only the section's first passage held it:
"How can the MSA be terminated?" never reached clause 12.3, two passages below it. Each passage now carries the heading over
its first character: the last Markdown heading line at or before it, or the heading of a numbered clause (Word documents)
that starts there. It is read from the version's own text and clause map, nothing is stored, and it goes to the BM25 index
(retrieval/terms.py), the cross-encoder's window and the answer's source tag.
"""
from __future__ import annotations

import bisect
import re
import threading
from collections import OrderedDict
from typing import Optional

_MARKDOWN = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t#]*$", re.M)
_EMPHASIS = re.compile(r"[*_`]+")
_CACHE_SIZE = 512

_lock = threading.Lock()
_cache: "OrderedDict[tuple[str, str], tuple[list[int], list[str]]]" = OrderedDict()


def _clean(heading: str) -> str:
    return re.sub(r"\s+", " ", _EMPHASIS.sub("", heading)).strip()[:120]


def _map(conn, version_id: str) -> tuple[list[int], list[str]]:
    row = conn.execute("SELECT text, created_at FROM legal_version_text WHERE version_id = ?", (version_id,)).fetchone()
    if not row:
        return [], []
    key = (version_id, row[1] or "")
    with _lock:
        got = _cache.get(key)
        if got is not None:
            _cache.move_to_end(key)
            return got
    found = {m.start(): _clean(m.group(1)) for m in _MARKDOWN.finditer(row[0] or "")}
    for number, heading, start in conn.execute("SELECT number, heading, char_start FROM legal_clauses WHERE version_id = ? "
                                               "AND heading IS NOT NULL AND level <= 1", (version_id,)):
        text = _clean(heading)
        if text and start not in found:
            found[start] = f"{number} {text}" if number and not text.startswith(number) else text
    offsets = sorted(o for o in found if found[o])
    got = (offsets, [found[o] for o in offsets])
    with _lock:
        _cache[key] = got
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return got


def at(conn, version_id: str, offset: Optional[int]) -> Optional[str]:
    """The heading over character `offset` of a version, or None (no heading before it, or no offsets)."""
    if offset is None:
        return None
    offsets, headings = _map(conn, version_id)
    i = bisect.bisect_right(offsets, offset) - 1
    return headings[i] if i >= 0 else None
