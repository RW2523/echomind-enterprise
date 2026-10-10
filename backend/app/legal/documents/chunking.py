"""Passages for search, cut from the canonical text by its blocks (S2.1; sections kept apart in S2.2).

Every passage is an exact slice of the stored text: `text[start:end]`. That is what lets a citation
point at the characters it quotes. Paragraphs are packed in reading order up to LEGAL_CHUNK_CHARS.
A paragraph longer than twice that is split at sentence ends. Headers, footers and footnotes are
never packed with body text. (The legacy chunker rewrites whitespace and, for short files, gives no
offsets at all, so Law no longer uses it.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ...core.config import settings
from .extract import Extraction

_APART = ("header", "footer", "footnote")


@dataclass
class Passage:
    char_start: int
    char_end: int
    text: str
    page_number: Optional[int] = None
    is_parent: bool = False


def _sentences(text: str, start: int, end: int, target: int) -> list[tuple[int, int]]:
    from ..research.prompt import document_sentences
    pieces, cur = [], None
    for a, b in document_sentences(text[start:end]):
        a, b = a + start, b + start
        while b - a > 2 * target:                             # one enormous "sentence": cut at a space
            cut = text.rfind(" ", a, a + target)
            cut = cut if cut > a else a + target
            pieces.append((a, cut))
            a = cut + 1
        if cur and b - cur[0] <= target:
            cur = (cur[0], b)
        else:
            if cur:
                pieces.append(cur)
            cur = (a, b)
    if cur:
        pieces.append(cur)
    return pieces


def _section_starts(x: Extraction) -> set[int]:
    """Where a new section begins: a heading, or a top-level clause (S2.2, D-038). A passage never
    crosses one, so a clause is scored with its own section, not with the next one's text."""
    from .structure import clauses
    starts = {b.start for b in x.blocks if b.kind == "heading"}
    starts |= {c.start for c in clauses(x) if c.level == 0}
    return starts


def passages(x: Extraction, target: Optional[int] = None) -> list[Passage]:
    target = target or settings.LEGAL_CHUNK_CHARS
    breaks = _section_starts(x)
    spans: list[tuple[int, int]] = []
    for zone in (lambda b: b.kind not in _APART, lambda b: b.kind in _APART):
        cur = None
        for b in (b for b in x.blocks if zone(b)):
            if cur and (b.start in breaks or any(cur[1] < s <= b.start for s in breaks)):
                spans.append(cur)
                cur = None
            if b.end - b.start > 2 * target:
                if cur:
                    spans.append(cur)
                    cur = None
                spans += _sentences(x.text, b.start, b.end, target)
            elif cur is None:
                cur = (b.start, b.end)
            elif b.end - cur[0] <= target:
                cur = (cur[0], b.end)
            else:
                spans.append(cur)
                cur = (b.start, b.end)
        if cur:
            spans.append(cur)
    return [Passage(a, b, x.text[a:b], x.page_at(a)) for a, b in spans if x.text[a:b].strip()]
