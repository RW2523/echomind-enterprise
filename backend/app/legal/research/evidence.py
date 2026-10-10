"""Quoted citations and their verification (S2.2b, RAG-01; decision D-037).

The model cites each statement with its source and a short exact quote: `[2: "capped at USD 250,000"]`
(a bare `[2]` is still read). After each paragraph, and before it is shown, every quote is looked
for in the passage it names:
- **verbatim:** found as written;
- **normalized:** found once Unicode forms, quotes, dashes, spacing and case are folded, or as the
  pieces of a quote with an ellipsis, in order;
- **failed:** not found, or the citation names no passage.

A claim (a sentence with citations) is supported when at least one of its quotes is found. It has
failed when it quotes and none is found. It is unquoted when it only gives source numbers.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

# A quote may contain escaped quotes, as models write them: [2: "Harrowgate Systems Inc. (\"Provider\")"].
_CITE = re.compile(r"\[(\d{1,2})(?::\s*[\"“”]((?:\\.|[^\"“”\\])+)[\"“”]\s*)?\]", re.S)
_TOKEN = re.compile("\x00(\\d+)\x01")
_FOLD = str.maketrans({"‘": "'", "’": "'", "‛": "'", "′": "'", "“": '"', "”": '"', "„": '"',
                       "–": "-", "—": "-", "−": "-", " ": " "})
_ELLIPSIS = re.compile(r"\s*(?:…|\.\.\.)\s*")


@dataclass
class Citation:
    n: int
    quote: Optional[str]
    verification: Optional[str] = None
    start: Optional[int] = None          # in the source passage
    end: Optional[int] = None


@dataclass
class Claim:
    sentence: str
    citations: list[Citation] = field(default_factory=list)

    @property
    def state(self) -> str:
        quoted = [c for c in self.citations if c.quote is not None]
        if any(c.verification in ("verbatim", "normalized") for c in quoted):
            return "supported"
        if quoted or any(c.verification == "failed" for c in self.citations):
            return "failed"
        return "unquoted"


@dataclass
class Paragraph:
    display: str
    claims: list[Claim]


@dataclass
class Found:
    verification: str
    start: Optional[int] = None
    end: Optional[int] = None


def parse(raw: str) -> Paragraph:
    """Citations out of the text: the display text keeps `[n]`; quotes go to the claims."""
    from .prompt import _ONLY_CITES, _sentences
    cites: list[Citation] = []

    def keep(m: re.Match) -> str:
        quote = m.group(2)
        cites.append(Citation(int(m.group(1)), re.sub(r"\\(.)", r"\1", quote).strip() if quote is not None else None))
        return f"\x00{len(cites) - 1}\x01"

    tokens = _CITE.sub(keep, raw or "")

    def shown(text: str) -> str:
        return _TOKEN.sub(lambda m: f"[{cites[int(m.group(1))].n}]", text)

    claims: list[Claim] = []
    for sentence in _sentences(tokens):
        ids = [int(i) for i in _TOKEN.findall(sentence)]
        bare = shown(sentence).strip()
        if ids and claims and _ONLY_CITES.match(bare):           # "… invoice. [2]": the citation belongs to the sentence before
            claims[-1].citations += [cites[i] for i in ids]
            claims[-1].sentence = f"{claims[-1].sentence} {bare}"
            continue
        if ids:
            claims.append(Claim(bare, [cites[i] for i in ids]))
    return Paragraph(shown(tokens), claims)


def _folded(text: str) -> tuple[str, list[int]]:
    """Folded text and, for each folded character, its index in `text`."""
    out, where, space = [], [], False
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch).translate(_FOLD).casefold():
            if c.isspace():
                if out and not space:
                    out.append(" ")
                    where.append(i)
                space = True
                continue
            out.append(c)
            where.append(i)
            space = False
    return "".join(out), where


def verify(quote: Optional[str], source: str) -> Found:
    if not quote or not quote.strip() or not source:
        return Found("failed")
    at = source.find(quote)
    if at >= 0:
        return Found("verbatim", at, at + len(quote))
    hay, where = _folded(source)
    pieces = [p for p in (_folded(part.strip(" .,;:\"'"))[0].strip() for part in _ELLIPSIS.split(quote.strip())) if p]
    if not pieces:
        return Found("failed")
    pos, first = 0, None
    for piece in pieces:
        k = hay.find(piece, pos)
        if k < 0:
            return Found("failed")
        first = k if first is None else first
        pos = k + len(piece)
    return Found("normalized", where[first], where[pos - 1] + 1)
