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

A sentence that cites nothing but quotes a source word for word, in quotation marks (D-073: the model
wrote `Section 1(1) states, "…"` with no [n]), is cited to the first source that contains the quote,
verified like any other and marked `matched`. A quote of fewer than MATCH_MIN_WORDS words is never
matched, and a quote found nowhere is left uncited, so the sentence is still marked as unsupported.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional

# A quote may contain escaped quotes, as models write them: [2: "Harrowgate Systems Inc. (\"Provider\")"].
# A quote may itself contain quotation marks (a clause quoting the words it inserts: [5: "… replaced with: "The Supplier …""]),
# so it runs to the first closing mark followed by "]", never past its own bracket (seen in Chrome, 2026-10-28).
_CITE = re.compile(r"\[(\d{1,2})(?::\s*[\"“”]((?:\\.|[^\\\]])+?)[\"“”]\s*)?\]", re.S)
_TOKEN = re.compile("\x00(\\d+)\x01")
# Single and double quotes of every style fold to one mark; Markdown markup (*, #, >, _, `) is not wording (S2.3).
_FOLD = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201b": "'", "\u2032": "'", "\u201c": "'", "\u201d": "'", "\u201e": "'",
                       '"': "'", "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00a0": " "})
_MARKUP = set("*#>_`")
_ELLIPSIS = re.compile(r"\s*(?:…|\.\.\.)\s*")
_QUOTED = re.compile(r"[\"“]([^\"“”\x00\x01]{8,800})[\"”]")
MATCH_MIN_WORDS = 4


@dataclass
class Citation:
    n: int
    quote: Optional[str]
    verification: Optional[str] = None
    start: Optional[int] = None          # in the source passage
    end: Optional[int] = None
    matched: bool = False                # cited by the system: the model quoted the source but gave no [n] (D-073)


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


def parse(raw: str, match: Optional[Callable[[str], Optional[int]]] = None) -> Paragraph:
    """Citations out of the text: the display text keeps `[n]`; quotes go to the claims. With `match` (quote ->
    source number or None), a quote in a sentence that cites nothing is cited where it is found (D-073)."""
    from .prompt import _ONLY_CITES, _sentences
    cites: list[Citation] = []

    def keep(m: re.Match) -> str:
        quote = m.group(2)
        cites.append(Citation(int(m.group(1)), re.sub(r"\\(.)", r"\1", quote).strip() if quote is not None else None))
        return f"\x00{len(cites) - 1}\x01"

    tokens = _CITE.sub(keep, raw or "")
    if match is not None:
        tokens = _matched(tokens, cites, match)

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


def _matched(tokens: str, cites: list[Citation], match: Callable[[str], Optional[int]]) -> str:
    """`tokens` with a citation placed after each quote of MATCH_MIN_WORDS words or more that a source contains, in
    sentences that cite nothing."""
    from .prompt import _spans
    out, last = [], 0
    for a, b in _spans(tokens):
        if "\x00" in tokens[a:b]:
            continue                                         # the model cited this sentence itself
        for m in _QUOTED.finditer(tokens, a, b):
            quote = m.group(1).strip()
            n = match(quote) if len(quote.split()) >= MATCH_MIN_WORDS else None
            if n is None:
                continue
            cites.append(Citation(n, quote, matched=True))
            out.append(tokens[last:m.end()] + f" \x00{len(cites) - 1}\x01")
            last = m.end()
    out.append(tokens[last:])
    return "".join(out)


def _folded(text: str) -> tuple[str, list[int]]:
    """Folded text and, for each folded character, its index in `text`."""
    out, where, space = [], [], False
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch).translate(_FOLD).casefold():
            if c in _MARKUP:
                continue
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
