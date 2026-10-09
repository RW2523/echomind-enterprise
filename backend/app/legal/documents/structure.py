"""Clauses and defined terms of a legal document (S2.1b, RAG-01; decision D-035).

Clause numbers come from Word's own numbering when a paragraph has it ("12.3", with its full path
"12.3(b)"), and otherwise from the markers drafting uses at the start of a line:
- "Article IV" and "Part 2" (level 0);
- "Schedule 2", "Exhibit A", "Annex B" and "Appendix 1" (level 0; they also end the clause tree
  before them);
- "Section 5.1", "Clause 4" and "§ 7" (the level follows the dots);
- bare numbers "4.1" or "12." followed by a capital, a quote or a bracket. A year (four digits),
  an amount ("12,000") or a date ("1 January") is not a clause;
- "(a)" and "(i)" under the last numbered clause, giving "4.1(a)" and "4.1(a)(i)".

A clause runs until the next clause at the same or a higher level, so it contains its sub-clauses.
Headers, footers and footnotes are never clauses.

Defined terms are found where they are defined: `"Term" means …` (also "shall mean", "has the
meaning", "includes", "refers to") and the parenthetical `Name (the "Term")`. Straight and curly
quotes are both accepted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .extract import Extraction

_APART = ("header", "footer", "footnote")
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_TOP = re.compile(r"(ARTICLE|Article|PART|Part)\s+([IVXLC]+|\d{1,3})\b")
_ANNEX = re.compile(r"(SCHEDULE|Schedule|EXHIBIT|Exhibit|ANNEX|Annex|APPENDIX|Appendix)\s+([A-Z]|\d{1,3})\b")
_SECTION = re.compile(r"(?:SECTION|Section|CLAUSE|Clause)\s+(\d{1,3}(?:\.\d{1,3})*)\b")
_SIGN = re.compile(r"§\s*(\d{1,3}(?:\.\d{1,3})*)\b")
_BARE = re.compile(rf"(\d{{1,3}}(?:\.\d{{1,3}})*)\.?(?=\s+(?!(?:{_MONTHS})\b)[A-Z(\"“])")
_SUB = re.compile(r"\(([a-z]{1,2}|[ivxlc]{1,6})\)(?=\s)")
_ROMAN = re.compile(r"^[ivxlc]+$")


@dataclass
class Clause:
    number: str
    label: str
    heading: Optional[str]
    level: int
    start: int
    end: int
    parent: Optional[str] = None
    page: Optional[int] = None


@dataclass
class Term:
    term: str
    start: int
    end: int
    term_start: int
    term_end: int


def _heading(rest: str) -> Optional[str]:
    rest = rest.strip().lstrip("—–-:.").strip()
    first = rest.split("\n", 1)[0].strip()
    if first and len(first) <= 80 and first[-1] not in ".;:," and not re.search(r"\.\s", first):
        return first
    m = re.match(r"([A-Z][^.\n]{0,60}?)\.\s", rest)
    return m.group(1).strip() if m else None


def _markers(text: str, start: int, end: int) -> list[tuple[int, str, str, int, str, str]]:
    """(offset, kind, value, level, label, rest-of-line) for each clause marker at a line start."""
    out = []
    pos = start
    for line in text[start:end].split("\n"):
        stripped = line.lstrip()
        at = pos + len(line) - len(stripped)
        rest_line = text[at:pos + len(line)]
        for kind, rx in (("top", _TOP), ("annex", _ANNEX), ("section", _SECTION), ("sign", _SIGN), ("bare", _BARE), ("sub", _SUB)):
            m = rx.match(rest_line)
            if not m:
                continue
            if kind == "top":
                out.append((at, kind, f"{m.group(1).title()} {m.group(2)}", 0, m.group(0), rest_line[m.end():]))
            elif kind == "annex":
                out.append((at, kind, f"{m.group(1).title()} {m.group(2)}", 0, m.group(0), rest_line[m.end():]))
            elif kind == "sign":
                out.append((at, kind, f"§ {m.group(1)}", 1 + m.group(1).count("."), m.group(0), rest_line[m.end():]))
            elif kind in ("section", "bare"):
                out.append((at, kind, m.group(1), 1 + m.group(1).count("."), m.group(0), rest_line[m.end():]))
            else:
                out.append((at, kind, m.group(1), -1, m.group(0), rest_line[m.end():]))
            break
        pos += len(line) + 1
    return out


def clauses(x: Extraction) -> list[Clause]:
    found: list[Clause] = []
    stack: list[Clause] = []
    numbered: Optional[Clause] = None                       # the last clause "(a)" would attach to
    last_sub: Optional[tuple[str, int]] = None               # (letter, level) of the last "(a)"-style item

    def push(c: Clause) -> None:
        while stack and stack[-1].level >= c.level:
            stack.pop()
        c.parent = stack[-1].number if stack else None
        stack.append(c)
        found.append(c)

    body = [b for b in x.blocks if b.kind not in _APART]
    for b in body:
        if b.path:                                          # Word numbering, as Word shows it
            rest = x.text[b.start:b.end][len(b.label or ""):]
            heading = x.text[b.start:b.end][len(b.label or ""):].strip() if b.kind == "heading" else _heading(rest)
            c = Clause(b.path, b.label or b.path, heading, b.level or 0, b.start, b.end, page=b.page)
            push(c)
            numbered = c if "(" not in b.path else numbered
            continue
        for at, kind, value, level, label, rest in _markers(x.text, b.start, b.end):
            if kind == "sub":
                if numbered is None:
                    continue
                roman = bool(_ROMAN.match(value)) and not (last_sub and last_sub[0] and len(value) == 1 and
                                                         ord(value) == ord(last_sub[0]) + 1)
                if roman and last_sub and not _ROMAN.match(last_sub[0] or ""):
                    base, lvl = f"{numbered.number}({last_sub[0]})", last_sub[1] + 1
                else:
                    base, lvl = numbered.number, numbered.level + 1
                    last_sub = (value, lvl)
                number = f"{base}({value})"
                push(Clause(number, label, _heading(rest), lvl, at, b.end, page=x.page_at(at)))
                continue
            if kind == "annex":
                stack.clear()
            c = Clause(value, label, _heading(rest), level, at, b.end, page=x.page_at(at))
            push(c)
            numbered, last_sub = c, None
    body_end = max((b.end for b in body), default=len(x.text))
    for i, c in enumerate(found):
        nxt = next((d.start for d in found[i + 1:] if d.level <= c.level), body_end)
        c.end = max(c.start, len(x.text[c.start:nxt].rstrip()) + c.start)
    return found


def clause_at(found: list[Clause], offset: int) -> Optional[Clause]:
    inside = [c for c in found if c.start <= offset < max(c.end, c.start + 1)]
    return max(inside, key=lambda c: (c.level, c.start)) if inside else None


_Q_OPEN, _Q_CLOSE = "[\"“]", "[\"”]"
_TERM = r"([A-Z][\w&'’\- ]{0,60}?)"
_MEANS = re.compile(_Q_OPEN + _TERM + _Q_CLOSE +
                    r"\s+(?:means|shall mean|has the meaning|have the meaning|includes|shall include|refers to)\b")
_PAREN = re.compile(r"\((?:the\s+|each\s+a\s+|together\s+the\s+|collectively\s+the\s+|hereinafter\s+(?:the\s+)?)?" +
                    _Q_OPEN + _TERM + _Q_CLOSE + r"\)")


def defined_terms(x: Extraction) -> list[Term]:
    from ..research.prompt import document_sentences
    sentences = document_sentences(x.text)

    def sentence_of(offset: int) -> tuple[int, int]:
        return next(((a, b) for a, b in sentences if a <= offset < b), (offset, len(x.text)))

    out: dict[str, Term] = {}
    for rx, from_quote in ((_MEANS, True), (_PAREN, False)):
        for m in rx.finditer(x.text):
            term = m.group(1).strip()
            if term in out:
                continue
            a, b = sentence_of(m.start())
            out[term] = Term(term, m.start() if from_quote else a, b, m.start(1), m.end(1))
    return sorted(out.values(), key=lambda t: t.start)
