"""Which of the matter's documents a question or a sentence speaks of (2026-10-28, held-out 2 h2-abstain-storage-law).

An answer said "The Kestrel storage agreement is governed by the law of England and Wales", resting on real quotes from the
Northbridge NDA and the Unit 7 Lease. Every quote was verified; the attribution was wrong. Two checks follow from this:

- a cited sentence that names a document must cite that document (at least one of its sources), or it does not stand;
- a question that names a document is answered only from that document: an answer whose every standing sentence cites
  other documents is the model reaching for a neighbour's clause, and is withheld.

A document is named by its type ("agreement", "lease", "NDA", "minutes", "letter", …) with the words that tell it from the
others of that type ("Kestrel storage agreement", "Northbridge NDA"). A party's name alone ("at Kestrel", "Calder & Wynne")
names no document, and a type shared by several documents with nothing to tell them apart names none: the checks only act
when the reference is clear.
"""
from __future__ import annotations

import re

# Types that name one document on their own when the matter has one of them ("the lease", "the MSA", "the NDA") …
_STRONG = frozenset("msa nda lease".split())
# … and types that are also ordinary nouns ("on thirty days' written notice", "within 30 minutes", "Schedule 2"): these name
# a document only with a word of its title beside them ("the change of control notice", "the witness statement").
_WEAK = frozenset("agreement letter minutes amendment statement notice memo memorandum contract engagement policy schedule deed".split())
_TYPES = _STRONG | _WEAK
_EXPAND = {"msa": ("master", "services", "agreement"), "nda": ("non", "disclosure", "agreement")}
_PHRASES = {("master", "services", "agreement"): "msa", ("non", "disclosure", "agreement"): "nda",
            ("nondisclosure", "agreement"): "nda"}
_NOISE = frozenset("the a an of to in on for by with and or no is are be as at that this which under from its it what when "
                   "who how does did do any each all their his her".split())


def _tokens(text: str) -> list[str]:
    """Words of the text in order, citation markers ("[1]", "[2: …]") left out."""
    return re.findall(r"[a-z0-9]+", re.sub(r"\[[^\]]*\]", " ", text or "").lower())


def _words(text: str) -> set[str]:
    toks = _tokens(text)
    out = set(toks)
    for t in toks:
        out.update(_EXPAND.get(t, ()))
    for phrase, short in _PHRASES.items():
        n = len(phrase)
        if any(tuple(toks[i:i + n]) == phrase for i in range(len(toks) - n + 1)):
            out.add(short)
    return out - _NOISE


def _qualifiers(title: str) -> set[str]:
    """The words of a title that tell it from others of its type: not the type, not a number, not a small word."""
    return {w for w in _words(title) - _TYPES if not w.isdigit()}


def referenced(text: str, titles: dict[str, str]) -> set[str]:
    """The ids of the documents (id → title) that the text clearly names; empty when it names none or is ambiguous.

    A strong type ("lease", "MSA", "NDA") names the matter's only document of that type. A weak type names a document only
    when a word of that document's title stands just before it, as in a name ("the Kestrel storage agreement", "the change
    of control notice"), not after it ("notice of a change of control" is an event, not the notice document)."""
    toks = _tokens(text)
    words = _words(text)
    out: set[str] = set()
    for t in words & _STRONG:
        holders = [did for did, title in titles.items() if t in _words(title)]
        if len(holders) == 1:
            out.update(holders)
    for i, t in enumerate(toks):
        if t not in _WEAK and t not in _STRONG:
            continue
        before = set(toks[max(0, i - 3):i])
        for phrase, short in _PHRASES.items():                      # "Master Services Agreement" stands for the MSA
            if tuple(toks[max(0, i - len(phrase) + 1):i + 1]) == phrase:
                before |= set(_EXPAND[short]) | {short}
        holders = {did: len(_qualifiers(title) & before) for did, title in titles.items() if t in _words(title)}
        best = max(holders.values(), default=0)
        if best >= 1:
            out.update(d for d, s in holders.items() if s == best)
    return out
