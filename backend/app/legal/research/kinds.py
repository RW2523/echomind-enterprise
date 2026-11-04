"""A question about "the supply agreement" in a matter with two (2026-11-04).

"What are the payment terms in the supply agreement?" in a matter holding two supply agreements was answered from
whichever one ranked first, as if it were the only one. When the question names a kind of document that two to four
documents of the matter are titled with, and none of their titles' own words, each of them is searched and the model is
told they are several, so that it answers for each and says which is which, or says which it means.

A party named in the question narrows them to the documents it is party to (2026-11-04 chat set 3: in five of six such
questions the party was named, "Northwind's supply agreement", "the Kestrel agreement", and was party to several); then the
kind may be a single word ("agreement", "letter"), and a title word every one of them shares (the party's name) tells none
of them apart.
"""
from __future__ import annotations

import re
from typing import Optional

# Words ending in a kind of document: "supply agreement", "side letter", "board minutes", "heads of terms".
_KIND = re.compile(r"\b((?:[a-z][a-z'-]+\s+){0,2}(?:agreement|contract|lease|licen[cs]e|letter|policy|minutes|memo(?:randum)?"
                   r"|note|notice|statement|report|amendment|addendum|deed|terms|procedures|register|articles)s?)\b", re.I)
_GENERIC = frozenset("the a an this that their its our your his her each any other same signed draft new old current".split())
MAX_SAME = 4


def same_kind(question: str, titles: dict[str, str], among: Optional[list[str]] = None) -> Optional[tuple[str, list[str]]]:
    """(the kind, the ids of the documents titled with it) when the question names a kind that two to MAX_SAME documents
    share, of those `among` when a party narrows them, and no words of one of their titles that the others lack; else None."""
    q = (question or "").lower().replace("’", "'")
    pool = {did: (t or "").lower() for did, t in titles.items() if among is None or did in among}
    least = 1 if among is not None else 2
    best = None
    for m in _KIND.finditer(q):
        words = [w for w in m.group(1).split() if w not in _GENERIC]
        for i in range(len(words) - least + 1):       # "northwind's supply agreement", then "supply agreement", …
            kind = " ".join(words[i:])
            docs = _titled(pool, kind)
            if not docs and kind.endswith("s"):       # "the supply agreements"
                kind = kind[:-1]
                docs = _titled(pool, kind)
            if 2 <= len(docs) <= MAX_SAME and (best is None or len(kind) > len(best[0])):
                best = (kind, docs)
    if best is None:
        return None
    kind, docs = best
    own = {did: set(_distinct(pool[did], kind).split()) for did in docs}
    common = set.intersection(*own.values())
    said = {re.sub(r"'s?$", "", w) for w in re.findall(r"[a-z0-9][a-z0-9'-]*", q)}
    return None if any((words := own[did] - common) and words <= said for did in docs) else best


def _titled(pool: dict[str, str], kind: str) -> list[str]:
    """The documents titled with the kind, not those titled as made to or under another of it ("Amendment No. 1 to the
    Kestrel storage agreement" is not a second agreement)."""
    return [did for did, t in pool.items()
            if any(not _OF_ANOTHER.search(t[:m.start()]) for m in re.finditer(rf"\b{re.escape(kind)}\b", t))]


_OF_ANOTHER = re.compile(r"\b(?:to|of|under|amending|supplementing)\s+(?:the\s+)?(?:[\w&'-]+\s+){0,3}$")


def _distinct(title: str, kind: str) -> str:
    """What a title says besides the kind and the demo label: "Supply Agreement - Northwind (DEMO)" -> "northwind"."""
    rest = re.sub(r"\((?:demo|draft|fictional)\)", " ", (title or "").lower()).replace(kind, " ")
    words = [w for w in re.findall(r"[a-z0-9][a-z0-9'-]+", rest) if w not in _GENERIC and w not in ("for", "of", "and", "with")]
    return " ".join(words)


def note(kind: str, titles: list[str]) -> str:
    listed = ", ".join(f'"{t}"' for t in titles)
    return (f'(This matter has {len(titles)} documents that are a{"n" if kind[:1] in "aeiou" else ""} "{kind}": {listed}. The '
            "question does not say which; answer for each one that the sources cover and name the document each answer comes "
            "from.)\n")
