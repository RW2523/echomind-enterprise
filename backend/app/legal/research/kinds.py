"""A question about "the supply agreement" in a matter with two (2026-11-04).

"What are the payment terms in the supply agreement?" in a matter holding two supply agreements was answered from
whichever one ranked first, as if it were the only one. When the question names a kind of document that two to four
documents of the matter are titled with, and names neither a party nor one of their titles, each of them is searched and
the model is told they are several, so that it answers for each and says which is which, or says which it means.
"""
from __future__ import annotations

import re
from typing import Optional

# Two words or more ending in a kind of document: "supply agreement", "side letter", "board minutes", "heads of terms".
_KIND = re.compile(r"\b((?:[a-z][a-z-]+\s+){1,2}(?:agreement|contract|lease|licen[cs]e|letter|policy|minutes|memo(?:randum)?"
                   r"|note|notice|statement|report|amendment|addendum|deed|terms|procedures|register|articles)s?)\b", re.I)
_GENERIC = frozenset("the a an this that their its our your his her each any other same signed draft new old current".split())
MAX_SAME = 4


def same_kind(question: str, titles: dict[str, str]) -> Optional[tuple[str, list[str]]]:
    """(the kind, the ids of the documents titled with it) when the question names a kind that two to MAX_SAME documents
    share and none of their titles in full; else None."""
    q = (question or "").lower()
    best = None
    for m in _KIND.finditer(q):
        words = [w for w in m.group(1).split() if w not in _GENERIC]
        if len(words) < 2:
            continue
        kind = " ".join(words)
        docs = [did for did, t in titles.items() if kind in (t or "").lower()]
        if not docs and kind.endswith("s"):                       # "the supply agreements"
            kind = kind[:-1]
            docs = [did for did, t in titles.items() if kind in (t or "").lower()]
        if 2 <= len(docs) <= MAX_SAME and (best is None or len(kind) > len(best[0])):
            best = (kind, docs)
    if best is None:
        return None
    named = [did for did in best[1] if _distinct(titles[did], best[0]) and _distinct(titles[did], best[0]) in q]
    return None if named else best


def _distinct(title: str, kind: str) -> str:
    """What a title says besides the kind and the demo label: "Supply Agreement - Northwind (DEMO)" -> "northwind"."""
    rest = re.sub(r"\((?:demo|draft|fictional)\)", " ", (title or "").lower()).replace(kind, " ")
    words = [w for w in re.findall(r"[a-z0-9][a-z0-9'-]+", rest) if w not in _GENERIC and w not in ("for", "of", "and", "with")]
    return " ".join(words)


def note(kind: str, titles: list[str]) -> str:
    listed = ", ".join(f'"{t}"' for t in titles)
    return (f'(This matter has {len(titles)} documents that are a "{kind}": {listed}. The question does not say which; answer '
            f"for each one that the sources cover and name the document each answer comes from.)\n")
