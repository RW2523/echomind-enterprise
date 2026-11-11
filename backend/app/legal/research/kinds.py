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


def same_kind(question: str, titles: dict[str, str], among: Optional[list[str]] = None, most: int = MAX_SAME) -> Optional[tuple[str, list[str]]]:
    """(the kind, the ids of the documents titled with it) when the question names a kind that two to `most` documents
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
            if 2 <= len(docs) <= most and (best is None or len(kind) > len(best[0])):
                best = (kind, docs)
    if best is None:
        return None
    kind, docs = best
    own = {did: set(_distinct(pool[did], kind).split()) for did in docs}
    common = set.intersection(*own.values())
    said = {re.sub(r"'s?$", "", w) for w in re.findall(r"[a-z0-9][a-z0-9'-]*", q)}
    return None if any((words := own[did] - common) and words <= said for did in docs) else best


def many_of_kind(question: str, titles: dict[str, str]) -> Optional[tuple[str, list[str]]]:
    """More documents of the kind named than are answered for each (2026-11-07 review point 10): "the letter" in a matter
    with seven letters. They are not all searched; the person is offered the most relevant to ask about one alone."""
    got = same_kind(question, titles, None, most=10_000)
    return got if got and len(got[1]) > MAX_SAME else None


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


# A question asking for every item of something (2026-11-12, evaluation fix list item 6: lists 3 of 9 on the chat set; search
# found 44% of their evidence, the items being spread over several passages of a document, or over several documents).
_PLURALS = (r"(?:clauses|provisions|parties|signatories|fees|charges|costs|conditions|obligations|duties|rights|exceptions|exclusions|"
            r"documents|dates|deadlines|amounts|payments|items|units|terms|events|people|persons|witnesses|changes|amendments|grounds|"
            r"reasons|actions|matters|steps|requirements|restrictions|remedies|risks|warranties|representations|services|products|"
            r"deliverables|properties|assets|lettings|comparables|options|issues|breaches|defects|claims|notices|sums|rates)")
_LIST = re.compile(r"\b(?:list|enumerate|itemi[sz]e)\b|\b(?:is|are)\s+(?:listed|itemi[sz]ed|scheduled)\b|\b(?:every|each\s+of\s+the|all\s+(?:of\s+)?the|all\s+(?:\w+\s+){0,2}" + _PLURALS + r")\b"
                   r"|\b(?:which|what)\s+(?:(?!(?:to|the|of|a|an|for|in|on|with|by|at|from|is|are|was|were|happens?|happened|"
                   r"does|do|did|has|have)\b)\w+\s+){0,2}" + _PLURALS + r"\b"
                   r"|\bwhat\s+(?:must|should|does|do|did|has|have)\s+(?:[\w'-]+\s+){0,6}?(?:include|contain|cover|list|set\s+out|ask\s+for|require)\b",
                   re.I)


def asks_list(question: str) -> bool:
    """Whether the question asks for every item of something: "List …", "every signatory", "Which actions need …", "What must
    a notice contain?"."""
    return bool(_LIST.search(question or ""))
