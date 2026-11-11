"""The words a matter's BM25 index and its queries are made of (2026-11-12, evaluation fix list item 1).

The base tokenizer (rag/sparse._tokenize) keeps words as written, so "How can the MSA be terminated?" shared no word with the
clause headed "Termination" or with "may terminate this Agreement": clause 12 of the demo's MSA was never found. Words are
reduced to their Porter stem here (terminate, terminated, termination: "termin"), for passages and queries alike, and a
passage's section heading is indexed with it, twice, so a heading counts more than a word in passing. Numbers and the base
tokenizer's compound references ("section_9") are kept as they are. Production's own index (rag/sparse.py) is unchanged.

VERSION is written into each partition's files: an index made with other words is rebuilt from the database on load.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from ...rag.sparse import _tokenize

VERSION = 2

_VOWELS = frozenset("aeiou")


def _cons(w: str, i: int) -> bool:
    c = w[i]
    if c in _VOWELS:
        return False
    if c == "y":
        return i == 0 or not _cons(w, i - 1)
    return True


def _m(stem: str) -> int:
    """Porter's measure: the number of vowel-consonant sequences in the stem."""
    n, i, size = 0, 0, len(stem)
    while i < size and _cons(stem, i):
        i += 1
    while i < size:
        while i < size and not _cons(stem, i):
            i += 1
        if i >= size:
            break
        while i < size and _cons(stem, i):
            i += 1
        n += 1
    return n


def _vowel_in(stem: str) -> bool:
    return any(not _cons(stem, i) for i in range(len(stem)))


def _double(w: str) -> bool:
    return len(w) >= 2 and w[-1] == w[-2] and _cons(w, len(w) - 1)


def _cvc(w: str) -> bool:
    n = len(w)
    return n >= 3 and _cons(w, n - 3) and not _cons(w, n - 2) and _cons(w, n - 1) and w[-1] not in "wxy"


def _replace(w: str, rules: tuple, min_m: int) -> str:
    """The longest suffix of `rules` that `w` ends with is replaced when the stem before it measures more than `min_m`."""
    for suffix, repl in rules:
        if w.endswith(suffix):
            stem = w[: len(w) - len(suffix)]
            return stem + repl if _m(stem) > min_m else w
    return w


_STEP2 = (("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"), ("izer", "ize"), ("bli", "ble"),
          ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
          ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"),
          ("iviti", "ive"), ("biliti", "ble"), ("logi", "log"))
_STEP3 = (("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", ""))
_STEP4 = ("ement", "ment", "ance", "ence", "able", "ible", "ant", "ent", "ism", "ate", "iti", "ous", "ive", "ize", "ion", "al",
          "er", "ic", "ou")


def _by_length(rules: tuple) -> tuple:
    return tuple(sorted(rules, key=lambda r: -len(r[0])))


_STEP2, _STEP3 = _by_length(_STEP2), _by_length(_STEP3)
_STEP4 = tuple(sorted(_STEP4, key=len, reverse=True))


@lru_cache(maxsize=65536)
def stem(word: str) -> str:
    """The Porter (1980) stem of a lower-case word."""
    w = word
    if len(w) <= 2:
        return w
    # 1a
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]
    # 1b
    again = False
    if w.endswith("eed"):
        if _m(w[:-3]) > 0:
            w = w[:-1]
    elif w.endswith("ed") and _vowel_in(w[:-2]):
        w, again = w[:-2], True
    elif w.endswith("ing") and _vowel_in(w[:-3]):
        w, again = w[:-3], True
    if again:
        if w.endswith(("at", "bl", "iz")):
            w += "e"
        elif _double(w) and w[-1] not in "lsz":
            w = w[:-1]
        elif _m(w) == 1 and _cvc(w):
            w += "e"
    # 1c
    if w.endswith("y") and _vowel_in(w[:-1]):
        w = w[:-1] + "i"
    w = _replace(w, _STEP2, 0)
    w = _replace(w, _STEP3, 0)
    # 4
    for suffix in _STEP4:
        if w.endswith(suffix):
            s = w[: len(w) - len(suffix)]
            if _m(s) > 1 and (suffix != "ion" or (s and s[-1] in "st")):
                w = s
            break
    # 5a, 5b
    if w.endswith("e"):
        s = w[:-1]
        if _m(s) > 1 or (_m(s) == 1 and not _cvc(s)):
            w = s
    if _m(w) > 1 and _double(w) and w.endswith("l"):
        w = w[:-1]
    return w


def tokens(text: str, heading: Optional[str] = None) -> list[str]:
    """A passage's (or a query's) index words: each word's stem, numbers and compound references as they are; the section
    heading's words twice over."""
    out = [stem(t) if t.isalpha() else t for t in _tokenize(text)]
    if heading:
        head = [stem(t) if t.isalpha() else t for t in _tokenize(heading)]
        out += head + head
    return out
