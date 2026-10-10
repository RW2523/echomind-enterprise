"""Two versions of a legal document, clause by clause (S2.3a, RAG-02; decision D-039).

A version is compared as a sequence of units: the paragraphs, headings, list items, table rows and
footnotes that reading found (S2.1). Headers and footers are left out because they repeat on every
page. Each unit knows the clause it starts ("8.1") and the clause it sits in.

Alignment runs in three passes, plus an optional fourth:
1. **Same wording.** Units whose words are the same once the clause number is set aside are matched
   in order, so a renumbered clause reads "unchanged, renumbered 4.2 → 4.3".
2. **Shared wording.** Between two stretches matched in pass 1, the remaining units are paired in
   order by how much wording they share: at least LEGAL_COMPARE_MATCH_MIN, with a small bonus for
   the same number.
3. **Moved.** Units left over on both sides that share most of their wording
   (LEGAL_COMPARE_MOVE_MIN) are a clause that moved.
4. **Same meaning** (`by_meaning`, run by the caller with embeddings): leftovers that say the same
   thing in other words are a rewritten clause.

A matched pair out of order relative to the others is marked as moved.

Within a pair, a word diff marks exactly what was removed and what was inserted, as character
offsets into each version's text. The change is then labelled:
- a number changed;
- an obligation changed ("shall" → "may");
- a negation was added or removed;
- formatting only: spacing, quotes, dashes or case.
"""
from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

_SKIP = ("header", "footer")
_TOKEN = re.compile(r"\w+(?:[.,'’]\w+)*|[^\w\s]")
_MARKUP = re.compile(r"[*_#>|`~]+")
_LEAD = re.compile(
    r"\s*(?:#{1,6}\s*)?(?:\*\*)?"
    r"(?:(?:ARTICLE|Article|PART|Part|SCHEDULE|Schedule|EXHIBIT|Exhibit|ANNEX|Annex|APPENDIX|Appendix|SECTION|Section|"
    r"CLAUSE|Clause)\s+[A-Z0-9]+(?:\.\d+)*\.?|§\s*\d+(?:\.\d+)*\.?|\d{1,3}(?:\.\d{1,3})*\.?|\((?:[a-z]{1,2}|[ivxlc]{1,6})\))"
    r"(?:\*\*)?(?=\s)")
_FOLD = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})
_NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
                 "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
                 "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion",
                 "half", "double", "twice", "percent", "per cent"}
_MODAL = {"shall", "must", "may", "will", "should", "can", "could", "might", "agrees", "undertakes", "required", "entitled",
          "obliged", "obligated", "permitted"}
_NEGATION = {"not", "no", "never", "neither", "nor", "none", "without", "except", "unless", "cannot"}


def fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(_FOLD).casefold()


@dataclass
class Unit:
    index: int
    start: int
    end: int
    kind: str
    label: Optional[str]                     # the clause that starts in this unit, e.g. "8.1"
    clause: Optional[str]                    # the deepest clause the unit sits in
    heading: Optional[str]
    lead_end: int                            # where the words after the clause number start
    words: list[tuple[str, int, int]] = field(default_factory=list)   # folded word, start, end (absolute)

    @property
    def key(self) -> str:
        return " ".join(w for w, _, _ in self.words)


@dataclass
class Pair:
    left: Optional[int]
    right: Optional[int]
    similarity: float = 1.0
    how: str = "same"                        # same | wording | moved | meaning
    moved: bool = False


def units(text: str, blocks: list[dict], clauses: list[dict]) -> list[Unit]:
    out: list[Unit] = []
    for b in blocks:
        s, e = b["start"], b["end"]
        if b.get("kind") in _SKIP or e <= s or not text[s:e].strip():
            continue
        inside = [c for c in clauses if c["start"] <= s < c["end"]]
        starting = [c for c in clauses if s <= c["start"] < e]
        clause = max(inside, key=lambda c: (c["level"], c["start"]))["number"] if inside else None
        label = min(starting, key=lambda c: (c["start"], c["level"]))["number"] if starting else None
        heading = next((c["heading"] for c in sorted(starting, key=lambda c: c["start"]) if c.get("heading")), None)
        m = _LEAD.match(text, s, e)
        lead = m.end() if m else s
        if label is None and m:                  # a heading such as "## 7. Warranties" numbers itself
            label = re.sub(r"^[#*\s]+|[*.\s]+$", "", m.group(0)) or None
        words = [(fold(t.group()), t.start(), t.end()) for t in _TOKEN.finditer(text, lead, e) if not _MARKUP.fullmatch(t.group())]
        out.append(Unit(len(out), s, e, b.get("kind") or "paragraph", label, clause, heading, lead, words))
    return out


# ── Alignment ────────────────────────────────────────────────────────────────────
def _words(u: Unit) -> list[str]:
    return [w for w, _, _ in u.words]


def wording(u: Unit, v: Unit, floor: float = 0.0) -> float:
    """Share of wording in common (0–1). Below `floor` the cheap upper bound is returned instead."""
    a, b = _words(u), _words(v)
    if not a and not b:
        return 1.0 if u.label == v.label else 0.0
    sm = SequenceMatcher(None, a, b, autojunk=False)
    bound = sm.real_quick_ratio()
    if bound < floor:
        return bound
    bound = sm.quick_ratio()
    return bound if bound < floor else sm.ratio()


def _score(u: Unit, v: Unit, floor: float) -> float:
    s = wording(u, v, floor - 0.1)
    return min(1.0, s + 0.1) if u.label and u.label == v.label else s


def _in_order(a: list[Unit], b: list[Unit], left: list[int], right: list[int], floor: float) -> list[Pair]:
    """Pair two stretches in order, maximising the wording they share (a weighted alignment)."""
    if not left or not right:
        return [Pair(i, None) for i in left] + [Pair(None, j) for j in right]
    n, m = len(left), len(right)
    wide = n * m > 40_000                       # a long rewritten stretch: only compare units near the diagonal
    sim = [[0.0] * m for _ in range(n)]
    for x, i in enumerate(left):
        for y, j in enumerate(right):
            if wide and abs(x / n - y / m) > 0.15 and a[i].label != b[j].label:
                continue
            s = _score(a[i], b[j], floor)
            sim[x][y] = s if s >= floor else 0.0
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    for x in range(1, n + 1):
        for y in range(1, m + 1):
            s = sim[x - 1][y - 1]
            best[x][y] = max(best[x - 1][y], best[x][y - 1], best[x - 1][y - 1] + s if s else 0.0)
    pairs, x, y, used_l, used_r = [], n, m, set(), set()
    while x > 0 and y > 0:
        s = sim[x - 1][y - 1]
        if s and best[x][y] == best[x - 1][y - 1] + s:
            i, j = left[x - 1], right[y - 1]
            pairs.append(Pair(i, j, wording(a[i], b[j]), "wording"))
            used_l.add(i)
            used_r.add(j)
            x, y = x - 1, y - 1
        elif best[x][y] == best[x - 1][y]:
            x -= 1
        else:
            y -= 1
    pairs += [Pair(i, None) for i in left if i not in used_l] + [Pair(None, j) for j in right if j not in used_r]
    return pairs


def _leftovers(pairs: list[Pair]) -> tuple[list[int], list[int]]:
    return [p.left for p in pairs if p.right is None], [p.right for p in pairs if p.left is None]


def _join(pairs: list[Pair], found: list[Pair]) -> list[Pair]:
    used_l = {p.left for p in found}
    used_r = {p.right for p in found}
    keep = [p for p in pairs if not (p.right is None and p.left in used_l) and not (p.left is None and p.right in used_r)]
    return keep + found


def align(a: list[Unit], b: list[Unit], match_min: float = 0.5, move_min: float = 0.8) -> list[Pair]:
    """Passes 1–3. The result is not yet arranged: call `arrange` (after `by_meaning`, if used)."""
    sm = SequenceMatcher(None, [u.key for u in a], [v.key for v in b], autojunk=False)
    pairs: list[Pair] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            pairs += [Pair(i1 + k, j1 + k) for k in range(i2 - i1)]
        else:
            pairs += _in_order(a, b, list(range(i1, i2)), list(range(j1, j2)), match_min)
    lefts, rights = _leftovers(pairs)
    candidates = sorted(((wording(a[i], b[j], move_min), i, j) for i in lefts for j in rights), reverse=True)
    found, used_l, used_r = [], set(), set()
    for s, i, j in candidates:
        if s < move_min:
            break
        if i in used_l or j in used_r:
            continue
        used_l.add(i)
        used_r.add(j)
        found.append(Pair(i, j, s, "moved"))
    return _join(pairs, found)


def leftovers(pairs: list[Pair]) -> tuple[list[int], list[int]]:
    return _leftovers(pairs)


def by_meaning(a: list[Unit], b: list[Unit], pairs: list[Pair], cosine: dict[tuple[int, int], float],
               cosine_min: float, wording_min: float = 0.15) -> list[Pair]:
    """Pass 4: pair leftovers that mean the same thing (cosine of their embeddings) and still share
    some wording, most similar first."""
    ranked = sorted(((c, i, j) for (i, j), c in cosine.items() if c >= cosine_min), reverse=True)
    found, used_l, used_r = [], set(), set()
    for c, i, j in ranked:
        if i in used_l or j in used_r:
            continue
        w = wording(a[i], b[j])
        if w < wording_min:
            continue
        used_l.add(i)
        used_r.add(j)
        found.append(Pair(i, j, w, "meaning"))
    return _join(pairs, found)


def arrange(pairs: list[Pair]) -> list[Pair]:
    """Mark pairs out of order as moved (those outside the longest run in order), then order everything
    as the right-hand version reads; a removed unit follows the place its neighbour went."""
    matched = sorted((p for p in pairs if p.left is not None and p.right is not None), key=lambda p: p.left)
    tails, tail_at, prev = [], [], [None] * len(matched)
    for k, p in enumerate(matched):
        pos = bisect.bisect_left(tails, p.right)
        if pos == len(tails):
            tails.append(p.right)
            tail_at.append(k)
        else:
            tails[pos], tail_at[pos] = p.right, k
        prev[k] = tail_at[pos - 1] if pos > 0 else None
    keep, k = set(), (tail_at[-1] if tail_at else None)
    while k is not None:
        keep.add(k)
        k = prev[k]
    for k, p in enumerate(matched):
        p.moved = k not in keep
    anchor = {p.left: p.right for p in matched if not p.moved}

    def key(p: Pair):
        if p.right is not None:
            return (p.right, 0, 0)
        i = p.left - 1
        while i >= 0 and i not in anchor:
            i -= 1
        return (anchor[i] if i >= 0 else -1, 1, p.left)
    return sorted(pairs, key=key)


# ── What changed inside a pair ───────────────────────────────────────────────────
@dataclass
class Diff:
    left_marks: list[tuple[int, int]]
    right_marks: list[tuple[int, int]]
    changes: list[dict]
    flags: list[str]


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def kind(u: Optional[Unit], v: Optional[Unit], ta: str, tb: str) -> str:
    if u is None:
        return "added"
    if v is None:
        return "removed"
    if u.key != v.key:
        return "changed"
    return "unchanged" if _squash(ta[u.lead_end:u.end]) == _squash(tb[v.lead_end:v.end]) else "formatting"


def _is_negation(w: str) -> bool:
    return w in _NEGATION or w.endswith("n't")


def _flags(removed: list[str], added: list[str]) -> list[str]:
    flags = []
    if any(re.search(r"\d", w) or w in _NUMBER_WORDS for w in removed + added):
        flags.append("number")
    if {w for w in removed if w in _MODAL} != {w for w in added if w in _MODAL}:
        flags.append("obligation")
    if {w for w in removed if _is_negation(w)} != {w for w in added if _is_negation(w)}:
        flags.append("negation")
    return flags


def diff(u: Unit, v: Unit, ta: str, tb: str) -> Diff:
    a, b = u.words, v.words
    ops = [op for op in SequenceMatcher(None, [w for w, _, _ in a], [w for w, _, _ in b], autojunk=False).get_opcodes()]
    # Group changes separated by at most two short words or marks ("thirty (30)" → "sixty (60)" is one change).
    groups: list[list[int]] = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            if groups and i2 - i1 <= 2 and all(len(w) <= 3 for w, _, _ in a[i1:i2]):
                groups[-1].append(-1)                # a small gap, possibly bridged
            else:
                groups.append([])
            continue
        if groups and groups[-1] and groups[-1][-1] == -1:
            groups[-1].pop()
            groups[-1] += [i1, i2, j1, j2]
        elif groups and groups[-1]:
            groups[-1] += [i1, i2, j1, j2]
        else:
            groups.append([i1, i2, j1, j2])
    left_marks, right_marks, changes, removed, added = [], [], [], [], []
    for g in groups:
        g = [x for x in g if x != -1]
        if not g:
            continue
        i1, i2 = min(g[0::4]), max(g[1::4])
        j1, j2 = min(g[2::4]), max(g[3::4])
        if i2 < len(a) and j2 < len(b) and a[i2][0] == ")" and b[j2][0] == ")":
            span_l = " ".join(w for w, _, _ in a[i1:i2])
            if span_l.count("(") > span_l.count(")"):
                i2, j2 = i2 + 1, j2 + 1
        left = (a[i1][1], a[i2 - 1][2]) if i2 > i1 else None
        right = (b[j1][1], b[j2 - 1][2]) if j2 > j1 else None
        if left:
            left_marks.append(left)
        if right:
            right_marks.append(right)
        changes.append({"from": ta[left[0]:left[1]] if left else "", "to": tb[right[0]:right[1]] if right else ""})
        removed += [w for w, _, _ in a[i1:i2]]
        added += [w for w, _, _ in b[j1:j2]]
    return Diff(left_marks, right_marks, changes, _flags(removed, added))
