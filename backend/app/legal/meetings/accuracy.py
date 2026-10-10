"""How accurate a legal transcript is (S2.4, LIVE-01; decision D-040).

A transcript is scored against a reference script, the way a lawyer would judge it:
- **Word error rate.** Two figures:
  - *verbatim:* case and punctuation count;
  - *normalised:* case-folded, punctuation and filler removed, numbers written as digits.
- **Entities.** Every name, amount, date and reference in the script must be in the transcript.
  "Eighteen thousand five hundred dollars" counts for "USD 18,500", and "March 2nd, 2026" counts for
  "2 March 2026". A missed or altered entity is listed.
- **Negations.** Every negated statement in the script ("we will not renew") must keep its negation.
  A transcript that drops a "not" says the opposite, and is listed.
- **Speakers.** The share of reference words attributed to the right speaker, after aligning the
  words.

Nothing here calls a model. The harness (`eval/law/transcription/`) feeds it the reference scripts
and whatever the speech service produced.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

_FOLD = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                                     "fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
             "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16,
             "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30}
_CURRENCY_WORDS = {"dollar": "usd", "dollars": "usd", "usd": "usd", "$": "usd", "euro": "eur", "euros": "eur", "eur": "eur",
                   "€": "eur", "pound": "gbp", "pounds": "gbp", "sterling": "gbp", "gbp": "gbp", "£": "gbp", "kronor": "sek",
                   "krona": "sek", "sek": "sek"}
_FILLER = {"um", "uh", "erm", "er", "ah", "hmm", "mm"}
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
           "december"]


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").translate(_FOLD)


def words(text: str) -> list[str]:
    """Verbatim words: case and punctuation kept, split on spaces."""
    return _fold(text).split()


def _numbers(tokens: list[str]) -> list[str]:
    """Spoken numbers as digits: "eighteen thousand five hundred" → "18500", "twenty-four" → "24",
    and spoken years: "twenty twenty-six" → "2026", "nineteen ninety-nine" → "1999"."""
    out: list[tuple[str, bool]] = []                 # (token, spoken as words)
    i = 0
    while i < len(tokens):
        total, current, used, j = 0, 0, False, i
        while j < len(tokens):
            t = tokens[j]
            parts = t.split("-")
            tens_unit = len(parts) == 2 and parts[0] in _TENS and parts[1] in _UNITS and _UNITS[parts[1]] < 10
            small = current % 100
            if tens_unit or t in _TENS:
                if used and small >= 10:             # "twenty twenty": a new number starts
                    break
                current += (_TENS[parts[0]] + _UNITS[parts[1]]) if tens_unit else _TENS[t]
            elif t in _UNITS:
                if used and (small % 10 or 10 <= small < 20 or (_UNITS[t] >= 10 and small)):
                    break
                current += _UNITS[t]
            elif t in _SCALES and used:
                if _SCALES[t] == 100:
                    current *= 100
                else:
                    total += current * _SCALES[t]
                    current = 0
            elif t == "and" and used and j + 1 < len(tokens) and (tokens[j + 1] in _UNITS or tokens[j + 1].split("-")[0] in _TENS):
                pass
            else:
                break
            used = True
            j += 1
        if used:
            if tokens[j - 1] == "and":                   # "… ninety-nine and one hundred": the "and" is not part of it
                j -= 1
            out.append((str(total + current), True))
            i = j
        else:
            out.append((tokens[i], False))
            i += 1
    merged: list[str] = []
    k = 0
    while k < len(out):                              # spoken years: two spoken numbers, 10–20 then 0–99
        a, spoken_a = out[k]
        if spoken_a and k + 1 < len(out) and out[k + 1][1] and a.isdigit() and 10 <= int(a) <= 20 and \
                out[k + 1][0].isdigit() and int(out[k + 1][0]) < 100:
            merged.append(str(int(a) * 100 + int(out[k + 1][0])))
            k += 2
            continue
        merged.append(a)
        k += 1
    return merged


def normalised(text: str) -> list[str]:
    """Words for the normalised error rate: case-folded, punctuation and filler removed, numbers as
    digits (thousands separators dropped), ordinals as numbers."""
    t = _fold(text).casefold()
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)                      # 18,500 → 18500
    t = re.sub(r"(\d+)(?:st|nd|rd|th)\b", r"\1", t)                   # 2nd → 2
    t = re.sub(r"[$€£]", lambda m: f" {m.group(0)} ", t)
    tokens = [w for w in re.findall(r"[$€£]|[a-z0-9]+(?:[-'.][a-z0-9]+)*", t) if w not in _FILLER]
    tokens = [str(_ORDINALS[w]) if w in _ORDINALS else w for w in tokens]
    tokens = [w[:-2] if w.endswith("'s") else w for w in _numbers(tokens)]
    # Amounts in one form: "USD 400000", "$400000" and "400000 dollars" all become "400000 usd".
    out: list[str] = []
    k = 0
    while k < len(tokens):
        w = tokens[k]
        code = _CURRENCY_WORDS.get(w)
        if code and k + 1 < len(tokens) and re.fullmatch(r"\d+(?:\.\d+)?", tokens[k + 1]):
            out += [tokens[k + 1], code]
            k += 2
            continue
        out.append(code if code and out and re.fullmatch(r"\d+(?:\.\d+)?", out[-1]) else w)
        k += 1
    return out


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(ref: list[str], hyp: list[str]) -> float:
    return edit_distance(ref, hyp) / max(1, len(ref))


# ── Entities ─────────────────────────────────────────────────────────────────────
def _amount(tokens: list[str]) -> Optional[tuple[str, str]]:
    cur = next((_CURRENCY_WORDS[t] for t in tokens if t in _CURRENCY_WORDS), None)
    num = next((t for t in tokens if re.fullmatch(r"\d+(?:\.\d+)?", t)), None)
    return (cur, num) if cur and num else None


def _dates(tokens: list[str]) -> set[str]:
    """Every date mentioned, as ISO: "2 march 2026", "march 2 2026"."""
    found = set()
    for i, t in enumerate(tokens):
        if t in _MONTHS:
            m = _MONTHS.index(t) + 1
            before = tokens[i - 2] if i > 1 and tokens[i - 1] == "of" else tokens[i - 1] if i > 0 else ""
            after = tokens[i + 1] if i + 1 < len(tokens) else ""
            day = before if before.isdigit() and int(before) <= 31 else after if after.isdigit() and int(after) <= 31 else None
            year = next((x for x in tokens[i + 1:i + 4] if re.fullmatch(r"(?:19|20)\d\d", x)), None)
            if day and year and day != year:
                try:
                    found.add(date(int(year), m, int(day)).isoformat())
                except ValueError:
                    pass
    return found


@dataclass
class Entity:
    kind: str                      # name | amount | date | reference
    text: str


def _contains(hay: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return n > 0 and any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def entity_found(e: Entity, hyp_norm: list[str]) -> bool:
    ref = normalised(e.text)
    if e.kind == "amount":
        want = _amount(ref)
        if want:
            cur, num = want
            idx = [i for i, t in enumerate(hyp_norm) if t == num]
            return any(_CURRENCY_WORDS.get(t) == cur for i in idx for t in hyp_norm[max(0, i - 3):i + 4])
    if e.kind == "date":
        want = _dates(ref)
        if want:
            return want <= _dates(hyp_norm)
    return _contains(hyp_norm, ref)


def negation_kept(statement: str, hyp_norm: list[str]) -> bool:
    """The negated statement is in the transcript word for word (normalised), negator included: a
    transcript with the same words but no "not" says the opposite and does not count."""
    return _contains(hyp_norm, normalised(statement))


# ── Speakers ─────────────────────────────────────────────────────────────────────
def _align(ref: list[str], hyp: list[str]) -> list[tuple[Optional[int], Optional[int]]]:
    """Word alignment (Levenshtein backtrace): pairs of (ref index, hyp index)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    out, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            out.append((i - 1, None))
            i -= 1
        else:
            out.append((None, j - 1))
            j -= 1
    return out[::-1]


@dataclass
class Score:
    wer_verbatim: float
    wer_normalised: float
    entities: int
    entities_found: int
    missed_entities: list[str] = field(default_factory=list)
    negations: int = 0
    negations_kept: int = 0
    lost_negations: list[str] = field(default_factory=list)
    speaker_accuracy: Optional[float] = None

    def as_dict(self) -> dict:
        return {"wer_verbatim": round(self.wer_verbatim, 4), "wer_normalised": round(self.wer_normalised, 4),
                "entity_accuracy": round(self.entities_found / self.entities, 4) if self.entities else None,
                "entities": self.entities, "missed_entities": self.missed_entities,
                "negation_accuracy": round(self.negations_kept / self.negations, 4) if self.negations else None,
                "negations": self.negations, "lost_negations": self.lost_negations,
                "speaker_accuracy": None if self.speaker_accuracy is None else round(self.speaker_accuracy, 4)}


def score(reference: list[dict], hypothesis: list[dict], entities: list[Entity], negations: list[str]) -> Score:
    """reference and hypothesis: ordered turns {"speaker": str | None, "text": str}."""
    ref_text = " ".join(t["text"] for t in reference)
    hyp_text = " ".join(t["text"] for t in hypothesis)
    hyp_norm = normalised(hyp_text)
    s = Score(wer(words(ref_text), words(hyp_text)), wer(normalised(ref_text), hyp_norm), len(entities), 0)
    for e in entities:
        if entity_found(e, hyp_norm):
            s.entities_found += 1
        else:
            s.missed_entities.append(e.text)
    s.negations = len(negations)
    for n in negations:
        if negation_kept(n, hyp_norm):
            s.negations_kept += 1
        else:
            s.lost_negations.append(n)
    if any(t.get("speaker") for t in hypothesis):
        ref_words = [(w, t.get("speaker")) for t in reference for w in normalised(t["text"])]
        hyp_words = [(w, t.get("speaker")) for t in hypothesis for w in normalised(t["text"])]
        pairs = [(i, j) for i, j in _align([w for w, _ in ref_words], [w for w, _ in hyp_words]) if i is not None]
        right = sum(1 for i, j in pairs if j is not None and ref_words[i][1] == hyp_words[j][1])
        s.speaker_accuracy = right / max(1, len(ref_words))
    return s
