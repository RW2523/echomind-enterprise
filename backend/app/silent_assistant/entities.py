"""Fast, LLM-free entity extraction for spoken sentences + identifier normalization.

Speech transcripts spell identifiers out ("A T four four eight two one", "double two",
"oh"), so we normalize spoken digits before matching. Names come from honorific/intro
cues ("this is Priya Nair", "my name is", "Mr Okafor") — the LLM verify pass may add
names regex missed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

_SPOKEN_DIGITS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
_MULTIPLIER = {"double": 2, "triple": 3}
_NUM_WORDS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUM_WORD_RE = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|double|triple|and)"


def words_to_number(phrase: str) -> Optional[int]:
    """'seventeen' -> 17, 'two thousand twenty four' -> 2024, 'twenty twenty four' -> 2024, 'one fifty' -> 150."""
    toks = [t for t in re.split(r"[\s\-]+", (phrase or "").lower()) if t and t != "and"]
    if not toks:
        return None
    total, cur = 0, 0
    year_style = []
    for t in toks:
        if t in _SPOKEN_DIGITS:
            cur = cur * 10 + int(_SPOKEN_DIGITS[t]) if cur and cur < 10 else (cur + int(_SPOKEN_DIGITS[t]) if cur >= 20 else int(_SPOKEN_DIGITS[t]) if not cur else cur * 10 + int(_SPOKEN_DIGITS[t]))
            year_style.append(int(_SPOKEN_DIGITS[t]))
        elif t in _NUM_WORDS:
            n = _NUM_WORDS[t]
            if cur >= 20 and n < 10:
                cur += n
            elif cur and cur < 10 and n >= 10:
                # 'two twenty' style -> 220 (year/id speech)
                cur = cur * 100 + n
            else:
                cur += n
            year_style.append(n)
        elif t == "hundred":
            cur = (cur or 1) * 100
        elif t == "thousand":
            total += (cur or 1) * 1000
            cur = 0
        elif t.isdigit():
            cur = cur * (10 ** len(t)) + int(t)
        else:
            return None
    total += cur
    # 'twenty twenty four' -> 2024
    if len(year_style) >= 2 and all(10 <= y < 100 for y in year_style[:1]) and total < 100 and toks and _NUM_WORDS.get(toks[0], 0) >= 20:
        pass
    if len(toks) == 2 and toks[0] in _NUM_WORDS and _NUM_WORDS[toks[0]] >= 20 and toks[1] in _NUM_WORDS and _NUM_WORDS[toks[1]] >= 20:
        return _NUM_WORDS[toks[0]] * 100 + _NUM_WORDS[toks[1]]
    if len(toks) == 3 and toks[0] in _NUM_WORDS and _NUM_WORDS[toks[0]] >= 20 and toks[1] in _NUM_WORDS and toks[2] in _SPOKEN_DIGITS:
        return _NUM_WORDS[toks[0]] * 100 + _NUM_WORDS[toks[1]] + int(_SPOKEN_DIGITS[toks[2]])
    return total


# ── Spoken-number normalization (STT/TTS verbalize digits) ─────────────────────
_UNITS = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TEENS_TENS = dict(_NUM_WORDS)
_ORDINAL = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
            "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16,
            "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30, "fortieth": 40}
_NUMBERISH = set(_UNITS) | set(_TEENS_TENS) | {"hundred", "thousand", "million", "billion", "point", "double", "triple", "and"} | set(_ORDINAL)


def _flush_group(units_seq, out):
    """Digit-string groups (phone/ID speech): consecutive unit words are concatenated."""
    if units_seq:
        out.append("".join(str(u) for u in units_seq))
        units_seq.clear()


def normalize_spoken_numbers(text: str) -> str:
    """Rewrite spoken numbers as digits, keeping everything else verbatim.
      'one hundred fifty dollars'                 -> '150 dollars'
      'five hundred fifty five zero one four two'  -> '555 0142'   (compound group, then digit string)
      'three point one percent'                    -> '3.1 percent'
      'thirty first of March two thousand twenty seven' -> '31 of March 2027'
      'twenty twenty four'                         -> '2024' (year style)
      'A T four four eight two one'                -> 'A T 44821'
      'the fourteen day cooling off period'        -> 'the 14 day cooling off period'
    Not perfect English number parsing — tuned for identifiers, amounts, dates, rates."""
    if not text:
        return text
    toks = re.findall(r"[A-Za-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*|[^\sA-Za-z\d]|\s+", text)
    out: List[str] = []
    i = 0
    n = len(toks)
    while i < n:
        t = toks[i]
        low = t.lower()
        if low not in _NUMBERISH or low == "and":
            out.append(t); i += 1; continue
        # collect a run of number-ish tokens (allowing single spaces / hyphens between)
        j = i
        run: List[str] = []
        while j < n:
            tj = toks[j]; lj = tj.lower()
            if lj in _NUMBERISH:
                run.append(lj); j += 1
            elif tj.isspace() or tj == "-":
                # lookahead: is the next real token number-ish?
                k = j + 1
                while k < n and (toks[k].isspace() or toks[k] == "-"):
                    k += 1
                if k < n and toks[k].lower() in _NUMBERISH:
                    j = k
                else:
                    break
            else:
                break
        # trailing 'and' / 'point' should not be part of the run
        while run and run[-1] in ("and", "point"):
            run.pop(); j -= 1
            while j > i and (toks[j - 1].isspace() or toks[j - 1] == "-"):
                j -= 1
        if not run or all(r == "and" for r in run):
            out.append(t); i += 1; continue
        out.append(_render_number_run(run))
        i = j
    return "".join(out)


def _render_number_run(run: List[str]) -> str:
    """Render a run of number words. Handles compound numbers (hundreds/tens/units, thousand),
    'point' decimals, digit strings, year-style pairs, ordinals."""
    # split at 'point' -> integer part . fractional digits
    if "point" in run:
        k = run.index("point")
        left = _render_number_run(run[:k]) if run[:k] else "0"
        frac = "".join(str(_UNITS.get(w, "")) for w in run[k + 1:] if w in _UNITS) or _render_number_run(run[k + 1:])
        return f"{left}.{frac}"
    groups: List[str] = []           # rendered pieces, joined by space when they are separate groups
    cur: Optional[int] = None        # current compound value
    digit_str: List[int] = []        # consecutive plain units (phone/ID style)
    pending_mult = 0
    def close_cur():
        nonlocal cur
        if cur is not None:
            groups.append(str(cur)); cur = None
    def close_digits():
        if digit_str:
            groups.append("".join(str(d) for d in digit_str)); digit_str.clear()
    prev = None
    for w in run:
        if w == "and":
            continue
        if w in ("double", "triple"):
            pending_mult = 2 if w == "double" else 3; prev = w; continue
        if w in _ORDINAL:
            close_digits()
            v = _ORDINAL[w]
            if cur is not None and cur >= 20 and v < 10:
                cur += v
            else:
                close_cur(); cur = v
            close_cur(); prev = w; continue
        if w in _UNITS:
            v = _UNITS[w]
            if pending_mult:
                close_cur()
                digit_str.extend([v] * pending_mult); pending_mult = 0; prev = w; continue
            if cur is not None and (prev in _TEENS_TENS and _TEENS_TENS.get(prev, 0) >= 20 or prev == "hundred"):
                cur += v; close_cur()          # 'fifty five' / 'hundred five' completes the compound
            elif cur is not None:
                close_cur(); digit_str.append(v)
            else:
                digit_str.append(v)
            prev = w; continue
        if w in _TEENS_TENS:
            v = _TEENS_TENS[w]
            close_digits()
            if cur is not None and prev in _TEENS_TENS and _TEENS_TENS[prev] >= 20 and v >= 20:
                # 'twenty twenty' year style -> 2020 (+ next unit)
                cur = cur * 100 + v
            elif cur is not None and prev == "hundred":
                cur += v
            elif cur is not None and prev in _UNITS and cur < 10:
                cur = cur * 100 + v            # 'two twenty' -> 220 (rare)
            else:
                close_cur(); cur = v
            if v < 20:                         # teens complete a compound
                close_cur()
            prev = w; continue
        if w == "hundred":
            base = digit_str.pop() if (cur is None and digit_str) else (cur if cur is not None else 1)
            close_digits(); cur = base * 100; prev = w; continue
        if w in ("thousand", "million", "billion"):
            mult = {"thousand": 1000, "million": 10**6, "billion": 10**9}[w]
            base = digit_str.pop() if (cur is None and digit_str) else (cur if cur is not None else 1)
            close_digits(); cur = base * mult; prev = w
            # keep cur open so 'two thousand twenty four' continues to add
            continue
        prev = w
    close_cur(); close_digits()
    # 'two thousand' + '24' style: merge consecutive groups where first is a round thousand and second < 1000
    merged: List[str] = []
    for g in groups:
        if merged and merged[-1].isdigit() and g.isdigit() and int(merged[-1]) % 1000 == 0 and int(merged[-1]) >= 1000 and int(g) < 1000:
            merged[-1] = str(int(merged[-1]) + int(g))
        elif merged and merged[-1].isdigit() and g.isdigit() and int(merged[-1]) % 100 == 0 and 100 <= int(merged[-1]) < 1000 and int(g) < 100 and len(g) <= 2:
            merged[-1] = str(int(merged[-1]) + int(g))
        else:
            merged.append(g)
    return " ".join(merged)


@dataclass
class Entity:
    kind: str
    value: str                 # as spoken (trimmed)
    normalized: str            # canonical key for matching
    span: tuple                # (start, end) in the sentence
    confidence: float = 0.8
    cue: str = ""
    id: str = ""
    subject_id: Optional[str] = None
    role: Optional[str] = None
    variants: List[str] = field(default_factory=list)   # alternative surface forms for exact lookup

    def public(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "value": self.value, "normalized": self.normalized,
            "confidence": round(self.confidence, 2), "subject_id": self.subject_id, "role": self.role,
        }


def spoken_to_digits(text: str) -> str:
    """'A T four four eight two one' -> 'AT44821'; 'double two five' -> '225'; keeps other words."""
    out: List[str] = []
    toks = re.split(r"(\s+|-)", text)
    pending_mult = 0
    for t in toks:
        low = t.lower().strip()
        if not low or low.isspace() or low == "-":
            continue
        if low in _MULTIPLIER:
            pending_mult = _MULTIPLIER[low]
            continue
        if low in _SPOKEN_DIGITS:
            d = _SPOKEN_DIGITS[low]
            out.append(d * (pending_mult or 1))
            pending_mult = 0
            continue
        pending_mult = 0
        out.append(t)
    return "".join(out) if all(re.fullmatch(r"[A-Za-z0-9]+", o) for o in out) else " ".join(out)


def normalize_identifier(raw: str) -> str:
    """Canonical identifier key: spoken digits -> digits, drop spaces/hyphens/slashes/dots, upper-case.
    '17 of 2024' -> '17/2024' is kept as a variant by the case extractor."""
    s = spoken_to_digits(raw)
    s = re.sub(r"[\s\-./,]", "", s)
    return s.upper()


_HONORIFIC_RE = re.compile(r"^(mr|mrs|ms|miss|dr|prof|sir|madam|mx)\.?\s+", re.IGNORECASE)


def normalize_person(name: str) -> str:
    n = _HONORIFIC_RE.sub("", (name or "").strip())
    n = re.sub(r"[^A-Za-z\s'\-]", "", n)
    return re.sub(r"\s+", " ", n).strip().lower()


# ── Patterns ───────────────────────────────────────────────────────────────────
_NAME_CUES = re.compile(
    r"(?:this is|my name is|i am|i'm|speaking with|speaking to|calling for|on behalf of|"
    r"the client is|client (?:name )?is|customer (?:name )?is|it's|its|name's|patient is)\s+"
    r"((?:[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?)(?:\s+(?:[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?)){0,3})",
    re.IGNORECASE,
)
_HONORIFIC_NAME = re.compile(r"\b((?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")
_ID_CUES = re.compile(
    r"\b(customer|account|acct|policy|order|ticket|case|matter|reference|ref|claim|contract|member|card|invoice|loan|file)"
    r"\s*(?:number|no\.?|id|#)?\s*(?:is|was|:)?\s*"
    r"((?:(?-i:[A-Z]{2,4})[\s-]?|(?:[A-Za-z]\s){1,4}|(?-i:[A-Z])[\s-]?)?(?:(?:\d{4,}(?:[\s,\-]+\d+)*|(?:(?:\d+|[A-Z]{1,2}\d+|\d+[A-Z]{1,2}|zero|oh|one|two|three|four|five|six|seven|eight|nine|double|triple)(?:[\s\-,]+|(?=[^A-Za-z0-9])|$)){2,24})))",
    re.IGNORECASE,
)
# Bare alphanumeric codes right after a cue word: "ticket TCK-9031", "order ORD-88-12", "policy PL/2210"
_ID_CODE = re.compile(
    r"\b(customer|account|acct|policy|order|ticket|case|matter|reference|ref|claim|contract|member|card|invoice|loan|file)"
    r"[.,]?\s*(?:number|no\.?|id|#)?\s*(?:is|was|:)?\s*((?-i:[A-Z]{2,5})[\-/ ]?\d{2,8}(?:[\-/]\d{1,6})?)\b",
    re.IGNORECASE,
)
# Standalone letter+digit codes without a cue word: "TCK 9031", "AT-44821", "PL/2210" (>=2 letters, >=4 digits)
_BARE_CODE = re.compile(r"\b([A-Z]{2,5}[\-/ ]?\d{4,8}(?:[\-/]\d{1,6})?)\b")
_CASE_OF = re.compile(r"\b(?:case|matter)(?:\s+(?:is|was|:))?\s*(?:number|no\.?)?\s*(\d{1,5})\s*(?:of|/|-)\s*(\d{4})\b", re.IGNORECASE)
_CASE_OF_SPOKEN = re.compile(
    r"\b(?:case|matter)(?:\s+(?:is|was|:))?\s*(?:number|no\.?)?\s*((?:\d+|" + _NUM_WORD_RE + r")(?:[\s\-](?:\d+|" + _NUM_WORD_RE + r")){0,4})\s+of\s+((?:\d{4}|" + _NUM_WORD_RE + r"(?:[\s\-]" + _NUM_WORD_RE + r"){1,4}))\b",
    re.IGNORECASE,
)
_PHONE = re.compile(r"\b(\+?\d[\d\s\-().]{5,}\d)\b")
_PHONE_SPOKEN = re.compile(r"\b(?:phone|number|mobile|cell|reach me (?:on|at)|call me (?:on|at|back on))\s*(?:is|:)?\s*((?:(?:\d|zero|oh|one|two|three|four|five|six|seven|eight|nine|double|triple)[\s\-]?){7,14})", re.IGNORECASE)
_EMAIL = re.compile(r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
_EMAIL_SPOKEN = re.compile(r"\b([A-Za-z0-9._-]+)\s+at\s+([A-Za-z0-9-]+)\s+dot\s+([A-Za-z]{2,})\b", re.IGNORECASE)
_DOB = re.compile(r"\b(?:date of birth|dob|born on|birthday)\s*(?:is|:)?\s*([A-Za-z0-9 ,/\-]{6,30})", re.IGNORECASE)
_AMOUNT = re.compile(r"(\$\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:dollars|usd|percent|%|k|thousand|million|lakh|crore|rupees|inr|eur|euros|pounds|gbp)\b|\b\d[\d,]*(?:\.\d+)?\s?(?:per ?cent))", re.IGNORECASE)
_DATE = re.compile(
    r"\b((?:\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}?(?:st|nd|rd|th)?,?\s*(?:\d{4})?|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:next|last|this)\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday)|tomorrow|today)\b",
    re.IGNORECASE,
)
_CLAUSE = re.compile(r"\b(clause|section|article|paragraph|schedule)\s+(\d+(?:\.\d+)*[a-z]?)\b", re.IGNORECASE)
_STATUTE = re.compile(r"\b(\d+\s+U\.?S\.?C\.?\s*§?\s*\d+[a-z]?|section\s+\d+\s+of\s+the\s+[A-Z][A-Za-z ]+Act(?:,?\s+\d{4})?)\b")
_ORG = re.compile(r"\b((?:[A-Z][A-Za-z&]+\s+){0,3}(?:Inc|Ltd|LLC|LLP|Corp|Corporation|Bank|Builders|Telecom|Group|Company|Co|Holdings|Partners|Trust|Fund)\b\.?)")


def _add(ents: List[Entity], kind: str, value: str, span, conf: float, cue: str, allowed: Iterable[str], normalized: str = "", variants=None):
    if kind not in allowed:
        return
    value = value.strip(" ,.;:")
    if not value:
        return
    norm = normalized or (normalize_person(value) if kind in ("person", "org") else normalize_identifier(value))
    if len(norm) < 2:
        return
    ents.append(Entity(kind=kind, value=value, normalized=norm, span=span, confidence=conf, cue=cue, variants=list(variants or [])))


def extract_entities_fast(text: str, allowed_kinds: Iterable[str]) -> List[Entity]:
    allowed = set(allowed_kinds or ())
    ents: List[Entity] = []
    if not text:
        return ents
    # names
    for m in _NAME_CUES.finditer(text):
        cand = m.group(1)
        words = cand.split()
        # keep only the leading run of Capitalized words (the cue is case-insensitive, the name is not)
        keep = []
        _NOT_NAME = {"speaking", "speak", "here", "calling", "from", "with", "at", "and", "again", "today", "sir", "madam", "there", "the", "this", "that", "your", "our", "customer", "support", "service", "team", "care", "bank", "telecom"}
        for w in words:
            if w[:1].isupper() and w.lower() not in _NOT_NAME:
                keep.append(w)
            else:
                break
        if keep and not (len(keep) == 1 and keep[0].lower() in ("i", "the", "a", "an", "not", "so", "very", "here", "just", "calling", "speaking", "sorry", "afraid", "going", "sure", "glad", "happy")):
            _add(ents, "person", " ".join(keep), (m.start(1), m.start(1) + len(" ".join(keep))), 0.85, "name_cue", allowed)
    for m in _HONORIFIC_NAME.finditer(text):
        _add(ents, "person", m.group(1), m.span(1), 0.8, "honorific", allowed)
    # spoken "case 17 of 2024"
    for m in _CASE_OF_SPOKEN.finditer(text):
        n_i, y_i = words_to_number(m.group(1)), words_to_number(m.group(2))
        n, y = (str(n_i) if n_i is not None else ""), (str(y_i) if y_i is not None else "")
        if n.isdigit() and y.isdigit() and len(y) == 4:
            _add(ents, "case", m.group(0), m.span(0), 0.9, "case_of", allowed,
                 normalized=f"{n}/{y}", variants=[f"{n}/{y}", f"{n}-{y}", f"{n} of {y}", f"{n}of{y}"])
    for m in _CASE_OF.finditer(text):
        n, y = m.group(1), m.group(2)
        _add(ents, "case", m.group(0), m.span(0), 0.9, "case_of", allowed,
             normalized=f"{n}/{y}", variants=[f"{n}/{y}", f"{n}-{y}", f"{n} of {y}"])
    # bare codes: ticket TCK-9031
    for m in _ID_CODE.finditer(text):
        cue_word = m.group(1).lower(); raw = m.group(2).strip()
        norm = normalize_identifier(raw)
        kind = {"customer": "account", "account": "account", "acct": "account", "member": "account", "loan": "account",
                "card": "card", "policy": "policy", "order": "order", "invoice": "order", "ticket": "ticket", "claim": "ticket",
                "case": "case", "matter": "case", "file": "case", "reference": "account", "ref": "account", "contract": "contract"}.get(cue_word, "account")
        if kind not in allowed and "account" in allowed:
            kind = "account"
        variants = {raw, norm, raw.replace(" ", "-"), raw.replace("-", " ")}
        _add(ents, kind, raw, m.span(2), 0.9, f"id_code:{cue_word}", allowed, normalized=norm, variants=sorted(variants))
    for m in _BARE_CODE.finditer(text):
        raw = m.group(1).strip(); norm = normalize_identifier(raw)
        if any(e.normalized == norm for e in ents):
            continue
        digits = re.sub(r"\D", "", norm)
        kind = "ticket" if raw.upper().startswith(("TCK", "TKT", "INC", "CASE")) else ("order" if raw.upper().startswith(("ORD", "INV")) else ("policy" if raw.upper().startswith(("PL", "POL")) else "account"))
        if kind not in allowed and "account" in allowed:
            kind = "account"
        _add(ents, kind, raw, m.span(1), 0.75, "bare_code", allowed, normalized=norm,
             variants=sorted({raw, norm, raw.replace(" ", "-"), raw.replace("-", " "), digits}))
    # generic identifiers with a cue word
    for m in _ID_CUES.finditer(text):
        cue_word = m.group(1).lower()
        raw = m.group(2).strip()
        norm = normalize_identifier(raw)
        if not re.search(r"\d", norm) or len(norm) < 3:
            continue
        kind = {
            "customer": "account", "account": "account", "acct": "account", "member": "account", "loan": "account",
            "card": "card", "policy": "policy", "order": "order", "invoice": "order", "ticket": "ticket", "claim": "ticket",
            "case": "case", "matter": "case", "file": "case", "reference": "account", "ref": "account", "contract": "contract",
        }.get(cue_word, "account")
        if kind not in allowed and "account" in allowed:
            kind = "account"
        # surface variants: hyphenated in blocks of 3-4 for LIKE searches
        variants = {norm, raw}
        letters = re.match(r"^([A-Z]+)(\d+)$", norm)
        if letters:
            variants.add(f"{letters.group(1)}-{letters.group(2)}")
            variants.add(f"{letters.group(1)} {letters.group(2)}")
        digits = re.sub(r"\D", "", norm)
        if len(digits) >= 5:
            variants.add(digits)
        _add(ents, kind, raw, m.span(2), 0.85, f"id_cue:{cue_word}", allowed, normalized=norm, variants=sorted(variants))
    for m in _PHONE_SPOKEN.finditer(text):
        d = re.sub(r"\D", "", spoken_to_digits(m.group(1)))
        if 7 <= len(d) <= 14:
            _add(ents, "phone", m.group(1), m.span(1), 0.8, "phone_spoken", allowed, normalized=d,
                 variants=[d, f"{d[:3]}-{d[3:]}", f"{d[:3]} {d[3:]}"] if len(d) == 7 else [d, f"{d[:3]}-{d[3:6]}-{d[6:]}", f"{d[:3]}-{d[3:]}"])
    for m in _PHONE.finditer(text):
        d = re.sub(r"\D", "", m.group(1))
        if 7 <= len(d) <= 14 and not any(e.kind == "phone" and e.normalized == d for e in ents):
            _add(ents, "phone", m.group(1), m.span(1), 0.85, "phone", allowed, normalized=d,
                 variants=[d, m.group(1), f"{d[-7:-4]}-{d[-4:]}"])
    for m in _EMAIL.finditer(text):
        _add(ents, "email", m.group(1), m.span(1), 0.95, "email", allowed, normalized=m.group(1).lower(), variants=[m.group(1)])
    for m in _EMAIL_SPOKEN.finditer(text):
        addr = f"{m.group(1)}@{m.group(2)}.{m.group(3)}".lower()
        _add(ents, "email", m.group(0), m.span(0), 0.8, "email_spoken", allowed, normalized=addr, variants=[addr])
    for m in _DOB.finditer(text):
        _add(ents, "dob", m.group(1), m.span(1), 0.7, "dob", allowed, normalized=m.group(1).strip().lower(), variants=[m.group(1).strip()])
    for m in _AMOUNT.finditer(text):
        _add(ents, "amount", m.group(1), m.span(1), 0.7, "amount", allowed, normalized=re.sub(r"[\s,]", "", m.group(1)).lower())
    for m in _DATE.finditer(text):
        _add(ents, "date", m.group(1), m.span(1), 0.6, "date", allowed, normalized=m.group(1).strip().lower())
    for m in _CLAUSE.finditer(text):
        _add(ents, "clause", m.group(0), m.span(0), 0.85, "clause", allowed, normalized=f"{m.group(1).lower()} {m.group(2)}",
             variants=[m.group(0), f"{m.group(1).title()} {m.group(2)}", m.group(2)])
    for m in _STATUTE.finditer(text):
        _add(ents, "statute", m.group(1), m.span(1), 0.8, "statute", allowed, normalized=m.group(1).lower(), variants=[m.group(1)])
    for m in _ORG.finditer(text):
        val = m.group(1).strip()
        if len(val.split()) >= 2:
            _add(ents, "org", val, m.span(1), 0.65, "org", allowed, variants=[val])
    # de-dup by (kind, normalized)
    seen = set()
    out: List[Entity] = []
    for e in ents:
        key = (e.kind, e.normalized)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


PERSONAL_DETAIL_KINDS = {"person", "phone", "email", "dob", "account", "policy", "order", "ticket", "case", "card", "address", "contract"}
LOOKUP_KINDS = {"person", "phone", "email", "account", "policy", "order", "ticket", "case", "card", "contract", "clause", "statute", "org"}
