"""Term and notice dates, calculated from what a document states (2026-10-29 functional QA, fix brief RAG-003).

Asked when the term ends and by when notice of non-renewal is due, the model said the end date "is not specified" and gave
"45 days before" — the dates follow from the document's words, but the model does not do date arithmetic reliably, and an
answer must not present a calculation as something the document says. So the calculation is done here, deterministically:

- the facts are read from the document's own text (start date, term, a stated end date, renewal term, notice period before
  the term ends), each with its exact words and characters, so the reader can open the clause;
- the dates are derived with calendar arithmetic and labelled as calculated, with every assumption written out;
- nothing is calculated when a fact is missing or the document states it more than once in different ways.

Counting rules (written into every result): a term of N months runs from its start date to the day before the
corresponding date N months later, or to the last day of that month when it has no corresponding date (31 January plus
one month ends on the last day of February; 29 February plus twelve months ends on 28 February). Notice "at least N days
before the term ends" is counted back in calendar days from the term's last day. Business days are not calculated: they
depend on a holiday calendar the document does not give.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

_MONTHS = ({m.lower(): i for i, m in enumerate(calendar.month_name) if m} | {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
           | {"sept": 9})
_M = r"(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?"
_DATE = (rf"(?:(?P<d1>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<m1>{_M})\s*,?\s+(?P<y1>\d{{4}})"
         rf"|(?P<m2>{_M})\s+(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s+(?P<y2>\d{{4}})"
         rf"|(?P<y3>\d{{4}})-(?P<m3>\d{{2}})-(?P<d3>\d{{2}}))")
_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
          "twelve": 12, "fifteen": 15, "eighteen": 18, "twenty": 20, "twenty-four": 24, "thirty": 30, "thirty-six": 36, "forty-five": 45,
          "sixty": 60, "ninety": 90}
_NUM = r"(?P<num>\d{1,3}|" + "|".join(sorted(map(re.escape, _WORDS), key=len, reverse=True)) + r")(?:\s*\((?P<paren>\d{1,3})\))?"
_UNIT = r"(?P<unit>calendar\s+months?|months?|years?|calendar\s+days?|days?|business\s+days?|working\s+days?)"

_START = re.compile(rf"\b(?:begins|commences|starts|starting|commencing|beginning|effective\s+(?:from|on|as\s+of))\s+(?:on\s+)?{_DATE}", re.I)
_TERM = re.compile(rf"\b(?:remains?\s+(?:effective|in\s+(?:force|effect))|continues?|in\s+force|initial\s+term\s+of|term\s+(?:is|of)|"
                   rf"for\s+an?\s+(?:initial\s+)?(?:term|period)\s+of|lasts?)\s+(?:for\s+)?(?:an?\s+)?(?:initial\s+term\s+of\s+)?{_NUM}\s*-?\s*{_UNIT}\b"
                   rf"(?!\s+(?:after|following|beyond|from\s+the\s+(?:end|termination|expiry)))",       # a survival period is not the term
                   re.I)
# "This Agreement begins on the Effective Date" with "entered into as of 2 March 2026 (the "Effective Date")"
_FROM_EFFECTIVE = re.compile(r"\b(?:begins|commences|starts|continues|lasts)\s+(?:on|from)\s+the\s+Effective\s+Date\b"
                             r"|\bfrom\s+the\s+Effective\s+Date\b", re.I)
_EFFECTIVE = re.compile(rf"\b(?:as\s+of|on|dated)\s+{_DATE}\s*\(\s*the\s+[\"“”']?Effective\s+Date[\"“”']?\s*\)", re.I)
_END = re.compile(rf"\b(?:ending|ends|expires|expiring|terminating|until)\s+(?:on\s+)?{_DATE}", re.I)
_RENEW = re.compile(rf"\brenews?\s+(?:automatically\s+)?(?:for\s+)?(?:an?\s+)?(?:additional\s+|successive\s+|further\s+)?(?:period\s+of\s+|term\s+of\s+)?"
                    rf"{_NUM}\s*-?\s*{_UNIT}", re.I)
_NOTICE = re.compile(rf"\b(?:at\s+least|no\s+(?:later|less)\s+than|not\s+less\s+than|a\s+minimum\s+of)\s+{_NUM}\s*{_UNIT}\s+"
                     rf"(?:prior\s+to|before)\s+(?:the\s+(?:end|expiry|expiration)\s+of\s+the\s+(?:then[- ])?(?:current\s+|initial\s+)?term|"
                     rf"the\s+(?:then[- ])?(?:current\s+)?term\s+(?:ends|expires)|(?:its\s+)?expiry|the\s+end\s+of\s+(?:the\s+)?term)", re.I)

_ASKS = re.compile(r"\b(?:when|what\s+date|which\s+date|deadline|by\s+when|last\s+day|end\s+date|expir\w*|how\s+long)\b", re.I)
_ABOUT = re.compile(r"\b(?:term|renew\w*|non-renewal|expir\w*|roll\s*over|end\s+of\s+the\s+(?:agreement|contract|term))\b", re.I)


def asks_term_dates(question: str) -> bool:
    """The question asks when a term ends, renews, or by when its notice is due."""
    q = question or ""
    return bool(_ASKS.search(q) and _ABOUT.search(q))


@dataclass
class Fact:
    label: str
    value: str
    start: int
    end: int

    def words(self, text: str) -> str:
        return text[self.start:self.end]


@dataclass
class Calculation:
    facts: list[Fact] = field(default_factory=list)
    derived: list[dict] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    not_calculated: list[str] = field(default_factory=list)


def _month(name: str) -> int:
    return _MONTHS[name.lower().rstrip(".")]


def _date(m: re.Match) -> Optional[date]:
    g = m.groupdict()
    try:
        if g.get("y1"):
            return date(int(g["y1"]), _month(g["m1"]), int(g["d1"]))
        if g.get("y2"):
            return date(int(g["y2"]), _month(g["m2"]), int(g["d2"]))
        if g.get("y3"):
            return date(int(g["y3"]), int(g["m3"]), int(g["d3"]))
    except (ValueError, KeyError):
        return None
    return None


def _count(m: re.Match) -> Optional[tuple[int, str]]:
    raw = m.group("paren") or m.group("num")
    n = int(raw) if raw.isdigit() else _WORDS.get(raw.lower())
    if m.group("paren") and not m.group("num").isdigit() and _WORDS.get(m.group("num").lower()) not in (None, int(m.group("paren"))):
        return None                                   # "thirty (60) days": the document contradicts itself
    unit = m.group("unit").lower().split()[-1].rstrip("s")
    kind = "business" if re.match(r"(?:business|working)", m.group("unit"), re.I) else unit
    return (n, kind) if n else None


def add_months(start: date, months: int) -> date:
    """The corresponding date `months` later, or the last day of that month when it has none."""
    y, m = divmod(start.month - 1 + months, 12)
    y, m = start.year + y, m + 1
    return date(y, m, min(start.day, calendar.monthrange(y, m)[1]))


def term_end(start: date, n: int, unit: str) -> date:
    """The last day of a term of n months, years or days starting on `start` (inclusive)."""
    if unit == "day":
        return start + timedelta(days=n - 1)
    months = n * 12 if unit == "year" else n
    corresponding = add_months(start, months)
    if corresponding.day < start.day:                 # no corresponding date: the month's last day
        return corresponding
    return corresponding - timedelta(days=1)


def _one(pattern: re.Pattern, text: str, value) -> tuple[Optional[re.Match], bool]:
    """The single statement of a fact, and whether the document states it more than once in different ways."""
    found = [m for m in pattern.finditer(text) if value(m) is not None]
    values = {value(m) for m in found}
    return (found[0] if found else None), len(values) > 1


def _show(d: date) -> str:
    return f"{d.day} {calendar.month_name[d.month]} {d.year}"


def calculate(text: str) -> Optional[Calculation]:
    """The term's dates, derived from `text` (one document version), or None when it states no start and term."""
    start_m, start_many = _one(_START, text, _date)
    if not start_m and _FROM_EFFECTIVE.search(text):
        start_m, start_many = _one(_EFFECTIVE, text, _date)
    term_m, term_many = _one(_TERM, text, _count)
    end_m, end_many = _one(_END, text, _date)
    renew_m, renew_many = _one(_RENEW, text, _count)
    notice_m, notice_many = _one(_NOTICE, text, _count)
    if not start_m and not end_m:
        return None
    out = Calculation()
    if start_many or term_many or end_many or renew_many or notice_many:
        which = [w for w, many in (("start date", start_many), ("term", term_many), ("end date", end_many),
                                   ("renewal term", renew_many), ("notice period", notice_many)) if many]
        out.not_calculated.append(f"The document states its {' and '.join(which)} more than once, differently; no date is calculated.")
        return out
    start = _date(start_m) if start_m else None
    if start_m:
        out.facts.append(Fact("Start", _show(start), start_m.start(), start_m.end()))
    end = None
    if end_m:
        end = _date(end_m)
        out.facts.append(Fact("End (stated)", _show(end), end_m.start(), end_m.end()))
    elif start and term_m:
        n, unit = _count(term_m)
        out.facts.append(Fact("Term", f"{n} {unit}{'s' if n != 1 else ''}", term_m.start(), term_m.end()))
        if unit == "business":
            out.not_calculated.append("The term is given in business days, which depend on a holiday calendar the document does not give.")
        else:
            end = term_end(start, n, unit)
            out.derived.append({"label": "Current term ends", "date": end.isoformat(), "display": _show(end), "from": ["Start", "Term"]})
            out.assumptions.append(f"The term runs for {n} {'calendar ' if unit != 'day' else ''}{unit}{'s' if n != 1 else ''} from and "
                                   f"including {_show(start)}, to the day before the corresponding date (or the month's last day "
                                   "when it has none).")
    if end is None:
        out.not_calculated.append("The document states no term length, so the end of the term is not calculated.")
        return out
    if notice_m:
        n, unit = _count(notice_m)
        out.facts.append(Fact("Notice before the term ends", f"at least {n} {unit if unit != 'business' else 'business day'}{'s' if n != 1 else ''}",
                              notice_m.start(), notice_m.end()))
        if unit == "business":
            out.not_calculated.append("The notice period is in business days, which depend on a holiday calendar the document does not give.")
        else:
            last = end - timedelta(days=n) if unit == "day" else add_months(end, -n * (12 if unit == "year" else 1))
            out.derived.append({"label": "Last day to give notice of non-renewal", "date": last.isoformat(), "display": _show(last),
                                "from": ["Start", "Term", "Notice before the term ends"] if not end_m else ["End (stated)", "Notice before the term ends"]})
            out.assumptions.append(f"{n} calendar {unit}{'s' if n != 1 else ''} counted back from {_show(end)}, the term's last day; notice "
                                   "is in time if it is received on or before that date. Rules on how notice is delivered or deemed "
                                   "received, and any business-day rule, are not applied.")
    if renew_m:
        n, unit = _count(renew_m)
        out.facts.append(Fact("Renewal", f"{n} {unit}{'s' if n != 1 else ''}", renew_m.start(), renew_m.end()))
        if unit != "business":
            first = end + timedelta(days=1)
            last = term_end(first, n, unit)
            out.derived.append({"label": "If no notice is given, the next term", "date": first.isoformat(), "display": f"{_show(first)} to {_show(last)}",
                                "from": ["Renewal"] + (["Start", "Term"] if not end_m else ["End (stated)"])})
    out.assumptions.append("No earlier termination, amendment or extension; only this document's text is used.")
    return out
