"""Reading obligations and deadlines from a contract (S3.6, DOC-05; decision D-050). Rules only: no model.

A sentence of a numbered clause is an obligation when it binds someone ("shall", "must", "may",
"is payable", "unless … notice") and gives a period ("within forty-eight (48) hours of becoming aware",
"at least ninety (90) days before the end of the term"), or is a consent right. Each keeps:
- its kind: renewal, termination, change of control, consent, payment, notification, post-termination,
  confidentiality or notice;
- the party bound, as the document names it ("Provider", "Either party");
- what must be done, and when, in the document's own words;
- the exact sentence it was read from, at its offsets.

**A date is computed only from dates the document gives**, and the way of counting is said. Today
that is the non-renewal deadline: a term of N months from a dated start ("the Effective Date"), read
as ending the day before the anniversary (the earlier, safer reading), less the notice period. Every
other obligation runs from an event (an invoice, a breach, termination) and has no date until it
happens. Nothing here is confirmed: a lawyer confirms each one.
"""
from __future__ import annotations

import calendar
import re
from datetime import date, timedelta
from typing import Optional

from ..compare.facts import _DATE, _PERIOD, _date, _word_number
from ..documents import numbering

VERSION = "obligations-s3.6f"
MODAL = re.compile(r"\b(?:shall|must|may(?!\s+not\s+be\b)|will|agrees? to|undertakes? to|is entitled to|are entitled to|"
                   r"is payable|are payable|is due|are due|falls? due|survives?|bears interest|accrues? interest)\b")
_UNLESS_NOTICE = re.compile(r"\bunless\b[^.]*\bnotice\b", re.I)
_CUE = re.compile(r"(?:\b(?:within|at least|not less than|no less than|no later than|not later than|on|by|for|before|after)\s+)$", re.I)
_AFTER_CUE = re.compile(r"^['’]?s?\s*(?:prior\s+)?(?:written\s+)?(?:notice\s+)?(?P<dir>before|prior to|after|following|from|of)\b", re.I)
_CUT = re.compile(r",|\s+(?:in accordance with|without undue delay|in any event|within|on|at least|no less than|not less than|no later than|"
                  r"subject to|without|unless|if|by written notice|for the term|upon|by)\b")
_PARTY_WORDS = r"(?:Provider|Client|Supplier|Customer|Seller|Buyer|Licensor|Licensee|Landlord|Tenant|Borrower|Lender|Issuer|Company|" \
               r"Consultant|Contractor|Employer|Employee|Purchaser|Vendor|Guarantor|Agent|Bank|Parent|Shareholder|Investor)"
_SUBJECT = re.compile(rf"\b(?:(?:Either|Each|Neither|Both|Any|The other|The)\s+part(?:y|ies)|(?:the\s+)?{_PARTY_WORDS}s?)\b", re.I)
KINDS = ("expiry", "renewal", "termination", "change_of_control", "consent", "payment", "notification", "post_termination", "confidentiality",
         "survival", "duty", "notice")
_FREQUENCY = re.compile(r"\b(?:monthly|quarterly|annually|yearly|weekly|half-yearly)(?:\s+in\s+(?:advance|arrears))?(?:\s+on\s+[^.,;]+)?"
                        r"|\bon each anniversary of [^.,;]+", re.I)
_EACH_PERIOD = re.compile(r"\bby the \w+(?:\s+(?:Business|working))?\s+[Dd]ay of each (?:month|quarter|year)\b|\bby the end of each (?:month|quarter|year)\b")
_ORDINAL = {w: i for i, w in enumerate("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth".split(), start=1)}


def kind_of(s: str) -> Optional[str]:
    if re.search(r"\brenew", s, re.I):
        return "renewal"
    if re.search(r"\b(?:may|right to|entitled to)\s+terminate\b|\buntil terminated by\b|\bterminated by (?:either|any|each) party\b", s, re.I):
        return "termination"
    if re.search(r"change of control", s, re.I):
        return "change_of_control"
    if re.search(r"\bconsent\b", s, re.I):
        return "consent"
    if re.search(r"\b(?:pay|payable|paid|unpaid|non-payment|repay|repayment|invoices?)\b", s, re.I):
        return "payment"
    if re.search(r"\bnotif(?:y|ies|ied|ication)\b", s, re.I) or (re.search(r"\bnotice\b", s, re.I) and re.search(r"\b(?:breach|incident|aware)", s, re.I)):
        return "notification"
    if re.search(r"\b(?:return|destroy|delete)\b", s, re.I):
        return "post_termination" if re.search(r"\bterminat|\bexpir", s, re.I) else "duty"
    if re.search(r"\bsurvives?\b", s, re.I):
        return "survival"
    if re.search(r"confidential", s, re.I):
        return "confidentiality"
    if re.search(r"\bnotice\b", s, re.I):
        return "notice"
    return None


def _periods(s: str) -> list[dict]:
    out = []
    for m in _PERIOD.finditer(s):
        n = int(m.group("d")) if m.group("d") else _word_number(m.group("n"))
        if n is None:
            continue
        lead = re.search(r"(?:\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|sixty|seventy|"
                         r"eighty|ninety|hundred|thousand|and)[\s-]+)+$", s[:m.start()], re.I) if m.group("d") else None
        pstart = lead.start() if lead else m.start()
        before = _CUE.search(s[:pstart])
        after = _AFTER_CUE.match(s[m.end():])
        unit = m.group("unit").lower().rstrip("s") + ("s" if n != 1 else "")
        out.append({"n": n, "unit": unit, "business": bool(m.group("kind")), "start": before.start() if before else pstart, "pstart": m.start(),
                    "end": m.end(), "score": (1 if before else 0) + (1 if after else 0),
                    "direction": ("before" if after and after.group("dir").lower() in ("before", "prior to") else "after") if after else None})
    return out


def _party(s: str, at: int) -> Optional[str]:
    found = [m for m in _SUBJECT.finditer(s, 0, at)]
    if not found:
        return None
    name = re.sub(r"^the\s+", "", found[-1].group(0), flags=re.I)
    return name[0].upper() + name[1:]


_UNLESS_GIVES = re.compile(r"\bunless\s+(?P<who>[^.,]*?)\s+gives?\s+(?P<what>[^.]*?)\s+(?=at least|no less than|not less than|within|by\b)", re.I)


def _action(s: str, modal: Optional[re.Match], period: Optional[dict]) -> str:
    body = re.sub(r"^\s*(?:\d+(?:\.\d+)*|\([a-z0-9]+\))\.?\s+", "", s).strip()
    if modal is None:
        g = _UNLESS_GIVES.search(body)                     # "… renews unless either party gives written notice of non-renewal at least …"
        return f"give {g.group('what').strip()}" if g else body.rstrip(".")
    m = MODAL.search(body)
    if m and m.group(0).startswith(("is ", "are ", "fall")):
        return body[:m.end()].strip()                      # "Invoices are payable"
    rest = body[m.end():].strip() if m else body
    if m and rest.startswith("be "):                       # "Holiday must be taken by …": the subject stays
        rest = f"{body[:m.start()].strip()} {m.group(0)} {rest}"
    if period:
        ptxt = s[period["start"]:period["end"]]
        if ptxt in rest:
            head = rest[:rest.index(ptxt)].strip()
            if len(head.split()) <= 1:                     # "provide no less than thirty (30) days' prior written notice of …"
                rest = (rest[:rest.index(ptxt)] + rest[rest.index(ptxt) + len(ptxt):]).replace("  ", " ")
                rest = re.sub(r"^\s*(\w+)\s+['’]s?\s*", r"\1 ", rest)
    cut = _CUT.search(rest)
    out = rest[:cut.start()] if cut and cut.start() > 0 else rest
    return out.strip().rstrip(".,;")


def _when(s: str, kind: str, period: Optional[dict]) -> str:
    if not period:
        m = re.search(r"\b(?:without|with)\b[^.]*", s)
        return m.group(0).strip().rstrip(".") if m else ""
    tail = s[period["end"]:]
    stop = re.search(r",|\.(?:\s|$)|;|\s+(?:Provider|Client|Supplier|Customer)\s+shall\b|\s+in accordance with\b|\s+of\s+any\b", tail)
    return (s[period["start"]:period["end"]] + (tail[:stop.start()] if stop else tail)).strip().rstrip(".")


def add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    year, month = d.year + y, m + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def _fmt(d: date) -> str:
    return f"{d.day} {d.strftime('%B')} {d.year}"


def anchor_date(text: str, name: str) -> Optional[tuple[date, str]]:
    """The date a named anchor stands for, as the document gives it: '"Term Commencement Date" means 25 March 2026',
    'made on 4 May 2026 (the "Effective Date")'."""
    if re.fullmatch(r"(?i)date of this \w+", name):          # "made on 12 January 2026", "dated as of March 3, 2026"
        m = re.search(r"\b(?:made|dated|entered into)(?:\s+(?:on|as of))?\s+" + _DATE.pattern, text[:600])
        if m and _date(m):
            d0 = date.fromisoformat(_date(m))
            return d0, f"{_fmt(d0)} (the date of this agreement)"
        return None
    for dm in re.finditer(rf"\((?:the\s+)?[\"“]{re.escape(name)}[\"”]\)", text):
        ds = list(_DATE.finditer(text[max(0, dm.start() - 120):dm.start()]))
        if ds and _date(ds[-1]):
            d0 = date.fromisoformat(_date(ds[-1]))
            return d0, f"{_fmt(d0)} (the {name})"
    dm = re.search(rf"[\"“]{re.escape(name)}[\"”]\s+means\s+(?P<v>[^.;]+)", text)
    if dm:
        d = _DATE.search(dm.group("v"))
        if d and _date(d):
            d0 = date.fromisoformat(_date(d))
            return d0, f"{_fmt(d0)} (the {name})"
    return None


_ON_TERMINATION = re.compile(r"\b(?:on|upon|following|at)\s+(?:the\s+)?(?:termination|expiry|expiration|end of (?:the|this) \w+)\b", re.I)
_YEARLY = re.compile(r"\b(?:by|before|on or before)\s+\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|"
                     r"December)\s+(?:in\s+)?(?:each|every)\s+(?:calendar\s+)?year\b", re.I)
_ON_NAMED = re.compile(r"\b(?:on|by|no later than)\s+the\s+(?P<name>[A-Z][\w ]{2,40}?Date)\b")
_ON_DATE = re.compile(r"\b(?:take place|occur|be completed|completes?|on|by|no later than|not later than|on or before)\s+(?:on\s+|by\s+|or before\s+)?" + _DATE.pattern)


def event_date(text: str, name: str) -> Optional[tuple[date, str]]:
    """The date of a named event or anchor the document fixes: 'Completion shall take place on 30 September 2026',
    or a defined date."""
    m = re.search(rf"\b{re.escape(name)}\s+(?:shall|will|is to)\s+(?:take place|occur)\s+on\s+" + _DATE.pattern, text)
    if m and _date(m):
        d0 = date.fromisoformat(_date(m))
        return d0, f"{name} ({_fmt(d0)})"
    return anchor_date(text, name) if name.endswith("Date") else None


_EXPIRY = re.compile(r"(?:\bcontinues?\b|\blasts?\b|\bruns?\b|\bmeans\b|\bremains? in force\b)[^.;]{0,60}?\bfor\s+(?:an?\s+initial\s+(?:term|period)\s+of\s+)?"
                     r"(?P<p>[^.;()]{0,30}(?:\(\d+\))?\s*(?:months?|years?))(?:\s+from\s+(?:the\s+)?(?P<anchor>[A-Z][\w ]{2,40}?Date|date of this \w+))?"
                     r"|[\"“](?:Term|Initial Term)[\"”]\s+means\s+(?P<p2>[^.;()]{0,30}(?:\(\d+\))?\s*(?:months?|years?))\s+from\s+(?:the\s+)?(?P<anchor2>[A-Z][\w ]{2,40}?Date)"
                     r"|\b(?:initial\s+)?(?:[A-Z][\w-]*\s+)?[Tt]erm(?:\s+of\s+this\s+\w+)?\s+(?:is|shall be|will be)\s+(?P<p3>[^.;()]{0,30}(?:\(\d+\))?\s*(?:months?|years?))\s+from\s+(?:and\s+including\s+)?(?:the\s+)?(?P<anchor3>[A-Z][\w ]{2,40}?Date)?",
                     re.I)
_ANNIV = re.compile(r"\bon the (?P<ord>first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|\d{1,2}(?:st|nd|rd|th))\s+"
                    r"anniversary of the (?P<anchor>[A-Z][\w ]{2,40}?Date)", re.I)


_FIRST = re.compile(r"\b[Tt]he first\s+(?P<p>[^.;()]{0,30}(?:\(\d+\))?\s*(?:months?|years?))\s+(?:of\s+\w+\s+)?from\s+(?:the\s+)?"
                    r"(?P<anchor>[A-Z][\w ]{2,40}?Date)\s+(?:are|is|shall be|will be)\s+(?:a|the)\s+(?P<what>[\w -]{3,40}?period)\b")


def first_period(s: str, text: str) -> Optional[dict]:
    """'The first six (6) months of employment from the Start Date are a probationary period': when that period ends."""
    m = _FIRST.search(s)
    p = next(iter(_periods(m.group("p"))), None) if m else None
    start = anchor_date(text, m.group("anchor")) if m and p else None
    if not start:
        return None
    end = add_months(start[0], p["n"] * (12 if p["unit"].startswith("year") else 1)) - timedelta(days=1)
    return {"when": f"{p['n']} {p['unit']} from the {m.group('anchor')}", "due": end.isoformat(), "at": m.start(), "what": m.group("what"),
            "note": f"The {m.group('what')} is read as {p['n']} {p['unit']} from {start[1]}, ending on {_fmt(end)} (counting the first day). "
                    f"Confirm before relying on it."}


def expiry(s: str, text: str) -> Optional[dict]:
    """A term that ends on a date the document fixes: 'continues for three (3) years from the Effective Date'."""
    m = _EXPIRY.search(s)
    if not m:
        return None
    ptxt, anchor = next(((m.group(p), m.group(an)) for p, an in (("p", "anchor"), ("p2", "anchor2"), ("p3", "anchor3")) if m.group(p)))
    p = next(iter(_periods(ptxt)), None)
    if not p:
        return None
    months = p["n"] * (12 if p["unit"].startswith("year") else 1)
    lit = re.search(r"\bfrom\s+(?:and\s+including\s+)?" + _DATE.pattern, s[m.start():m.end() + 60])   # "from 1 July 2024"
    start = (anchor_date(text, anchor) if anchor else
             (date.fromisoformat(_date(lit)), _fmt(date.fromisoformat(_date(lit)))) if lit and _date(lit) else
             _start_date(text) if re.search(r"\b(?:begins|commences|starts)\b", s) else None)
    when = f"{p['n']} {p['unit']}" + (f" from the {anchor}" if anchor else f" from {start[1]}" if start else "")
    if not start:
        return {"when": when, "due": None, "at": m.start(), "note": f"The term is {p['n']} {p['unit']}, but the document gives no date it starts from."}
    end = add_months(start[0], months) - timedelta(days=1)
    return {"when": when, "due": end.isoformat(), "at": m.start(),
            "note": f"The term is read as {p['n']} {p['unit']} from {start[1]}, ending on {_fmt(end)} (counting the first day). Confirm before relying on it."}


def _start_date(text: str) -> Optional[tuple[date, str]]:
    """The date the term starts from, as the document gives it: 'begins on 2 March 2026', or a defined
    'Effective Date' / 'Commencement Date' / 'Start Date' whose definition carries a date."""
    m = re.search(r"\b(?:begins|commences|starts|takes effect)\s+on\s+(?:the\s+)?(?P<what>[^.,;]{0,40})", text)
    if m:
        d = _DATE.search(m.group("what"))
        if d and _date(d):
            return date.fromisoformat(_date(d)), _fmt(date.fromisoformat(_date(d)))
        named = re.match(r"(?P<t>(?:Effective|Commencement|Start|Signature)\s+Date)", m.group("what"))
        if named:
            term = named.group("t")
            for dm in re.finditer(rf"\((?:the\s+)?[\"“]{term}[\"”]\)", text):
                win = text[max(0, dm.start() - 120):dm.start()]
                ds = list(_DATE.finditer(win))
                if ds and _date(ds[-1]):
                    d0 = date.fromisoformat(_date(ds[-1]))
                    return d0, f"{_fmt(d0)} (the {term})"
            dm = re.search(rf"[\"“]{term}[\"”]\s+means\s+(?P<v>[^.;]+)", text)
            if dm:
                d = _DATE.search(dm.group("v"))
                if d and _date(d):
                    d0 = date.fromisoformat(_date(d))
                    return d0, f"{_fmt(d0)} (the {term})"
    return None


def _term(text: str) -> Optional[tuple[int, str]]:
    """The initial term in months: 'an initial term of thirty-six (36) months', 'for two (2) years'."""
    m = re.search(r"\binitial\s+(?:term|period)\s+of\s+(?P<p>[^.;()]{0,30}(?:\(\d+\))?\s*(?:months?|years?))", text, re.I)
    if not m:
        return None
    p = next(iter(_periods(m.group("p"))), None)
    if not p:
        return None
    months = p["n"] * (12 if p["unit"].startswith("year") else 1)
    return months, f"{p['n']} {p['unit']}"


def due(kind: str, period: Optional[dict], text: str, sentence: Optional[str] = None) -> tuple[Optional[str], str]:
    """(due date, how it was reached — or why there is none)."""
    if kind == "renewal" and period and period["direction"] == "before":
        from ..research.prompt import document_sentences
        ends = [e for e in (expiry(text[x:y], text) for x, y in document_sentences(text)) if e and e["due"]]
        if ends:                                          # the term's end, as the document fixes it (whatever the term is called)
            end = date.fromisoformat(ends[0]["due"])
            if period["unit"].startswith("day") and not period["business"]:
                deadline = end - timedelta(days=period["n"])
            elif period["unit"].startswith("month"):
                deadline = add_months(end, -period["n"])
            else:
                return None, f"Notice is due {period['n']} {period['unit']} before {_fmt(end)}; business days are not counted here."
            return deadline.isoformat(), (f"{period['n']} {period['unit']} before the end of the term. {ends[0]['note'].split(' Confirm')[0]} "
                                          f"So notice is due by {_fmt(deadline)}. Confirm how the days are counted before relying on it.")
        start, term = _start_date(text), _term(text)
        if not term:
            return None, "The document does not state the length of the term, so no date is computed."
        if not start:
            return None, (f"No start date in the document: notice is due {period['n']} {period['unit']} before the end of the "
                          f"{term[1]} term, which this document does not date.")
        end = add_months(start[0], term[0]) - timedelta(days=1)
        if period["unit"].startswith("day") and not period["business"]:
            deadline = end - timedelta(days=period["n"])
        elif period["unit"].startswith("month"):
            deadline = add_months(end, -period["n"])
        else:
            return None, f"Notice is due {period['n']} {period['unit']} before {_fmt(end)}; business days are not counted here."
        return deadline.isoformat(), (
            f"{period['n']} {period['unit']} before the end of the initial term. The initial term is read as {term[1]} from {start[1]}, "
            f"ending on {_fmt(end)} (counting the first day), so notice is due by {_fmt(deadline)}. Confirm how the days are counted "
            f"before relying on it.")
    if kind == "termination" and period and _ANNIV.search(sentence or ""):
        m = _ANNIV.search(sentence)
        o = m.group("ord").lower()
        n = _ORDINAL.get(o) or int(re.match(r"\d+", o).group(0))
        start = anchor_date(text, m.group("anchor"))
        if not start:
            return None, f"The break date is the {o} anniversary of the {m.group('anchor')}, which the document does not date."
        brk = add_months(start[0], 12 * n)
        if period["unit"].startswith("month"):
            deadline = add_months(brk, -period["n"])
        elif period["unit"].startswith("day") and not period["business"]:
            deadline = brk - timedelta(days=period["n"])
        else:
            return None, f"The break date is {_fmt(brk)}; business days are not counted here."
        return deadline.isoformat(), (f"The break date is the {o} anniversary of {start[1]}: {_fmt(brk)}. Notice of {period['n']} {period['unit']} "
                                      f"is due by {_fmt(deadline)}. Confirm how the period is counted before relying on it.")
    on = _ON_DATE.search(sentence or "")
    if on and _date(on) and not period:
        d0 = date.fromisoformat(_date(on))
        return d0.isoformat(), f"On {_fmt(d0)}, as the clause says."
    if period and period["direction"] == "after" and not period["business"]:
        m = re.search(r"\b(?:after|following|from)\s+(?:the\s+)?(?P<a>[A-Z][\w ]{1,40}?)(?=[\s,.;]|$)", (sentence or "")[period["end"]:period["end"] + 60])
        start = event_date(text, m.group("a").strip()) if m else None
        if start:
            end = add_months(start[0], period["n"] * (12 if period["unit"].startswith("year") else 1)) if period["unit"].startswith(("month", "year")) \
                else start[0] + timedelta(days=period["n"]) if period["unit"].startswith("day") else None
            if end:
                return end.isoformat(), (f"{period['n']} {period['unit']} after {start[1]}: {_fmt(end)}. Confirm how the period is counted "
                                         f"before relying on it.")
    if not period:
        return None, ("Recurring, on the dates the clause gives." if kind == "payment" else "A right, not a deadline: no date.")
    trigger = {"payment": "each invoice", "notification": "an event (becoming aware)", "post_termination": "termination or expiry",
               "termination": "an event (the notice or the breach)", "change_of_control": "an event (the change of control)",
               "confidentiality": "the end of this agreement", "survival": "the end of this agreement",
               "duty": "an event (a request or the date the clause names)"}.get(kind, "an event")
    return None, f"Runs from {trigger}; no date until it happens."


def read(text: str, clauses: list[dict]) -> list[dict]:
    from ..research.prompt import document_sentences
    rows = numbering.complete(text, clauses)
    numbered = [c for c in rows if numbering.split(c["number"])]
    out = []
    for a, b in document_sentences(text):
        s = text[a:b]
        modal = MODAL.search(s)
        ex = expiry(s, text) if not modal or re.search(r"\bcontinues?\b|\bmeans\b|\bremains? in force\b", s) else None
        ex = ex or first_period(s, text)
        anchor = a + (modal.start() if modal else ex["at"] if ex else 0)
        inside = [c for c in numbered if c["start"] <= anchor < max(c["end"], c["start"] + 1)]
        if not inside:
            continue
        leaf = max(inside, key=lambda c: (c["level"], c["start"]))
        clause = leaf["number"]
        if a < leaf["start"] < b:                         # a heading line read into the sentence ("3. Break\n3.1 The Tenant …")
            a, s = leaf["start"], text[leaf["start"]:b]
            modal = MODAL.search(s)
        quote = s.rstrip()
        evidence = {"start": a, "end": a + len(quote), "quote": quote}
        if ex:
            out.append({"kind": "expiry", "clause": clause, "party": None, "action": f"The {ex['what']} ends" if ex.get("what") else "The term ends",
                        "when": ex["when"], "due_date": ex["due"],
                        "due_note": ex["note"], "rule": {"period": None, "engine": VERSION}, "evidence": evidence})
            # the same sentence may also give a right ("… and thereafter until terminated by either party giving
            # twelve (12) months' notice"): it goes on to be read, without the term's own period
        if not modal and not _UNLESS_NOTICE.search(s):
            continue
        main = re.split(r",?\s+(?:unless|provided that|if|except|save that)\b", s)[0]
        kind = kind_of(main)                              # what the main clause binds, not its condition ("…, unless a Claim Notice …")
        if not kind and not (_ON_DATE.search(main) or any(p["score"] for p in _periods(main))):
            kind = kind_of(s)
        ps = [p for p in _periods(s) if not (ex and ex["when"].startswith(f"{p['n']} {p['unit']}"))]
        period = max(ps, key=lambda p: (p["score"], p["pstart"])) if ps else None
        freq = _FREQUENCY.search(s) if kind == "payment" and re.search(r"\b(?:pay|payable|paid)\b", s, re.I) else None
        each = _EACH_PERIOD.search(s)
        on = _ON_DATE.search(s) if modal else None
        named = _ON_NAMED.search(s) if modal else None
        named_date = anchor_date(text, named.group("name")) if named else None
        yearly = _YEARLY.search(s)
        term_end = _ON_TERMINATION.search(s) if not period else None
        if not kind and period and re.search(r"\b(?:after|following)\s+(?:the\s+)?(?:termination|expiry|expiration)\b", s, re.I):
            kind = "post_termination"                     # "may sell its remaining stock for six (6) months after termination"
        if not kind and (period and period["score"] or on or named_date or yearly or term_end or each) and re.search(r"\b(?:shall|must)\b", s):
            kind = "duty"                                 # "shall provide a copy within ten (10) Business Days of a request"
        if kind == "duty" and (term_end or re.search(r"\b(?:after|following|upon|on)\s+(?:the\s+)?(?:termination|expiry|expiration|Termination Date)\b", s, re.I)):
            kind = "post_termination"
        if period and not period["score"]:
            period = None                                 # "in the twelve (12) months preceding": a look-back, not a time to act
        if not kind or (not period and kind != "consent" and not freq and not on and not named_date and not yearly and not each
                        and not (term_end and kind in ("duty", "post_termination"))):
            continue
        date_, note = due(kind, period, text, s)
        if named_date and not period and not date_:
            date_, note = named_date[0].isoformat(), f"On {named_date[1]}, as the clause says."
        g = None if modal else _UNLESS_GIVES.search(s)
        party = (g.group("who").strip()[0].upper() + g.group("who").strip()[1:]) if g else _party(s, modal.start() if modal else len(s))
        when = (_when(s, kind, period) if period else freq.group(0).strip() if freq
                else f"on {_fmt(date.fromisoformat(_date(on)))}" if on and _date(on) else named.group(0) if named_date
                else yearly.group(0) if yearly else each.group(0) if each else term_end.group(0) if term_end else _when(s, kind, None))
        out.append({"kind": kind, "clause": clause, "party": party,
                    "action": _action(s, modal, period), "when": when, "due_date": date_, "due_note": note,
                    "rule": {"period": {k: period[k] for k in ("n", "unit", "business", "direction")} if period else None, "engine": VERSION},
                    "evidence": evidence})
    return out
