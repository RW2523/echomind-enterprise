"""Facts that documents in one matter can disagree on (S2.3c, RAG-02; decision D-039).

From each sentence of a version's body text (headers and footers are left out), this finds:
- amounts of money ("USD 18,500", "EUR 4.2 million", "£650");
- periods ("thirty (30) days", "five (5) business days", "36 months", "48 hours");
- percentages ("five percent (5%)");
- dates ("2 March 2026", "March 2, 2026", "2026-03-02");
- the governing law ("governed by the laws of the State of Delaware");
- companies by name and legal form ("Harrowgate Systems Inc.");
- references to another agreement by its date ("the Master Services Agreement dated 2 March 2026").

Each fact keeps its exact characters and its sentence, so a finding can quote both documents exactly.

`candidates` pairs facts from two related documents that could contradict each other:
- same kind and a comparable unit, but different values;
- sentences about the same subject.

Company names are compared between related documents: they name the same two companies, or one
mentions the other's title. Amounts, periods, percentages, dates and the governing law are compared
only between documents with the same parties (between, To/From, Signed for, a defined role, the
letterhead), each statement with its closest counterpart. A reference by date is checked against the date the referenced agreement gives itself.
This step only proposes. A company's legal form, an agreement's date and the governing law are
reported as found; the model checks amounts, periods, percentages and dates in context (`scan.py`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from .align import fold

_MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                "November", "December")
_MONTH = {m.lower(): i for i, m in enumerate(_MONTH_NAMES, start=1)}
_M = "|".join(_MONTH_NAMES)
_DATE_RE = (rf"(?:(?P<d1>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<m1>{_M})\s*,?\s+(?P<y1>\d{{4}})"
            rf"|(?P<m2>{_M})\s+(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s+(?P<y2>\d{{4}})"
            r"|(?P<iso>\d{4}-\d{2}-\d{2}))")
_DATE = re.compile(rf"\b{_DATE_RE}\b")
_CURRENCY = {"$": "USD", "US$": "USD", "€": "EUR", "£": "GBP"}
_SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9}
_MONEY = re.compile(r"(?P<cur>US\$|USD|EUR|GBP|SEK|CHF|JPY|NOK|DKK|CAD|AUD|[$€£])\s?"
                    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?:\s?(?P<scale>thousand|million|billion|bn|mn|m|k)\b)?")
_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                                     "fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_PERIOD = re.compile(r"\b(?P<n>\d{1,4}|[A-Za-z]+(?:-[A-Za-z]+)?)(?:\s*\((?P<d>\d{1,4})\))?\s+"
                     r"(?P<kind>business\s+|working\s+|calendar\s+|banking\s+)?(?P<unit>days?|weeks?|months?|years?|hours?)\b", re.I)
_PERCENT = re.compile(r"(?P<n>\d{1,3}(?:\.\d+)?)\s?(?:%|per\s?cent\b|percent\b)")
_LAW = re.compile(r"governed by(?: and (?:construed|interpreted) in accordance with)?\s+(?:the\s+)?laws?\s+of\s+(?:the\s+)?"
                  r"(?P<j>(?:State|Commonwealth|Province|Republic|Kingdom|Federal Republic) of [A-Z][a-z]+(?: [A-Z][a-z]+)?"
                  r"|[A-Z][a-z]+(?:(?: and | )[A-Z][a-z]+)?)")
_FORMS = {"inc": "inc", "inc.": "inc", "incorporated": "inc", "ltd": "ltd", "ltd.": "ltd", "limited": "ltd", "llc": "llc",
          "l.l.c.": "llc", "llp": "llp", "plc": "plc", "ab": "ab", "nv": "nv", "n.v.": "nv", "bv": "bv", "b.v.": "bv",
          "gmbh": "gmbh", "ag": "ag", "sa": "sa", "s.a.": "sa", "sas": "sas", "corp": "corp", "corp.": "corp",
          "corporation": "corp"}
_ENTITY = re.compile(r"(?P<name>[A-Z][\w'’-]*(?:\s+(?:&\s+)?[A-Z][\w'’-]*){0,4})\s+"
                     r"(?P<form>Inc\.?|INC\.?|Incorporated|Ltd\.?|LTD\.?|Limited|LIMITED|LLC|L\.L\.C\.|LLP|plc|PLC|AB|NV|N\.V\.|"
                     r"BV|B\.V\.|GmbH|AG|SA|S\.A\.|SAS|Corp\.?|CORP\.?|Corporation)(?=[\s,.;:)\"”*]|$)")
_NOT_NAME = {"the", "and", "between", "for", "to", "by", "from", "of", "signed", "agreed", "dear", "this", "that", "a", "an"}
_TITLE_WORDS = r"(?:Agreement|Contract|Lease|Letter|Deed|Notice|Order Form|Statement of Work|Addendum|Amendment|Side Letter)"
_REFERENCE = re.compile(rf"(?P<title>(?:[A-Z][A-Za-z-]+\s+){{0,5}}{_TITLE_WORDS})\s+(?:dated|made on|entered into on|of)\s+"
                        rf"(?:the\s+)?{_DATE_RE}")
_OWN_DATE = re.compile(rf"(?:dated(?:\s+as\s+of)?|entered\s+into(?:\s+as\s+of|\s+on)?|made(?:\s+on|\s+as\s+of)?|Date:)\s*"
                       rf"(?:the\s+)?{_DATE_RE}", re.I)
_STOP = set("""a an the and or of to in on at by for from with without within under over into than then that this these those
such any each every all both either neither other another its it is are was were be been being has have had do does did not no
nor shall will may must can could would should might per as if so upon after before between during until unless which who whom
whose what when where while our your their his her they them we you he she us party parties agreement provided provides
including include includes under hereby herein thereof whereby also only more less least most same set out given give gives
day days month months year years hour hours week weeks business working calendar""".split())
_ANCHORS = {
    "fee": {"fee", "price", "subscription", "charge", "rate", "rent"},
    "payment": {"invoice", "payable", "payment", "pay", "paid", "net"},
    "liability": {"liability", "liable", "cap", "aggregate", "exceed"},
    "notice": {"notice", "notify", "notification"},
    "term": {"term", "renew", "renewal", "initial", "expire", "expiry", "duration"},
    "termination": {"terminate", "termination"},
    "interest": {"interest", "late"},
    "confidentiality": {"confidential", "confidentiality"},
    "service": {"credit", "availability", "uptime", "maintenance"},
    "deposit": {"deposit"},
    "breach": {"breach"},
    "law": {"governed", "law", "laws", "jurisdiction", "court", "courts"},
    "completion": {"completion", "closing"},
    "start": {"effective", "commence", "commencement", "begin"},
}


@dataclass
class Fact:
    kind: str                    # money | period | percent | date | law | entity | reference
    value: str                   # comparable: "USD 18500", "30 days", "5%", "2026-03-02", "delaware", "inc"
    shown: str                   # as written
    start: int
    end: int
    sentence: tuple[int, int]
    topic: frozenset = field(default_factory=frozenset)
    subject: Optional[str] = None   # entity: its name without the legal form; reference: the agreement's title
    unit: Optional[str] = None      # money: currency; period: days | business days | months | hours
    near: frozenset = field(default_factory=frozenset)   # content words within eight words of the value


def _word_number(word: str) -> Optional[int]:
    w = word.lower()
    if w.isdigit():
        return int(w)
    if w in _UNITS:
        return _UNITS[w]
    if w in _TENS:
        return _TENS[w]
    if "-" in w:
        tens, unit = w.split("-", 1)
        if tens in _TENS and unit in _UNITS and _UNITS[unit] < 10:
            return _TENS[tens] + _UNITS[unit]
    return None


def _date(m: re.Match) -> Optional[str]:
    try:
        if m.group("iso"):
            return date.fromisoformat(m.group("iso")).isoformat()
        if m.group("d1"):
            return date(int(m.group("y1")), _MONTH[m.group("m1").lower()], int(m.group("d1"))).isoformat()
        return date(int(m.group("y2")), _MONTH[m.group("m2").lower()], int(m.group("d2"))).isoformat()
    except ValueError:
        return None


def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def topic(text: str) -> frozenset:
    return frozenset(_stem(w) for w in re.findall(r"[a-z][a-z'-]{2,}", fold(text)) if w not in _STOP and _word_number(w) is None)


_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]+")


def _near(sentence: str, a: int, b: int, n: int = 8) -> frozenset:
    """What a value is about: the content words within n words of it, in its sentence."""
    return topic(" ".join(_WORD.findall(sentence[:a])[-n:] + _WORD.findall(sentence[b:])[:n]))


def _sentences(text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    from ..research.prompt import document_sentences
    out = []
    for s, e in spans:
        out += [(s + a, s + b) for a, b in document_sentences(text[s:e])]
    return out


def extract(text: str, spans: list[tuple[int, int]]) -> list[Fact]:
    """Facts in the given body spans of `text` (offsets are the text's)."""
    facts: list[Fact] = []
    for s, e in _sentences(text, spans):
        sent = text[s:e]
        t = topic(sent)

        def add(kind, value, m, group=0, **kw):
            facts.append(Fact(kind, value, m.group(group), s + m.start(group), s + m.end(group), (s, e), t,
                              near=_near(sent, m.start(group), m.end(group)), **kw))
        taken: list[tuple[int, int]] = []
        for m in _MONEY.finditer(sent):
            cur = _CURRENCY.get(m.group("cur"), m.group("cur"))
            amount = float(m.group("num").replace(",", "")) * _SCALE.get((m.group("scale") or "").lower(), 1)
            add("money", f"{cur} " + (f"{amount:.2f}".rstrip("0").rstrip(".")), m, unit=cur)
            taken.append(m.span())
        for m in _PERIOD.finditer(sent):
            n = int(m.group("d")) if m.group("d") else _word_number(m.group("n"))
            if n is None or any(a <= m.start() < b for a, b in taken):
                continue
            unit, kind = m.group("unit").lower().rstrip("s"), (m.group("kind") or "").strip().lower()
            if unit == "year":
                n, unit = n * 12, "month"
            elif unit == "week":
                n, unit = n * 7, "day"
            unit = ("business day" if kind in ("business", "working", "banking") and unit == "day" else unit) + "s"
            add("period", f"{n} {unit}", m, unit=unit)
        for m in _PERCENT.finditer(sent):
            add("percent", f"{float(m.group('n')):g}%", m)
        referenced = []
        for m in _REFERENCE.finditer(sent):
            d = _date(m)
            title = re.sub(r"^(?:the|this|that|our|your)\s+", "", m.group("title").strip(), flags=re.I)
            if d and len(title.split()) >= 2:
                add("reference", d, m, subject=fold(title))
                referenced.append(m.span())
        for m in _DATE.finditer(sent):
            if any(a <= m.start() < b for a, b in referenced):
                continue
            d = _date(m)
            if d:
                add("date", d, m)
        for m in _LAW.finditer(sent):
            j = re.sub(r"^(?:state|commonwealth|province|republic|kingdom|federal republic) of ", "", fold(m.group("j")))
            add("law", j, m, group="j")
        names: set[str] = set()
        for m in _ENTITY.finditer(sent):
            words = m.group("name").split()
            if m.group("name").isupper() and "OF" in words:         # "BOARD OF DIRECTORS OF VOSS-LINDQVIST AB"
                words = words[len(words) - words[::-1].index("OF"):]
            while words and words[0].lower() in _NOT_NAME:
                words.pop(0)
            if not words:
                continue
            form = _FORMS.get(m.group("form").lower())
            names |= {_stem(w) for w in fold(" ".join(words)).split()} | {m.group("form").lower().rstrip(".")}
            facts.append(Fact("entity", form or m.group("form").lower(), m.group(0)[m.group(0).index(words[0]):],
                              s + m.start("name") + m.group("name").index(words[0]), s + m.end("form"), (s, e), t,
                              subject=fold(" ".join(words))))
        if names:                               # who is named is not what a value is about
            for f in facts:
                if f.sentence == (s, e) and f.kind in _VALUES:
                    f.near = f.near - names
    return facts


# ── A document as the scan sees it ───────────────────────────────────────────────
@dataclass
class Doc:
    document_id: str
    title: str
    version_id: str
    text: str
    facts: list[Fact]
    name: str                    # its own title, folded ("master services agreement")
    own_date: Optional[str]
    entities: set[str]


def _name(text: str, title: str) -> str:
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    first = re.sub(r"^[#\s*]+|[*]+$", "", first)
    first = re.split(r"\s+[—–-]\s+|\(", first)[0]
    return fold(first.strip() or title)


def document(document_id: str, title: str, version_id: str, text: str, spans: list[tuple[int, int]]) -> Doc:
    facts = extract(text, spans)
    head = text[:1200]
    m = _OWN_DATE.search(head)
    own = _date(m) if m else None
    return Doc(document_id, title, version_id, text, facts, _name(text, title), own,
               {f.subject for f in facts if f.kind == "entity" and f.subject})


_PARTY_CUE = re.compile(r"\b(?:between|to:|from:|signed for|accepted for|agreed by|on behalf of|for and on behalf of)\s", re.I)
_ROLE_AFTER = re.compile(r"[\s*,]*(?:an?\s[^()]{0,60}?)?\(\s*(?:the\s+)?[\"“][A-Z]")


def parties(doc: Doc) -> list[str]:
    """The companies a document is between, as far as its text says: named after "between", "To:",
    "From:", "Signed for", "Accepted for" or "on behalf of", or followed by a defined role
    ("Provider"), or the letterhead (the first company, near the top)."""
    out: list[str] = []
    for k, f in enumerate(e for e in doc.facts if e.kind == "entity"):
        before = doc.text[max(f.sentence[0], f.start - 120):f.start]
        letterhead = k == 0 and f.start < 300
        if letterhead or _PARTY_CUE.search(before) or _ROLE_AFTER.match(doc.text, f.end, f.end + 90):
            if f.shown not in out:
                out.append(f.shown)
    return out


def _party_names(doc: Doc) -> set[str]:
    named = set(parties(doc))
    return {f.subject for f in doc.facts if f.kind == "entity" and f.shown in named and f.subject}


def same_parties(a: Doc, b: Doc) -> bool:
    """Both documents are between the same companies (at least two in common): only then can their
    amounts, periods, dates or governing law contradict each other."""
    return len(_party_names(a) & _party_names(b)) >= 2


def related(a: Doc, b: Doc) -> bool:
    shared = len(a.entities & b.entities)
    mentions = (len(b.name.split()) >= 2 and b.name in fold(a.text)) or (len(a.name.split()) >= 2 and a.name in fold(b.text))
    return shared >= 2 or mentions


def _comparable(f: Fact, g: Fact) -> bool:
    if f.kind != g.kind or f.value == g.value:
        return False
    if f.kind in ("money", "period"):
        return f.unit == g.unit
    if f.kind == "entity":
        return bool(f.subject) and f.subject == g.subject           # the same name with another legal form
    return True


_VALUES = ("money", "period", "percent", "date")


def _about_the_same(f: Fact, g: Fact) -> float:
    """How closely two statements are about the same thing (0 = not at all). Amounts, periods,
    percentages and dates are compared by the words around the value; the governing law and company
    names always concern the same thing."""
    if f.kind == "law":
        return 1.0
    if f.kind == "entity":
        return 0.6
    a, b = f.near, g.near
    if not a or not b:
        return 0.0
    jaccard = len(a & b) / len(a | b)
    anchors = sum(1 for words in _ANCHORS.values() if a & words and b & words)
    if f.kind == "date" and not anchors:         # dates are compared only when they mark the same kind of event
        return 0.0
    return jaccard + 0.1 * anchors


def _best(f: Fact, among: list[Fact]) -> tuple[Optional[Fact], float]:
    found = [(g, _about_the_same(f, g)) for g in among if g.kind == f.kind and g.unit == f.unit]
    return max(found, key=lambda x: x[1], default=(None, 0.0))


_NEAR_MIN = 0.2


@dataclass
class Candidate:
    kind: str
    a: Doc
    fa: Fact
    b: Doc
    fb: Fact
    score: float


def candidates(docs: list[Doc], limit: int = 40) -> list[Candidate]:
    out: list[Candidate] = []
    for i, x in enumerate(docs):
        for y in docs[i + 1:]:
            if x.document_id == y.document_id:
                continue
            # A reference by date, checked against the date the referenced agreement gives itself.
            for p, q in ((x, y), (y, x)):
                if not q.own_date:
                    continue
                for f in p.facts:
                    if f.kind == "reference" and f.value != q.own_date and _refers_to(f.subject or "", q):
                        own = _own_date_fact(q)
                        if own:
                            out.append(Candidate("reference", p, f, q, own, 2.0))
            if not related(x, y):
                continue
            named: set = set()
            parties_agree = same_parties(x, y)
            for f in x.facts:
                if f.kind in ("law", "entity"):
                    if f.kind == "law" and not parties_agree:
                        continue
                    for g in y.facts:
                        key = (f.kind, f.subject, f.value, g.value)
                        if _comparable(f, g) and key not in named:      # once per document pair, not per sentence
                            named.add(key)
                            out.append(Candidate(f.kind, x, f, y, g, _about_the_same(f, g)))
                    continue
                if f.kind not in _VALUES or not parties_agree:
                    continue
                # The statement's counterpart is the most similar statement of the same kind in the other
                # document, and only if that one's own best match is this statement (mutual). A counterpart
                # with the same value means the two documents agree.
                g, score = _best(f, y.facts)
                if g is None or score < _NEAR_MIN or f.value == g.value:
                    continue
                back, _ = _best(g, x.facts)
                if back is f:
                    out.append(Candidate(f.kind, x, f, y, g, score))
    out.sort(key=lambda c: c.score, reverse=True)
    seen, kept = set(), []
    for c in out:                                  # one candidate per pair of sentences
        key = (c.a.version_id, c.fa.sentence, c.b.version_id, c.fb.sentence)
        if key in seen:
            continue
        seen.add(key)
        kept.append(c)
    return kept[:limit]


def _refers_to(title: str, doc: Doc) -> bool:
    words = [w for w in title.split() if w not in ("the", "this")]
    return len(words) >= 2 and all(w in doc.name.split() or w in fold(doc.title).split() for w in words)


def _own_date_fact(doc: Doc) -> Optional[Fact]:
    m = _OWN_DATE.search(doc.text[:1200])
    if not m:
        return None
    from ..research.prompt import document_sentences
    sentence = next(((a, b) for a, b in document_sentences(doc.text[:1200]) if a <= m.start() < b), (m.start(), m.end()))
    return Fact("date", _date(m) or "", m.group(0), m.start(), m.end(), sentence, topic(doc.text[sentence[0]:sentence[1]]))
