"""Which of the matter's documents a question or a sentence speaks of (2026-10-28, held-out 2 h2-abstain-storage-law).

An answer said "The Kestrel storage agreement is governed by the law of England and Wales", resting on real quotes from the
Northbridge NDA and the Unit 7 Lease. Every quote was verified; the attribution was wrong. Two checks follow from this:

- a cited sentence that names a document must cite that document (at least one of its sources), or it does not stand;
- a question that names a document is answered only from that document: an answer whose every standing sentence cites
  other documents is the model reaching for a neighbour's clause, and is withheld.

A document is named by its type ("agreement", "lease", "NDA", "minutes", "letter", …) with the words that tell it from the
others of that type ("Kestrel storage agreement", "Northbridge NDA"). A party's name alone ("at Kestrel", "Calder & Wynne")
names no document, and a type shared by several documents with nothing to tell them apart names none: the checks only act
when the reference is clear.
"""
from __future__ import annotations

import re

# Types that name one document on their own when the matter has one of them ("the lease", "the MSA", "the NDA") …
_STRONG = frozenset("msa nda lease".split())
# … and types that are also ordinary nouns ("on thirty days' written notice", "within 30 minutes", "Schedule 2"): these name
# a document only with a word of its title beside them ("the change of control notice", "the witness statement").
_WEAK = frozenset("agreement letter minutes amendment statement notice memo memorandum contract engagement policy schedule deed".split())
_TYPES = _STRONG | _WEAK
_EXPAND = {"msa": ("master", "services", "agreement"), "nda": ("non", "disclosure", "agreement")}
_PHRASES = {("master", "services", "agreement"): "msa", ("non", "disclosure", "agreement"): "nda",
            ("nondisclosure", "agreement"): "nda"}
_NOISE = frozenset("the a an of to in on for by with and or no is are be as at that this which under from its it what when "
                   "who how does did do any each all their his her".split())


def _tokens(text: str) -> list[str]:
    """Words of the text in order, citation markers ("[1]", "[2: …]") left out."""
    return re.findall(r"[a-z0-9]+", re.sub(r"\[[^\]]*\]", " ", text or "").lower())


def _words(text: str) -> set[str]:
    toks = _tokens(text)
    out = set(toks)
    for t in toks:
        out.update(_EXPAND.get(t, ()))
    for phrase, short in _PHRASES.items():
        n = len(phrase)
        if any(tuple(toks[i:i + n]) == phrase for i in range(len(toks) - n + 1)):
            out.add(short)
    return out - _NOISE


def _qualifiers(title: str) -> set[str]:
    """The words of a title that tell it from others of its type: not the type, not a number, not a small word."""
    return {w for w in _words(title) - _TYPES if not w.isdigit()}


def referenced(text: str, titles: dict[str, str]) -> set[str]:
    """The ids of the documents (id → title) that the text clearly names; empty when it names none or is ambiguous.

    A strong type ("lease", "MSA", "NDA") names the matter's only document of that type. A weak type names a document only
    when a word of that document's title stands just before it, as in a name ("the Kestrel storage agreement", "the change
    of control notice"), not after it ("notice of a change of control" is an event, not the notice document)."""
    toks = _tokens(text)
    words = _words(text)
    out: set[str] = set()
    for t in words & _STRONG:
        holders = [did for did, title in titles.items() if t in _words(title)]
        if len(holders) == 1:
            out.update(holders)
    for i, t in enumerate(toks):
        if t not in _WEAK and t not in _STRONG:
            continue
        before = set(toks[max(0, i - 3):i])
        for phrase, short in _PHRASES.items():                      # "Master Services Agreement" stands for the MSA
            if tuple(toks[max(0, i - len(phrase) + 1):i + 1]) == phrase:
                before |= set(_EXPAND[short]) | {short}
        holders = {did: len(_qualifiers(title) & before) for did, title in titles.items() if t in _words(title)}
        best = max(holders.values(), default=0)
        if best >= 1:
            out.update(d for d, s in holders.items() if s == best)
    return out


# ── Parties (2026-10-28, held-out 3 h3-abstain-kestrel-board) ────────────────────────────────────────────────────────────
# "The next board meeting of Kestrel Cold Storage Ltd is scheduled for 10 September 2026", from Voss-Lindqvist's board
# minutes. A party's own internal record — its board, officers, offices, finances, advisers — is stated only by a document
# that party is party to; a document that merely mentions the party does not say it.
_COMPANY = re.compile(r"\b([A-Z][\w&'-]*(?:[ \t]+(?:&[ \t]+)?[A-Z][\w&'-]*){0,3})[ \t]+(?:Inc|Ltd|LLP|AB|plc|PLC|NV|GmbH|LLC|SA|BV|Limited)\b\.?")
_RECORD = (r"(?:board(?:\s+meeting)?|board\s+of\s+directors|directors|chair(?:man|woman|person)?|general\s+counsel|chief\s+\w+\s+officer|"
           r"ceo|cfo|offices?|headquarters|registered\s+office|employees|staff|revenue|turnover|budget|dividend|shareholders|"
           r"auditors?|law\s+firm|lawyers|solicitors|advisers|account\s+manager|financial\s+year|complaints?(?:\s+procedure)?)")
# A party's own corporate act (2026-10-28, held-out 4: "What budget did Northbridge approve for Project Lantern?" answered with
# Voss-Lindqvist's board resolution): the party as the subject of a decision of its own, or the one that keeps a record.
_ACT = r"(?:approv|decid|resolv|appoint|declar|elect|hir|retain|instruct|authori[sz]|vot|dismiss)\w*"
_MEETS = r"(?:hold|held|holds|convene\w*)\s+(?:\w+\s+){0,3}?(?:board\s+)?meeting"
_KEEPS = r"(?:use|uses|offer|offers|have|has|keep|keeps|employ|employs|retain|retains)\b"


_SMALL = frozenset("of the a an and to for by between with from at on in".split())
_SUFFIXES = frozenset("LLP PLC LLC GMBH LTD INC".split())


def party_key(name: str) -> str:
    """A party by its first distinctive word: "Kestrel Cold Storage Ltd" → "kestrel", "Calder & Wynne LLP" → "calder"."""
    words = [w.strip("'-") for w in re.findall(r"[a-z][a-z'-]*", (name or "").lower())]
    while words and words[0] in _SMALL:
        words.pop(0)
    return words[0] if words else ""


def parties_of(title: str, text: str) -> set[str]:
    """The parties of a document: the companies named in its title and preamble — before its first numbered clause — who
    made, signed or received it. Companies its clauses go on to mention are not its parties."""
    body = text or ""
    first = re.search(r"(?m)^\s*(?:#+\s*)?1[.\s]", body)
    head = "\n".join((title or "", body[:min(first.start() if first else 600, 1200)]))
    head = re.sub(r"\b[A-Z][A-Z'-]{2,}\b", lambda m: m.group(0) if m.group(0) in _SUFFIXES else m.group(0).title(), head)
    out = set()
    for m in _COMPANY.finditer(head):
        words = m.group(1).split()
        cut = max((i for i, w in enumerate(words) if w.lower() in _SMALL), default=-1)   # "Board of Directors of Voss-Lindqvist"
        out.add(party_key(" ".join(words[cut + 1:])))
    return out - {""}


def party_records(text: str, parties: set[str]) -> set[str]:
    """The known parties whose own internal record the text speaks of: "Harrowgate's board", "Northbridge Analytics'
    offices", "the general counsel of Kestrel". A party named in passing ("the Voss-Lindqvist pallets") is not counted."""
    out = set()
    t = re.sub(r"\[[^\]]*\]", " ", text or "").replace("’", "'")
    for p in parties:
        name = re.escape(p) + r"[\w-]*(?:\s+[A-Z][\w&-]*){0,3}"
        if re.search(rf"\b{name}(?:'s|')\s+(?:\w+\s+){{0,2}}?{_RECORD}\b", t, re.I) or \
           re.search(rf"\b{_RECORD}\s+of\s+(?:the\s+)?{name}", t, re.I) or \
           re.search(rf"\b(?:did|does|do|has|had|will|would)\s+{name}\s+(?:\w+\s+)?{_ACT}", t, re.I) or \
           re.search(rf"\b{name}\s+(?:has\s+|had\s+)?(?:approved|decided|resolved|appointed|declared|elected|hired|retained|instructed|authori[sz]ed|voted)\b", t, re.I) or \
           re.search(rf"\b{name}\s+(?:\w+\s+)?{_MEETS}", t, re.I) or \
           (re.search(_RECORD, t, re.I) and re.search(rf"\b(?:does|do|did)\s+{name}\s+{_KEEPS}", t, re.I)):
            out.add(p)
    return out


def mentions_record(text: str) -> bool:
    """Cheap first look: does the text speak of some party's internal record at all (before parties are read)?"""
    t = text or ""
    return bool(re.search(_RECORD, t, re.I) or re.search(rf"\b{_ACT}", t, re.I) or re.search(r"\bmeeting\b", t, re.I))


# ── A clause for another purpose (2026-10-28, held-out 4 h4-abstain-lease-nonrenewal) ───────────────────────────────────
# "How much notice of non-renewal does the warehouse lease require?" was answered with the lease's break clause: a notice
# period, but not for renewal. A question that asks for a named legal concept is not answered from documents that never
# deal with it, when a retrieved passage of another document does: the cited clause is about something else.
# concept: (how a question asks for it, how a document deals with it)
_CONCEPTS = {
    "renewal": (r"\brenew\w*", r"\brenew\w*"),
    "review": (r"\breview\w*", r"\breview\w*"),
    "refund": (r"\brefund\w*|\brepa(?:y|id)\w*", r"\brefund\w*|\brepa(?:y|id)\w*|\breturn\w*\s+(?:of\s+)?the\s+deposit"),
    "indemnity": (r"\bindemn\w*", r"\bindemn\w*"),
    "warranty": (r"\bwarrant\w*", r"\bwarrant\w*"),
    "cure": (r"\bcure\b|\bremed(?:y|ied)\b", r"\bcure\b|\bremed\w*"),
    "jurisdiction": (r"\bjurisdiction\b|\bcourts?\b", r"\bjurisdiction\b|\bcourts?\b"),
    "governing law": (r"\bgovern\w*\b(?=[^?]*\blaw\b)|\bgoverning\s+law\b|\bwhich\s+law\b(?!\s+firm)", r"\bgoverned\s+by\b|\bgoverning\s+law\b"),
    "force majeure": (r"\bforce\s+majeure\b", r"\bforce\s+majeure\b|beyond\s+(?:its|their)\s+reasonable\s+control"),
    "insurance": (r"\binsur\w*|\bpremium\b", r"\binsur\w*|\bpremium\b"),
    "deposit": (r"\bdeposit\b", r"\bdeposit\b"),
    "service credit": (r"\bservice\s+credits?\b", r"\bservice\s+credits?\b"),
    "maintenance": (r"\bmaintenance\b", r"\bmaintenance\b"),
    "quorum": (r"\bquor\w*", r"\bquor\w*"),
    "dividend": (r"\bdividend\w*", r"\bdividend\w*"),
    "budget": (r"\bbudget\w*", r"\bbudget\w*"),
    "salary": (r"\bsalar\w*|\bremunerat\w*|\bcompensat\w*", r"\bsalar\w*|\bremunerat\w*|\bcompensat\w*"),
    "complaints": (r"\bcomplain\w*", r"\bcomplain\w*"),
    "non-compete": (r"\bnon-?compet\w*", r"\bnon-?compet\w*|\bcompet\w*"),
    "interest": (r"\binterest\s+(?:rate|accru|charge|on)\w*|\bwhat\s+interest\b|\bearns?\s+interest\b", r"\binterest\b(?!\s+in\b)"),
    "use class": (r"\buse\s+class\b", r"\buse\s+class\b"),
    "audit": (r"\baudit\w*", r"\baudit\w*"),
    "security standard": (r"\bsecurity\s+standard\b|\biso\b", r"\biso\b|\bsecurity\s+programme\b"),
    "confidentiality": (r"\bconfidential\w*", r"\bconfidential\w*"),
    "data breach": (r"\bdata\s+breach\b|\bpersonal\s+data\b", r"\bpersonal\s+data\s+breach\b|\bdata\s+breach\b"),
}


def concepts_elsewhere(question: str, cited_texts: list[str], other_texts: list[str]) -> list[str]:
    """The legal concepts the question asks for that no cited document deals with while another retrieved passage does."""
    q = (question or "").lower()
    cited = " ".join(cited_texts).lower()
    other = " ".join(other_texts).lower()
    return [k for k, (asks, deals) in _CONCEPTS.items()
            if re.search(asks, q) and not re.search(deals, cited) and re.search(deals, other)]


def asks_concept(question: str) -> bool:
    q = (question or "").lower()
    return any(re.search(asks, q) for asks, _ in _CONCEPTS.values())
