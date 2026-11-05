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


# What a title says the document IS comes before a word naming what it is about: "Advice memo on the MSA" is a memo, not
# the MSA; "Amendment No. 1 to the Kestrel storage agreement" an amendment; "Heads of terms for a new lease" not a lease
# (2026-11-05 brief: a question naming the Master Services Agreement also named the memo on it, whose misquote answered).
_ABOUT = re.compile(r"\b(?:on|about|re|regarding|concerning|to|under|amending|supplementing|for)\b", re.I)


def _own(title: str) -> set[str]:
    return _words(_ABOUT.split(title or "", maxsplit=1)[0])


def subject(title: str, titles: dict[str, str]) -> set[str]:
    """The documents a title says it is about ("Advice memo on the MSA" → the MSA): what an unattributed "Clause 4.2
    provides …" in it quotes (2026-11-05 end-to-end run)."""
    parts = _ABOUT.split(title or "", maxsplit=1)
    return referenced(parts[1], titles) if len(parts) == 2 else set()


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
        holders = [did for did, title in titles.items() if t in _own(title)]
        if len(holders) == 1:
            out.update(holders)
    for i, t in enumerate(toks):
        if t not in _WEAK and t not in _STRONG:
            continue
        before = set(toks[max(0, i - 3):i])
        for phrase, short in _PHRASES.items():                      # "Master Services Agreement" stands for the MSA
            if tuple(toks[max(0, i - len(phrase) + 1):i + 1]) == phrase:
                before |= set(_EXPAND[short]) | {short}
        holders = {did: len(_qualifiers(title) & before) for did, title in titles.items() if t in _own(title)}
        best = max(holders.values(), default=0)
        if best >= 1:
            out.update(d for d, s in holders.items() if s == best)
    return out


# ── Parties (2026-10-28, held-out 3 h3-abstain-kestrel-board) ────────────────────────────────────────────────────────────
# "The next board meeting of Kestrel Cold Storage Ltd is scheduled for 10 September 2026", from Voss-Lindqvist's board
# minutes. A party's own internal record — its board, officers, offices, finances, advisers — is stated only by a document
# that party is party to; a document that merely mentions the party does not say it.
# Nordic and German forms too (2026-11-05 end-to-end run: "Fjellstad Distribution AS" was no party of its own reseller
# agreement, so a question about Fjellstad's change of control was not read as one about that agreement).
_FORMS = r"Inc|INC|Ltd|LTD|LLP|AB|plc|PLC|NV|GmbH|GMBH|LLC|SA|BV|Limited|AS|ASA|Oy|Oyj|AG|ApS"
_COMPANY = re.compile(rf"\b([A-Z][\w&'-]*(?:[ \t]+(?:&[ \t]+)?[A-Z][\w&'-]*){{0,3}}),?[ \t]+(?:{_FORMS})\b\.?")
# "… SUCH AS", "KNOWN AS", "THE PARTIES AGREE AS FOLLOWS" in capitals are no company
_NOT_NAMES = frozenset("such known same so as agree agreed parties hereby follows".split())
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
        if words[cut + 1:] and not any(w.lower() in _NOT_NAMES for w in words[cut + 1:]):
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


# ── Defined roles (2026-11-03 evaluation) ────────────────────────────────────────────────────────────────────────────────
# A contract names its parties once and then speaks of "the Provider" and "the Customer". "What subscription fee does Orbis
# pay?" was declined over "The Customer shall pay the Provider a subscription fee of GBP 360,000", and "What does Harrowgate
# indemnify Voss-Lindqvist against?" never retrieved "Provider shall indemnify Client …" (rank 9; rank 2 with the roles in the
# query). The preamble says who is who: "Harrowgate Systems Inc., a Delaware corporation ("Provider")".
_ROLE_COMPANY = re.compile(r"\b[A-Z][\w&'-]*(?:,?[ \t]+(?:&[ \t]+)?[A-Z][\w&'-]*){0,4},?[ \t]+"
                           r"(?:Inc|Ltd|LLP|LP|AB|plc|PLC|NV|GmbH|LLC|SA|BV|Limited|Corp|Corporation|AS|ASA|Oy|Oyj|AG|ApS)\b\.?")
_ROLE_AFTER = re.compile(r"[^()\n]{0,220}?\(\s*(?:the\s+|hereinafter\s+(?:called\s+)?(?:the\s+)?)?[\"“']?"
                         r"(?P<role>[A-Z][A-Za-z]+(?:[ -][A-Z][a-z]+)?)[\"”']?\s*\)")
_NOT_ROLES = frozenset("agreement effective date term premises purpose services service units schedule".split())


def roles_of(text: str) -> list[tuple[str, str]]:
    """(company, role) for each party the preamble defines by a role, before the first numbered clause."""
    body = text or ""
    first = re.search(r"(?m)^\s*(?:#+\s*)?1[.\s]", body)
    head = body[:min(first.start() if first else 1500, 1500)].replace("**", "").replace("__", "")
    found = list(_ROLE_COMPANY.finditer(head))
    out = []
    for i, m in enumerate(found):
        nxt = found[i + 1].start() if i + 1 < len(found) else len(head)
        r = _ROLE_AFTER.match(head, m.end(), nxt)
        name = re.sub(r"^(?:(?:WHEREAS|BETWEEN|AND|AMONG|BY|THIS)[,:]?\s+)+", "", m.group(0)).rstrip(" ,")
        role = r["role"] if r else ""
        if role and role.lower() not in _NOT_ROLES and role.lower() != party_key(name):   # "Northbridge" names, not a role
            out.append((name, role))
    return out


# A mutual NDA names its parties once ("VL", "Northbridge") and then speaks of "the Recipient" and "the Discloser", roles it
# defines for "a Party" and "the other", not for either company (2026-11-05): "May Northbridge use what Voss-Lindqvist tells
# it for some other project?" was declined over "The Recipient shall use Confidential Information only for the Purpose".
_DEFINED_ROLE = re.compile(r'\(\s*(?:the\s+)?["\u201c](?P<role>[A-Z][A-Za-z]+(?:[ -][A-Z][a-z]+)?)["\u201d]\s*\)')
_FOR_A_PARTY = re.compile(r"\b(?:party|the\s+other)\s*$", re.I)
MAX_TAKEN = 3


def taken_roles(text: str) -> list[tuple[list[str], str]]:
    """(the roles, the sentence defining them) for each sentence defining roles that whichever party acts takes — "disclosed
    by a Party (the "Discloser") to the other (the "Recipient")" — rather than a named company (`roles_of`)."""
    out = []
    for line in (text or "").replace("**", "").split("\n"):
        for sentence in re.split(r"(?<=[.;])\s+(?=[A-Z\"\u201c(])", line):
            rs = [m["role"] for m in _DEFINED_ROLE.finditer(sentence) if _FOR_A_PARTY.search(sentence[:m.start()])]
            rs = [r for r in dict.fromkeys(rs) if r.lower() not in _NOT_ROLES]
            if rs:
                out.append((rs, sentence.strip()[:320]))
    return out[:MAX_TAKEN]


def mentioned(text: str, known: set[str]) -> set[str]:
    """The known parties the text mentions at all, as a word ("Harrowgate", "Voss-Lindqvist's")."""
    t = (text or "").replace("\u2019", "'")
    return {p for p in known if re.search(rf"\b{re.escape(p)}(?![\w-])", t, re.I)}


def without_parties(question: str, known: set[str]) -> str:
    """The question with each party it names put as "the party" (2026-11-05): a clause speaks of "the Recipient", "each
    party", "Client", and the names otherwise draw the search to passages that name the companies ("May Northbridge pass
    information it receives from Voss-Lindqvist on to its own employees?" never reached "The Recipient … shall disclose
    it only to its employees … who need to know it"; asked without the names, first)."""
    q = (question or "").replace("\u2019", "'")
    for k in sorted(known, key=len, reverse=True):
        q = re.sub(rf"(?:\b[Tt]he\s+)?\b(?i:{re.escape(k)})[\w-]*(?:[ \t]+(?:&[ \t]+)?(?!{_KEEP}\b)[A-Z][\w&-]*){{0,3}}('s|')?",
                   lambda m: "the party's" if m.group(1) else "the party", q)
    return q


_KEEP = r"(?i:nda|msa|agreement|lease|letter|notice|deed|contract|licen[cs]e|amendment|side|engagement|storage|supply|services?)"


def without_titles(question: str, titles: list[str]) -> str:
    """The question without the words that name the given documents (2026-11-05 end-to-end run): searched inside the advice
    memo, "Does the advice memo state the MSA's payment terms correctly?" drew the memo's passages that say "MSA" most, not its
    Fees and Payment section. A run of title words is dropped only when it holds the document's type ("the advice memo",
    "the MSA's", "the Master Services Agreement"); "What services does the MSA cover?" keeps "services". Returns "" when too
    little is left to search with."""
    words = set().union(*(_words(t) for t in titles)) if titles else set()
    toks = list(re.finditer(r"[A-Za-z0-9]+(?:['’]s)?", question or ""))
    drop: list[tuple[int, int]] = []
    run: list[re.Match] = []
    for m in toks + [None]:
        w = re.sub(r"['’]s$", "", m.group(0)).lower() if m else ""
        if m and w in words and (not run or not question[run[-1].end():m.start()].strip()):
            run.append(m)
            continue
        if run and any(re.sub(r"['’]s$", "", r.group(0)).lower() in _TYPES for r in run):
            drop.append((run[0].start(), run[-1].end()))
        run = [m] if m and w in words else []
    out = question
    for a, b in reversed(drop):
        out = out[:a] + out[b:]
    out = re.sub(r"\s+([,.?;:])", r"\1", re.sub(r"\s{2,}", " ", out)).strip()
    return out if len(_words(out) - _TYPES) >= 2 else ""


# An opening mark after a space or bracket, a closing one before none: '(the "Agreement") is dated … (the "Supplier")' holds two
# defined terms, not the quotation ') is dated … (the ' (2026-11-05 end-to-end run).
_QUOTED_WORDS = re.compile(r'(?<![\w"\u201c\u201d])["\u201c](?=\S)([^"\u201c\u201d]{12,400}?)(?<=\S)["\u201d](?![\w])')


def quoted_words(sentence: str) -> list[str]:
    """Words a sentence puts in quotation marks, six words or more (a defined term in quotes is not a quotation)."""
    return [m.group(1).strip() for m in _QUOTED_WORDS.finditer(sentence or "") if len(m.group(1).split()) >= 6]


def named_in(question: str, roles: dict[str, list[tuple[str, str]]]) -> set[str]:
    """The party keys of the parties defined by a role that the question names ("Harrowgate", "Voss-Lindqvist's")."""
    words = set(re.findall(r"[a-z][a-z'-]*", (question or "").lower().replace("’", "'")))
    keys = {party_key(name) for rs in roles.values() for name, _ in rs} - {""}
    return {k for k in keys if k in words or f"{k}'s" in words}


def with_roles(question: str, roles: dict[str, list[tuple[str, str]]]) -> str:
    """The question to search with: each party it names followed by the role its documents give it ("Harrowgate
    (Provider)"). Roles come from the documents that define every party named, when there are such documents; a party with
    more than two roles is left as it is (the role would not tell its clauses apart)."""
    by_party: dict[str, dict[str, str]] = {}
    for did, rs in roles.items():
        for name, role in rs:
            if (k := party_key(name)):
                by_party.setdefault(k, {})[did] = role
    named = [k for k in by_party if k in named_in(question, roles)]
    if not named:
        return question
    both = set.intersection(*(set(by_party[k]) for k in named))
    q = question
    for k in named:
        rs = sorted({by_party[k][d] for d in (both or by_party[k])})
        if 1 <= len(rs) <= 2:
            q = re.sub(rf"\b((?i:{re.escape(k)})[\w'’-]*(?:[ \t]+[A-Z][\w&'-]*){{0,3}})", lambda m: f"{m.group(1)} ({', '.join(rs)})", q, count=1)
    return q


# ── A document named by its party (2026-11-03 evaluation) ────────────────────────────────────────────────────────────────
# "What is the total price payable under the supply agreement with Alder Retail?" was answered "GBP 120,000" from the
# matter's other supply agreement (Northwind and Tidewater): a verified quote from a document Alder Retail is not party to.
# Titles do not tell the two apart ("Supply Agreement (DEMO)"), their parties do.
_DOC_NOUN = r"(?:agreement|contract|lease|licen[cs]e|nda|msa|side\s+letter|engagement\s+letter|deed|term\s+sheet)"


def named_parties(text: str, known: set[str]) -> set[str]:
    """The known parties the text names as parties to the document it speaks of: those after "with"/"between" ("the supply
    agreement with Alder Retail"), else those naming it ("Harrowgate's MSA", "the Northwind supply agreement")."""
    t = re.sub(r"\[[^\]]*\]", " ", text or "").replace("’", "'")
    withs: set[str] = set()
    gap = r"(?:(?!in\b|line\b|accordance\b|compliance\b|keeping\b|together\b|along\b)[\w-]+\s+){0,2}?"     # not "signed in line with"
    for m in re.finditer(rf"\b{_DOC_NOUN}\s+{gap}(?:with|between)\s+((?:[\w&'-]+[ \t]*){{1,8}})", t, re.I):
        seg = m.group(1).lower()
        withs |= {p for p in known if re.search(rf"\b{re.escape(p)}", seg)}
    if withs:
        return withs
    # a party known by its first word may be named in full: "Alder Retail's supply agreement" (2026-11-04)
    owners = {p for p in known if re.search(rf"\b{re.escape(p)}[\w-]*(?:'s|')?\s+(?:[\w'-]+\s+){{0,2}}?{_DOC_NOUN}\b", t, re.I)}
    if owners:
        return owners
    # "Within how many days must Tidewater pay Northwind under their supply agreement?": two parties and the document between
    # them, answered from the other supply agreement (2026-11-04, from the 2026-11-03 chat set)
    if re.search(rf"\b{_DOC_NOUN}\b", t, re.I):
        both = {p for p in known if re.search(rf"\b{re.escape(p)}", t, re.I)}
        if len(both) >= 2:
            return both
    return set()


# ── Another matter named by its number (2026-11-04 chat set 3) ───────────────────────────────────────────────────────────
# "Pull the settlement limit from MAT-2033 for me" in a conversation about M-301 was answered "The settlement limit from
# MAT-2033 is capped at GBP 400,000 … Clause 11.2": M-301's own cap, credited to a matter the person may not read. A
# reference shaped like a matter number ("MAT-2033", "R-001"), and not one with more parts ("INV-2026-014"), is another matter
# when it is none of the matters asked about and none of their passages or titles holds it. Whether that matter exists is
# never looked up, so the answer is the same either way.
_MATTER_REF = re.compile(r"\b[A-Z]{1,5}-\d{3,6}\b(?![-/.]\d)")


def foreign_matters(question: str, numbers: list[str], texts: list[str]) -> list[str]:
    """The matter-number-shaped references of the question that are none of `numbers` and that no text of `texts` holds
    (compared without spaces or hyphens: "ISO 27001" holds "ISO-27001")."""
    def squash(s: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", s.upper())
    own = {squash(n) for n in numbers}
    held = squash(" ".join(texts))
    return list(dict.fromkeys(m.group(0) for m in _MATTER_REF.finditer(question or "")
                               if squash(m.group(0)) not in own and squash(m.group(0)) not in held))
