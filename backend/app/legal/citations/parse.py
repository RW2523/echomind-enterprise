"""Reading what a document cites (S3.4, DOC-03; decision D-048). Deterministic: no model.

- **External authorities:** US Code and CFR sections, US cases with reporter citations, UK neutral
  citations and law reports, UK Acts (with a section), EU regulations, directives and CJEU cases.
  A citation that cannot be real is flagged: an unknown reporter, a court that does not exist (a
  circuit above the 11th), a year in the future or before the court sat, a US Code title above 54.
- **References:** clause, section, paragraph, article, § (not part of an authority), schedule,
  exhibit, annex and appendix, with "of the <document>" when another document is named.
- **Quotations:** words in quotation marks (four words or more), with the reference before them in
  the same sentence that attributes them, if any.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

_YEAR_NOW = datetime.now(timezone.utc).year
REPORTERS = {
    "U.S.", "S. Ct.", "S.Ct.", "L. Ed.", "L. Ed. 2d", "F.", "F.2d", "F.3d", "F.4th", "F. Supp.", "F. Supp. 2d", "F. Supp. 3d", "F. App'x",
    "B.R.", "A.", "A.2d", "A.3d", "N.E.", "N.E.2d", "N.E.3d", "N.W.", "N.W.2d", "P.", "P.2d", "P.3d", "S.E.", "S.E.2d", "So.", "So. 2d",
    "So. 3d", "S.W.", "S.W.2d", "S.W.3d", "Cal. Rptr.", "Cal. Rptr. 2d", "Cal. Rptr. 3d", "N.Y.S.", "N.Y.S.2d", "N.Y.S.3d", "Del. Ch.", "A.D.3d",
}
_CIRCUIT = re.compile(r"^(\d{1,2})(?:st|nd|rd|th) Cir\.$")
_UK_COURTS = {"UKSC": 2009, "UKPC": 1830, "UKHL": 1876, "EWCA Civ": 2001, "EWCA Crim": 2001, "EWHC": 2001, "UKUT": 2008, "UKFTT": 2008,
              "CSIH": 2005, "CSOH": 2005}

_RULE_SETS = {"civ": ("the Federal Rules of Civil Procedure", 1, 87), "evid": ("the Federal Rules of Evidence", 101, 1103),
              "app": ("the Federal Rules of Appellate Procedure", 1, 48), "crim": ("the Federal Rules of Criminal Procedure", 1, 61),
              "bankr": ("the Federal Rules of Bankruptcy Procedure", 1001, 9037)}
AUTHORITIES = [
    # S3.9: court rules. Federal Rules (Fed. R. Civ. P. 81(c), Fed. R. Evid. 401) and the English Civil Procedure Rules
    # (CPR r. 3.4, CPR 31.16, CPR Part 36, CPR PD 51U).
    ("us_rule", re.compile(r"\bFed\.\s?R\.\s?(?P<set>Civ|Evid|App|Crim|Bankr)\.(?:\s?P\.)?\s?(?P<num>\d{1,4})(?:\([a-z0-9]{1,4}\))*")),
    ("uk_rule", re.compile(r"\bCPR\s+(?:(?P<kind>r\.|rule|Part|PD)\s?)?(?P<num>\d{1,3}[A-Z]?)(?:\.\d{1,3})*(?:\(\d+\))*")),
    ("us_code", re.compile(r"\b(?P<title>\d{1,3})\s+U\.\s?S\.\s?C\.(?:\s?A\.)?\s*(?:§§?|[Ss]ec(?:tion)?\.?)\s*(?P<sec>\d[\w\-]*(?:\.[\w\-]+)*(?:\([\w]+\))*)")),
    ("cfr", re.compile(r"\b(?P<title>\d{1,3})\s+C\.\s?F\.\s?R\.\s*(?:§§?|[Pp]art)\s*(?P<sec>\d[\w\-]*(?:\.[\w\-]+)*(?:\([\w]+\))*)")),
    ("us_case", re.compile(r"(?P<name>(?:[A-Z][\w.'&-]*\s+){0,4}[A-Z][\w.'&-]*\s+v\.?\s+(?:[A-Z][\w.'&-]*\s+){0,4}[A-Z][\w.'&-]*),\s+"
                           r"(?P<vol>\d{1,4})\s+(?P<rep>[A-Z][A-Za-z.' ]*?(?:\s?\d(?:d|th|nd|rd))?)\s+(?P<page>\d{1,5})(?:,\s*\d{1,5})?\s*"
                           r"\((?P<court>[^()]*?)\s*(?P<year>\d{4})\)")),
    ("uk_neutral", re.compile(r"\[(?P<year>\d{4})\]\s+(?P<court>UKSC|UKPC|UKHL|EWCA\s+Civ|EWCA\s+Crim|EWHC|UKUT|UKFTT|CSIH|CSOH)\s+(?P<num>\d{1,5})"
                              r"(?:\s*\((?:Ch|QB|KB|Comm|Fam|Admin|TCC|Pat|IPEC)\))?")),
    ("uk_report", re.compile(r"\[(?P<year>\d{4})\]\s+(?:(?P<vol>\d{1,2})\s+)?(?P<rep>AC|QB|KB|Ch|WLR|All ER|Lloyd's Rep|BCLC|BCC)\s+(?P<page>\d{1,5})")),
    ("uk_report_old", re.compile(r"\((?P<year>1[5-9]\d{2}|20\d{2})\)\s+(?P<vol>\d{1,3})\s+(?P<rep>Exch|ER|QBD|Ch D|App Cas|CB|M & W|B & Ald|"
                                 r"Cr App R|P & CR|TLR|LT|LJ Ch|LJQB|Lloyd's Rep|WLR|All ER|BCLC|BCC)\s+(?P<page>\d{1,5})")),
    ("uk_act", re.compile(r"(?:(?:[Ss]ection|s\.?)\s+(?P<sec>\d+[A-Z]?(?:\(\d+\))*)\s+of\s+the\s+)?(?P<act>(?:[A-Z][a-z]+\s+){1,6}Act\s+(?P<year>\d{4}))"
                          r"(?:,\s+s(?:ection)?\.?\s*(?P<sec2>\d+[A-Z]?))?")),
    ("eu_act", re.compile(r"(?P<kind>Regulation|Directive|Decision)\s+\((?P<sys>EU|EC|EEC|Euratom)\)\s+(?:No\s+)?(?P<a>\d{2,4})/(?P<b>\d{1,4})"
                          r"|Directive\s+(?P<a2>\d{2,4})/(?P<b2>\d{1,4})/(?P<sys2>EU|EC|EEC)")),
    ("cjeu_case", re.compile(r"\bCase\s+(?P<court>C|T)-(?P<num>\d{1,4})/(?P<yy>\d{2})")),
]
REFERENCE = re.compile(
    r"(?P<word>\b[Cc]lauses?|\b[Ss]ections?|\b[Pp]aragraphs?|\b[Aa]rticles?|§)\s*(?P<num>\d+(?:\.\d+)*(?:\([a-z0-9]{1,4}\))*|[IVXLC]{1,6}\b)"
    r"(?:\s+of\s+(?:the\s+|(?P<dthis>this\s+))?(?P<doc>(?:[A-Z][\w-]*\s+){0,6}(?:Agreement|Contract|Lease|Deed|Letter|Policy|Terms|Memorandum|Order Form|Licence|License)\b))?"
    r"|(?P<attach>\b(?:Schedule|Exhibit|Annex|Appendix))\s+(?P<anum>\d{1,3}|[A-Z]\b)"
    r"(?:\s+(?:of|to)\s+(?:the\s+|(?P<athis>this\s+))?(?P<adoc>(?:[A-Z][\w-]*\s+){0,6}(?:Agreement|Contract|Lease|Deed|Letter|Policy|Terms)\b))?")
DOC_NAME = re.compile(r"\b(?P<det>[Tt]he|[Tt]his)\s+(?P<doc>(?:[A-Z][\w-]*\s+){0,6}(?:Agreement|Contract|Lease|Deed|Letter|Policy|Terms|Memorandum|Order Form|Licence|License))\b")
MORE = re.compile(r"(?:\s*,\s*(?:and\s+|or\s+)?|\s+and\s+|\s+or\s+|\s*(?:to|-|–)\s*)(?P<num>\d+(?:\.\d+)*(?:\([a-z0-9]{1,4}\))*)(?![\w.]*\d)")


@dataclass
class Found:
    kind: str                       # authority | reference | quote
    sub: str                        # us_case, clause, …
    text: str
    start: int
    end: int
    data: dict = field(default_factory=dict)
    malformed: list = field(default_factory=list)


def _sentence(text: str, at: int) -> tuple[int, int]:
    a = max(text.rfind(". ", 0, at), text.rfind("\n", 0, at)) + 1
    m = re.search(r"(?<=[.!?])[\"”]?\s|\n", text[at:])
    return a, at + (m.start() + 1 if m else len(text) - at)


def _malformed(sub: str, m: re.Match) -> list[str]:
    g = m.groupdict()
    out = []
    year = int(g["year"]) if g.get("year") and g["year"].isdigit() else None
    if year and year > _YEAR_NOW:
        out.append(f"the year {year} is in the future")
    if sub == "us_code" and int(g["title"]) > 54:
        out.append(f"the US Code has no title {g['title']}")
    if sub == "cfr" and int(g["title"]) > 50:
        out.append(f"the CFR has no title {g['title']}")
    if sub == "us_case":
        rep = " ".join(g["rep"].split())
        if rep not in REPORTERS:
            out.append(f"{rep} is not a known reporter")
        court = " ".join(g["court"].split())
        c = _CIRCUIT.match(court)
        if c and int(c.group(1)) > 11:
            out.append(f"there is no {court}")
    if sub == "uk_neutral":
        court = " ".join(g["court"].split())
        began = _UK_COURTS.get(court)
        if year and began and year < began:
            out.append(f"{court} gave no judgments before {began}")
        if court == "UKHL" and year and year > 2009:
            out.append("the House of Lords stopped sitting in 2009")
    if sub == "us_rule":
        name, lo, hi = _RULE_SETS[g["set"].casefold()]
        if not lo <= int(g["num"]) <= hi:
            out.append(f"there is no Rule {g['num']} of {name}")
    if sub == "uk_rule" and (g.get("kind") or "").casefold() in ("part", "") and g["num"].isdigit() and int(g["num"]) > 89:
        out.append(f"the Civil Procedure Rules have no Part {g['num']}")
    if sub == "eu_act":
        years = [int(v) for k, v in g.items() if k in ("a", "b", "a2", "b2") and v and len(v) == 4]
        out += [f"the year {y} is in the future" for y in years if y > _YEAR_NOW]
    if sub == "cjeu_case" and int(g["yy"]) > _YEAR_NOW % 100 and int(g["yy"]) < 50:
        out.append(f"the year {g['yy']} is in the future")
    return out


def _heading(text: str, m: re.Match) -> bool:
    """A clause or schedule's own heading: first on its line, followed by a short title, not a sentence."""
    if text[text.rfind("\n", 0, m.start()) + 1:m.start()].strip(" \t#*>_"):
        return False
    eol = text.find("\n", m.end())
    rest = text[m.end():eol if eol >= 0 else len(text)].strip(" \t*_—–-:.")
    return len(rest.split()) <= 6 and not re.search(r"[.;:,\"“]$|\b(?:says?|provides?|states?|is|are|shall|may|must)\b", rest)


_DOC_WORDS = r"(?:Agreement|Contract|Lease|Deed|Letter|Policy|Terms|Memorandum|Order Form|Licence|License)"
ALIAS = re.compile(r"(?P<doc>(?:[A-Z][\w-]*\s+){0,6}" + _DOC_WORDS + r")\s*\(\s*(?:the\s+|this\s+)?[\"“‘](?P<abbr>[A-Z][\w&-]{1,24})[\"”’]\s*\)")
WITHIN = re.compile(r"\s+(?:of|to|in)\s+(?:the\s+)?(?P<att>Schedule|Annex|Exhibit|Appendix)\s+(?P<an>\d{1,3}|[A-Z])\b")


def aliases(text: str) -> dict[str, str]:
    """Short names a document gives others: 'the Share Purchase Agreement (the "SPA")' → {"SPA": "Share Purchase Agreement"}."""
    return {m.group("abbr"): " ".join(m.group("doc").split()) for m in ALIAS.finditer(text)
            if not re.match(r"(?i)this\b", m.group("doc"))}            # 'This Agreement (the "Agreement")' names itself


def last_name(text: str, a: int, b: int, short: dict[str, str]) -> Optional[str]:
    """The last other document named between a and b ("the Lease", "the SPA"), if any."""
    hits = [(m.start(), m.group("doc")) for m in DOC_NAME.finditer(text, a, b) if m.group("det").lower() == "the"]
    if short:
        rx = re.compile(r"\b(?:the\s+)?(" + "|".join(re.escape(k) for k in short) + r")\b")
        hits += [(m.start(), short[m.group(1)]) for m in rx.finditer(text, a, b) if not _in_alias(text, m.start())]
    return max(hits)[1] if hits else None


def _in_alias(text: str, at: int) -> bool:
    return any(m.start() <= at < m.end() for m in ALIAS.finditer(text, max(0, at - 200), min(len(text), at + 40)))


def quote_spans(text: str) -> list[tuple[int, int]]:
    """Quotation marks paired in order: “ ” and ‘ ’ by shape, straight quotes alternately. A quotation
    never runs across a blank line, so one stray mark cannot pair everything after it the wrong way."""
    out, open_at, kind = [], None, None
    for m in re.finditer(r"[\"“”‘’]", text):
        ch, at = m.group(), m.start()
        if ch == "’" and re.match(r"\w", text[at + 1:at + 2] or " "):
            continue                                       # an apostrophe (party’s), not a closing quote
        if ch == "‘" and at and re.match(r"\w", text[at - 1]):
            continue
        if open_at is not None and "\n\n" in text[open_at:at]:
            open_at = None
        if open_at is None:
            if ch in "“‘\"":
                open_at, kind = at, ch
            continue
        closes = {"“": "”", "‘": "’", "\"": "\""}[kind]
        if ch == closes:
            out.append((open_at, at + 1))
            open_at = None
        elif ch in "“‘":
            open_at, kind = at, ch
    return out


def read(text: str) -> list[Found]:
    found: list[Found] = []
    taken: list[tuple[int, int]] = []
    short = aliases(text)
    short_rx = re.compile(r"\s+(?:of|to|in)\s+(?:the\s+)?(" + "|".join(re.escape(k) for k in short) + r")\b") if short else None

    def free(a: int, b: int) -> bool:
        return all(b <= x or a >= y for x, y in taken)

    def by_alias(at: int) -> tuple[Optional[str], int]:
        x = short_rx.match(text, at) if short_rx else None
        return (short[x.group(1)], x.end()) if x else (None, at)
    for sub, rx in AUTHORITIES:
        for m in rx.finditer(text):
            if not free(m.start(), m.end()):
                continue
            taken.append((m.start(), m.end()))
            found.append(Found("authority", sub, m.group(0).strip(), m.start(), m.end(), {k: v for k, v in m.groupdict().items() if v},
                               _malformed(sub, m)))
    refs = []
    for m in REFERENCE.finditer(text):
        if not free(m.start(), m.end()):
            continue
        if _heading(text, m):
            continue                                       # a heading ("Schedule 1 Fees", "Article 2"): the thing itself
        g = m.groupdict()
        if g.get("word"):
            word = g["word"].lower().rstrip("s") if g["word"] != "§" else "section"
            doc = None if g.get("dthis") else (g.get("doc") or "").strip() or None
            end, within = m.end(), None
            if not doc and not g.get("dthis"):
                w = WITHIN.match(text, m.end("num"))
                if w:                                      # "paragraph 1 of Schedule 4": a paragraph inside the schedule
                    within, end = f"{w.group('att')} {w.group('an')}", w.end()
                else:
                    doc, end = by_alias(m.end("num"))
                    end = end if doc else m.end()
            nums = [(g["num"], m.start(), end, text[m.start():end].strip())]
            if g["word"].endswith("s") and not g.get("doc") and not within:   # "Clauses 2.1 and 9.3": each one is checked
                pos = m.end("num")
                while (x := MORE.match(text, pos)) and free(x.start("num"), x.end("num")):
                    nums.append((x.group("num"), x.start("num"), x.end("num"), f"{g['word'].rstrip('s')} {x.group('num')}"))
                    pos = x.end()
                if len(nums) > 1:
                    nums[0] = (g["num"], m.start(), m.end("num"), f"{g['word'].rstrip('s')} {g['num']}")
            for num, a0, b0, label in nums:
                base = re.sub(r"\(.*$", "", num)
                refs.append(Found("reference", word, label, a0, b0, {"number": base, "sub": num[len(base):] or None, "document": doc,
                                                                    "within": within}))
        else:
            doc = None if g.get("athis") else (g.get("adoc") or "").strip() or None
            end = m.end()
            if not doc and not g.get("athis"):
                doc, end = by_alias(m.end("anum"))
                end = end if doc else m.end()
            refs.append(Found("reference", g["attach"].lower(), text[m.start():end].strip(), m.start(), end,
                              {"number": g["anum"], "sub": None, "document": doc, "attachment": True}))
    for r in refs:                                         # "paragraph 1 of Schedule 4 of the SPA": the schedule's document
        if r.data.get("within"):
            att = next((x for x in refs if x.data.get("attachment") and x.start > r.start and x.start <= r.end), None)
            r.data["document"] = att.data["document"] if att else None
    quotes = []
    for a0, b0 in quote_spans(text):
        q = text[a0 + 1:b0 - 1].strip()
        a, _ = _sentence(text, a0)
        lead = [r for r in refs if a <= r.start < a0]
        words = len(q.split())
        if words < 2 or (words < 4 and not lead) or _in_alias(text, a0) or re.match(r"\s*(?:means|shall mean|has the meaning)\b", text[b0:b0 + 30]):
            continue                                       # a defined term, not a quotation
        named = last_name(text, a, a0, short)
        quotes.append(Found("quote", "quotation", q, a0, b0,
                            {"reference": lead[-1] if lead else None, "document": named if not lead else None}))
    used = {id(q.data["reference"]) for q in quotes if q.data.get("reference")}
    found += [r for r in refs if id(r) not in used]
    found += quotes
    return sorted(found, key=lambda f: f.start)
