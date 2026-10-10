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

AUTHORITIES = [
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
QUOTE = re.compile(r"[\"“](?P<q>[^\"“”]{12,1200}?)[\"”]")


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


def read(text: str) -> list[Found]:
    found: list[Found] = []
    taken: list[tuple[int, int]] = []

    def free(a: int, b: int) -> bool:
        return all(b <= x or a >= y for x, y in taken)
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
            nums = [(g["num"], m.start(), m.end(), m.group(0).strip())]
            if g["word"].endswith("s") and not g.get("doc"):          # "Clauses 2.1 and 9.3": each one is checked
                pos = m.end("num")
                while (x := MORE.match(text, pos)) and free(x.start("num"), x.end("num")):
                    nums.append((x.group("num"), x.start("num"), x.end("num"), f"{g['word'].rstrip('s')} {x.group('num')}"))
                    pos = x.end()
                if len(nums) > 1:
                    nums[0] = (g["num"], m.start(), m.end("num"), f"{g['word'].rstrip('s')} {g['num']}")
            for num, a0, b0, label in nums:
                base = re.sub(r"\(.*$", "", num)
                refs.append(Found("reference", word, label, a0, b0, {"number": base, "sub": num[len(base):] or None, "document": doc}))
        else:
            refs.append(Found("reference", g["attach"].lower(), m.group(0).strip(), m.start(), m.end(),
                              {"number": g["anum"], "sub": None, "document": None if g.get("athis") else (g.get("adoc") or "").strip() or None,
                               "attachment": True}))
    quotes = []
    for m in QUOTE.finditer(text):
        q = m.group("q").strip()
        if len(q.split()) < 4:
            continue
        a, _ = _sentence(text, m.start())
        lead = [r for r in refs if a <= r.start < m.start()]
        named = [d for d in DOC_NAME.finditer(text, a, m.start()) if d.group("det").lower() == "the"]
        quotes.append(Found("quote", "quotation", q, m.start(), m.end(),
                            {"reference": lead[-1] if lead else None, "document": named[-1].group("doc") if named and not lead else None}))
    used = {id(q.data["reference"]) for q in quotes if q.data.get("reference")}
    found += [r for r in refs if id(r) not in used]
    found += quotes
    return sorted(found, key=lambda f: f.start)
