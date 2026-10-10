"""Defined terms and cross-references (S3.5, DOC-04; decision D-049). Rules only: no model.

A document is read for its own consistency:
- **Definitions:** every definition ('"X" means …', '(the "X")'); a term defined twice, and whether the
  meanings differ; a definition never used; a capitalised term that reads as a defined term but is
  never defined; a defined term used in another form ("the fees" for "Fees"); a term used before it is
  defined.
- **Numbering:** gaps and repeats at every level, items such as "(a)" included, and schedules. The
  numbers are read from the clauses themselves, so a Markdown heading the reader does not see still
  counts (`documents.numbering`).
- **References:** a clause of this document that does not exist (S3.4's reader of references), and a
  reference whose heading no longer matches ("clause 4 (Liability)" when clause 4 is Confidentiality).

Each finding has exact offsets in the version's text. A finding may carry fixes; nothing changes until
a lawyer picks them, they become a redline (S3.3) and a second lawyer approves it. Numbering is never
renumbered automatically.
"""
from __future__ import annotations

import re
from typing import Optional

from ..citations import parse as cparse
from ..citations.check import _GENERIC
from ..documents import numbering
from ..documents.structure import _MEANS, _PAREN

VERSION = "definitions-s3.5b"
# Words a defined term usually ends with: a capitalised phrase ending with one of them reads as a defined term.
HEADS = {"date", "period", "day", "fee", "price", "charge", "information", "data", "service", "term", "notice", "event", "order",
         "product", "deliverable", "material", "right", "loss", "claim", "control", "default", "amount", "level", "credit", "plan",
         "territory", "premises", "property", "rent", "cap", "year", "month", "quarter", "account", "software", "documentation",
         "specification", "milestone", "release", "requirement", "standard", "party", "parties", "affiliate", "group", "personnel",
         "equipment", "facility", "site", "change", "interest", "payment", "invoice", "expense", "cost", "value", "warranty",
         "warranties", "obligation", "liabilities", "indemnity", "budget", "policy", "procedure", "report", "deliverables"}
_DETERMINERS = r"(?:the|any|each|all|such|that|those|these|its|their|every|no)"
_LEADING = {"The", "This", "These", "Those", "That", "Each", "Either", "Neither", "No", "Any", "All", "If", "In", "On", "Under", "Subject",
            "For", "A", "An", "Where", "When", "Without", "Notwithstanding", "Upon", "Unless", "Save", "Except", "Following", "Before", "After"}
COMPANY = re.compile(r"^(?:Ltd|Limited|Inc|LLC|LLP|plc|PLC|AB|GmbH|AG|SA|BV|NV|Corp|Corporation|Company|Co|SE|SAS|Pty|PLLC)\.?$")
_TOKEN = r"[A-Z][A-Za-z'’\-]*"
PHRASE = re.compile(rf"(?<![\w\"“]){_TOKEN}(?:[ \t]+(?:(?:of|and|for|to|in|on|under)[ \t]+)?{_TOKEN})*")
_EACH = re.compile(r",?\s+(?:each|together|collectively|jointly)\s+(?:a|an|the)\s+[\"“]([A-Z][\w&'’\- ]{0,60}?)[\"”]")
_PARTY_BEFORE = re.compile(r"(?:Ltd|Limited|Inc|LLC|LLP|plc|PLC|AB|GmbH|AG|SA|BV|NV|Corp|Corporation|Company|Co|SE|SAS|Pty|PLLC)\.?\s*$")
DESCRIPTOR = re.compile(r"\s*\((?P<h>[A-Z][^()\n]{1,60})\)")


# ── Definitions ───────────────────────────────────────────────────────────────────────
def _sentences(text: str) -> list[tuple[int, int]]:
    from ..research.prompt import document_sentences
    return document_sentences(text)


def definitions(text: str) -> list[dict]:
    """Every definition, duplicates included: term, offsets of the term, the definition sentence and its meaning."""
    sents = _sentences(text)
    out = []
    for rx, kind in ((_MEANS, "means"), (_PAREN, "paren"), (_EACH, "paren")):
        for m in rx.finditer(text):
            a, b = next(((x, y) for x, y in sents if x <= m.start() < y), (m.start(), len(text)))
            meaning = text[m.end():b] if kind == "means" else text[a:m.start()]
            out.append({"term": m.group(1).strip(), "start": m.start(1), "end": m.end(1), "sentence": (a, b), "kind": kind,
                        "meaning": " ".join(re.sub(r"[\"“”'’.,;:()]", " ", meaning).casefold().split())})
    return sorted(out, key=lambda d: d["start"])


def _forms(term: str) -> list[str]:
    forms = [term]
    if not term.endswith("s"):
        forms.append(term + "s")
    elif term.endswith("ies"):
        forms.append(term[:-3] + "y")
    else:
        forms.append(term[:-1])
    return forms


def _heading_spans(text: str, clauses: list[dict]) -> list[tuple[int, int]]:
    spans = []
    for c in clauses:
        if c.get("heading"):
            eol = text.find("\n", c["start"])
            spans.append((c["start"], eol if eol >= 0 else len(text)))
    spans += [(m.start(), m.end()) for m in re.finditer(r"^[ \t]*#{1,6}[^\n]*$", text, re.M)]
    return spans


def _inside(at: int, spans) -> bool:
    return any(a <= at < b for a, b in spans)


def _quoted(text: str, a: int, b: int) -> bool:
    return text[a - 1:a] in "\"“‘" and text[b:b + 1] in "\"”’"


def analyse(text: str, clauses: list[dict]) -> list[dict]:
    """All findings for one version's text and its clause rows."""
    clauses = numbering.complete(text, clauses)
    defs = definitions(text)
    def_sentences = [d["sentence"] for d in defs]
    heads = _heading_spans(text, clauses)
    findings: list[dict] = []

    def add(kind: str, type_: str, severity: str, text_: str, start: int, end: int, message: str, **kw) -> dict:
        f = {"kind": kind, "type": type_, "severity": severity, "text": text_, "start": start, "end": end, "message": message,
             "places": kw.pop("places", []), "fixes": kw.pop("fixes", []), **kw}
        findings.append(f)
        return f

    by_term: dict[str, list[dict]] = {}
    for d in defs:
        by_term.setdefault(d["term"], []).append(d)
    defined = set(by_term)

    # Uses of each defined term: the exact form (or its plural), not inside a definition of that term,
    # a heading, quotation marks, or a longer capitalised name ("Services Agreement").
    uses: dict[str, list[tuple[int, int]]] = {}
    for term, ds in by_term.items():
        own = [d["sentence"] for d in ds]
        rx = re.compile(r"(?<![\w-])(" + "|".join(re.escape(f) for f in sorted(_forms(term), key=len, reverse=True)) + r")(?![\w-])")
        for m in rx.finditer(text):
            a, b = m.span(1)
            if _inside(a, own) or _inside(a, heads) or _quoted(text, a, b):
                continue
            nxt = re.match(r"[ \t]+([A-Z][a-z]+)", text[b:b + 30])
            if nxt and nxt.group(1) not in _LEADING and f"{term} {nxt.group(1)}" not in defined:
                continue                                    # part of a longer name: "Services Agreement"
            uses.setdefault(term, []).append((a, b))

    # Defined terms used in another form: "the fees" for "Fees", "confidential information".
    for term, ds in by_term.items():
        own = [d["sentence"] for d in ds]
        words = term.split()
        variants = [term] + ([term + "s"] if not term.endswith("s") else [])   # not "service levels" for "Services"
        pat = r"(?<![\w-])(" + "|".join(re.escape(f) for f in sorted(variants, key=len, reverse=True)) + r")(?![\w-])"
        for m in re.finditer(pat, text, re.I):
            a, b = m.span(1)
            got = m.group(1)
            if got in variants or _inside(a, own) or _inside(a, heads) or _quoted(text, a, b) or got.isupper():
                continue
            if len(words) == 1 and not re.search(rf"\b{_DETERMINERS}[ \t]+$", text[max(0, a - 12):a], re.I):
                continue                                    # a single common word only after "the", "any", "such"…
            if text[a - 1:a].isalpha() or (a and text[a - 1] == "-"):
                continue
            right = term if got.casefold() == term.casefold() else term + "s"
            add("definition", "inconsistent", "low", got, a, b, term=term, message=f"“{got}” is used where the defined term is “{right}”.",
                fixes=[{"label": f"Use “{right}”", "start": a, "end": b, "original": got, "revised": right,
                        "clause": _leaf(clauses, a)}])

    for term, ds in by_term.items():
        if len(ds) > 1:
            conflicting = len({d["meaning"] for d in ds}) > 1
            add("definition", "duplicate", "high" if conflicting else "medium", term, ds[1]["start"], ds[1]["end"],
                f"“{term}” is defined {'twice' if len(ds) == 2 else f'{len(ds)} times'}" + (", with different meanings." if conflicting else ", in the same words."),
                places=[{"start": d["start"], "end": d["end"]} for d in ds], conflicting=conflicting)
        party = all(_PARTY_BEFORE.search(text[max(0, d["start"] - 40):d["start"]].rstrip(" (\"“")) for d in ds)
        if not uses.get(term) and not party and not any(f["type"] == "inconsistent" and f["term"] == term for f in findings):
            d = ds[0]
            fixes = []
            c = _whole_clause(text, clauses, d)
            if c and len(ds) == 1:
                fixes.append({"label": "Delete the definition", "start": c["start"], "end": c["end"], "original": text[c["start"]:c["end"]],
                              "revised": "", "clause": c["number"]})
            add("definition", "unused", "low", term, d["start"], d["end"], f"“{term}” is defined but never used.", fixes=fixes)
        elif uses.get(term) and uses[term][0][0] < ds[0]["start"] and not _in_definitions_clause(ds[0], clauses, text):
            a, b = uses[term][0]
            add("definition", "before_definition", "low", term, a, b,
                f"“{term}” is used before it is defined (in {_where(clauses, ds[0]['start'])}).",
                places=[{"start": ds[0]["start"], "end": ds[0]["end"]}])

    # Capitalised terms that read as defined terms but are not defined.
    seen_undefined: dict[str, dict] = {}
    segments = []
    for m in PHRASE.finditer(text):                        # "Party's Confidential Information": two names, not one
        pos = m.start()
        for seg in re.split(r"(?<=['’]s)[ \t]+", m.group(0)):
            at = text.index(seg, pos)
            segments.append((at, re.sub(r"['’]s$", "", seg)))
            pos = at + len(seg)
    # A single capitalised word reads as a term when it follows "the", "a" or a number at least twice
    # ("five Directors", "a Director"), and is not used as a name in front of a noun ("the Harrowgate platform").
    singles: dict[str, int] = {}
    for a0, seg in segments:
        w = seg.split()[-1] if seg.split() else ""
        if len(seg.split()) == 1 or seg.split()[0] in _LEADING:
            if _single_term(text, a0 + seg.rfind(w), w):
                singles[_singular(w)] = singles.get(_singular(w), 0) + 1
    for a, phrase0 in segments:
        words = phrase0.split()
        while words and words[0] in _LEADING:
            a = text.index(words[1], a + len(words[0])) if len(words) > 1 else a
            words = words[1:]
        if not words:
            continue
        phrase = " ".join(words)
        span = re.match(r"[ \t]+".join(map(re.escape, words)), text[a:])
        b = a + (span.end() if span else len(phrase))
        last = words[-1].rstrip("’'s").casefold() if words[-1].endswith(("’s", "'s")) else words[-1].casefold()
        base = last[:-1] if last.endswith("s") and last[:-1] in HEADS else last
        starts_defined = (len(words) > 1 and any(words[0] in _forms(t) for t in defined)       # "Shareholder Consent"
                          and all(w[:1].isupper() for w in words) and words[-1] not in ("Clause", "Section", "Schedule", "Article", "Paragraph"))
        repeated = len(words) == 1 and singles.get(_singular(words[0]), 0) >= 2 and words[0] not in _NAMES
        if base not in HEADS and last not in HEADS and not starts_defined and not repeated:
            continue
        if any(COMPANY.match(w) for w in words) or re.match(r"[ \t]+(?:Ltd|Limited|Inc|LLC|LLP|plc|AB|GmbH)\b", text[b:b + 10]):
            continue
        if _inside(a, heads) or _quoted(text, a, b) or _inside(a, def_sentences) and _defined_here(text, a, b):
            continue
        if words[-1] in ("Agreement", "Form", "Contract", "Lease", "Deed", "Letter", "Terms", "Memorandum", "Schedule", "Annex",
                         "Exhibit", "Appendix", "Policy"):
            continue
        if text[max(0, a - 1):a] == "(" and text[b:b + 1] == ")":
            continue                                        # a heading named in a reference: "clause 9 (Liability)"
        if phrase in defined or any(phrase in _forms(t) for t in defined) or any(p in defined for p in _forms(phrase)):
            continue
        if len(words) == 1 and not _single_term(text, a, words[0]):
            continue                                        # one capitalised word only reads as a term after "the", "a", "five"…
        if _sentence_start(text, a) and len(words) == 1:
            continue
        key = _singular(phrase) if len(words) == 1 else phrase
        if key in seen_undefined:
            seen_undefined[key]["places"].append({"start": a, "end": b})
            continue
        seen_undefined[key] = add("definition", "undefined", "medium", phrase, a, b,
                                     f"“{phrase}” reads as a defined term but is not defined in this document.",
                                     places=[{"start": a, "end": b}])

    findings += numbering_findings(clauses)
    findings += reference_findings(text, clauses)
    for f in findings:                                     # where each finding (and each place) is, as a lawyer says it
        f["where"] = _where(clauses, f["start"])
        for p in f["places"]:
            p["where"] = _where(clauses, p["start"])
    order = {"definition": 0, "reference": 1, "numbering": 2}
    findings.sort(key=lambda f: (order[f["kind"]], f["start"]))
    return findings


_QUANT = re.compile(r"(?i)\b(?:the|a|an|any|each|all|such|that|those|these|its|their|every|no|one|two|three|four|five|six|seven|eight|nine|"
                    r"ten|several|more|other|\d+)[ \t]+$")
_FOLLOW = re.compile(r"[ \t]+(?:shall|may|must|will|is|are|was|were|has|have|does|do|agrees?|warrants?|undertakes?|confirms?|can|cannot|"
                     r"should|would|means|with|of|to|for|in|on|at|by|from|under|and|or|but|as|if|who|which|that|whose|unless|until|"
                     r"before|after|within|without|between|against|during|upon|into|than)\b|\s*[.,;:)]|\s*$|[ \t]+[A-Z0-9(]")
_NAMES = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January", "February", "March", "April", "May",
          "June", "July", "August", "September", "October", "November", "December", "England", "Wales", "Scotland", "Ireland", "Europe",
          "Delaware", "Sweden", "Court", "Crown", "Government", "Parliament", "Internet", "State", "States", "Kingdom", "Union"}


def _to_roman(n: int) -> str:
    out = ""
    for v, r in ((100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= v:
            out, n = out + r, n - v
    return out


def _singular(w: str) -> str:
    return w[:-1] if w.endswith("s") and not w.endswith("ss") else w


def _single_term(text: str, a: int, w: str) -> bool:
    """A capitalised word after "the", "a" or a number, standing as a noun (not "the Harrowgate platform")."""
    return bool(_QUANT.search(text[max(0, a - 14):a])) and bool(_FOLLOW.match(text, a + len(w))) and w not in _NAMES


def _defined_here(text: str, a: int, b: int) -> bool:
    """The name a definition is made from: '"X" means', or 'the Building Regulations (the "Regulations")'."""
    return bool(re.match(r"\s*(?:\)|\"|”|means\b|\((?:the\s+|each\s+a\s+|together\s+the\s+)?[\"“])", text[b:b + 24]))


def _sentence_start(text: str, a: int) -> bool:
    before = text[max(0, a - 4):a]
    return not before.strip() or bool(re.search(r"[.;:!?\n]\s*$|^\s*$", before)) or bool(re.search(r"\n\s*(?:\d+(?:\.\d+)*\.?|\([a-z0-9]+\))\s*$", text[max(0, a - 12):a]))


def _leaf(clauses: list[dict], at: int) -> Optional[str]:
    inside = [c for c in clauses if c["start"] <= at < max(c["end"], c["start"] + 1)]
    return max(inside, key=lambda c: (c["level"], c["start"]))["number"] if inside else None


def _where(clauses: list[dict], at: int) -> str:
    n = _leaf(clauses, at)
    if n:
        return f"clause {n}" if not numbering.ATTACHMENT.match(n) else n
    return "the opening text" if clauses and at < min(c["start"] for c in clauses) else "this document"


def _whole_clause(text: str, clauses: list[dict], d: dict) -> Optional[dict]:
    """The clause that is only this definition ('1.6 "Territory" means …'), to offer its deletion."""
    leaf = _leaf(clauses, d["start"])
    c = next((c for c in clauses if c["number"] == leaf), None)
    if not c or d["kind"] != "means":
        return None
    body = text[c["start"]:c["end"]]
    return c if body.count("\n") == 0 and len(re.findall(r"\bmeans\b", body)) == 1 else None


def _in_definitions_clause(d: dict, clauses: list[dict], text: str) -> bool:
    """A definitions clause (headed "Definitions" or "Interpretation") is read first whatever its place."""
    for c in clauses:
        if c["start"] <= d["start"] < c["end"] and re.match(r"(?i)(?:definitions|interpretation|defined terms)", c.get("heading") or ""):
            return True
    return False


# ── Numbering ───────────────────────────────────────────────────────────────────────
def _letters(a: str, b: str) -> list[str]:
    return [chr(x) for x in range(ord(a) + 1, ord(b))]


def _seq_msg(prefix: str, missing: list[str], lo: str, hi: str, item: bool, owner: str) -> str:
    if item:
        names = [f"({x})" for x in missing]
        what = (f"Item {names[0]} of {owner} is missing" if len(names) == 1 else
                f"Items {names[0]} and {names[1]} of {owner} are missing" if len(names) == 2 else
                f"Items {names[0]} to {names[-1]} of {owner} are missing")
        return f"{what}: the items go from ({lo}) to ({hi})."
    names = [prefix + x for x in missing]
    noun = "Clause" if not owner.startswith(("Schedule", "Annex", "Exhibit", "Appendix")) else "Paragraph"
    where = "" if noun == "Clause" else f" of {owner}"
    what = (f"{noun} {names[0]}{where} is missing" if len(names) == 1 else
            f"{noun}s {names[0]} and {names[1]}{where} are missing" if len(names) == 2 else
            f"{noun}s {names[0]} to {names[-1]}{where} are missing")
    return f"{what}: the numbering goes from {prefix}{lo} to {prefix}{hi}."


def numbering_findings(clauses: list[dict]) -> list[dict]:
    out = []
    rows = [c for c in clauses if not c.get("implied")]
    boxes = [c for c in clauses if numbering.ATTACHMENT.match(c["number"])]

    def owner_of(c):
        box = next((b for b in boxes if b["start"] < c["start"] < max(b["end"], b["start"] + 1)), None)
        return box["number"] if box else ""
    groups: dict[tuple[str, str, str], list[dict]] = {}       # (attachment, parent number, kind) → children in order
    articles = [c for c in clauses if re.fullmatch(r"Article (\d{1,3}|[IVXLC]+)", c["number"])]
    for c in articles:
        groups.setdefault(("", "", "Article"), []).append({**c, "_val": c["number"].split()[1]})
    for c in clauses:
        if numbering.ATTACHMENT.match(c["number"]):
            kind_, _, val = c["number"].partition(" ")
            groups.setdefault(("", "", kind_.title()), []).append({**c, "_val": val})
            continue
        sp = numbering.split(c["number"])
        if not sp:
            continue
        nums, items = sp
        box = owner_of(c)
        if items:
            parent = ".".join(map(str, nums)) + "".join(f"({x})" for x in items[:-1])
            groups.setdefault((box, parent, "item"), []).append({**c, "_val": items[-1]})
        else:
            parent = ".".join(map(str, nums[:-1]))
            groups.setdefault((box, parent, "num"), []).append({**c, "_val": str(nums[-1])})
    for (box, parent, kind_), kids in groups.items():
        kids = sorted(kids, key=lambda c: c["start"])
        if articles and kind_ == "num" and not parent and not box and all(k.get("implied") for k in kids):
            continue                                       # the articles' own numbers, checked as articles
        vals = [k["_val"] for k in kids]
        real = [k for k in kids if not k.get("implied")]
        for v in sorted({k["_val"] for k in real}):
            same = [k for k in real if k["_val"] == v]
            if len(same) > 1:
                n = same[1]["number"]
                out.append({"kind": "numbering", "type": "repeat", "severity": "medium", "text": n, "start": same[1]["start"],
                            "end": same[1]["start"] + len(n), "message": f"{'Clause' if kind_ != 'item' else 'Item'} {n} appears "
                            f"{'twice' if len(same) == 2 else f'{len(same)} times'}.", "places": [{"start": k["start"], "end": k["start"]} for k in same],
                            "fixes": []})
        prefix = f"{parent}." if parent and kind_ == "num" else ""
        owner = box or (f"clause {parent}" if parent else "")
        if kind_ in ("num", "Schedule", "Annex", "Exhibit", "Appendix", "Article"):
            seq = []
            for k in kids:
                if k["_val"].isdigit():
                    seq.append((int(k["_val"]), k))
                elif kind_ == "Article" and re.fullmatch(r"[IVXLC]+", k["_val"]):
                    from ..documents.structure import _roman_value
                    seq.append((_roman_value(k["_val"].lower()), k))
                elif len(k["_val"]) == 1 and k["_val"].isalpha():
                    seq.append((ord(k["_val"].upper()) - 64, k))
            seq = sorted({v: k for v, k in seq}.items())
            for (a, ka), (b, kb) in zip(seq, seq[1:]):
                if b > a + 1:
                    romans = kind_ == "Article" and not ka["_val"].isdigit()
                    letters = not ka["_val"].isdigit() and not romans
                    miss = [_to_roman(x) if romans else chr(64 + x) if letters else str(x) for x in range(a + 1, b)]
                    if kind_ == "num":
                        msg = _seq_msg(prefix, miss, ka["_val"], kb["_val"], False, box or "")
                    else:
                        msg = (f"{kind_} {miss[0]} is missing" if len(miss) == 1 else f"{kind_}s {miss[0]} to {miss[-1]} are missing") + \
                              f": the {kind_.lower()}s go from {ka['_val']} to {kb['_val']}."
                    out.append({"kind": "numbering", "type": "gap", "severity": "low", "text": kb["number"], "start": kb["start"],
                                "end": kb["start"] + len(kb["number"]), "message": msg, "places": [], "fixes": []})
        elif vals and vals[0] == "i" and all(re.fullmatch(r"[ivxlc]+", v) for v in vals):
            from ..documents.structure import _roman_value
            romans = [(_roman_value(k["_val"]), k) for k in kids]
            for (a, ka), (b, kb) in zip(romans, romans[1:]):
                if b > a + 1:
                    out.append({"kind": "numbering", "type": "gap", "severity": "low", "text": kb["number"], "start": kb["start"],
                                "end": kb["start"] + len(kb["number"]), "message": f"Items between ({ka['_val']}) and ({kb['_val']}) of "
                                f"{owner} are missing.", "places": [], "fixes": []})
        else:
            letters = [k for k in kids if len(k["_val"]) == 1 and k["_val"].isalpha()]
            for ka, kb in zip(letters, letters[1:]):
                if ord(kb["_val"]) > ord(ka["_val"]) + 1:
                    out.append({"kind": "numbering", "type": "gap", "severity": "low", "text": kb["number"], "start": kb["start"],
                                "end": kb["start"] + len(kb["number"]),
                                "message": _seq_msg("", _letters(ka["_val"], kb["_val"]), ka["_val"], kb["_val"], True, owner),
                                "places": [], "fixes": []})
    return out


# ── References ──────────────────────────────────────────────────────────────────────
def _norm(h: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", h.casefold().replace("&", " and ")).split())


def _lookup(clauses: list[dict], f: cparse.Found) -> Optional[dict]:
    from ..citations.check import _keys
    keys = _keys(f)
    pool = clauses
    if f.data.get("within"):
        box = next((c for c in clauses if c["number"].casefold() == f.data["within"].casefold()), None)
        if not box:
            return None
        pool = [c for c in clauses if box["start"] < c["start"] < max(box["end"], box["start"] + 1)]
    for k in keys:
        hit = next((c for c in pool if c["number"].casefold() == k.casefold()), None)
        if hit:
            return hit
    return None


def reference_findings(text: str, clauses: list[dict]) -> list[dict]:
    out = []
    for f in cparse.read(text):
        if f.kind != "reference" or (f.data.get("document") and f.data["document"].casefold() not in _GENERIC):
            continue                                        # references to other documents are S3.4's
        target = _lookup(clauses, f)
        d = DESCRIPTOR.match(text, f.end)
        desc = d.group("h").strip() if d and len(d.group("h").split()) <= 6 else None
        word = f.text.split()[0]
        by_heading = [c for c in clauses if desc and c.get("heading") and _norm(c["heading"]) == _norm(desc)
                      and not numbering.ATTACHMENT.match(c["number"])]
        span_end = d.end() if desc else f.end
        if target is None:
            label = f"{f.sub} {f.data['number']}{f.data.get('sub') or ''}" if not f.data.get("attachment") else f.text
            fixes = [{"label": f"Point to clause {c['number']} ({c['heading']})", "start": f.start, "end": span_end,
                      "original": text[f.start:span_end], "revised": f"{word} {c['number']} ({desc})", "clause": _leaf(clauses, f.start)}
                     for c in by_heading]
            msg = f"This document has no {label}."
            same = next((x for x in out if x["type"] == "broken" and x["message"] == msg and not fixes and not x["fixes"]), None)
            if same:                                       # the same missing clause cited again: one finding, every place
                same["places"].append({"start": f.start, "end": span_end})
                continue
            out.append({"kind": "reference", "type": "broken", "severity": "high", "text": text[f.start:span_end], "start": f.start,
                        "end": span_end, "message": msg, "places": [{"start": f.start, "end": span_end}] if not fixes else [], "fixes": fixes})
            continue
        if desc and target.get("heading") and _norm(target["heading"]) != _norm(desc) and not _norm(target["heading"]).startswith(_norm(desc)):
            fixes = [{"label": f"Keep {word.lower()} {target['number']}; name it {target['heading']}", "start": f.start, "end": span_end,
                      "original": text[f.start:span_end], "revised": f"{text[f.start:f.end]} ({target['heading']})", "clause": _leaf(clauses, f.start)}]
            fixes += [{"label": f"Point to clause {c['number']} ({c['heading']})", "start": f.start, "end": span_end,
                       "original": text[f.start:span_end], "revised": f"{word} {c['number']} ({desc})", "clause": _leaf(clauses, f.start)}
                      for c in by_heading if c["number"] != target["number"]]
            out.append({"kind": "reference", "type": "outdated", "severity": "medium", "text": text[f.start:span_end], "start": f.start,
                        "end": span_end, "message": f"{word.title() if word.islower() else word} {target['number']} is headed {target['heading']}, "
                        f"not {desc}.", "places": [{"start": target["start"], "end": target["start"] + len(target["number"])}], "fixes": fixes})
    return out


def summary(findings: list[dict], defs: Optional[int] = None) -> dict:
    out = {"definitions": defs or 0}
    for t in ("duplicate", "unused", "undefined", "inconsistent", "before_definition", "gap", "repeat", "broken", "outdated"):
        out[t] = sum(1 for f in findings if f["type"] == t)
    return out
