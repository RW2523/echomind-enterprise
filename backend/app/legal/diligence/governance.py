"""Governance checks (S3.8, DOC-07; decision D-052). Rules only: no model, and no legal decision.

A company's own documents are read against each other. Each finding quotes both documents (the second as
"related") and is advisory: preliminary until a reviewer verifies or dismisses it (the S3.7 register).

- **share_capital:** the shares in the register of members (or cap table) against the shares the board or
  the shareholders approved, by class, and against the classes the articles create.
- **authority:** each contract's total value against the signing limit, in the delegation of authority, of
  the person who signed it for the company.
- **constitution:** a policy or procedure against the articles: the board quorum, the notice of board
  meetings, the minimum number of directors.
- **ip_chain:** each founder the company's documents name against the IP assignments.

Documents are recognised by their own title line ("REGISTER OF MEMBERS …", "DELEGATION OF AUTHORITY …").
"""
from __future__ import annotations

import re
from typing import Optional

from ..compare.facts import _MONEY, _word_number

CHECKS = ("share_capital", "authority", "constitution", "ip_chain")
_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
ROLES = {
    "articles": r"articles of association|bylaws|by-laws|constitution",
    "register": r"register of members|cap table|share register|shareholder register",
    "resolutions": r"minutes|written resolution|resolutions? of the (?:board|shareholders)|written consent|board consent|consent of the (?:board|stockholders|shareholders)",
    "delegation": r"delegation of authority|authority matrix|signing authority|schedule of delegated authorit",
    "policy": r"\bpolicy\b|procedures|terms of reference|board charter",
    "ip": r"ip assignment|intellectual property assignment|assignment of intellectual property|invention assignment",
}


def role_of(title: str, first_line: str) -> set[str]:
    hay = f"{title} {first_line}".casefold()
    return {r for r, rx in ROLES.items() if re.search(rx, hay)}


def _sentence(doc: dict, at: int) -> dict:
    """The sentence (or numbered line) around `at`, as evidence."""
    from .run import _sentence as sentence
    ev = sentence(doc["text"], doc["rows"], at)
    return {"start": ev["start"], "end": ev["end"], "quote": ev["quote"], "clause": ev["clause"]}


def _rel(doc: dict, ev: dict) -> dict:
    return {"document_id": doc["id"], "version_id": doc["version_id"], "start": ev["start"], "end": ev["end"], "quote": ev["quote"]}


def _money(s: str) -> Optional[tuple[float, str]]:
    m = _MONEY.search(s)
    if not m:
        return None
    return float(m.group("num").replace(",", "")), m.group(0).strip()


def _cls(s: str) -> str:
    """'ordinary shares', 'shares of common stock', 'Series A preferred stock' → 'ordinary', 'common', 'series a preferred'."""
    s = re.sub(r"\b(?:shares?|stock|of)\b", " ", s, flags=re.I)
    return " ".join(s.split()).casefold()


def share_capital(docs: list[dict]) -> list[dict]:
    out = []
    regs = [d for d in docs if "register" in d["roles"]]
    arts = [d for d in docs if "articles" in d["roles"]]
    res = [d for d in docs if "resolutions" in d["roles"]]
    holding = re.compile(r"^[ \t]*(?:\d+\.\s*)?(?P<holder>[^:\n—–]{2,80}?)\s*[:—–-]\s*(?P<n>\d[\d,]*)\s+"
                         r"(?:(?P<cls>[A-Za-z][A-Za-z ]{0,30}?)\s+shares\b|shares\s+of\s+(?P<cls2>[A-Za-z][A-Za-z ]{0,30}?\s+stock)\b)", re.M)
    approved_rx = re.compile(r"\b(?:approved|resolved to approve|authori[sz]ed|resolved)\s+(?:the\s+)?(?:allotment|issue|issuance)\s+of\s+(?P<n>\d[\d,]*)\s+"
                             r"(?:(?P<cls>[A-Za-z][A-Za-z ]{0,30}?)\s+shares\b|shares\s+of\s+(?P<cls2>[A-Za-z][A-Za-z ]{0,30}?\s+stock)\b)", re.I)
    classes: Optional[set[str]] = None
    class_ev, class_words = None, ""
    for a in arts:
        m = re.search(r"(?:consists? of|divided into|comprises?)\s+(?P<list>[^.]{3,200}?)\s*(?:\.|$)", a["text"], re.I | re.M)
        if m and re.search(r"\b(?:shares|stock)\b", m.group("list")):
            nominal = r"\bof\s+(?:GBP|USD|EUR|£|\$|€)\s?[\d.,]+(?:\s+each)?|\beach\b"          # "ordinary shares of GBP 1 each"
            labels = [" ".join(re.sub(r"\b(?:shares|stock)\b", " ", re.sub(nominal, " ", x, flags=re.I)).split())
                      for x in re.split(r",\s*|\s+and\s+", m.group("list")) if x.strip()]
            classes = {lab.casefold() for lab in labels}
            class_words = " and ".join([", ".join(labels[:-1]), labels[-1]] if len(labels) > 1 else labels)   # as the articles write them
            class_ev = (a, _sentence(a, m.start()))
            break
    approvals: dict[str, list] = {}
    for r in res:
        for m in approved_rx.finditer(r["text"]):
            approvals.setdefault(_cls(m.group("cls") or m.group("cls2")), []).append((int(m.group("n").replace(",", "")), r, _sentence(r, m.start())))
    for reg in regs:
        lines = [(m, _cls(m.group("cls") or m.group("cls2")), int(m.group("n").replace(",", ""))) for m in holding.finditer(reg["text"])]
        by_class: dict[str, list] = {}
        for m, c, n in lines:
            by_class.setdefault(c, []).append((m, n))
        for c, rows in by_class.items():
            last_m = rows[-1][0]
            ev = {"start": last_m.start("holder"), "end": last_m.end(), "quote": reg["text"][last_m.start("holder"):last_m.end()].rstrip("."), "clause": None}
            if classes is not None and c not in classes:
                label = (last_m.group("cls") or last_m.group("cls2")).strip()
                out.append({"kind": "share_capital", "severity": "high", "document": reg, "evidence": ev,
                            "title": f"{label[0].upper() + label[1:]}{'' if label.endswith('stock') else ' shares'} is not a class the articles create"
                            if label.endswith("stock") else f"{label[0].upper() + label[1:]} shares are not a class the articles create",
                            "detail": f"The register shows {sum(n for _, n in rows):,} {label} shares; the articles create {class_words} shares only.",
                            "related": [_rel(class_ev[0], class_ev[1])] if class_ev else []})
                continue
            if res and c in approvals:
                issued, ok = sum(n for _, n in rows), sum(n for n, _, _ in approvals[c])
                if issued > ok:
                    label = (last_m.group("cls") or last_m.group("cls2")).strip()
                    first = approvals[c][0]
                    out.append({"kind": "share_capital", "severity": "high", "document": reg, "evidence": ev,
                                "title": f"{issued - ok:,} {label}{'' if label.endswith('stock') else ' shares'} in the register have no approval",
                                "detail": f"The register shows {issued:,} {label} shares; the approvals found allow {ok:,}.",
                                "related": [_rel(first[1], first[2])]})
            elif res and approvals and c not in approvals:
                label = (last_m.group("cls") or last_m.group("cls2")).strip()
                out.append({"kind": "share_capital", "severity": "medium", "document": reg, "evidence": ev,
                            "title": f"No approval found for the {label} shares", "detail": "No resolution in this matter approves their issue.", "related": []})
    return out


def authority(docs: list[dict]) -> list[dict]:
    out = []
    dels = [d for d in docs if "delegation" in d["roles"]]
    limits: dict[str, tuple] = {}
    for d in dels:
        for m in re.finditer(r"(?:^|\.\s+|\n[\d.\s]*)(?:The\s+)?(?P<role>[A-Z][A-Za-z ]{2,40}?)\s+may\s+sign\s+[^.]*?\bup\s+to\s+(?P<amt>[^.;]{2,30})", d["text"]):
            amt = _money(m.group("amt"))
            if amt:
                limits[m.group("role").strip().casefold()] = (amt, d, _sentence(d, m.start("role")))
    if not limits:
        return out
    sign = re.compile(r"Signed (?:for|on behalf of) [^\n]{2,80}? by (?P<name>[^,\n]{2,60}),\s*(?P<role>[A-Z][A-Za-z ]{2,40}?)\s*(?:\.|$)", re.M)
    value = re.compile(r"\btotal\s+(?:price|fees|value|consideration|amount|contract value)\s+(?:payable\s+)?(?:under\s+this\s+\w+\s+)?(?:is|are|shall be)\s+(?P<amt>[^.;]{2,30})", re.I)
    for doc in docs:
        if doc["roles"] & {"delegation", "articles", "register", "policy"}:
            continue
        s = sign.search(doc["text"])
        v = value.search(doc["text"])
        if not s or not v:
            continue
        amt = _money(v.group("amt"))
        role = s.group("role").strip()
        if not amt:
            continue
        lim = limits.get(role.casefold())
        ev = _sentence(doc, s.start())
        if lim is None:
            continue                                        # a role the delegation does not list is for a reviewer, not a rule
        (limit, limit_text), ddoc, dev = lim
        if amt[0] > limit:
            out.append({"kind": "authority", "severity": "high", "document": doc, "evidence": ev,
                        "title": f"Signed above the signer's authority (clause {ev['clause'] or '—'})",
                        "detail": f"Signed by {s.group('name').strip()}, {role}, for a contract of {amt[1]}; the delegation of authority allows the "
                                  f"{role} up to {limit_text}.",
                        "related": [_rel(ddoc, dev)]})
    return out


def _n(word: str) -> Optional[int]:
    word = word.strip("()")
    return int(word) if word.isdigit() else _word_number(word)


def _amount(g: dict):
    """A number, or for notice periods hours (so 5 days and 24 hours compare): from n/d/u or n2/d2/u2."""
    for suf in ("", "2"):
        n, d, u = g.get("n" + suf), g.get("d" + suf), g.get("u" + suf)
        if n or d:
            v = int(d) if d else _n(n)
            if v is None:
                return None
            return v * 24 if (u or "").casefold() == "days" else v if u else v
    return None


def _notice_words(hours) -> str:
    return f"{hours // 24} days" if hours % 24 == 0 and hours >= 48 else f"{hours} hours"   # "24 hours", "2 days", "7 days"


def constitution(docs: list[dict]) -> list[dict]:
    out = []
    arts = [d for d in docs if "articles" in d["roles"]]
    pols = [d for d in docs if "policy" in d["roles"] and "articles" not in d["roles"]]
    if not arts or not pols:
        return out
    rules = [
        ("quorum", "Board quorum",
         re.compile(r"quorum for (?:board|directors['’]?) meetings (?:is|shall be) (?P<n>\w+)(?: \(\d+\))? directors?", re.I),
         re.compile(r"(?:quorate when (?P<n>\w+)(?: \(\d+\))? directors? (?:is|are) present|quorum for (?:board|directors['’]?) meetings (?:is|shall be) (?P<n2>\w+)(?: \(\d+\))? directors?)", re.I),
         lambda p, a: f"{_WORDS.get(p, p)} director{'s' if p != 1 else ''} in the policy, {_WORDS.get(a, a)} in the articles"),
        ("notice", "Notice of board meetings",
         re.compile(r"(?:(?P<n>\w+)(?: \((?P<d>\d+)\))? (?P<u>days|hours)['’]? notice shall be given of (?:every|each|any) board meeting"
                    r"|board meetings? require[s]? not less than (?P<n2>\w+)(?: \((?P<d2>\d+)\))? (?P<u2>days|hours)['’]? notice)", re.I),
         re.compile(r"(?:board meetings? may be called|(?:convene|call) a board meeting) on (?P<n>\w+)(?: \((?P<d>\d+)\))? (?P<u>days|hours)['’]? notice", re.I),
         lambda p, a: f"{p} in the policy, {a} in the articles"),
        ("directors", "Number of directors",
         re.compile(r"number of directors shall not be less than (?P<n>\w+)", re.I),
         re.compile(r"(?:at least|not less than|minimum of) (?P<n>\w+)(?: \(\d+\))? directors?", re.I),
         lambda p, a: f"at least {_WORDS.get(p, p)} in the policy, {_WORDS.get(a, a)} in the articles"),
    ]
    for key, label, art_rx, pol_rx, say in rules:
        am = next(((a, m) for a in arts for m in [art_rx.search(a["text"])] if m), None)
        if not am:
            continue
        a, m = am
        an = _amount(m.groupdict())
        for p in pols:
            pm = pol_rx.search(p["text"])
            if not pm:
                continue
            pn = _amount(pm.groupdict())
            if an is None or pn is None or an == pn:
                continue
            words = (lambda v: v) if key != "notice" else _notice_words
            out.append({"kind": "constitution", "severity": "medium", "document": p, "evidence": _sentence(p, pm.start()),
                        "title": f"{label}: {say(words(pn), words(an))}", "detail": "The policy and the articles disagree; the articles normally prevail.",
                        "related": [_rel(a, _sentence(a, m.start()))]})
    return out


def ip_chain(docs: list[dict]) -> list[dict]:
    out = []
    ips = [d for d in docs if "ip" in d["roles"]]
    name = r"(?:[A-Z]\.\s?)*[A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)*"                       # "A. Moreau", "Ines Faraday"
    names_rx = re.compile(rf"\bfounders?\s+(?:of\s+the\s+company\s+)?(?:are|is|were|was|:)\s+(?P<names>{name}(?:\s*(?:,\s*(?:and\s+)?|\s+and\s+){name})*)")
    seen = set()
    for d in docs:
        if d in ips:
            continue
        for m in names_rx.finditer(d["text"]):
            names = [x.strip() for x in re.split(r",\s*(?:and\s+)?|\s+and\s+", m.group("names")) if x.strip()]
            for name in names:
                if name in seen or not re.match(r"[A-Z]", name):
                    continue
                seen.add(name)
                assigned = any(re.search(rf"{re.escape(name)}[^.]{{0,120}}\bassign", ip["text"]) for ip in ips)
                if not assigned:
                    out.append({"kind": "ip_chain", "severity": "medium", "document": d, "evidence": _sentence(d, m.start()),
                                "title": f"No IP assignment from {name}",
                                "detail": f"{name} is named as a founder; no IP assignment in this matter is made by them.", "related": []})
    return out


def run(checks: list[str], docs: list[dict]) -> list[dict]:
    """Every finding of the checks asked for, over the documents read in a run."""
    out = []
    for c in checks:
        out += {"share_capital": share_capital, "authority": authority, "constitution": constitution, "ip_chain": ip_chain}[c](docs)
    return out
