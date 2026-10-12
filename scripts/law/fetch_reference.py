#!/usr/bin/env python3
"""Fetch the public-law reference texts for the EchoMind Law demo (D-064), once, from official sources.

    python3 scripts/law/fetch_reference.py            # writes eval/fixtures/law/reference/*.txt and PROVENANCE.md

Owner-approved one-time network use (2026-10-12). Only public law, never client material:
- United States Code, from govinfo.gov: works of the US Government are in the public domain (17 U.S.C. 105).
- UK Acts, from legislation.gov.uk, under the Open Government Licence v3.0 (attribution below).
The text is the operative wording only (headings and sections); editorial notes and commentary are left out. These
documents are REAL public law, not fictional, and are labelled so wherever they are shown.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import pathlib
import re
import sys
import urllib.request

from lxml import etree, html

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "eval" / "fixtures" / "law" / "reference"
AGENT = "EchoMind-Law-reference-fetch/1.0 (one-time, owner-approved)"
OGL = ("Contains public sector information licensed under the Open Government Licence v3.0 "
       "(https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).")
SOURCES = [
    ("Federal Arbitration Act (9 U.S.C. ch. 1)", "us",
     "https://www.govinfo.gov/content/pkg/USCODE-2023-title9/html/USCODE-2023-title9-chap1.htm"),
    ("Electronic Signatures in Global and National Commerce Act (15 U.S.C. ch. 96)", "us",
     "https://www.govinfo.gov/content/pkg/USCODE-2023-title15/html/USCODE-2023-title15-chap96.htm"),
    ("Late Payment of Commercial Debts (Interest) Act 1998", "uk", "https://www.legislation.gov.uk/ukpga/1998/20/data.xml"),
    ("Unfair Contract Terms Act 1977", "uk", "https://www.legislation.gov.uk/ukpga/1977/50/data.xml"),
    ("Contracts (Rights of Third Parties) Act 1999", "uk", "https://www.legislation.gov.uk/ukpga/1999/31/data.xml"),
]


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": AGENT}), timeout=60) as r:
        return r.read()


def us_code(raw: bytes) -> str:
    """Section headings and statutory text from a govinfo US Code chapter page; notes left out."""
    doc = html.fromstring(raw)
    lines = []
    for el in doc.iter("h1", "h2", "h3", "h4", "p"):
        cls = el.get("class") or ""
        text = " ".join(el.text_content().split())
        if not text:
            continue
        if el.tag in ("h1", "h2") and "head" in cls and "note" not in cls:
            lines.append(f"\n{text.upper()}\n")
        elif el.tag == "h3" and "section-head" in cls:
            lines.append(f"\n{text}\n")
        elif el.tag == "p" and cls.startswith("statutory-body"):
            lines.append(text)
    return "\n".join(lines).strip() + "\n"


def uk_act(raw: bytes) -> str:
    """Sections from legislation.gov.uk's CLML: titles, numbers and text; commentary and annotations left out."""
    root = etree.fromstring(raw)
    ns = {"l": "http://www.legislation.gov.uk/namespaces/legislation"}
    lines = []
    title = root.find(".//l:PrimaryPrelims/l:Title", ns)
    if title is not None:
        lines.append(" ".join("".join(title.itertext()).split()).upper() + "\n")
    skip = {f"{{{ns['l']}}}{t}" for t in ("Commentaries", "Commentary", "CommentaryRef", "Addition", "Repeal", "Substitution")}

    def words(el) -> str:
        parts = [el.text or ""]
        for c in el:
            if c.tag not in skip or c.tag.endswith(("Addition", "Substitution")):
                parts.append(words(c))
            parts.append(c.tail or "")
        return " ".join("".join(parts).split())

    body = root.find(".//l:Body", ns)
    if body is None:
        raise SystemExit("no body in the act")
    for el in body.iter():
        tag = etree.QName(el).localname
        if tag in ("Part", "Pblock") and el.find("l:Title", ns) is not None:
            lines.append(f"\n{words(el.find('l:Title', ns)).upper()}\n")
        elif tag in ("P1", "P2", "P3"):
            num = el.find("l:Pnumber", ns)
            n = words(num) if num is not None else ""
            para = el.find(f"l:{tag}para", ns)
            text = " ".join(words(t) for t in para.findall("l:Text", ns)) if para is not None else ""
            if tag == "P1":                                   # "1. Statutory interest." then its text, if any
                group = el.getparent()
                t = group.find("l:Title", ns) if etree.QName(group).localname == "P1group" else None
                lines.append(f"\n{n}. {words(t) if t is not None else ''}".rstrip())
                if text:
                    lines.append(text)
            elif text:
                lines.append(f"{'  ' if tag == 'P2' else '    '}({n}) {text}")
    out = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    rows = []
    for title, kind, url in SOURCES:
        raw = fetch(url)
        text = us_code(raw) if kind == "us" else uk_act(raw)
        notice = ("REAL PUBLIC LAW (not fictional). United States Code, from govinfo.gov; a work of the US Government in the "
                  "public domain (17 U.S.C. 105)." if kind == "us" else f"REAL PUBLIC LAW (not fictional). {OGL}")
        doc = f"{title}\n\n{notice}\nSource: {url} (fetched {today}; operative text only, notes and commentary left out)\n\n{text}"
        name = re.sub(r"[^A-Za-z0-9]+", " ", title).strip() + ".txt"
        (OUT / name).write_text(doc, encoding="utf-8")
        rows.append(f"| {title} | {url} | {'US public domain' if kind == 'us' else 'OGL v3.0'} | {today} | "
                    f"`{hashlib.sha256(doc.encode()).hexdigest()[:16]}` | {len(text.split())} words |")
        print(f"{name}: {len(text.split())} words")
    (OUT / "PROVENANCE.md").write_text(
        "# Reference library: real public law (D-064)\n\n"
        "Fetched once by `scripts/law/fetch_reference.py` with the owner's approval. **These are real public law, not "
        "fictional**, kept apart from the fictional fixtures and labelled as such in every document.\n\n"
        f"- United States Code: works of the US Government, public domain (17 U.S.C. 105).\n- UK Acts: {OGL}\n\n"
        "| Document | Source | Licence | Fetched | SHA-256 (first 16) | Size |\n|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
