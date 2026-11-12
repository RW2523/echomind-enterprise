#!/usr/bin/env python3
"""Render the synthetic legal dataset (2026-11-12) from its text sources into the files a firm holds.

    python3 scripts/law/synthetic/render.py [--only SYN-01] [--list]

Reads eval/fixtures/synthetic/src/<matter>/NN-*.src.md (the format: eval/fixtures/synthetic/FORMAT.md) and writes:
- eval/fixtures/synthetic/<matter>/<file>: Word files with real list numbering, tracked changes and comments; typeset
  PDFs with running headers and footers; scanned PDFs (images only, so the product has to OCR them: skew, speckle,
  blur, uneven toner, stamps, handwriting and signatures); plain text; and files the product does not import (CSV,
  .eml, .xlsx);
- eval/fixtures/synthetic/text/<matter>/<file>.txt: the words a reader sees on each document (tracked changes
  accepted, comments, handwriting, stamps and signatures marked), which the answer keys quote;
- eval/fixtures/synthetic/manifest.json: the matters (src/<matter>/matter.json) and each document's title, kind,
  status, format, features and checksum.

Deterministic: every random choice is seeded by the file name. Every document is fictional, and every page of a
rendered Word or PDF file says so in its footer.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "eval" / "fixtures" / "synthetic"
SRC = BASE / "src"
MARK = "Fictional test document - EchoMind synthetic set"
FONTS = Path("/usr/share/fonts/truetype/liberation")
HAND_FONT = Path("/usr/share/fonts/opentype/urw-base35/Z003-MediumItalic.otf")
STAMP_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


# ── Sources ─────────────────────────────────────────────────────────────────────
@dataclass
class Doc:
    src: Path
    matter: str
    meta: dict
    body: str

    @property
    def render(self) -> str:
        return self.meta.get("render", "pdf")

    @property
    def file(self) -> str:
        return self.meta["file"]


def load(path: Path) -> Doc:
    raw = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---[ \t]*\n?", raw, re.S)
    if not m:
        raise SystemExit(f"{path}: no front matter")
    meta = {}
    for line in m.group(1).splitlines():
        line = re.sub(r"\s{2,}#.*$", "", line).rstrip()
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip().strip('"').strip("'")
    for need in ("file", "title", "kind", "status", "render"):
        if need not in meta:
            raise SystemExit(f"{path}: front matter has no {need!r}")
    return Doc(path, path.parent.name, meta, raw[m.end():])


@dataclass
class Block:
    kind: str                     # h1 h2 h3 para table page num_off num_on
    text: str = ""
    rows: list = field(default_factory=list)


def blocks(body: str) -> list[Block]:
    out, buf = [], []
    lines = body.replace("\r\n", "\n").split("\n")

    def flush():
        if buf:
            out.append(Block("para", "\n".join(buf)))
            buf.clear()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s == "<<<PAGE>>>":
            flush()
            out.append(Block("page"))
        elif s in ("<<<NUMBERING OFF>>>", "<<<NUMBERING ON>>>"):
            flush()
            out.append(Block("num_off" if s.endswith("OFF>>>") else "num_on"))
        elif s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            out.append(Block("table", rows=rows))
            continue
        elif re.match(r"^#{1,3} ", s):
            flush()
            n = len(s) - len(s.lstrip("#"))
            out.append(Block(f"h{n}", s[n:].strip()))
        elif not s:
            flush()
        else:
            buf.append(lines[i].rstrip())
        i += 1
    flush()
    return out


_TOK = re.compile(r"\{\+\+(.*?)\+\+\}|\{--(.*?)--\}|\{~~(.*?)~>(.*?)~~\}|\{==(.*?)==\}\s*\{>>(.*?)<<\}|\{>>(.*?)<<\}"
                  r"|\[\[(hand|stamp|sign):\s*(.*?)\]\]|\*\*(.+?)\*\*", re.S)


def segments(text: str) -> list[tuple]:
    """Inline markup: text, ins, del, comment (words, comment), hand, stamp, sign, bold."""
    out, pos = [], 0
    for m in _TOK.finditer(text):
        if m.start() > pos:
            out.append(("text", text[pos:m.start()]))
        g = m.groups()
        if g[0] is not None:
            out.append(("ins", g[0]))
        elif g[1] is not None:
            out.append(("del", g[1]))
        elif g[2] is not None:
            out += [("del", g[2]), ("ins", g[3])]
        elif g[4] is not None:
            out.append(("comment", g[4], g[5]))
        elif g[6] is not None:
            out.append(("comment", "", g[6]))
        elif g[7] is not None:
            out.append((g[7], g[8].strip()))
        else:
            out.append(("bold", g[9]))
        pos = m.end()
    if pos < len(text):
        out.append(("text", text[pos:]))
    return out


def accepted(text: str, mode: str) -> str:
    """The text with tracked changes accepted. mode 'reader' marks what is on the page; 'plain' writes it as text."""
    parts = []
    for s in segments(text):
        k = s[0]
        if k in ("text", "ins", "bold"):
            parts.append(s[1])
        elif k == "comment":
            parts.append(s[1] + (f" [Word comment: {s[2]}]" if mode == "reader" else ""))
        elif k in ("hand", "stamp"):
            parts.append(f"[{'handwritten' if k == 'hand' else 'stamp'}: {s[1]}]" if mode == "reader" else s[1])
        elif k == "sign":
            parts.append(f"[signature: {s[1]}]" if mode == "reader" else f"/s/ {s[1]}")
    return "".join(parts)


# ── Clause numbers (Word list numbering) ───────────────────────────────────────
_DEC0 = re.compile(r"^(\d{1,3})\.[ \t]+(?=\S)")
_DECN = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){1,3})\.?[ \t]+(?=\S)")
_PAREN = re.compile(r"^\(([a-z]{1,2}|[ivx]{1,5})\)[ \t]+(?=\S)")
_ROMAN = re.compile(r"^(?=[ivx])(x{0,3})(ix|iv|v?i{0,3})$")


def _roman(n: int) -> str:
    out = ""
    for v, s in ((10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
        while n >= v:
            out, n = out + s, n - v
    return out


class Numberer:
    """Which list level each numbered paragraph is, and whether Word's counters give the typed number."""

    def __init__(self, paras: list[str]):
        self.depth = max([len(m.group(1).split(".")) for p in paras if (m := _DECN.match(p))] + [1])
        self.counts: dict[int, int] = {}
        self.last_alpha: str | None = None
        self.warnings: list[str] = []

    def levels(self) -> list[tuple[str, str]]:
        lv = [("decimal", "%1."), ("decimal", "%1.%2"), ("decimal", "%1.%2.%3"), ("decimal", "%1.%2.%3.%4")][:self.depth]
        n = len(lv)
        return lv + [("lowerLetter", f"(%{n + 1})"), ("lowerRoman", f"(%{n + 2})")]

    def classify(self, text: str) -> tuple[int, str, str] | None:
        """(level, typed label, text after it), or None for an unnumbered paragraph."""
        if m := _DEC0.match(text):
            return 0, m.group(1) + ".", text[m.end():]
        if m := _DECN.match(text):
            return len(m.group(1).split(".")) - 1, m.group(1), text[m.end():]
        if m := _PAREN.match(text):
            v = m.group(1)
            alpha_next = chr(ord(self.last_alpha) + 1) if self.last_alpha and len(self.last_alpha) == 1 else None
            if _ROMAN.match(v) and v != alpha_next:
                return self.depth + 1, f"({v})", text[m.end():]
            return self.depth, f"({v})", text[m.end():]
        return None

    def take(self, level: int, label: str) -> bool:
        """Count the paragraph as Word would; False (and a warning) when Word would show another number."""
        n = self.counts.get(level, 0) + 1
        if level < self.depth:
            parts = [self.counts.get(k, 0) for k in range(level)] + [n]
            want = ".".join(map(str, parts)) + ("." if level == 0 else "")
        elif level == self.depth:
            want = f"({chr(ord('a') + n - 1) if n <= 26 else 'z' + str(n)})"
        else:
            want = f"({_roman(n)})"
        if want != label:
            self.warnings.append(f"typed {label}, Word would number it {want}: kept as typed text")
            return False
        self.counts[level] = n
        for deeper in [k for k in self.counts if k > level]:
            del self.counts[deeper]
        if level == self.depth:
            self.last_alpha = label.strip("()")
        elif level < self.depth:
            self.last_alpha = None
        return True


# ── Word ────────────────────────────────────────────────────────────────────────
def render_docx(doc: Doc, rnd: random.Random) -> tuple[bytes, dict]:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.opc.packuri import PackURI
    from docx.opc.part import Part
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt

    meta = doc.meta
    d = Document()
    normal = d.styles["Normal"]
    normal.font.name = {"serif": "Times New Roman", "sans": "Arial", "mono": "Courier New"}.get(meta.get("font", "serif"))
    normal.font.size = Pt(11)
    author = meta.get("author", "Author")
    when = meta.get("changes_date", "2025-01-01") + "T09:00:00Z"
    ids = iter(range(1, 100000))
    comments: list[tuple[int, str]] = []
    features = {"tracked_insertions": 0, "tracked_deletions": 0, "comments": 0, "auto_numbered": 0, "tables": 0}

    def w(tag, **attrs):
        el = OxmlElement(tag)
        for k, v in attrs.items():
            el.set(qn(k), str(v))
        return el

    def run(text, bold=False, italic=False, deleted=False):
        r = w("w:r")
        if bold or italic:
            rpr = w("w:rPr")
            if bold:
                rpr.append(w("w:b"))
            if italic:
                rpr.append(w("w:i"))
            r.append(rpr)
        for i, part in enumerate(text.split("\n")):
            if i:
                r.append(w("w:br"))
            t = w("w:delText" if deleted else "w:t")
            t.set(qn("xml:space"), "preserve")
            t.text = part
            r.append(t)
        return r

    def fill(p, text, bold=False):
        for s in segments(text):
            k = s[0]
            if k in ("text", "bold"):
                p._p.append(run(s[1], bold=bold or k == "bold"))
            elif k == "ins":
                el = w("w:ins", **{"w:id": next(ids), "w:author": author, "w:date": when})
                el.append(run(s[1], bold=bold))
                p._p.append(el)
                features["tracked_insertions"] += 1
            elif k == "del":
                el = w("w:del", **{"w:id": next(ids), "w:author": author, "w:date": when})
                el.append(run(s[1], bold=bold, deleted=True))
                p._p.append(el)
                features["tracked_deletions"] += 1
            elif k == "comment":
                cid = next(ids)
                p._p.append(w("w:commentRangeStart", **{"w:id": cid}))
                if s[1]:
                    p._p.append(run(s[1], bold=bold))
                p._p.append(w("w:commentRangeEnd", **{"w:id": cid}))
                ref = w("w:r")
                ref.append(w("w:commentReference", **{"w:id": cid}))
                p._p.append(ref)
                comments.append((cid, s[2]))
                features["comments"] += 1
            elif k == "hand":
                p._p.append(run(s[1], italic=True))
            elif k == "stamp":
                p._p.append(run(f"[{s[1]}]", bold=True))
            elif k == "sign":
                p._p.append(run(f"/s/ {s[1]}", italic=True))

    sec = d.sections[0]
    if meta.get("header"):
        sec.header.paragraphs[0].text = meta["header"]
    fp = sec.footer.paragraphs[0]
    fp.text = (meta["footer"] + "    " if meta.get("footer") else "") + "Page "
    fld = w("w:fldSimple", **{"w:instr": "PAGE"})
    fld.append(run("1"))
    fp._p.append(fld)
    mark = sec.footer.add_paragraph(MARK)
    mark.runs[0].font.size = Pt(7)

    if meta.get("letterhead"):
        parts = [x.strip() for x in meta["letterhead"].split("|")]
        p = d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(parts[0])
        r.bold, r.font.size = True, Pt(15)
        if parts[1:]:
            p2 = d.add_paragraph(" · ".join(parts[1:]))
            p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p2.runs[0].font.size = Pt(8.5)

    bl = blocks(doc.body)
    auto = meta.get("numbering") == "auto"
    nb = Numberer([b.text for b in bl if b.kind == "para"])
    num_id = None
    if auto:
        numbering = d.part.numbering_part.element
        absn = w("w:abstractNum", **{"w:abstractNumId": 90})
        absn.append(w("w:multiLevelType", **{"w:val": "multilevel"}))
        for ilvl, (fmt, text) in enumerate(nb.levels()):
            lvl = w("w:lvl", **{"w:ilvl": ilvl})
            lvl.append(w("w:start", **{"w:val": 1}))
            lvl.append(w("w:numFmt", **{"w:val": fmt}))
            lvl.append(w("w:lvlText", **{"w:val": text}))
            lvl.append(w("w:lvlJc", **{"w:val": "left"}))
            ppr = w("w:pPr")
            ppr.append(w("w:ind", **{"w:left": 567 * (ilvl + 1), "w:hanging": 567}))
            lvl.append(ppr)
            absn.append(lvl)
        first_num = numbering.find(qn("w:num"))
        if first_num is not None:
            first_num.addprevious(absn)
        else:
            numbering.append(absn)
        num = w("w:num", **{"w:numId": 90})
        num.append(w("w:abstractNumId", **{"w:val": 90}))
        numbering.append(num)
        num_id = 90

    numbering_on = True
    for b in bl:
        if b.kind == "page":
            d.add_page_break()
        elif b.kind in ("num_off", "num_on"):
            numbering_on = b.kind == "num_on"
        elif b.kind in ("h1", "h2", "h3"):
            p = d.add_paragraph(style={"h1": "Heading 1", "h2": "Heading 2", "h3": "Heading 3"}[b.kind])
            if b.kind == "h1":
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            fill(p, b.text)
        elif b.kind == "table":
            features["tables"] += 1
            cols = max(len(r) for r in b.rows) if b.rows else 1
            t = d.add_table(rows=len(b.rows), cols=cols)
            t.style = "Table Grid"
            for i, row in enumerate(b.rows):
                for j in range(cols):
                    cell = t.cell(i, j)
                    cell.text = ""
                    fill(cell.paragraphs[0], row[j] if j < len(row) else "", bold=i == 0)
        else:
            text = b.text
            c = nb.classify(text) if auto and numbering_on else None
            p = d.add_paragraph()
            if c and nb.take(c[0], c[1]):
                level, _, rest = c
                pPr = p._p.get_or_add_pPr()
                numPr = pPr.get_or_add_numPr()
                numPr.get_or_add_ilvl().val = level
                numPr.get_or_add_numId().val = num_id
                heading = level == 0 and len(rest) < 70 and rest.upper() == rest and re.search(r"[A-Z]", rest)
                fill(p, rest, bold=bool(heading))
                features["auto_numbered"] += 1
            else:
                fill(p, text)

    if comments:
        W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        from xml.sax.saxutils import escape
        initials = "".join(x[0] for x in author.split() if x[:1].isalpha())[:3] or "A"
        body = "".join(f'<w:comment w:id="{cid}" w:author="{escape(author)}" w:date="{when}" w:initials="{initials}">'
                       f'<w:p><w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p></w:comment>' for cid, text in comments)
        xml = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:comments xmlns:w="{W}">{body}</w:comments>'.encode()
        part = Part(PackURI("/word/comments.xml"),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml", xml, d.part.package)
        d.part.relate_to(part, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments")
    d.core_properties.author = author if meta.get("author") else "Calder & Wynne LLP"
    d.core_properties.title = meta["title"]
    d.core_properties.comments = MARK
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue(), {k: v for k, v in features.items() if v} | ({"numbering_warnings": nb.warnings} if nb.warnings else {})


# ── PDF (typeset) and scans ─────────────────────────────────────────────────────
_REGISTERED = False


def _fonts():
    global _REGISTERED
    if _REGISTERED:
        return
    from reportlab.lib.fonts import addMapping
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    for fam, base in (("Serif", "LiberationSerif"), ("Sans", "LiberationSans"), ("Mono", "LiberationMono")):
        for suffix, (bold, italic) in (("Regular", (0, 0)), ("Bold", (1, 0)), ("Italic", (0, 1)), ("BoldItalic", (1, 1))):
            name = f"L{fam}-{suffix}"
            pdfmetrics.registerFont(TTFont(name, str(FONTS / f"{base}-{suffix}.ttf")))
            addMapping(f"L{fam}", bold, italic, name)
    _REGISTERED = True


def _xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_pdf(doc: Doc, rnd: random.Random, paper: str, scan: bool) -> tuple[bytes, dict, list]:
    """A typeset PDF. With scan=True, every mark becomes a white token whose place is found afterwards, so the
    rasteriser can draw handwriting, stamps and signatures there the way they look on paper."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4, LETTER, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.platypus import (Flowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

    _fonts()
    meta = doc.meta
    style = meta.get("style", "contract")
    fam = {"serif": "LSerif", "sans": "LSans", "mono": "LMono"}[meta.get("font", "sans" if style in ("email", "table", "form") else "serif")]
    size = {"email": 9.5, "table": 9.5, "form": 10}.get(style, 10.5)
    page = A4 if paper == "a4" else LETTER
    bl = blocks(doc.body)

    def table_min(rows: list, fsize: float) -> float:
        plain = [[accepted(c, "plain") for c in r] for r in rows]
        cols = max(len(r) for r in plain)
        return sum(max([stringWidth(wd, f"{fam}-Bold", fsize) for r in plain if j < len(r) for wd in r[j].split()] + [12]) + 7
                   for j in range(cols))
    if any(b.kind == "table" and b.rows and table_min(b.rows, 7) > page[0] - 50 * mm for b in bl):
        page = landscape(page)                       # a wide sheet is printed landscape, as an office would
    base = ParagraphStyle("base", fontName=f"{fam}-Regular", fontSize=size, leading=size * 1.32, spaceAfter=size * 0.55,
                          alignment=TA_JUSTIFY if style in ("contract", "policy", "statement") else TA_LEFT)
    styles = {
        "h1": ParagraphStyle("h1", parent=base, fontSize=size + 3.5, leading=(size + 3.5) * 1.3,
                             alignment=TA_CENTER, spaceBefore=6, spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=base, fontSize=size + 1, leading=(size + 1) * 1.3, spaceBefore=8, spaceAfter=5, alignment=TA_LEFT),
        "h3": ParagraphStyle("h3", parent=base, spaceBefore=5, spaceAfter=3, alignment=TA_LEFT),
    }
    cell = ParagraphStyle("cell", parent=base, fontSize=size - 1.5, leading=(size - 1.5) * 1.25, spaceAfter=0, alignment=TA_LEFT)
    tokens: list[list] = []                                       # [token, kind, text, standalone] for scans
    features = {"tables": 0, "pages": 0}

    def inline(text: str) -> str:
        out = []
        for s in segments(text):
            k = s[0]
            if k in ("text", "ins"):
                out.append(_xml(s[1]))
            elif k == "bold":
                out.append(f"<b>{_xml(s[1])}</b>")
            elif k == "comment":
                out.append(_xml(s[1]))
            elif scan and k in ("hand", "stamp", "sign"):
                tok = f"QQMK{len(tokens):03d}QQ"
                tokens.append([tok, k, s[1], False])
                out.append(f'<font color="white" size="6">{tok}</font>')
            elif k == "hand":
                out.append(f'<font color="#1d3f8f"><i>{_xml(s[1])}</i></font>')
            elif k == "stamp":
                out.append(f'<font color="#b3261e"><b>[{_xml(s[1])}]</b></font>')
            elif k == "sign":
                out.append(f"<i>/s/ {_xml(s[1])}</i>")
        return "".join(out).replace("\n", "<br/>")

    class Scribble(Flowable):
        def __init__(self, seed: int):
            super().__init__()
            self.seed, self.width, self.height = seed, 150, 34

        def draw(self):
            r = random.Random(self.seed)
            c = self.canv
            c.setStrokeColor(colors.HexColor("#1d3f8f"))
            c.setLineWidth(1.1)
            p = c.beginPath()
            x, y = 4, 12 + r.uniform(-3, 3)
            p.moveTo(x, y)
            while x < 135:
                nx = x + r.uniform(10, 24)
                p.curveTo(x + r.uniform(2, 8), y + r.uniform(8, 20), nx - r.uniform(2, 8), y - r.uniform(6, 16), nx, 12 + r.uniform(-5, 5))
                x, y = nx, 12
            c.drawPath(p, stroke=1, fill=0)

    class Stamp(Flowable):
        def __init__(self, text: str, seed: int):
            super().__init__()
            self.text, self.seed, self.width, self.height = text, seed, 190, 40

        def draw(self):
            r = random.Random(self.seed)
            c = self.canv
            c.saveState()
            c.translate(20, 6)
            c.rotate(r.uniform(-6, 6))
            c.setStrokeColor(colors.HexColor("#b3261e"))
            c.setFillColor(colors.HexColor("#b3261e"))
            c.setLineWidth(1.6)
            c.rect(0, 0, 165, 28)
            c.setFont("LSans-Bold", 9.5)
            for i, line in enumerate(self.text.split(" / ")[:2]):
                c.drawCentredString(82, 17 - i * 11 if len(self.text.split(" / ")) > 1 else 10, line.upper())
            c.restoreState()

    story = []
    if meta.get("letterhead"):
        parts = [x.strip() for x in meta["letterhead"].split("|")]
        story.append(Paragraph(f"<b>{_xml(parts[0])}</b>", ParagraphStyle("lh", parent=base, fontName="LSerif-Regular", fontSize=17,
                                                                            leading=21, alignment=TA_CENTER, spaceAfter=2)))
        if parts[1:]:
            story.append(Paragraph(_xml("  ·  ".join(parts[1:])), ParagraphStyle("lh2", parent=base, fontName="LSans-Regular", fontSize=8,
                                                                                leading=10, alignment=TA_CENTER, spaceAfter=14)))
    nb = Numberer([b.text for b in bl if b.kind == "para"])
    for b in bl:
        if b.kind == "page":
            story.append(PageBreak())
        elif b.kind in ("num_off", "num_on"):
            continue
        elif b.kind in ("h1", "h2", "h3"):
            txt = inline(b.text)
            story.append(Paragraph(f"<b>{txt}</b>" if b.kind != "h3" else f"<b><i>{txt}</i></b>", styles[b.kind]))
        elif b.kind == "table":
            features["tables"] += 1
            cols = max(len(r) for r in b.rows) if b.rows else 1
            avail = page[0] - 50 * mm
            plain = [[accepted(r[j], "plain") if j < len(r) else "" for j in range(cols)] for r in b.rows]
            fsize = cell.fontSize
            while True:                                      # no word is ever split: shrink the type until each fits
                minw = [max([stringWidth(wd, f"{fam}-Bold", fsize) for r in plain for wd in r[j].split()] + [12]) + 7
                        for j in range(cols)]
                if sum(minw) <= avail or fsize <= 6:
                    break
                fsize -= 0.5
            cst = ParagraphStyle("cellfit", parent=cell, fontSize=fsize, leading=fsize * 1.25)
            lens = [max(8, min(60, max(len(r[j]) for r in plain))) for j in range(cols)]
            extra = max(0, avail - sum(minw))
            widths = [minw[j] + extra * lens[j] / sum(lens) for j in range(cols)]
            rows = [[Paragraph(("<b>%s</b>" if i == 0 else "%s") % inline(r[j] if j < len(r) else ""), cst) for j in range(cols)]
                    for i, r in enumerate(b.rows)]
            t = Table(rows, colWidths=widths, repeatRows=1)
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#777777")),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e6e6")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
            story += [t, Spacer(1, 8)]
        else:
            text = b.text
            only = re.fullmatch(r"\s*\[\[(sign|stamp|hand):\s*(.*?)\]\]\s*", text, re.S)
            if only and only.group(1) == "hand" and not scan:
                only = None
            if only and not scan:
                story.append(Scribble(rnd.randrange(10**9)) if only.group(1) == "sign" else Stamp(only.group(2), rnd.randrange(10**9)))
                continue
            c = nb.classify(text)
            st = base
            if c:
                level = min(c[0], 4)
                indent = 22 * (level + 1) if style in ("contract", "policy", "statement", "minutes") else 18 * (level + 1)
                st = ParagraphStyle(f"n{level}", parent=base, leftIndent=indent, firstLineIndent=-22 if level == 0 else -24)
                rest = c[2]
                heading = c[0] == 0 and len(rest) < 70 and rest.upper() == rest and re.search(r"[A-Z]", rest)
                body = inline(rest)
                text_xml = f"{_xml(c[1])}&nbsp;&nbsp;" + (f"<b>{body}</b>" if heading else body)
            else:
                text_xml = inline(text)
                if style == "email":
                    text_xml = re.sub(r"(^|<br/>)(From|Sent|To|Cc|CC|Subject|Date|Attachments):", r"\1<b>\2:</b>", text_xml)
                    if text.lstrip().startswith(">"):
                        st = ParagraphStyle("quote", parent=base, textColor=colors.HexColor("#444444"), leftIndent=8)
            story.append(Paragraph(text_xml, st))
            if only and scan:
                tokens[-1][3] = True
                if only.group(1) != "hand":
                    story.append(Spacer(1, 30))

    class Decorated(rl_canvas.Canvas):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._pages = []

        def showPage(self):
            self._pages.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._pages)
            for state in self._pages:
                self.__dict__.update(state)
                self.setFont("LSans-Regular", 7.5)
                self.setFillColor(colors.HexColor("#555555"))
                if meta.get("header"):
                    self.drawString(25 * mm, page[1] - 14 * mm, meta["header"])
                foot = (meta.get("footer", "") + "    " if meta.get("footer") else "") + f"Page {self._pageNumber} of {total}"
                self.drawString(25 * mm, 12 * mm, foot)
                self.setFont("LSans-Regular", 6)
                self.drawRightString(page[0] - 25 * mm, 12 * mm, MARK)
                super().showPage()
            super().save()

    buf = io.BytesIO()
    tmpl = SimpleDocTemplate(buf, pagesize=page, leftMargin=25 * mm, rightMargin=25 * mm, topMargin=22 * mm, bottomMargin=22 * mm,
                             title=meta["title"], author="", subject=MARK)
    tmpl.build(story, canvasmaker=Decorated)
    data = buf.getvalue()
    import fitz
    features["pages"] = fitz.open(stream=data, filetype="pdf").page_count
    return data, {k: v for k, v in features.items() if v}, tokens


def _scan_options(meta: dict) -> dict:
    opts = {"skew": 0.6, "noise": 0.3, "dpi": 200, "mode": "gray"}
    for part in (meta.get("scan") or "").split():
        k, _, v = part.partition("=")
        if k in opts:
            opts[k] = v if k == "mode" else float(v)
    return opts


def render_scan(doc: Doc, rnd: random.Random, paper: str) -> tuple[bytes, dict]:
    import fitz
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    pdf, features, tokens = render_pdf(doc, rnd, paper, scan=True)
    opts = _scan_options(doc.meta)
    dpi = int(opts["dpi"])
    scale = dpi / 72
    src = fitz.open(stream=pdf, filetype="pdf")
    hand_font = ImageFont.truetype(str(HAND_FONT), int(12.5 * scale))
    stamp_font = ImageFont.truetype(str(STAMP_FONT), int(10 * scale))
    pages = []
    for pno in range(src.page_count):
        pg = src[pno]
        pix = pg.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        draw = ImageDraw.Draw(img)
        for tok, kind, text, alone in tokens:
            for rect in pg.search_for(tok):
                x, y = rect.x0 * scale, rect.y0 * scale
                if kind == "hand":                                  # written just above the typed line
                    box = hand_font.getbbox(text)
                    layer = Image.new("L", (box[2] + 16, box[3] + 12), 0)
                    ImageDraw.Draw(layer).text((8, 4), text, font=hand_font, fill=255)
                    layer = layer.rotate(rnd.uniform(-3, 3), resample=Image.BICUBIC, expand=True)
                    ink = Image.new("L", layer.size, rnd.randint(40, 80))
                    top = y - 4 * scale if alone else y - layer.size[1] + 3 * scale   # on its own line, or above the typed one
                    img.paste(ink, (int(x), int(top)), layer)
                elif kind == "sign":
                    pts, cx = [], x
                    base_y = y + (14 if alone else 2) * scale
                    while cx < x + 140 * scale:
                        pts.append((cx, base_y + rnd.uniform(-9, 9) * scale))
                        cx += rnd.uniform(5, 13) * scale
                    draw.line(pts, fill=rnd.randint(30, 70), width=max(2, int(1.3 * scale)), joint="curve")
                elif kind == "stamp":
                    w, h = int(175 * scale), int(34 * scale)
                    layer = Image.new("L", (w, h), 0)
                    ld = ImageDraw.Draw(layer)
                    ld.rectangle([2, 2, w - 3, h - 3], outline=255, width=int(2 * scale))
                    lines = [s.upper() for s in text.split(" / ")][:2]
                    for i, line in enumerate(lines):
                        tw = ld.textlength(line, font=stamp_font)
                        ld.text(((w - tw) / 2, (h / 2 - 7 * scale) + (i - (len(lines) - 1) / 2) * 12 * scale), line,
                                font=stamp_font, fill=255)
                    holes = np.asarray(layer).copy()
                    mask = np.random.default_rng(rnd.randrange(10**9)).random(holes.shape) < 0.18
                    holes[mask] = 0
                    layer = Image.fromarray(holes).rotate(rnd.uniform(-8, 8), resample=Image.BICUBIC, expand=True)
                    ink = Image.new("L", layer.size, rnd.randint(95, 130))
                    img.paste(ink, (int(x), int(y)), layer)
        pages.append(_degrade(img, rnd, opts))
    out = io.BytesIO()
    first, rest = pages[0], pages[1:]
    if opts["mode"] == "bw":
        first.save(out, "PDF", resolution=dpi, save_all=True, append_images=rest)
    else:
        first.save(out, "PDF", resolution=dpi, save_all=True, append_images=rest, quality=50)
    features.update({"scanned": True, "scan": f"{opts['mode']} {dpi} dpi, skew up to {opts['skew']} degrees",
                     "handwriting": sum(t[1] == "hand" for t in tokens), "stamps": sum(t[1] == "stamp" for t in tokens),
                     "signatures": sum(t[1] == "sign" for t in tokens)})
    return out.getvalue(), {k: v for k, v in features.items() if v}


def _degrade(img, rnd: random.Random, opts: dict):
    import numpy as np
    from PIL import Image, ImageFilter
    g = np.random.default_rng(rnd.randrange(10**9))
    a = np.asarray(img).astype(np.float32)
    h, w = a.shape
    a = 255 - (255 - a) * rnd.uniform(0.82, 1.0)                                     # toner strength
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = rnd.uniform(0.3, 0.7) * w, rnd.uniform(0.3, 0.7) * h
    shade = ((xx - cx) ** 2 + (yy - cy) ** 2) / (w ** 2 + h ** 2)
    a = a * (0.96 - 0.10 * shade) + 4                                                 # paper and uneven light
    if rnd.random() < 0.35:                                                           # a copier's dark edge
        band = int(rnd.uniform(0.01, 0.03) * w)
        a[:, :band] *= np.linspace(0.35, 1, band)[None, :]
    a += g.normal(0, 2 + 6 * opts["noise"], a.shape)
    a = np.clip(a, 0, 255)
    im = Image.fromarray(a.astype(np.uint8)).filter(ImageFilter.GaussianBlur(rnd.uniform(0.5, 0.9)))
    im = im.rotate(rnd.uniform(-opts["skew"], opts["skew"]), resample=Image.BICUBIC, fillcolor=238,
                   translate=(rnd.randint(-12, 12), rnd.randint(-10, 10)))
    a = np.asarray(im).astype(np.float32)
    n = int(opts["noise"] * h * w / 2500)                                             # speckle
    ys, xs = g.integers(0, h - 3, n), g.integers(0, w - 3, n)
    for dy in range(2):
        for dx in range(2):
            a[ys + dy, xs + dx] = np.minimum(a[ys + dy, xs + dx], g.integers(20, 120, n))
    if opts["mode"] == "bw":
        return Image.fromarray(np.where(a < rnd.uniform(140, 165), 0, 255).astype(np.uint8)).convert("1")
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# ── Text, raw files, spreadsheets ───────────────────────────────────────────────
def reader_text(doc: Doc) -> str:
    out = []
    for b in blocks(doc.body):
        if b.kind == "page":
            out.append("--- page break ---")
        elif b.kind in ("num_off", "num_on"):
            continue
        elif b.kind == "table":
            out.append("\n".join("| " + " | ".join(accepted(c, "reader") for c in r) + " |" for r in b.rows))
        elif b.kind.startswith("h"):
            out.append(accepted(b.text, "reader"))
        else:
            out.append(accepted(b.text, "reader"))
    return "\n\n".join(out) + "\n"


def render_text(doc: Doc, markdown: bool) -> bytes:
    out = []
    for b in blocks(doc.body):
        if b.kind == "page":
            out.append("")
        elif b.kind in ("num_off", "num_on"):
            continue
        elif b.kind == "table":
            rows = [[accepted(c, "plain") for c in r] for r in b.rows]
            if markdown and rows:
                rows.insert(1, ["---"] * len(rows[0]))
            out.append("\n".join("| " + " | ".join(r) + " |" for r in rows))
        elif b.kind.startswith("h"):
            out.append(("#" * int(b.kind[1]) + " " if markdown else "") + accepted(b.text, "plain"))
        else:
            out.append(accepted(b.text, "plain"))
    return ("\n\n".join(out).strip() + "\n").encode()


def render_xlsx(doc: Doc) -> bytes:
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = (doc.meta.get("sheet") or "Sheet1")[:31]
    for b in blocks(doc.body):
        if b.kind == "table":
            for r in b.rows:
                ws.append([_cell(accepted(c, "plain")) for c in r])
            ws.append([])
        elif b.kind == "para":
            ws.append([accepted(b.text, "plain")])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _cell(v: str):
    t = v.replace(",", "")
    if re.fullmatch(r"-?\d+(\.\d+)?", t):
        return float(t) if "." in t else int(t)
    return v


# ── Main ────────────────────────────────────────────────────────────────────────
def render(doc: Doc, paper: str) -> tuple[bytes, dict]:
    rnd = random.Random(int(hashlib.sha256(doc.file.encode()).hexdigest()[:12], 16))
    r = doc.render
    ext = Path(doc.file).suffix.lower()
    want = {"docx": ".docx", "pdf": ".pdf", "scan": ".pdf", "txt": ".txt", "md": ".md", "xlsx": ".xlsx"}.get(r)
    if want and ext != want:
        raise SystemExit(f"{doc.src}: render {r} needs a {want} file, not {doc.file}")
    if r == "docx":
        return render_docx(doc, rnd)
    if r == "pdf":
        data, feats, _ = render_pdf(doc, rnd, paper, scan=False)
        return data, feats
    if r == "scan":
        return render_scan(doc, rnd, paper)
    if r in ("txt", "md"):
        return render_text(doc, r == "md"), {}
    if r == "xlsx":
        return render_xlsx(doc), {}
    if r == "raw":
        return doc.body.lstrip("\n").encode(), {}
    raise SystemExit(f"{doc.src}: unknown render {r!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma-separated matters (default: all)")
    ap.add_argument("--list", action="store_true", help="list the sources and stop")
    ap.add_argument("--base", default=str(BASE), help="another dataset folder (with src/) to render, for a test")
    args = ap.parse_args()
    base = Path(args.base)
    src = base / "src"
    wanted = {m for m in args.only.split(",") if m}
    manifest_path = base / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"cases": []}
    cases = {c["number"]: c for c in old.get("cases", [])}
    warnings = 0
    for mdir in sorted(p for p in src.iterdir() if p.is_dir()):
        if wanted and mdir.name not in wanted:
            continue
        info = json.loads((mdir / "matter.json").read_text())
        paper = info.get("paper", "a4")
        docs = [load(p) for p in sorted(mdir.glob("[0-9][0-9]-*.src.md"))]
        if args.list:
            for d in docs:
                print(f"{mdir.name}  {d.render:5} {d.file}")
            continue
        out_dir, text_dir = base / mdir.name, base / "text" / mdir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)
        by_src = {d.src.name: d for d in docs}
        entries = []
        for d in docs:
            data, feats = render(d, paper)
            (out_dir / d.file).write_bytes(data)
            (text_dir / (d.file + ".txt")).write_text(f"{d.meta['title']}\n(file: {d.file}; {d.render}; {d.meta['kind']}, "
                                                      f"{d.meta['status']})\n\n" + reader_text(d))
            for wmsg in feats.get("numbering_warnings", []):
                print(f"  WARN {mdir.name}/{d.src.name}: {wmsg}")
                warnings += 1
            entry = {"file": f"{mdir.name}/{d.file}", "title": d.meta["title"], "kind": d.meta["kind"], "status": d.meta["status"],
                     "render": d.render, "import": d.meta.get("import", "yes") != "no", "source": f"src/{mdir.name}/{d.src.name}",
                     "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "features": feats}
            if d.meta.get("version_of"):
                parent = by_src.get(d.meta["version_of"])
                if not parent:
                    raise SystemExit(f"{d.src}: version_of {d.meta['version_of']!r} is not a source of {mdir.name}")
                entry["version_of"] = f"{mdir.name}/{parent.file}"
            entries.append(entry)
            print(f"{mdir.name}  {d.render:5} {len(data):>9,} B  {d.file}")
        cases[mdir.name] = {**{k: v for k, v in info.items() if k != "paper"}, "number": mdir.name, "documents": entries}
    if args.list:
        return 0
    manifest = {"note": "FICTIONAL synthetic legal dataset (2026-11-12): every party, person, address, figure and event is "
                        "invented. Rendered by scripts/law/synthetic/render.py from src/; answer keys in answer_key_*.jsonl.",
                "cases": [cases[k] for k in sorted(cases)]}
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"manifest: {sum(len(c['documents']) for c in manifest['cases'])} documents in {len(manifest['cases'])} matters; "
          f"{warnings} numbering warnings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
