"""Legal document extraction (S2.1a, RAG-01; decisions D-010 and D-034; THREAT_MODEL T-06).

One canonical text per document version, with exact maps back to it:
- **pages:** each with where it starts and ends, and whether it came from the text layer or from
  OCR (with Tesseract's mean confidence);
- **lines:** numbered per page, counting only lines with text, as a reader counts them;
- **blocks:** paragraphs, headings, numbered items with Word's clause number as shown ("12.3") and
  its full path ("12.3(b)"), table rows, headers, footers and footnotes;
- **marks:** tracked insertions and deletions (author, date), comments beside the words they are
  about, low-confidence OCR words, sentences addressed to an AI, and text nobody can see.

Rules:
- **Readable only:** what a reader can see is what EchoMind reads.
  - White, tiny, invisible or off-page PDF text, and hidden or white Word runs, are left out of the
    canonical text and reported as `hidden_text` marks, with the text kept in the mark for review.
  - White text on a dark fill, and a searchable scan's invisible OCR layer over its image, are
    legitimate and kept.
- **Tracked changes:** read as accepted (insertions in, deletions out), with every change recorded.
- **Comments:** never part of the text.
- **Original order:** nothing is normalised in a way that would move offsets after the text is
  built. PDFs keep their visual lines, so page and line citations match the page.

OCR runs only on pages without a usable text layer, one page at a time, with the local `tesseract`
program (LEGAL_TESSERACT_CMD). Nothing leaves the host.
"""
from __future__ import annotations

import bisect
import io
import logging
import re
import subprocess
import zipfile
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Callable, Optional

from ...core.config import settings

logger = logging.getLogger(__name__)
Progress = Optional[Callable[[dict], None]]


@dataclass
class Page:
    number: int
    start: int
    end: int
    source: str                     # text | ocr | image (OCR off or over the page limit) | empty
    confidence: Optional[float] = None


@dataclass
class Block:
    kind: str                       # heading | paragraph | list_item | table_row | header | footer | footnote
    start: int
    end: int
    label: Optional[str] = None     # the number as shown, e.g. "12.3" or "(b)"
    path: Optional[str] = None      # the full clause number, e.g. "12.3(b)"
    level: Optional[int] = None
    page: Optional[int] = None


@dataclass
class Mark:
    kind: str                       # hidden_text | instructions | tracked_insert | tracked_delete | comment | ocr_low_confidence
    start: Optional[int]
    end: Optional[int]
    page: Optional[int] = None
    meta: dict = field(default_factory=dict)


@dataclass
class Extraction:
    filetype: str
    text: str
    pages: list[Page]
    blocks: list[Block]
    marks: list[Mark]
    engines: dict
    line_starts: list[int] = field(default_factory=list)

    def page_at(self, offset: int) -> Optional[int]:
        for p in self.pages:
            if p.start <= offset <= p.end:
                return p.number
        return None

    def line_at(self, offset: int) -> tuple[Optional[int], int]:
        """(page, line) for a character offset; lines with text only, counted per page."""
        page = self.page_at(offset)
        base = next((p.start for p in self.pages if p.number == page), 0) if page else 0
        first = bisect.bisect_left(self.line_starts, base)
        return page, max(1, bisect.bisect_right(self.line_starts, offset) - first)

    def page_offsets(self) -> list[tuple[int, int]]:
        return [(p.start, p.number) for p in self.pages]

    def as_json(self) -> dict:
        return {"pages": [asdict(p) for p in self.pages], "blocks": [asdict(b) for b in self.blocks],
                "lines": self.line_starts, "engines": self.engines}


class _Writer:
    """Builds the canonical text; blocks are separated by a blank line."""

    def __init__(self):
        self.parts: list[str] = []
        self.length = 0
        self.blocks: list[Block] = []
        self.marks: list[Mark] = []

    def add(self, text: str, kind: str, marks: list[Mark] = (), **attrs) -> Optional[Block]:
        if not text.strip():
            return None
        if self.length:
            self.parts.append("\n\n")
            self.length += 2
        start = self.length
        self.parts.append(text)
        self.length += len(text)
        block = Block(kind, start, self.length, **attrs)
        self.blocks.append(block)
        for m in marks:
            self.marks.append(Mark(m.kind, None if m.start is None else m.start + start,
                                   None if m.end is None else m.end + start, m.page, m.meta))
        return block

    def text(self) -> str:
        return "".join(self.parts)


def _line_starts(text: str) -> list[int]:
    out, pos = [], 0
    for line in text.split("\n"):
        if line.strip():
            out.append(pos + len(line) - len(line.lstrip()))
        pos += len(line) + 1
    return out


_LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", " ": " "})


def extract(filename: str, data: bytes, progress: Progress = None) -> Extraction:
    low = filename.lower()
    if low.endswith(".pdf"):
        x = _pdf(data, progress)
    elif low.endswith(".docx"):
        x = _docx(data)
    else:
        x = _plain(data.decode("utf-8-sig", errors="replace"), "md" if low.endswith(".md") else "txt")
    from ..research import prompt
    for a, b in prompt.addressed_to_ai_spans(x.text):          # T-06: flagged where it is, at import
        x.marks.append(Mark("instructions", a, b, x.page_at(a)))
    for m in x.marks:
        if m.kind == "hidden_text":
            m.meta["instructions"] = prompt.addressed_to_ai(m.meta.get("text", ""))
    x.line_starts = _line_starts(x.text)
    return x


# ── Plain text and Markdown ──────────────────────────────────────────────────────
def _plain(text: str, filetype: str) -> Extraction:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks, pos = [], 0
    for para in re.split(r"(\n\s*\n)", text):
        if para.strip() and not re.fullmatch(r"\n\s*\n", para):
            lead = len(para) - len(para.lstrip())
            kind = "heading" if filetype == "md" and para.lstrip().startswith("#") else "paragraph"
            blocks.append(Block(kind, pos + lead, pos + len(para.rstrip())))
        pos += len(para)
    return Extraction(filetype, text, [], blocks, [], {"text": "utf-8"})


# ── PDF ──────────────────────────────────────────────────────────────────────────
def _rgb(color: int) -> tuple[float, float, float]:
    return ((color >> 16) & 255) / 255, ((color >> 8) & 255) / 255, (color & 255) / 255


def _covered(bbox, rects, share: float = 0.5) -> bool:
    import fitz
    r = fitz.Rect(bbox)
    area = max(r.get_area(), 1e-6)
    return any((r & x).get_area() / area >= share for x in rects)


def _hidden_reason(span: dict, page_rect, dark_fills: list, images: list) -> Optional[str]:
    import fitz
    bbox = fitz.Rect(span["bbox"])
    if not bbox.intersects(page_rect) or (bbox & page_rect).get_area() < 0.5 * max(bbox.get_area(), 1e-6):
        return "off_page"
    if span.get("alpha", 255) == 0:
        return None if _covered(bbox, images) else "invisible"   # a searchable scan's OCR layer sits on its image
    if span.get("size", 12) < 2.0:
        return "tiny"
    if min(_rgb(span.get("color", 0))) >= 0.94 and not _covered(bbox, dark_fills + images, 0.6):
        return "white"
    return None


def _pdf(data: bytes, progress: Progress) -> Extraction:
    import fitz
    from ...rag.parse import detect_header_footer_patterns, strip_header_footer_lines
    doc = fitz.open(stream=data, filetype="pdf")
    pages_raw: list[dict] = []
    for i, page in enumerate(doc):
        dark = [d["rect"] for d in page.get_drawings() if d.get("fill") and sum(d["fill"][:3]) / 3 < 0.5]
        images = [r for img in page.get_images(full=True) for r in page.get_image_rects(img[0])]
        d = page.get_text("dict", clip=fitz.INFINITE_RECT(), flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_MEDIABOX_CLIP)
        blocks, hidden = [], []
        for b in d["blocks"]:
            if b.get("type") != 0:
                continue
            lines = []
            for line in b["lines"]:
                kept, gone = [], []
                for s in line["spans"]:
                    reason = _hidden_reason(s, page.rect, dark, images)
                    if reason:
                        if gone and gone[-1][0] == reason:
                            gone[-1][1] += s["text"]
                        else:
                            gone.append([reason, s["text"]])
                    else:
                        kept.append(s["text"])
                hidden += [(r, t.strip()) for r, t in gone if t.strip()]
                text = "".join(kept).translate(_LIGATURES).rstrip()
                if text.strip():
                    lines.append(text)
            if lines:
                blocks.append(lines)
        pages_raw.append({"page": page, "blocks": blocks, "hidden": hidden, "images": bool(images)})
    # Running headers and footers repeat on every page; they are left out as before (legacy behaviour).
    page_texts = ["\n".join("\n".join(b) for b in p["blocks"]) for p in pages_raw]
    heads, foots = detect_header_footer_patterns(page_texts)

    w = _Writer()
    pages: list[Page] = []
    ocr_engine = None
    ocr_done = 0
    to_ocr = [i for i, p in enumerate(pages_raw)
              if sum(len(l.strip()) for b in p["blocks"] for l in b) < settings.LEGAL_OCR_MIN_CHARS and p["images"]]
    for i, p in enumerate(pages_raw):
        number = i + 1
        page_start = w.length + (2 if w.length else 0)
        source, confidence = "text", None
        for reason, text in p["hidden"]:
            w.marks.append(Mark("hidden_text", None, None, number, {"reason": reason, "text": text}))
        if i in to_ocr:
            if not settings.LEGAL_OCR_ENABLED or ocr_done >= settings.LEGAL_OCR_MAX_PAGES:
                source = "image"
            else:
                if progress:
                    progress({"stage": "ocr", "page": ocr_done + 1, "pages": min(len(to_ocr), settings.LEGAL_OCR_MAX_PAGES)})
                result = _ocr_page(p["page"])
                ocr_done += 1
                ocr_engine = result["engine"]
                source, confidence = "ocr", result["confidence"]
                for para, low in result["paragraphs"]:
                    w.add(para, "paragraph", [Mark("ocr_low_confidence", a, b, number, {"confidence": c}) for a, b, c in low],
                          page=number)
        else:
            kept = strip_header_footer_lines(page_texts[i], heads, foots).split("\n")
            keep = {l.strip() for l in kept if l.strip()}
            for lines in p["blocks"]:
                text = "\n".join(l for l in lines if l.strip() in keep)
                w.add(text, "paragraph", page=number)
        end = w.length
        if end < page_start:                                  # nothing on this page
            page_start = end
            source = "empty" if source == "text" else source
        pages.append(Page(number, page_start, end, source, confidence))
    doc.close()
    engines = {"pdf": f"pymupdf {fitz.version[0]}"}
    if ocr_engine:
        engines["ocr"] = ocr_engine
    return Extraction("pdf", w.text(), pages, w.blocks, w.marks, engines)


@lru_cache(maxsize=4)
def _tessdata_args(cmd: str, configured: str, lang: str) -> tuple[str, ...]:
    """--tessdata-dir when the language data is not where this tesseract build looks by default
    (a conda build looks in its compile-time prefix). Configured, or <prefix>/share/tessdata."""
    import os
    import shutil
    if configured:
        return ("--tessdata-dir", configured)
    exe = shutil.which(cmd)
    if exe:
        guess = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(exe))), "share", "tessdata")
        if os.path.exists(os.path.join(guess, f"{lang}.traineddata")):
            return ("--tessdata-dir", guess)
    return ()


@lru_cache(maxsize=1)
def _tesseract_version() -> str:
    out = subprocess.run([settings.LEGAL_TESSERACT_CMD, "--version"], capture_output=True, text=True, timeout=30)
    return (out.stdout or out.stderr).splitlines()[0].strip()


def _ocr_page(page) -> dict:
    """OCR one page: paragraphs of text, each with its low-confidence words, and the mean confidence."""
    # A huge page (a poster, a forged media box) is rendered at a lower resolution, never as a giant image (review F4).
    area = max(1.0, float(page.rect.width) * float(page.rect.height)) / (72.0 * 72.0)      # square inches
    dpi = max(36, min(settings.LEGAL_OCR_DPI, int((settings.LEGAL_OCR_MAX_PIXELS / area) ** 0.5)))
    pix = page.get_pixmap(dpi=dpi, colorspace="gray")
    run = subprocess.run([settings.LEGAL_TESSERACT_CMD, "stdin", "stdout",
                          *_tessdata_args(settings.LEGAL_TESSERACT_CMD, settings.LEGAL_TESSDATA_DIR, settings.LEGAL_OCR_LANG),
                          "-l", settings.LEGAL_OCR_LANG, "--psm", "3", "tsv"],
                         input=pix.tobytes("png"), capture_output=True, timeout=settings.LEGAL_OCR_TIMEOUT_S)
    if run.returncode != 0:
        raise RuntimeError("ocr_failed")
    paragraphs: dict[tuple, dict[int, list]] = {}
    confs: list[float] = []
    for row in run.stdout.decode("utf-8", errors="replace").splitlines()[1:]:
        cols = row.split("\t")
        if len(cols) < 12 or cols[0] != "5" or not cols[11].strip():
            continue
        conf = float(cols[10])
        key = (int(cols[2]), int(cols[3]))                    # block, paragraph
        paragraphs.setdefault(key, {}).setdefault(int(cols[4]), []).append((cols[11], conf))
        if conf >= 0:
            confs.append(conf)
    out = []
    for key in sorted(paragraphs):
        text, low = "", []
        for n, line_no in enumerate(sorted(paragraphs[key])):
            if n:
                text += "\n"
            for k, (word, conf) in enumerate(paragraphs[key][line_no]):
                if k:
                    text += " "
                if 0 <= conf < settings.LEGAL_OCR_LOW_CONFIDENCE:
                    low.append((len(text), len(text) + len(word), round(conf, 1)))
                text += word
        out.append((text, low))
    return {"paragraphs": out, "confidence": round(sum(confs) / len(confs), 1) if confs else 0.0,
            "engine": _tesseract_version()}


# ── Word (.docx) ─────────────────────────────────────────────────────────────────
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_W = "{%s}" % W_NS


def _xml(data: bytes):
    from lxml import etree
    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False, remove_blank_text=False)
    return etree.fromstring(data, parser)


def _attr(el, name: str) -> Optional[str]:
    return None if el is None else el.get(_W + name)


def _roman(n: int) -> str:
    vals = ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
            (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))
    out = ""
    for v, s in vals:
        while n >= v:
            out, n = out + s, n - v
    return out


def _letters(n: int) -> str:
    # Word repeats the letter: a … z, aa … zz, aaa …
    return chr(ord("a") + (n - 1) % 26) * ((n - 1) // 26 + 1)


def _format(n: int, fmt: str) -> str:
    return {"decimal": str(n), "lowerLetter": _letters(n), "upperLetter": _letters(n).upper(), "lowerRoman": _roman(n).lower(),
            "upperRoman": _roman(n), "decimalZero": f"{n:02d}", "none": "", "bullet": ""}.get(fmt, str(n))


class _Numbering:
    """Word list numbering: counters per list, levels reset below the level just used."""

    def __init__(self, xml: Optional[bytes]):
        self.abstract: dict[str, dict[int, dict]] = {}
        self.nums: dict[str, tuple[str, dict[int, int]]] = {}
        self.counts: dict[str, dict[int, int]] = {}
        self.paths: dict[str, dict[int, str]] = {}
        if not xml:
            return
        root = _xml(xml)
        for a in root.findall(_W + "abstractNum"):
            levels = {}
            for lvl in a.findall(_W + "lvl"):
                levels[int(_attr(lvl, "ilvl") or 0)] = {
                    "start": int(_attr(lvl.find(_W + "start"), "val") or 1),
                    "fmt": _attr(lvl.find(_W + "numFmt"), "val") or "decimal",
                    "text": _attr(lvl.find(_W + "lvlText"), "val") or "",
                    "legal": lvl.find(_W + "isLgl") is not None}
            self.abstract[_attr(a, "abstractNumId")] = levels
        for n in root.findall(_W + "num"):
            overrides = {}
            for o in n.findall(_W + "lvlOverride"):
                so = o.find(_W + "startOverride")
                if so is not None:
                    overrides[int(_attr(o, "ilvl") or 0)] = int(_attr(so, "val") or 1)
            self.nums[_attr(n, "numId")] = (_attr(n.find(_W + "abstractNumId"), "val"), overrides)

    def label(self, num_id: str, ilvl: int) -> tuple[Optional[str], Optional[str]]:
        if num_id not in self.nums:
            return None, None
        abstract_id, overrides = self.nums[num_id]
        levels = self.abstract.get(abstract_id, {})
        lvl = levels.get(ilvl)
        if lvl is None or lvl["fmt"] == "bullet":
            return None, None
        counts = self.counts.setdefault(num_id, {})
        start = overrides.get(ilvl, lvl["start"])
        counts[ilvl] = counts.get(ilvl, start - 1) + 1
        for deeper in [k for k in counts if k > ilvl]:
            del counts[deeper]
        text = lvl["text"]
        for k in range(ilvl + 1):
            fmt = "decimal" if lvl["legal"] else levels.get(k, {}).get("fmt", "decimal")
            value = counts.get(k, levels.get(k, {}).get("start", 1))
            text = text.replace(f"%{k + 1}", _format(value, fmt))
        label = text.strip()
        paths = self.paths.setdefault(num_id, {})
        for deeper in [k for k in paths if k >= ilvl]:
            del paths[deeper]
        if "%1" in lvl["text"] or ilvl == 0 or (ilvl - 1) not in paths:
            path = label.rstrip(".").strip()
        else:
            path = paths[ilvl - 1] + label.rstrip(".").strip()
        paths[ilvl] = path
        return label, path


class _Styles:
    def __init__(self, xml: Optional[bytes]):
        self.styles: dict[str, dict] = {}
        if not xml:
            return
        for s in _xml(xml).findall(_W + "style"):
            ppr = s.find(_W + "pPr")
            numpr = ppr.find(_W + "numPr") if ppr is not None else None
            self.styles[_attr(s, "styleId")] = {
                "based": _attr(s.find(_W + "basedOn"), "val"),
                "num": (_attr(numpr.find(_W + "numId"), "val") if numpr is not None and numpr.find(_W + "numId") is not None else None,
                        int(_attr(numpr.find(_W + "ilvl"), "val") or 0) if numpr is not None and numpr.find(_W + "ilvl") is not None else None),
                "outline": ppr is not None and ppr.find(_W + "outlineLvl") is not None}

    def numbering(self, style_id: Optional[str]) -> tuple[Optional[str], Optional[int]]:
        seen = set()
        while style_id and style_id in self.styles and style_id not in seen:
            seen.add(style_id)
            num_id, ilvl = self.styles[style_id]["num"]
            if num_id:
                return num_id, ilvl
            style_id = self.styles[style_id]["based"]
        return None, None


class _Paragraph:
    """The visible text of one paragraph and the marks inside it (offsets relative to the paragraph)."""

    def __init__(self):
        self.text = ""
        self.marks: list[Mark] = []
        self.hidden: list[tuple[str, str]] = []
        self.comment_starts: dict[str, int] = {}
        self.comment_ends: dict[str, int] = {}
        self.comment_refs: dict[str, int] = {}
        self.footnotes: list[str] = []
        self._field = 0                                         # inside a field's instruction part


def _run_props(r) -> tuple[Optional[str], bool]:
    rpr = r.find(_W + "rPr")
    if rpr is None:
        return None, False
    if rpr.find(_W + "vanish") is not None and _attr(rpr.find(_W + "vanish"), "val") not in ("0", "false"):
        return "hidden_run", False
    size = _attr(rpr.find(_W + "sz"), "val")
    if size and int(size) < 4:                                  # half-points: under 2 pt
        return "tiny", False
    color = (_attr(rpr.find(_W + "color"), "val") or "").upper()
    shaded = rpr.find(_W + "shd") is not None or rpr.find(_W + "highlight") is not None
    if color in ("FFFFFF", "FEFEFE") and not shaded:
        return "white", False
    return None, False


def _collect(el, p: _Paragraph, deleted: Optional[list] = None) -> None:
    for child in el:
        tag = child.tag
        if tag == _W + "r":
            _run(child, p, deleted)
        elif tag in (_W + "ins", _W + "moveTo"):
            start = len(p.text)
            _collect(child, p)
            if len(p.text) > start:
                p.marks.append(Mark("tracked_insert", start, len(p.text), meta={"author": _attr(child, "author"),
                                                                                 "date": _attr(child, "date")}))
        elif tag in (_W + "del", _W + "moveFrom"):
            gone: list[str] = []
            _collect(child, p, gone)
            if "".join(gone):
                p.marks.append(Mark("tracked_delete", len(p.text), len(p.text),
                                    meta={"author": _attr(child, "author"), "date": _attr(child, "date"), "text": "".join(gone)}))
        elif tag == _W + "commentRangeStart":
            p.comment_starts[_attr(child, "id")] = len(p.text)
        elif tag == _W + "commentRangeEnd":
            p.comment_ends[_attr(child, "id")] = len(p.text)
        elif tag in (_W + "hyperlink", _W + "smartTag", _W + "fldSimple", _W + "customXml", _W + "sdtContent", _W + "sdt"):
            _collect(child, p, deleted)


def _run(r, p: _Paragraph, deleted: Optional[list]) -> None:
    hidden, _ = _run_props(r)
    buf = ""
    for c in r:
        tag = c.tag
        if tag == _W + "fldChar":
            kind = _attr(c, "fldCharType")
            p._field = 1 if kind == "begin" else 0 if kind in ("separate", "end") else p._field
        elif p._field or tag == _W + "instrText":
            continue
        elif tag == _W + "t":
            buf += c.text or ""
        elif tag == _W + "delText" and deleted is not None:
            deleted.append(c.text or "")
        elif tag == _W + "tab":
            buf += "\t"
        elif tag in (_W + "br", _W + "cr"):
            buf += "\n"
        elif tag == _W + "noBreakHyphen":
            buf += "-"
        elif tag == _W + "footnoteReference":
            p.footnotes.append(_attr(c, "id"))
            buf += f"[fn {_attr(c, 'id')}]"
        elif tag == _W + "endnoteReference":
            buf += f"[en {_attr(c, 'id')}]"
        elif tag == _W + "commentReference":
            p.comment_refs[_attr(c, "id")] = len(p.text) + len(buf)
    if deleted is not None:
        return
    if hidden:
        if buf.strip():
            p.hidden.append((hidden, buf))
        return
    p.text += buf


def _notes(z: zipfile.ZipFile, name: str, tag: str) -> dict[str, str]:
    if name not in z.namelist():
        return {}
    out = {}
    for n in _xml(z.read(name)).findall(_W + tag):
        if _attr(n, "type") in ("separator", "continuationSeparator", "continuationNotice"):
            continue
        out[_attr(n, "id")] = " ".join(_plain_paragraphs(n))
    return out


def _plain_paragraphs(el) -> list[str]:
    out = []
    for para in el.iter(_W + "p"):
        p = _Paragraph()
        _collect(para, p)
        if p.text.strip():
            out.append(p.text.strip())
    return out


def _docx(data: bytes) -> Extraction:
    z = zipfile.ZipFile(io.BytesIO(data))
    names = set(z.namelist())
    rels: dict[str, str] = {}
    if "word/_rels/document.xml.rels" in names:
        for rel in _xml(z.read("word/_rels/document.xml.rels")):
            rels[rel.get("Id")] = rel.get("Target", "").lstrip("/").removeprefix("word/")
    numbering = _Numbering(z.read("word/numbering.xml") if "word/numbering.xml" in names else None)
    styles = _Styles(z.read("word/styles.xml") if "word/styles.xml" in names else None)
    comments: dict[str, dict] = {}
    if "word/comments.xml" in names:
        for c in _xml(z.read("word/comments.xml")).findall(_W + "comment"):
            comments[_attr(c, "id")] = {"author": _attr(c, "author"), "date": _attr(c, "date"),
                                        "text": " ".join(_plain_paragraphs(c))}
    footnotes = _notes(z, "word/footnotes.xml", "footnote")
    endnotes = _notes(z, "word/endnotes.xml", "endnote")
    body = _xml(z.read("word/document.xml")).find(_W + "body")

    w = _Writer()
    open_comments: dict[str, int] = {}
    closed_comments: dict[str, tuple[int, int]] = {}
    note_order: list[str] = []
    section_parts: list[tuple[str, str]] = []

    def paragraph(el) -> tuple[_Paragraph, Optional[str], Optional[str], Optional[int], str]:
        ppr = el.find(_W + "pPr")
        style = _attr(ppr.find(_W + "pStyle"), "val") if ppr is not None and ppr.find(_W + "pStyle") is not None else None
        num_id, ilvl = styles.numbering(style)
        numpr = ppr.find(_W + "numPr") if ppr is not None else None
        if numpr is not None:
            if numpr.find(_W + "numId") is not None:
                num_id = _attr(numpr.find(_W + "numId"), "val")
            if numpr.find(_W + "ilvl") is not None:
                ilvl = int(_attr(numpr.find(_W + "ilvl"), "val") or 0)
        if ppr is not None:
            for s in ppr.findall(_W + "sectPr"):
                section_parts.extend(_section_refs(s))
        p = _Paragraph()
        _collect(el, p)
        label = path = None
        if num_id and num_id != "0" and p.text.strip():
            label, path = numbering.label(num_id, ilvl or 0)
        heading = bool(style and (style.lower().startswith("heading") or style.lower() == "title")) or \
            bool(style and styles.styles.get(style, {}).get("outline"))
        kind = "heading" if heading else "list_item" if label else "paragraph"
        return p, label, path, ilvl if label else None, kind

    def emit(p: _Paragraph, prefix: str, kind: str, **attrs) -> None:
        shift = len(prefix)
        marks = [Mark(m.kind, m.start + shift, m.end + shift, None, m.meta) for m in p.marks]
        block = w.add(prefix + p.text, kind, marks, **attrs)
        base = (block.start if block else w.length) + shift
        for cid, at in p.comment_starts.items():
            open_comments[cid] = base + at
        for cid, at in p.comment_ends.items():
            closed_comments[cid] = (open_comments.get(cid, base + at), base + at)
        for cid, at in p.comment_refs.items():                 # a comment without a range sits at its reference
            closed_comments.setdefault(cid, (open_comments.get(cid, base + at), base + at))
        for reason, text in p.hidden:
            w.marks.append(Mark("hidden_text", None, None, None, {"reason": reason, "text": text.strip()}))
        note_order.extend(n for n in p.footnotes if n not in note_order)

    def walk(container) -> None:
        for el in container:
            if el.tag == _W + "p":
                p, label, path, level, kind = paragraph(el)
                emit(p, f"{label} " if label else "", kind, label=label, path=path, level=level)
            elif el.tag == _W + "tbl":
                for tr in el.iter(_W + "tr"):
                    cells, marks, shift = [], [], 0
                    for tc in tr.findall(_W + "tc"):
                        parts = []
                        for para in tc.findall(_W + "p"):
                            p = _Paragraph()
                            _collect(para, p)
                            if p.text.strip():
                                parts.append(p)
                        cell_text = ""
                        for k, p in enumerate(parts):
                            if k:
                                cell_text += " / "
                            for m in p.marks:
                                marks.append(Mark(m.kind, m.start + shift + len(cell_text), m.end + shift + len(cell_text), None, m.meta))
                            cell_text += p.text
                        cells.append(cell_text)
                        shift += len(cell_text) + 3
                    w.add(" | ".join(cells), "table_row", marks)
            elif el.tag in (_W + "sdt",):
                content = el.find(_W + "sdtContent")
                if content is not None:
                    walk(content)
            elif el.tag == _W + "sectPr":
                section_parts.extend(_section_refs(el))

    walk(body)
    for n in note_order:
        if n in footnotes:
            w.add(f"[fn {n}] {footnotes[n]}", "footnote", label=f"fn {n}")
    for n, text in endnotes.items():
        w.add(f"[en {n}] {text}", "footnote", label=f"en {n}")
    seen = set()
    for kind, rid in section_parts:
        target = rels.get(rid)
        if not target or target in seen or f"word/{target}" not in names:
            continue
        seen.add(target)
        text = "\n".join(_plain_paragraphs(_xml(z.read(f"word/{target}"))))
        w.add(text, kind)
    for cid, (a, b) in closed_comments.items():
        if cid in comments:
            w.marks.append(Mark("comment", a, b, None, comments[cid]))
    return Extraction("docx", w.text(), [], w.blocks, w.marks, {"docx": "ooxml"})


def _section_refs(sect) -> list[tuple[str, str]]:
    out = []
    for tag, kind in (("headerReference", "header"), ("footerReference", "footer")):
        for ref in sect.findall(_W + tag):
            out.append((kind, ref.get("{%s}id" % R_NS)))
    return out
