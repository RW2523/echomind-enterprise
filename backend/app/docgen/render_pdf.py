"""
PDF renderer for Document Studio documents (v2).

Turns a normalized document dict (see :mod:`app.docgen.models`) into a polished,
professional, human-quality multi-page PDF using reportlab's platypus layout engine.
Everything is theme-driven via :mod:`app.docgen.theme`, so the same document renders in
the deep-slate "midnight" palette, the warm "counsel" serif palette, etc., staying cohesive
with the PPTX export and the on-screen preview.

Public API:
    render_pdf(doc: dict) -> bytes

Visual language:
  * A full-bleed COVER page (hero image band or cover-colour band + a title block with
    title / subtitle / org · doc_type · Generated date), with NO running header/footer.
  * A real TABLE OF CONTENTS with dot leaders and LIVE page numbers (reportlab
    TableOfContents + BaseDocTemplate.multiBuild). Falls back to a clean styled list if the
    live build proves fragile.
  * Running HEADER (muted doc title, left) + FOOTER (org left, "Page N" right) + a thin
    accent rule on every page except the cover (onPage canvas callbacks).
  * Professionally rendered blocks: numbered heading badges with a top rule, justified prose,
    accent-square / decimal bullets, primary-banded zebra tables, real "flow" step diagrams
    (rounded boxes + down-arrow connectors), accent-barred callouts, and embedded images
    (decoded from data URLs, with a tasteful placeholder when no image is available).

Robustness contract (mirrors models.normalize_document's "never crash" promise):
  * Every dynamic string is XML-escaped before it touches a reportlab Paragraph — reportlab
    parses a mini-XML markup, so a stray '<', '>' or '&' would otherwise raise during build.
  * Malformed / unexpected blocks are skipped individually rather than aborting the build.
  * Bad / oversized / missing images degrade to a placeholder; an image never crashes.
  * A non-dict doc is coerced to {}; an empty doc still produces a valid one-line PDF.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents
from xml.sax.saxutils import escape as _xml_escape

from . import models, theme as theme_mod
from ..utils.ids import now_iso

logger = logging.getLogger(__name__)

# ── Page geometry ───────────────────────────────────────────────────────────
_PAGE_W, _PAGE_H = letter
_LR_MARGIN = inch
_TOP_MARGIN = 0.95 * inch        # leaves room for the running header
_BOTTOM_MARGIN = 0.9 * inch      # leaves room for the running footer
_CONTENT_WIDTH = _PAGE_W - 2 * _LR_MARGIN
# Usable height of the body frame (one full page). A flowable taller than this cannot be
# placed by reportlab and aborts the build, so oversized diagrams must scale down to fit.
_BODY_FRAME_H = _PAGE_H - _TOP_MARGIN - _BOTTOM_MARGIN

# A bad image data_url can be large; cap the bytes we try to decode (~6MB).
_MAX_IMAGE_BYTES = 6_000_000


def _esc(v: Any) -> str:
    """XML-escape any value for safe inclusion in a reportlab Paragraph."""
    return _xml_escape(str(v if v is not None else ""))


def _c(hex_no_hash: str) -> colors.Color:
    """Theme hex (no leading #) -> reportlab Color. Never raises."""
    try:
        return colors.HexColor(theme_mod.hexc(hex_no_hash))
    except Exception:
        return colors.HexColor("#000000")


# ---------------------------------------------------------------------------
# Palette / styles derived from the document theme
# ---------------------------------------------------------------------------
class _Palette:
    """Resolved reportlab colours + fonts for one theme."""

    def __init__(self, pal: Dict[str, str]) -> None:
        g = pal.get
        self.primary = _c(g("primary", "0EA5B7"))
        self.primary_dark = _c(g("primary_dark", "0E7490"))
        self.accent = _c(g("accent", "22D3EE"))
        self.ink = _c(g("ink", "0F172A"))
        self.body = _c(g("body", "1E293B"))
        self.muted = _c(g("muted", "64748B"))
        self.bg = _c(g("bg", "FFFFFF"))
        self.panel = _c(g("panel", "F1F5F9"))
        self.panel_alt = _c(g("panel_alt", "F8FAFC"))
        self.cover_bg = _c(g("cover_bg", "0F172A"))
        self.cover_ink = _c(g("cover_ink", "FFFFFF"))
        self.cover_accent = _c(g("cover_accent", "22D3EE"))
        self.font = g("font", "Helvetica") or "Helvetica"
        self.font_bold = g("font_bold", "Helvetica-Bold") or "Helvetica-Bold"
        # Soft hairline for grids / rules.
        self.hairline = colors.HexColor("#e2e8f0")
        self.white = colors.white

    def callout(self, style: str) -> Tuple[colors.Color, colors.Color]:
        accent_hex, soft_hex = theme_mod.CALLOUT_COLORS.get(style, theme_mod.CALLOUT_COLORS["info"])
        return _c(accent_hex), _c(soft_hex)


def _styles(p: _Palette) -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    f, fb = p.font, p.font_bold
    return {
        # Cover
        "cover_title": ParagraphStyle(
            "CoverTitle", parent=base["Title"], fontName=fb, fontSize=34, leading=40,
            textColor=p.cover_ink, alignment=TA_LEFT, spaceAfter=12,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle", parent=base["Normal"], fontName=f, fontSize=15, leading=22,
            textColor=p.cover_ink, alignment=TA_LEFT, spaceAfter=8,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta", parent=base["Normal"], fontName=f, fontSize=10.5, leading=16,
            textColor=p.cover_ink, alignment=TA_LEFT,
        ),
        # TOC
        "toc_title": ParagraphStyle(
            "TocTitle", parent=base["Heading1"], fontName=fb, fontSize=22, leading=27,
            textColor=p.ink, spaceAfter=6,
        ),
        "toc1": ParagraphStyle(
            "Toc1", parent=base["Normal"], fontName=fb, fontSize=12, leading=24,
            textColor=p.body, leftIndent=0,
        ),
        "toc2": ParagraphStyle(
            "Toc2", parent=base["Normal"], fontName=f, fontSize=11, leading=21,
            textColor=p.muted, leftIndent=18,
        ),
        # Headings
        "h1": ParagraphStyle(
            "DH1", parent=base["Heading1"], fontName=fb, fontSize=21, leading=26,
            textColor=p.ink, spaceBefore=2, spaceAfter=0,
        ),
        "h2": ParagraphStyle(
            "DH2", parent=base["Heading2"], fontName=fb, fontSize=15.5, leading=20,
            textColor=p.primary_dark, spaceBefore=12, spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "DH3", parent=base["Heading3"], fontName=fb, fontSize=12.5, leading=17,
            textColor=p.body, spaceBefore=9, spaceAfter=4,
        ),
        # Body / lists
        "body": ParagraphStyle(
            "DBody", parent=base["Normal"], fontName=f, fontSize=10.5, leading=16,
            textColor=p.body, spaceAfter=7, alignment=TA_JUSTIFY,
        ),
        "bullet": ParagraphStyle(
            "DBullet", parent=base["Normal"], fontName=f, fontSize=10.5, leading=15.5,
            textColor=p.body,
        ),
        # Table
        "cell": ParagraphStyle(
            "Cell", parent=base["Normal"], fontName=f, fontSize=9.5, leading=13,
            textColor=p.body,
        ),
        "cell_header": ParagraphStyle(
            "CellHeader", parent=base["Normal"], fontName=fb, fontSize=9.5, leading=13,
            textColor=p.white,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["Italic"], fontName=f, fontSize=9, leading=12,
            textColor=p.muted, spaceAfter=4,
        ),
        "img_caption": ParagraphStyle(
            "ImgCaption", parent=base["Italic"], fontName=f, fontSize=9, leading=12,
            textColor=p.muted, spaceBefore=3, spaceAfter=6, alignment=TA_CENTER,
        ),
        "placeholder": ParagraphStyle(
            "Placeholder", parent=base["Normal"], fontName=f, fontSize=10, leading=14,
            textColor=p.muted, alignment=TA_CENTER,
        ),
        # Flow / callout
        "flow_step": ParagraphStyle(
            "FlowStep", parent=base["Normal"], fontName=f, fontSize=10.5, leading=14.5,
            textColor=p.body, alignment=TA_LEFT,
        ),
        "callout_title": ParagraphStyle(
            "CalloutTitle", parent=base["Normal"], fontName=fb, fontSize=10.5, leading=15,
            spaceAfter=3,
        ),
        "callout_text": ParagraphStyle(
            "CalloutText", parent=base["Normal"], fontName=f, fontSize=10, leading=15,
            textColor=p.body,
        ),
    }


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def _decode_data_url(data_url: Any) -> Optional[bytes]:
    """Decode a data:image/...;base64,<b64> string to raw bytes. Never raises."""
    if not isinstance(data_url, str) or not data_url.startswith("data:image"):
        return None
    if len(data_url) > _MAX_IMAGE_BYTES + 1024:
        return None
    try:
        b64 = data_url.split(",", 1)[1] if "," in data_url else ""
        if not b64:
            return None
        raw = base64.b64decode(b64, validate=False)
        return raw if raw and len(raw) <= _MAX_IMAGE_BYTES else None
    except Exception:
        return None


def _measure_image(raw: bytes) -> Optional[Tuple[int, int]]:
    """Validate + measure PNG/JPEG bytes with Pillow. Returns (w, h) or None."""
    try:
        from PIL import Image as PILImage  # Pillow 11 available per spec
        with PILImage.open(io.BytesIO(raw)) as im:
            im.verify()  # raises on a corrupt image
        with PILImage.open(io.BytesIO(raw)) as im2:
            iw, ih = im2.size
        return (iw, ih) if iw > 0 and ih > 0 else None
    except Exception:
        return None


def _image_flowable(raw: bytes, max_w: float, max_h: float) -> Optional[RLImage]:
    """Build a scaled, aspect-preserving reportlab Image from PNG/JPEG bytes."""
    size = _measure_image(raw)
    if size is None:
        return None
    iw, ih = size
    try:
        scale = min(max_w / iw, max_h / ih, 1.0)
        if scale <= 0:
            return None
        img = RLImage(io.BytesIO(raw), width=iw * scale, height=ih * scale)
        img.hAlign = "CENTER"
        return img
    except Exception:
        return None


class _ImagePlaceholder(Flowable):
    """A tasteful placeholder panel for image blocks with no usable data: a soft rounded
    panel in the theme panel colour, a small centred picture glyph, and the caption/alt
    text — so the layout still looks intentional."""

    def __init__(self, label: str, p: _Palette, st: Dict[str, ParagraphStyle],
                 width: float, height: float = 2.1 * inch) -> None:
        super().__init__()
        self.p = p
        self.caption = Paragraph(_esc(label) if label else "Illustration", st["placeholder"])
        self._w = width
        self._h = height

    def wrap(self, availWidth, availHeight):
        self._w = min(self._w, availWidth)
        return self._w, self._h

    def draw(self):
        c = self.canv
        p = self.p
        w, h = self._w, self._h
        c.setFillColor(p.panel)
        c.setStrokeColor(p.accent)
        c.setLineWidth(0.75)
        c.roundRect(0, 0, w, h, 10, stroke=1, fill=1)
        # centred picture glyph: framed rectangle + "sun" + "mountain"
        gx, gy, gs = w / 2, h / 2 + 16, 34
        c.setStrokeColor(p.muted)
        c.setLineWidth(1.5)
        c.roundRect(gx - gs / 2, gy - gs / 2, gs, gs, 3, stroke=1, fill=0)
        c.setFillColor(p.muted)
        c.circle(gx - gs / 4, gy + gs / 5, 3.2, stroke=0, fill=1)
        path = c.beginPath()
        path.moveTo(gx - gs / 2 + 3, gy - gs / 2 + 3)
        path.lineTo(gx - 3, gy + 1)
        path.lineTo(gx + gs / 5, gy - gs / 2 + 8)
        path.lineTo(gx + gs / 2 - 3, gy - gs / 2 + 3)
        c.setStrokeColor(p.muted)
        c.setLineWidth(1.5)
        c.drawPath(path, stroke=1, fill=0)
        # caption below the glyph
        cap_w = w - 36
        _, ch = self.caption.wrap(cap_w, h)
        self.caption.drawOn(c, 18, gy - gs / 2 - 10 - ch)


# ---------------------------------------------------------------------------
# Custom flowables: heading badge + flow diagram
# ---------------------------------------------------------------------------
class _HeadingBadge(Flowable):
    """Level-1 heading: a thin top accent rule, then a numbered square badge + a large
    accent-underlined title. Reads like a real numbered section header."""

    _BADGE = 24
    _GAP = 12          # badge -> title gap
    _RULE_PAD = 9      # gap below the top rule

    def __init__(self, number: int, text: str, p: _Palette, st: Dict[str, ParagraphStyle]) -> None:
        super().__init__()
        self.number = number
        self.p = p
        self.title = Paragraph(_esc(text), st["h1"])
        self._w = _CONTENT_WIDTH
        self._title_h = 0.0

    def wrap(self, availWidth, availHeight):
        self._w = availWidth
        title_w = max(20, availWidth - self._BADGE - self._GAP)
        _, self._title_h = self.title.wrap(title_w, availHeight)
        row_h = max(self._BADGE, self._title_h)
        # top rule (2) + pad + row + underline gap (5) + underline (2) + trailing (4)
        self._h = 2 + self._RULE_PAD + row_h + 5 + 2 + 4
        return availWidth, self._h

    def draw(self):
        c = self.canv
        p = self.p
        w = self._w
        # top accent rule
        c.setStrokeColor(p.accent)
        c.setLineWidth(2)
        top = self._h - 1
        c.line(0, top, w, top)
        # content row
        row_h = max(self._BADGE, self._title_h)
        row_top = top - self._RULE_PAD
        # numbered badge (square, primary fill)
        bs = self._BADGE
        by = row_top - row_h + (row_h - bs) / 2
        c.setFillColor(p.primary)
        c.roundRect(0, by, bs, bs, 4, stroke=0, fill=1)
        c.setFillColor(p.white)
        c.setFont(p.font_bold, 13)
        c.drawCentredString(bs / 2, by + bs / 2 - 4.6, str(self.number))
        # title to the right of the badge
        tx = bs + self._GAP
        ty = row_top - self._title_h
        self.title.drawOn(c, tx, ty)
        # accent underline under the title block
        uy = ty - 5
        c.setStrokeColor(p.accent)
        c.setLineWidth(2)
        c.line(tx, uy, min(w, tx + 0.32 * (w - tx) + 120), uy)


class _FlowDiagram(Flowable):
    """Ordered steps rendered as connected rounded STEP boxes with down-arrow connectors,
    so a 'flow' block looks like a real process diagram rather than a numbered list.

    Each step is a soft-filled rounded rectangle with a numbered accent chip + the step
    text; an accent down-arrow connects consecutive steps.
    """

    _GAP = 16          # vertical gap between boxes (room for the arrow)
    _PAD_X = 12
    _PAD_Y = 9
    _CHIP = 19         # numbered chip diameter

    def __init__(self, steps: List[str], p: _Palette, st: Dict[str, ParagraphStyle]) -> None:
        super().__init__()
        self.p = p
        self.st = st
        self._steps = steps
        self._paras: List[Paragraph] = []
        self._heights: List[float] = []
        self._w = _CONTENT_WIDTH

    def _layout(self, width: float):
        self._w = width
        text_w = max(20, width - 2 * self._PAD_X - self._CHIP - 8)
        self._paras = []
        self._heights = []
        for step in self._steps:
            para = Paragraph(_esc(step), self.st["flow_step"])
            _, ph = para.wrap(text_w, 10_000)
            self._paras.append(para)
            self._heights.append(max(self._CHIP, ph) + 2 * self._PAD_Y)

    def wrap(self, availWidth, availHeight):
        self._layout(availWidth)
        total = sum(self._heights) + self._GAP * max(0, len(self._heights) - 1) + 4
        self._nat_h = max(1.0, total)
        # A flowable taller than a full page can't be placed and aborts the build. If this
        # diagram is taller than a page, scale it down uniformly to fit; otherwise keep its
        # natural size and let reportlab paginate it normally.
        frame_h = _BODY_FRAME_H - 6
        if self._nat_h > frame_h:
            self._scale = frame_h / self._nat_h
            self._h = frame_h
        else:
            self._scale = 1.0
            self._h = self._nat_h
        return availWidth, self._h

    def draw(self):
        c = self.canv
        p = self.p
        w = self._w
        s = getattr(self, "_scale", 1.0)
        if s < 1.0:
            # Uniformly shrink the whole diagram (boxes + text) to fit one page, centered.
            c.saveState()
            c.translate(w * (1.0 - s) / 2.0, 0)
            c.scale(s, s)
        y = getattr(self, "_nat_h", self._h) - 2
        n = len(self._paras)
        for i, (para, box_h) in enumerate(zip(self._paras, self._heights)):
            box_top = y
            box_bottom = y - box_h
            # rounded step box
            c.setFillColor(p.panel)
            c.setStrokeColor(p.accent)
            c.setLineWidth(1)
            c.roundRect(0, box_bottom, w, box_h, 6, stroke=1, fill=1)
            # numbered chip
            chip = self._CHIP
            cx = self._PAD_X + chip / 2
            cy = box_top - self._PAD_Y - chip / 2
            c.setFillColor(p.primary)
            c.circle(cx, cy, chip / 2, stroke=0, fill=1)
            c.setFillColor(p.white)
            c.setFont(p.font_bold, 10)
            c.drawCentredString(cx, cy - 3.4, str(i + 1))
            # step text
            tx = self._PAD_X + chip + 8
            _, ph = para.wrap(w - tx - self._PAD_X, box_h - 2 * self._PAD_Y)
            para.drawOn(c, tx, box_top - self._PAD_Y - ph)
            # down-arrow connector
            if i < n - 1:
                ax = w / 2
                a_top = box_bottom
                a_bot = box_bottom - self._GAP + 3
                c.setStrokeColor(p.accent)
                c.setLineWidth(2)
                c.line(ax, a_top, ax, a_bot + 3)
                c.setFillColor(p.accent)
                path = c.beginPath()
                path.moveTo(ax - 4, a_bot + 4)
                path.lineTo(ax + 4, a_bot + 4)
                path.lineTo(ax, a_bot - 1)
                path.close()
                c.drawPath(path, stroke=0, fill=1)
            y = box_bottom - self._GAP
        if s < 1.0:
            c.restoreState()


# ---------------------------------------------------------------------------
# Block renderers. Each returns a list of flowables (possibly empty) and never raises.
# Heading flowables get a ``_toc`` attribute so afterFlowable can notify the live TOC.
# ---------------------------------------------------------------------------
def _render_heading(b, p, st, counters) -> List[Any]:
    text = b.get("text", "")
    if not _esc(text):
        return []
    level = int(b.get("level", 2) or 2)
    if level <= 1:
        counters["h1"] += 1
        badge = _HeadingBadge(counters["h1"], text, p, st)
        badge._toc = (0, text)
        badge.keepWithNext = True       # don't strand the heading at the page bottom
        gap = Spacer(1, 4)
        gap.keepWithNext = True
        return [Spacer(1, 8), badge, gap]
    style = st["h2"] if level == 2 else st["h3"]
    para = Paragraph(_esc(text), style)
    para.keepWithNext = True            # keep sub-heading with its following content
    if level == 2:
        para._toc = (1, text)
    return [para]


def _render_paragraph(b, p, st, counters) -> List[Any]:
    text = _esc(b.get("text", ""))
    return [Paragraph(text, st["body"])] if text else []


def _render_bullets(b, p, st, counters) -> List[Any]:
    items = [i for i in (b.get("items") or []) if str(i).strip()]
    if not items:
        return []
    ordered = bool(b.get("ordered"))
    flow_items = [
        ListItem(Paragraph(_esc(i), st["bullet"]), value=idx + 1 if ordered else None)
        for idx, i in enumerate(items)
    ]
    lf = ListFlowable(
        flow_items,
        bulletType="1" if ordered else "bullet",
        bulletColor=p.accent,
        bulletFontName=p.font_bold,
        bulletFontSize=10 if ordered else 11,
        bulletChar="•",                 # clear round bullet for unordered (readable, aligned)
        bulletOffsetY=-1 if not ordered else 0,
        bulletFormat="%s." if ordered else None,
        leftIndent=22,
        start="1" if ordered else None,
    )
    return [lf, Spacer(1, 8)]


def _render_table(b, p, st, counters) -> List[Any]:
    columns = [c for c in (b.get("columns") or [])]
    rows = b.get("rows") or []
    if not columns and not rows:
        return []
    ncols = len(columns) if columns else (max((len(r) for r in rows), default=0))
    if ncols <= 0:
        return []

    out: List[Any] = []
    caption = _esc(b.get("caption", ""))
    if caption:
        out.append(Paragraph(caption, st["caption"]))

    data: List[List[Any]] = []
    if columns:
        data.append([Paragraph(_esc(c), st["cell_header"]) for c in columns])
    for r in rows:
        cells = list(r) if isinstance(r, (list, tuple)) else [r]
        cells = (cells + [""] * ncols)[:ncols]
        data.append([Paragraph(_esc(c), st["cell"]) for c in cells])
    if not data:
        return out

    col_width = _CONTENT_WIDTH / ncols
    table = Table(data, colWidths=[col_width] * ncols, repeatRows=1 if columns else 0)

    style_cmds = [
        ("GRID", (0, 0), (-1, -1), 0.5, p.hairline),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    if columns:
        style_cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), p.primary),
            ("TEXTCOLOR", (0, 0), (-1, 0), p.white),
            ("LINEBELOW", (0, 0), (-1, 0), 1.2, p.accent),
        ]
        for i in range(1, len(data)):
            if i % 2 == 0:
                style_cmds.append(("BACKGROUND", (0, i), (-1, i), p.panel_alt))
    else:
        for i in range(len(data)):
            if i % 2 == 1:
                style_cmds.append(("BACKGROUND", (0, i), (-1, i), p.panel_alt))

    table.setStyle(TableStyle(style_cmds))
    out.append(table)
    out.append(Spacer(1, 12))
    return out


def _render_flow(b, p, st, counters) -> List[Any]:
    steps = []
    for s in (b.get("steps") or []):
        txt = models.flatten_step(s).strip()[:220]
        if txt:
            steps.append(txt)
    steps = steps[:14]
    if not steps:
        return []
    out: List[Any] = []
    title = _esc(b.get("title", ""))
    if title:
        out.append(Paragraph(title, st["h3"]))
    out.append(_FlowDiagram(steps, p, st))
    out.append(Spacer(1, 12))
    return out


def _render_callout(b, p, st, counters) -> List[Any]:
    style = str(b.get("style", "info")).lower()
    accent, soft_bg = p.callout(style)
    title = _esc(b.get("title", ""))
    text = _esc(b.get("text", ""))
    if not title and not text:
        return []

    inner: List[Any] = []
    if title:
        title_style = ParagraphStyle("CT", parent=st["callout_title"], textColor=accent)
        inner.append(Paragraph(title, title_style))
    if text:
        inner.append(Paragraph(text, st["callout_text"]))

    table = Table([[inner]], colWidths=[_CONTENT_WIDTH])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), soft_bg),
        ("LINEBEFORE", (0, 0), (0, -1), 4, accent),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return [table, Spacer(1, 12)]


def _render_image(b, p, st, counters) -> List[Any]:
    caption = _esc(b.get("caption") or "")
    alt = b.get("alt") or b.get("caption") or b.get("prompt") or "Illustration"
    raw = _decode_data_url(b.get("data_url"))

    block: List[Any]
    img = _image_flowable(raw, _CONTENT_WIDTH, 4.8 * inch) if raw is not None else None
    if img is not None:
        block = [img]
    else:
        block = [_ImagePlaceholder(str(alt), p, st, _CONTENT_WIDTH)]

    if caption:
        block.append(Paragraph(caption, st["img_caption"]))
    block.append(Spacer(1, 8))
    # Keep the image + caption together on one page where possible.
    return [KeepTogether(block)]


def _render_divider(b, p, st, counters) -> List[Any]:
    return [
        Spacer(1, 4),
        HRFlowable(width="100%", thickness=1, color=p.hairline),
        Spacer(1, 8),
    ]


_BLOCK_RENDERERS = {
    "heading": _render_heading,
    "paragraph": _render_paragraph,
    "bullets": _render_bullets,
    "table": _render_table,
    "flow": _render_flow,
    "callout": _render_callout,
    "image": _render_image,
    "divider": _render_divider,
}


# ---------------------------------------------------------------------------
# Cover page: background is painted on the canvas (onPage), the title block lives
# in a dedicated lower frame so it can never spill onto a second page.
# ---------------------------------------------------------------------------
_COVER_BAND_FRAC = 0.46                       # hero / colour band height (top of page)
_COVER_TITLE_TOP = _PAGE_H * (1 - _COVER_BAND_FRAC) - 0.35 * inch  # title field starts here


def _paint_cover_background(canvas, p: _Palette, hero_raw: Optional[bytes]) -> None:
    """Paint the full-bleed cover background directly on the page canvas: a hero image band
    (or a cover-colour band) across the top, a solid cover_bg field below, and an accent
    divider between them. Never raises — a bad hero degrades to a colour band."""
    canvas.saveState()
    canvas.setFillColor(p.cover_bg)
    canvas.rect(0, 0, _PAGE_W, _PAGE_H, stroke=0, fill=1)
    band_h = _PAGE_H * _COVER_BAND_FRAC
    band_y = _PAGE_H - band_h
    drew_hero = False
    if hero_raw is not None:
        try:
            from reportlab.lib.utils import ImageReader
            ir = ImageReader(io.BytesIO(hero_raw))
            iw, ih = ir.getSize()
            if iw > 0 and ih > 0:
                scale = max(_PAGE_W / iw, band_h / ih)  # cover-fit (crop overflow)
                dw, dh = iw * scale, ih * scale
                dx = (_PAGE_W - dw) / 2
                dy = band_y + (band_h - dh) / 2
                canvas.saveState()
                path = canvas.beginPath()
                path.rect(0, band_y, _PAGE_W, band_h)
                canvas.clipPath(path, stroke=0, fill=0)
                canvas.drawImage(ir, dx, dy, width=dw, height=dh, mask="auto")
                canvas.restoreState()
                drew_hero = True
        except Exception:
            drew_hero = False
    if not drew_hero:
        canvas.setFillColor(p.primary_dark)
        canvas.rect(0, band_y, _PAGE_W, band_h, stroke=0, fill=1)
    canvas.setStrokeColor(p.cover_accent)
    canvas.setLineWidth(3)
    canvas.line(0, band_y, _PAGE_W, band_y)
    canvas.restoreState()


def _build_cover_title(doc_dict: Dict[str, Any], p: _Palette, st: Dict[str, ParagraphStyle]) -> List[Any]:
    """The cover title block flowables, rendered on top of the painted cover background
    inside the lower cover frame (no header/footer, no page break here)."""
    block: List[Any] = [
        HRFlowable(width="22%", thickness=3, color=p.cover_accent, hAlign="LEFT", spaceAfter=14),
        Paragraph(_esc(doc_dict.get("title") or "Untitled Document"), st["cover_title"]),
    ]
    subtitle = _esc(doc_dict.get("subtitle", ""))
    if subtitle:
        block.append(Paragraph(subtitle, st["cover_subtitle"]))
    block.append(Spacer(1, 12))
    meta_bits = [b for b in (
        _esc(doc_dict.get("org", "")),
        _esc(doc_dict.get("doc_type", "")),
        f"Generated: {_esc(now_iso())}",
    ) if b]
    block.append(Paragraph("&nbsp;&nbsp;|&nbsp;&nbsp;".join(meta_bits), st["cover_meta"]))
    return block


# ---------------------------------------------------------------------------
# Table of contents
# ---------------------------------------------------------------------------
def _make_toc(p: _Palette) -> TableOfContents:
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOCL0", fontName=p.font_bold, fontSize=12, leading=24,
            textColor=p.body, leftIndent=0, firstLineIndent=0,
        ),
        ParagraphStyle(
            "TOCL1", fontName=p.font, fontSize=11, leading=21,
            textColor=p.muted, leftIndent=18, firstLineIndent=0,
        ),
    ]
    toc.dotsMinLevel = 0  # dot leaders at every level
    return toc


def _build_toc_fallback(doc_dict: Dict[str, Any], p: _Palette, st: Dict[str, ParagraphStyle]) -> List[Any]:
    """A clean styled TOC list (no live page numbers); used only if multiBuild fails."""
    try:
        entries = models.build_toc(doc_dict, 2)
    except Exception:
        entries = []
    if not entries:
        return []
    story: List[Any] = [
        Paragraph("Table of Contents", st["toc_title"]),
        HRFlowable(width="100%", thickness=1, color=p.hairline, spaceAfter=10),
    ]
    for e in entries:
        text = _esc(e.get("text", ""))
        if not text:
            continue
        style = st["toc1"] if int(e.get("level", 1)) <= 1 else st["toc2"]
        story.append(Paragraph(text, style))
    return story


# ---------------------------------------------------------------------------
# Document template: cover frame (no chrome) + body frame (header/footer/accent rule)
# ---------------------------------------------------------------------------
class _DocTemplate(BaseDocTemplate):
    def __init__(self, buf, doc_dict: Dict[str, Any], p: _Palette, **kw):
        super().__init__(buf, **kw)
        self._p = p
        self._title = str(doc_dict.get("title") or "Document")
        self._org = str(doc_dict.get("org") or "")
        self._hero = _decode_data_url(doc_dict.get("cover_image_data_url"))

        # Cover: a lower frame for the title block (background is painted on the canvas).
        cover_frame = Frame(
            _LR_MARGIN, _BOTTOM_MARGIN, _CONTENT_WIDTH,
            _COVER_TITLE_TOP - _BOTTOM_MARGIN, id="cover",
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        )
        body_frame = Frame(
            _LR_MARGIN, _BOTTOM_MARGIN, _CONTENT_WIDTH,
            _PAGE_H - _TOP_MARGIN - _BOTTOM_MARGIN, id="body",
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        )
        self.addPageTemplates([
            PageTemplate(id="Cover", frames=[cover_frame], onPage=self._on_cover),
            PageTemplate(id="Body", frames=[body_frame], onPage=self._on_body),
        ])

    def _on_cover(self, canvas, doc):
        # Full-bleed cover background; no running header/footer on the cover.
        try:
            _paint_cover_background(canvas, self._p, self._hero)
        except Exception:
            logger.debug("docgen pdf: cover background paint failed", exc_info=True)

    def _on_body(self, canvas, doc):
        p = self._p
        canvas.saveState()
        # Header: muted doc title (left) + a thin accent rule beneath it.
        canvas.setFont(p.font, 8.5)
        canvas.setFillColor(p.muted)
        canvas.drawString(_LR_MARGIN, _PAGE_H - _TOP_MARGIN + 22, self._title[:90])
        rule_y = _PAGE_H - _TOP_MARGIN + 14
        canvas.setStrokeColor(p.accent)
        canvas.setLineWidth(0.75)
        canvas.line(_LR_MARGIN, rule_y, _PAGE_W - _LR_MARGIN, rule_y)
        # Footer: a thin rule, then org (left) + "Page N" (right). Page numbering excludes
        # the cover (page 1), so the first body/TOC page reads "Page 1".
        foot_rule_y = _BOTTOM_MARGIN - 16
        canvas.setStrokeColor(p.hairline)
        canvas.setLineWidth(0.5)
        canvas.line(_LR_MARGIN, foot_rule_y, _PAGE_W - _LR_MARGIN, foot_rule_y)
        canvas.setFont(p.font, 8.5)
        canvas.setFillColor(p.muted)
        if self._org:
            canvas.drawString(_LR_MARGIN, _BOTTOM_MARGIN - 28, self._org[:80])
        canvas.drawRightString(_PAGE_W - _LR_MARGIN, _BOTTOM_MARGIN - 28, f"Page {max(1, doc.page - 1)}")
        canvas.restoreState()

    def afterFlowable(self, flowable):
        """Register heading entries with the live TableOfContents (real page numbers)."""
        toc = getattr(flowable, "_toc", None)
        if not toc:
            return
        try:
            level, text = toc
            self.notify("TOCEntry", (int(level), str(text), max(1, self.page - 1)))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Story assembly
# ---------------------------------------------------------------------------
def _build_body_story(doc_dict: Dict[str, Any], p: _Palette, st: Dict[str, ParagraphStyle]) -> List[Any]:
    counters = {"h1": 0}
    story: List[Any] = []
    for block in doc_dict.get("blocks", []) or []:
        try:
            if not isinstance(block, dict):
                continue
            btype = str(block.get("type", "")).lower()
            if btype == "pagebreak":
                story.append(PageBreak())
                continue
            renderer = _BLOCK_RENDERERS.get(btype)
            if renderer is None:
                continue
            story.extend(renderer(block, p, st, counters))
        except Exception:
            logger.debug("docgen pdf: skipped a block that failed to render", exc_info=True)
            continue
    if not story:
        story.append(Paragraph("(empty document)", st["body"]))
    return story


def _new_template(doc_dict: Dict[str, Any], p: _Palette) -> Tuple[io.BytesIO, _DocTemplate]:
    buf = io.BytesIO()
    pdf = _DocTemplate(
        buf, doc_dict, p,
        pagesize=letter,
        leftMargin=_LR_MARGIN, rightMargin=_LR_MARGIN,
        topMargin=_TOP_MARGIN, bottomMargin=_BOTTOM_MARGIN,
        title=str(doc_dict.get("title") or "Document"),
        author=str(doc_dict.get("org") or "EchoMind"),
    )
    return buf, pdf


def _flyer_content(doc: Dict[str, Any]) -> Tuple[str, List[str], str, str]:
    """Extract (tagline, bullets, cta_title, cta_text) from a flyer document's blocks."""
    blocks = [b for b in (doc.get("blocks") or []) if isinstance(b, dict)]
    tagline = str(doc.get("subtitle") or "").strip()
    bullets: List[str] = []
    cta_title = cta_text = ""
    paras: List[str] = []
    for b in blocks:
        t = str(b.get("type", "")).lower()
        if t == "bullets" and not bullets:
            bullets = [str(x).strip() for x in (b.get("items") or []) if str(x).strip()][:5]
        elif t == "callout" and not (cta_title or cta_text):
            cta_title = str(b.get("title") or "").strip()
            cta_text = str(b.get("text") or "").strip()
        elif t == "paragraph":
            tx = str(b.get("text") or "").strip()
            if tx:
                paras.append(tx)
    if not tagline and paras:
        tagline = paras[0]
    if not bullets and len(paras) > 1:
        bullets = paras[1:6]
    if not (cta_title or cta_text):
        cta_title = "Get in touch"
        cta_text = str(doc.get("org") or "")
    return tagline, bullets, cta_title, cta_text


def _render_flyer_pdf(doc: Dict[str, Any], p: _Palette) -> bytes:
    """A single-page promotional flyer/poster: full-bleed hero band, headline, tagline, a
    punchy benefit list, and a bottom call-to-action band. Never raises."""
    from reportlab.pdfgen import canvas as _canvas
    from reportlab.lib.utils import ImageReader

    W, H = letter
    margin = 0.85 * inch
    cw = W - 2 * margin
    band_h = H * 0.50          # hero occupies the top half
    band_y = H - band_h
    cta_h = 1.25 * inch        # bottom call-to-action band

    tagline, bullets, cta_title, cta_text = _flyer_content(doc)
    hero = _decode_data_url(doc.get("cover_image_data_url"))

    buf = io.BytesIO()
    c = _canvas.Canvas(buf, pagesize=letter)
    c.setTitle(str(doc.get("title") or "Flyer"))

    # Background field.
    c.setFillColor(p.cover_bg)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # Hero band (cover-fit image, else a primary_dark band).
    drew = False
    if hero is not None:
        try:
            ir = ImageReader(io.BytesIO(hero))
            iw, ih = ir.getSize()
            if iw > 0 and ih > 0:
                scale = max(W / iw, band_h / ih)
                dw, dh = iw * scale, ih * scale
                c.saveState()
                path = c.beginPath()
                path.rect(0, band_y, W, band_h)
                c.clipPath(path, stroke=0, fill=0)
                c.drawImage(ir, (W - dw) / 2, band_y + (band_h - dh) / 2, dw, dh, mask="auto")
                c.restoreState()
                drew = True
        except Exception:
            drew = False
    if not drew:
        c.setFillColor(p.primary_dark)
        c.rect(0, band_y, W, band_h, fill=1, stroke=0)

    # Legibility scrim across the lower hero, then an accent rule.
    try:
        c.saveState()
        c.setFillColor(p.cover_bg)
        c.setFillAlpha(0.60)
        c.rect(0, band_y, W, band_h * 0.46, fill=1, stroke=0)
        c.restoreState()
    except Exception:
        pass
    c.setStrokeColor(p.cover_accent)
    c.setLineWidth(4)
    c.line(0, band_y, W, band_y)

    # Headline over the lower hero (with an accent tab above it).
    title_style = ParagraphStyle("flyTitle", fontName=p.font_bold, fontSize=34, leading=38,
                                 textColor=p.cover_ink, alignment=TA_LEFT)
    pT = Paragraph(_esc(doc.get("title") or "Untitled"), title_style)
    _tw, th = pT.wrap(cw, band_h * 0.6)
    title_y = band_y + 0.42 * inch
    c.setFillColor(p.cover_accent)
    c.rect(margin, title_y + th + 12, 1.0 * inch, 5, fill=1, stroke=0)
    pT.drawOn(c, margin, title_y)

    # Body zone (between the hero and the CTA band): tagline + benefit bullets.
    y = band_y - 0.5 * inch
    if tagline:
        tag_style = ParagraphStyle("flyTag", fontName=p.font, fontSize=14.5, leading=20,
                                   textColor=p.cover_ink, alignment=TA_LEFT)
        pTag = Paragraph(_esc(tagline[:260]), tag_style)
        _w, htag = pTag.wrap(cw, 2 * inch)
        pTag.drawOn(c, margin, y - htag)
        y -= htag + 0.32 * inch

    bul_style = ParagraphStyle("flyBul", fontName=p.font, fontSize=13.5, leading=18,
                               textColor=p.cover_ink, alignment=TA_LEFT)
    for it in bullets:
        if y < cta_h + 0.85 * inch:
            break
        pB = Paragraph(_esc(it[:180]), bul_style)
        _w, hB = pB.wrap(cw - 0.30 * inch, 1.5 * inch)
        c.setFillColor(p.accent)
        c.rect(margin, y - 13, 8, 8, fill=1, stroke=0)
        pB.drawOn(c, margin + 0.30 * inch, y - hB)
        y -= max(hB, 0.24 * inch) + 0.14 * inch

    # Call-to-action band at the bottom.
    c.setFillColor(p.primary)
    c.rect(0, 0, W, cta_h, fill=1, stroke=0)
    c.setStrokeColor(p.cover_accent)
    c.setLineWidth(3)
    c.line(0, cta_h, W, cta_h)
    if cta_title:
        pC = Paragraph(_esc(cta_title[:120]),
                       ParagraphStyle("flyCT", fontName=p.font_bold, fontSize=17, leading=20,
                                      textColor=colors.white, alignment=TA_CENTER))
        _w, hC = pC.wrap(cw, cta_h)
        pC.drawOn(c, margin, cta_h - 0.40 * inch - hC + 0.18 * inch)
    if cta_text:
        pX = Paragraph(_esc(cta_text[:220]),
                       ParagraphStyle("flyCX", fontName=p.font, fontSize=12.5, leading=16,
                                      textColor=colors.white, alignment=TA_CENTER))
        _w, hX = pX.wrap(cw, cta_h)
        pX.drawOn(c, margin, 0.30 * inch)

    c.showPage()
    c.save()
    return buf.getvalue()


def render_pdf(doc: Dict[str, Any]) -> bytes:
    """Render a (normalized) document dict to professional PDF bytes. Never raises."""
    doc = doc if isinstance(doc, dict) else {}
    p = _Palette(theme_mod.get_theme(doc.get("theme")))
    st = _styles(p)

    # Single-page promotional flyer layout.
    if str(doc.get("layout", "")).lower() == "flyer":
        try:
            return _render_flyer_pdf(doc, p)
        except Exception as e:
            logger.warning("docgen pdf: flyer render failed (%s); falling back to document", e)

    try:
        has_toc = bool(models.build_toc(doc, 2))
    except Exception:
        has_toc = False

    def _assemble(toc_flowable: Optional[Any]) -> List[Any]:
        story: List[Any] = [NextPageTemplate("Body")]   # cover uses Cover; switch after it
        story.extend(_build_cover_title(doc, p, st))
        story.append(PageBreak())
        if toc_flowable is not None:
            story.append(Paragraph("Table of Contents", st["toc_title"]))
            story.append(HRFlowable(width="100%", thickness=1, color=p.hairline, spaceAfter=10))
            story.append(toc_flowable)
            story.append(PageBreak())
        story.extend(_build_body_story(doc, p, st))
        return story

    # 1) Preferred: real TOC with live page numbers via multiBuild.
    if has_toc:
        try:
            buf, pdf = _new_template(doc, p)
            pdf.multiBuild(_assemble(_make_toc(p)))
            return buf.getvalue()
        except Exception as e:
            logger.warning("docgen pdf: live TOC build failed (%s); using static TOC", e)

    # 2) Fallback: static styled TOC (or no TOC) via a single build.
    try:
        buf, pdf = _new_template(doc, p)
        story: List[Any] = [NextPageTemplate("Body")]
        story.extend(_build_cover_title(doc, p, st))
        story.append(PageBreak())
        fallback = _build_toc_fallback(doc, p, st) if has_toc else []
        if fallback:
            story.extend(fallback)
            story.append(PageBreak())
        story.extend(_build_body_story(doc, p, st))
        pdf.build(story)
        return buf.getvalue()
    except Exception as e:
        logger.error("docgen pdf: build failed (%s); emitting minimal PDF", e)
        return _minimal_pdf(doc, p, st)


def _minimal_pdf(doc: Dict[str, Any], p: _Palette, st: Dict[str, ParagraphStyle]) -> bytes:
    """Absolute last resort: a single-page PDF that cannot fail on layout."""
    buf = io.BytesIO()
    try:
        SimpleDocTemplate(buf, pagesize=letter).build([
            Paragraph(_esc(doc.get("title") or "Document"), st["h1"]),
            Spacer(1, 12),
            Paragraph("(document could not be fully rendered)", st["body"]),
        ])
    except Exception:
        buf = io.BytesIO()
        SimpleDocTemplate(buf, pagesize=letter).build([Spacer(1, 1)])
    return buf.getvalue()
