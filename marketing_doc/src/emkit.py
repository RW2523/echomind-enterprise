"""Shared design kit for EchoMind document generation (reportlab)."""
import math
from reportlab.pdfgen import canvas as _canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.colors import HexColor, Color
from reportlab.lib.utils import simpleSplit

W, H = A4
LW, LH = landscape(A4)

INK   = HexColor('#0B1F2A')
ACC   = HexColor('#0E8F9E')
ACC2  = HexColor('#17607A')
TEAL  = HexColor('#0E8F9E')
BLUE  = HexColor('#1D6FB8')
PURP  = HexColor('#4F52C4')
AMBER = HexColor('#A2620B')
GREEN = HexColor('#12805F')
ROSE  = HexColor('#B03A5B')
SLATE = HexColor('#3B4C7A')
BODY  = HexColor('#2B363D')
MUTE  = HexColor('#6C7C86')
LINE  = HexColor('#D3DFE4')
FILL  = HexColor('#F4F8F9')
FILL2 = HexColor('#E9F3F6')
WARM  = HexColor('#FBF6EE')
WHT   = HexColor('#FFFFFF')
SOFT  = HexColor('#93AEB8')

F   = 'Helvetica'
FB  = 'Helvetica-Bold'
FO  = 'Helvetica-Oblique'
FC  = 'Courier'
FCB = 'Courier-Bold'


def tint(c, a):
    return Color(c.red, c.green, c.blue, a)


class Doc:
    """Portrait document with running header band + footer."""

    def __init__(self, path, title, author="Ajace AI", subject="", land=False):
        self.land = land
        self.pw, self.ph = (LW, LH) if land else (W, H)
        self.c = _canvas.Canvas(path, pagesize=(self.pw, self.ph))
        self.c.setTitle(title); self.c.setAuthor(author); self.c.setSubject(subject)
        self.n = 0
        self.footer_text = ""

    # ---------- page furniture ----------
    def page(self, kicker, title, tagline="", accent=ACC, band=96):
        self.n += 1
        c = self.c
        c.setFillColor(INK); c.rect(0, self.ph - band, self.pw, band, stroke=0, fill=1)
        c.setFillColor(accent); c.rect(0, self.ph - band - 4, self.pw, 4, stroke=0, fill=1)
        if kicker:
            c.setFillColor(accent); c.setFont(FB, 8.5)
            c.drawString(42, self.ph - 34, kicker.upper())
        c.setFillColor(WHT)
        ts = 21.0
        while ts > 13 and c.stringWidth(title, FB, ts) > self.pw - 84:
            ts -= 0.5
        c.setFont(FB, ts); c.drawString(42, self.ph - 60, title)
        if tagline:
            c.setFillColor(SOFT); c.setFont(F, 9.6)
            for i, ln in enumerate(simpleSplit(tagline, F, 9.6, self.pw - 84)):
                c.drawString(42, self.ph - 78 - i * 12.0, ln)
        self.footer()
        return self.ph - band - 32

    def footer(self):
        c = self.c
        c.setStrokeColor(LINE); c.setLineWidth(0.6); c.line(42, 44, self.pw - 42, 44)
        c.setFillColor(MUTE); c.setFont(F, 7.4)
        c.drawString(42, 33, self.footer_text)
        c.drawRightString(self.pw - 42, 33, str(self.n))

    def end(self):
        self.c.showPage()

    def save(self):
        self.c.save()

    # ---------- text ----------
    def h2(self, x, y, text, col=ACC, size=11.5):
        c = self.c
        c.setFillColor(col); c.setFont(FB, size); c.drawString(x, y, text)
        c.setStrokeColor(tint(col, .55)); c.setLineWidth(1.5)
        c.line(x, y - 4.5, x + c.stringWidth(text, FB, size), y - 4.5)
        return y - 18

    def h3(self, x, y, text, col=INK, size=9.8):
        self.c.setFillColor(col); self.c.setFont(FB, size)
        self.c.drawString(x, y, text)
        return y - 13

    def p(self, x, y, text, w, size=9.4, lead=13.0, col=BODY, font=F):
        c = self.c
        c.setFillColor(col); c.setFont(font, size)
        for ln in simpleSplit(text, font, size, w):
            c.drawString(x, y, ln); y -= lead
        return y

    def bullets(self, x, y, items, w, accent=ACC, size=9.0, lead=12.2, gap=3.6, marker="sq"):
        c = self.c
        for it in items:
            if marker == "sq":
                c.setFillColor(accent); c.rect(x, y + 1.6, 3.4, 3.4, stroke=0, fill=1)
            else:
                c.setFillColor(accent); c.circle(x + 1.8, y + 3.2, 1.8, stroke=0, fill=1)
            c.setFillColor(BODY); c.setFont(F, size)
            for ln in simpleSplit(it, F, size, w - 12):
                c.drawString(x + 11, y, ln); y -= lead
            y -= gap
        return y

    def kv(self, x, y, rows, w, kw=140, size=8.9, lead=12.0, gap=3.6, kcol=ACC2, kfont=FB, ksize=8.2):
        c = self.c
        for k, v in rows:
            c.setFillColor(kcol); c.setFont(kfont, ksize); c.drawString(x, y, k)
            c.setFillColor(BODY); c.setFont(F, size)
            lines = simpleSplit(v, F, size, w - kw)
            for i, ln in enumerate(lines):
                c.drawString(x + kw, y - i * lead, ln)
            y -= lead * max(1, len(lines)) + gap
        return y

    def table(self, x, y, headers, rows, widths, size=8.2, hsize=8.2, pad=5, accent=ACC,
              zebra=True, lead=10.8):
        """Simple grid table. widths = list of column widths."""
        c = self.c
        tw = sum(widths)
        # header
        hh = 17
        c.setFillColor(accent); c.rect(x, y - hh, tw, hh, stroke=0, fill=1)
        cx = x
        c.setFillColor(WHT); c.setFont(FB, hsize)
        for h, wd in zip(headers, widths):
            c.drawString(cx + pad, y - hh + 5.5, h)
            cx += wd
        y -= hh
        # rows
        for ri, row in enumerate(rows):
            cellL = [simpleSplit(str(v), F, size, wd - 2 * pad) for v, wd in zip(row, widths)]
            rh = max(len(l) for l in cellL) * lead + 7
            if zebra and ri % 2 == 0:
                c.setFillColor(FILL); c.rect(x, y - rh, tw, rh, stroke=0, fill=1)
            cx = x
            for lines, wd in zip(cellL, widths):
                c.setFillColor(BODY); c.setFont(F, size)
                for li, ln in enumerate(lines):
                    c.drawString(cx + pad, y - 12 - li * lead + 1, ln)
                cx += wd
            c.setStrokeColor(LINE); c.setLineWidth(0.4)
            c.line(x, y - rh, x + tw, y - rh)
            y -= rh
        return y

    def callout(self, x, y, w, title, text, accent=ACC, size=8.8, lead=11.6):
        c = self.c
        lines = simpleSplit(text, F, size, w - 24)
        bh = 26 + len(lines) * lead
        c.setFillColor(tint(accent, .07)); c.setStrokeColor(accent); c.setLineWidth(0.9)
        c.roundRect(x, y - bh, w, bh, 5, stroke=1, fill=1)
        c.setFillColor(accent); c.setFont(FB, 8.2); c.drawString(x + 12, y - 14, title.upper())
        c.setFillColor(BODY); c.setFont(F, size)
        yy = y - 26
        for ln in lines:
            c.drawString(x + 12, yy, ln); yy -= lead
        return y - bh

    def statrow(self, x, y, w, stats, accent=ACC):
        """stats = [(big, label), ...] evenly spread."""
        c = self.c
        n = len(stats)
        cw = w / n
        for i, (big, label) in enumerate(stats):
            cx = x + i * cw
            c.setFillColor(accent); c.setFont(FB, 17)
            c.drawString(cx, y, big)
            c.setFillColor(MUTE); c.setFont(F, 7.8)
            for j, ln in enumerate(simpleSplit(label, F, 7.8, cw - 8)):
                c.drawString(cx, y - 12 - j * 9.4, ln)
        return y - 34


# ══════════════════════ diagram primitives ══════════════════════

def rbox(c, x, y, w, h, fill=FILL, stroke=LINE, lw=0.9, r=5, dash=None):
    c.setFillColor(fill); c.setStrokeColor(stroke); c.setLineWidth(lw)
    if dash: c.setDash(dash, 3)
    c.roundRect(x, y, w, h, r, stroke=1, fill=1)
    c.setDash()


def node(c, x, y, w, h, title, lines=None, fill=FILL, stroke=ACC, tcol=INK,
         tsize=8.4, lsize=6.6, lw=1.0, r=5, lcol=MUTE):
    rbox(c, x, y, w, h, fill, stroke, lw, r)
    lines = lines or []
    n = len(lines)
    ty = y + h / 2 + (2.6 if n == 0 else 1.6 + n * 4.2)
    c.setFillColor(tcol); c.setFont(FB, tsize)
    c.drawCentredString(x + w / 2, ty, title)
    c.setFillColor(lcol); c.setFont(F, lsize)
    yy = ty - 9.2
    for ln in lines:
        c.drawCentredString(x + w / 2, yy, ln); yy -= 8.0


def diamond(c, cx, cy, w, h, text, stroke=AMBER, fill=WARM, size=7.0):
    c.setFillColor(fill); c.setStrokeColor(stroke); c.setLineWidth(1.0)
    p = c.beginPath()
    p.moveTo(cx, cy + h / 2); p.lineTo(cx + w / 2, cy)
    p.lineTo(cx, cy - h / 2); p.lineTo(cx - w / 2, cy); p.close()
    c.drawPath(p, stroke=1, fill=1)
    c.setFillColor(INK); c.setFont(FB, size)
    lines = simpleSplit(text, FB, size, w - 16)
    yy = cy + (len(lines) - 1) * 4.2
    for ln in lines:
        c.drawCentredString(cx, yy - 2.4, ln); yy -= 8.4


def arrow(c, x1, y1, x2, y2, col=ACC2, lw=1.1, head=4.2, dash=None):
    c.setStrokeColor(col); c.setFillColor(col); c.setLineWidth(lw)
    if dash: c.setDash(dash, 2)
    ang = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - head * 1.6 * math.cos(ang), y2 - head * 1.6 * math.sin(ang)
    c.line(x1, y1, bx, by)
    c.setDash()
    p = c.beginPath()
    p.moveTo(x2, y2)
    p.lineTo(x2 - head * 2.0 * math.cos(ang - .42), y2 - head * 2.0 * math.sin(ang - .42))
    p.lineTo(x2 - head * 2.0 * math.cos(ang + .42), y2 - head * 2.0 * math.sin(ang + .42))
    p.close()
    c.drawPath(p, stroke=0, fill=1)


def elbow(c, x1, y1, x2, y2, col=ACC2, lw=1.1, via=None, head=4.2, dash=None):
    """Orthogonal 3-segment connector: down/up to mid, across, then into target."""
    c.setStrokeColor(col); c.setLineWidth(lw)
    if dash: c.setDash(dash, 2)
    my = via if via is not None else (y1 + y2) / 2
    c.line(x1, y1, x1, my)
    c.line(x1, my, x2, my)
    c.setDash()
    arrow(c, x2, my, x2, y2, col, lw, head)


def badge(c, x, y, n, col=ACC2, r=6.2, size=7.0):
    c.setFillColor(WHT); c.setStrokeColor(col); c.setLineWidth(1.1)
    c.circle(x, y, r, stroke=1, fill=1)
    c.setFillColor(col); c.setFont(FB, size)
    c.drawCentredString(x, y - 2.4, str(n))


def lane(c, x, y, w, h, label, col=ACC, fill=None):
    c.setFillColor(fill or tint(col, .05)); c.setStrokeColor(tint(col, .35))
    c.setLineWidth(0.8); c.roundRect(x, y, w, h, 4, stroke=1, fill=1)
    c.saveState()
    c.setFillColor(col); c.setFont(FB, 7.6)
    c.translate(x + 11, y + h / 2)
    c.rotate(90)
    c.drawCentredString(0, -2.6, label.upper())
    c.restoreState()


def tag(c, x, y, text, col=ACC, size=6.6, padx=5, h=11.5):
    tw = c.stringWidth(text, FB, size)
    c.setFillColor(tint(col, .12)); c.setStrokeColor(col); c.setLineWidth(0.7)
    c.roundRect(x, y, tw + padx * 2, h, 3, stroke=1, fill=1)
    c.setFillColor(col); c.setFont(FB, size)
    c.drawString(x + padx, y + 3.4, text)
    return tw + padx * 2


def label(c, x, y, text, col=MUTE, size=6.8, font=F, center=False, right=False):
    c.setFillColor(col); c.setFont(font, size)
    if center: c.drawCentredString(x, y, text)
    elif right: c.drawRightString(x, y, text)
    else: c.drawString(x, y, text)
