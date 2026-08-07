#!/usr/bin/env python3
"""EchoMind Enterprise — Architecture Brief v2 (full feature coverage)."""
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.colors import HexColor, Color
from reportlab.lib.utils import simpleSplit

OUT = "/home/echomind/Documents/echomind/echomind-enterprise/docs/arch/EchoMind_Architecture_Brief_v2.pdf"

W, H = A4                      # 595 x 842 portrait
LW, LH = landscape(A4)         # 842 x 595

INK   = HexColor('#0B1F2A')
ACC   = HexColor('#0E8F9E')
ACC2  = HexColor('#17607A')
GREEN = HexColor('#12805F')
AMBER = HexColor('#A2620B')
PURP  = HexColor('#4F52C4')
ROSE  = HexColor('#B03A5B')
BODY  = HexColor('#2E3A42')
MUTE  = HexColor('#6C7C86')
LINE  = HexColor('#C4D5DC')
FILL  = HexColor('#F5FAFB')
FILL2 = HexColor('#E9F3F6')
FILL3 = HexColor('#FBF6EE')
WHT   = HexColor('#FFFFFF')

F  = 'Helvetica'
FB = 'Helvetica-Bold'
FO = 'Helvetica-Oblique'
FC = 'Courier'
FCB = 'Courier-Bold'

PAGE = [0]


# ────────────────────────────── primitives ──────────────────────────────

def header(c, title, sub, w=W, h=H):
    c.setFillColor(INK)
    c.rect(0, h - 62, w, 62, stroke=0, fill=1)
    c.setFillColor(WHT)
    ts = 18.0
    maxw = w - 84 - 150
    while ts > 12.5 and c.stringWidth(title, FB, ts) > maxw:
        ts -= 0.5
    c.setFont(FB, ts)
    c.drawString(42, h - 34, title)
    c.setFillColor(HexColor('#93AEB8'))
    c.setFont(F, 9)
    c.drawString(42, h - 49, sub)
    c.setFillColor(ACC)
    c.setFont(FB, 7.5)
    c.drawRightString(w - 42, h - 30, "ECHOMIND ENTERPRISE")
    c.setFillColor(HexColor('#93AEB8'))
    c.setFont(F, 7.5)
    c.drawRightString(w - 42, h - 44, "Architecture Brief v2  ·  page %d" % PAGE[0])


def footer(c, w=W, fy=40):
    c.setStrokeColor(LINE)
    c.setLineWidth(0.6)
    c.line(42, fy, w - 42, fy)
    c.setFillColor(MUTE)
    c.setFont(F, 7.5)
    c.drawString(42, fy - 11, "Ajace AI  —  EchoMind Enterprise  ·  internal architecture brief")
    c.drawRightString(w - 42, fy - 11, str(PAGE[0]))


def page(c, title, sub, land=False):
    PAGE[0] += 1
    w, h = (LW, LH) if land else (W, H)
    c.setPageSize((w, h))
    header(c, title, sub, w, h)
    footer(c, w, 26 if land else 40)
    return w, h


def h2(c, x, y, text, col=ACC2, size=12.5):
    c.setFillColor(col)
    c.setFont(FB, size)
    c.drawString(x, y, text)
    tw = c.stringWidth(text, FB, size)
    c.setStrokeColor(ACC)
    c.setLineWidth(1.8)
    c.line(x, y - 5, x + tw, y - 5)
    return y - 20


def para(c, x, y, text, w, size=9.4, lead=13.2, col=BODY, font=F):
    c.setFillColor(col)
    c.setFont(font, size)
    for ln in simpleSplit(text, font, size, w):
        c.drawString(x, y, ln)
        y -= lead
    return y


def bullets(c, x, y, items, w, size=9.3, lead=12.8, gap=4.5, col=BODY, mark=ACC):
    for it in items:
        lines = simpleSplit(it, F, size, w - 14)
        c.setFillColor(mark)
        c.rect(x + 1, y + 2.4, 4, 4, stroke=0, fill=1)
        c.setFillColor(col)
        c.setFont(F, size)
        for i, ln in enumerate(lines):
            c.drawString(x + 14, y, ln)
            y -= lead
        y -= gap
    return y


def rbox(c, x, y, w, h, fill=FILL, stroke=LINE, lw=0.9, r=6, dash=None):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(lw)
    if dash:
        c.setDash(dash, 3)
    c.roundRect(x, y, w, h, r, stroke=1, fill=1)
    c.setDash()


def boxed(c, x, y, w, h, title, lines=None, fill=FILL, stroke=LINE,
          tcol=INK, tsize=8.6, lsize=6.9, lw=0.9, lcol=MUTE, r=6):
    rbox(c, x, y, w, h, fill, stroke, lw, r)
    lines = lines or []
    n = len(lines)
    ty = y + h / 2 + (2.5 if n == 0 else 2.0 + n * 4.1)
    c.setFillColor(tcol)
    c.setFont(FB, tsize)
    c.drawCentredString(x + w / 2, ty, title)
    c.setFillColor(lcol)
    c.setFont(F, lsize)
    yy = ty - 9.5
    for ln in lines:
        c.drawCentredString(x + w / 2, yy, ln)
        yy -= 8.2


def arrow(c, x1, y1, x2, y2, col=ACC2, lw=1.2, head=4.6, dash=None):
    c.setStrokeColor(col)
    c.setFillColor(col)
    c.setLineWidth(lw)
    if dash:
        c.setDash(dash, 2)
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - head * 1.7 * math.cos(ang), y2 - head * 1.7 * math.sin(ang)
    c.line(x1, y1, bx, by)
    c.setDash()
    p = c.beginPath()
    p.moveTo(x2, y2)
    p.lineTo(x2 - head * 2.0 * math.cos(ang - 0.42), y2 - head * 2.0 * math.sin(ang - 0.42))
    p.lineTo(x2 - head * 2.0 * math.cos(ang + 0.42), y2 - head * 2.0 * math.sin(ang + 0.42))
    p.close()
    c.drawPath(p, stroke=0, fill=1)


def badge(c, x, y, n, col=ACC2, r=6.4):
    c.setFillColor(WHT)
    c.setStrokeColor(col)
    c.setLineWidth(1.1)
    c.circle(x, y, r, stroke=1, fill=1)
    c.setFillColor(col)
    c.setFont(FB, 7.2)
    c.drawCentredString(x, y - 2.5, str(n))


def tag(c, x, y, text, col=ACC, fill=None, size=6.6, padx=5, h=11):
    tw = c.stringWidth(text, FB, size)
    c.setFillColor(fill or Color(col.red, col.green, col.blue, 0.10))
    c.setStrokeColor(col)
    c.setLineWidth(0.7)
    c.roundRect(x, y, tw + padx * 2, h, 3, stroke=1, fill=1)
    c.setFillColor(col)
    c.setFont(FB, size)
    c.drawString(x + padx, y + 3.2, text)
    return tw + padx * 2


def label(c, x, y, text, col=MUTE, size=6.8, font=F, center=False):
    c.setFillColor(col)
    c.setFont(font, size)
    (c.drawCentredString if center else c.drawString)(x, y, text)


def kv_table(c, x, y, rows, w, kw=150, size=8.8, lead=12.4, rowgap=4.2, kfont=FCB, ksize=7.6):
    for k, v in rows:
        lines = simpleSplit(v, F, size, w - kw)
        c.setFillColor(ACC2)
        c.setFont(kfont, ksize)
        c.drawString(x, y, k)
        c.setFillColor(BODY)
        c.setFont(F, size)
        for i, ln in enumerate(lines):
            c.drawString(x + kw, y - i * lead, ln)
        y -= lead * max(1, len(lines)) + rowgap
    return y


def feature(c, x, y, w, name, tags, what, how, why, col=ACC):
    """One feature card. Returns new y."""
    inner = w - 22
    body_lines = 0
    for t in (what, how, why):
        body_lines += len(simpleSplit(t[1], F, 8.6, inner - 46))
    h = 30 + body_lines * 11.6 + 10
    rbox(c, x, y - h, w, h, WHT, LINE, 0.9)
    c.setFillColor(col)
    c.rect(x, y - h, 3.2, h, stroke=0, fill=1)
    c.setFillColor(INK)
    c.setFont(FB, 10.4)
    c.drawString(x + 13, y - 17, name)
    tx = x + 15 + c.stringWidth(name, FB, 10.4)
    for t in tags:
        tx += tag(c, tx, y - 20, t, col) + 4
    yy = y - 33
    for lab, txt in (what, how, why):
        c.setFillColor(col)
        c.setFont(FCB, 6.8)
        c.drawString(x + 13, yy, lab)
        c.setFillColor(BODY)
        c.setFont(F, 8.6)
        for ln in simpleSplit(txt, F, 8.6, inner - 46):
            c.drawString(x + 59, yy, ln)
            yy -= 11.6
    return y - h - 13


# ══════════════════════════════ PAGE 1 — overview ══════════════════════════════

def p1(c):
    w, h = page(c, "EchoMind Enterprise", "What it is, what it does, and how to read this document")
    x, y = 42, h - 92
    y = para(c, x, y,
             "EchoMind is a private AI assistant. It answers questions from your own documents, listens to "
             "your meetings, holds a spoken conversation, and writes finished documents — all on one machine "
             "you own. No cloud, no external API calls, no data leaving the building.", w - 84, 10.4, 15)
    y -= 12

    y = h2(c, x, y, "The five things it does")
    cols = [
        ("Knowledge Chat", "Ask questions, get cited answers from your documents."),
        ("Live Transcript", "Real-time meeting transcription with a fact-checker running alongside."),
        ("Conversation", "Full spoken dialogue — you talk, it answers out loud."),
        ("Document Studio", "Generates PDF and PowerPoint deliverables from a brief."),
        ("Boardroom", "Records multi-speaker sessions, separates speakers, produces a report."),
    ]
    cw = (w - 84 - 2 * 10) / 3
    for i, (t, d) in enumerate(cols):
        cx = x + (i % 3) * (cw + 10)
        cy = y - (i // 3) * 62
        boxed(c, cx, cy - 54, cw, 54, t, simpleSplit(d, F, 6.9, cw - 16),
              FILL, LINE, INK, 9.2, 6.9)
    y = y - 62 * 2 - 18

    y = h2(c, x, y, "Key facts")
    y = kv_table(c, x, y, [
        ("Deployment", "One NVIDIA DGX Spark (GB10 Grace-Blackwell, 128 GB unified memory). Five Docker containers, one command."),
        ("Language model", "nvidia/Llama-3.3-70B-Instruct-FP4 served by TensorRT-LLM over an OpenAI-compatible endpoint."),
        ("Embeddings", "nomic-embed-text (768 dimensions), served locally by Ollama."),
        ("Speech", "NVIDIA Nemotron streaming ASR for live captions, Parakeet-TDT for the accurate final transcript, Piper for speech output."),
        ("Images", "SDXL-Turbo running on the same GPU for Document Studio artwork."),
        ("Retrieval", "FAISS dense vectors + BM25 keyword index, fused and re-ranked by a cross-encoder."),
        ("Storage", "SQLite plus local disk. Nothing is written outside the machine."),
        ("Network", "Runs with the cable unplugged. Optional public access via Cloudflare Tunnel behind a login wall."),
    ], w - 84, kw=96)
    y -= 6

    y = h2(c, x, y, "How to read this document")
    y = bullets(c, x, y, [
        "Page 2 is the full architecture on one landscape page — start there.",
        "Pages 3–4 describe every feature: what it does, how it works, why it matters.",
        "Pages 5–7 are the three pipelines in detail: ingestion, answering, and speech.",
        "Page 8 is the security and trust model. Page 9 is the code map and operations.",
    ], w - 84)

    c.setFillColor(MUTE)
    c.setFont(FO, 8.2)
    c.drawString(x, 58, "The model named above is the model specified for this deployment; the platform is model-agnostic "
                        "behind a single endpoint setting.")
    c.showPage()


# ══════════════════════════ PAGE 2 — master architecture ══════════════════════════

def p2(c):
    w, h = page(c, "System Architecture", "Every component, every path, one machine", land=True)
    LX, RX = 34, w - 34
    CX0 = 116                      # content column start
    CW = RX - CX0                  # content width

    def lane(c, y0, y1, name, sub):
        c.setFillColor(ACC2)
        c.setFont(FB, 8.0)
        c.drawString(LX, y1 - 11, name)
        c.setFillColor(MUTE)
        c.setFont(F, 6.4)
        yy = y1 - 21
        for ln in simpleSplit(sub, F, 6.4, CX0 - LX - 12):
            c.drawString(LX, yy, ln)
            yy -= 7.6
        c.setStrokeColor(HexColor('#E1ECF0'))
        c.setLineWidth(0.7)
        c.line(LX, y0 - 6, RX, y0 - 6)

    # --- band geometry ---
    b_cli = (474, 528)
    b_edg = (410, 456)
    b_svc = (250, 392)
    b_inf = (160, 232)
    b_dat = (82, 148)
    SP = CX0 + 15                  # left spine: backend <-> data, routed clear of inference
    IX = CX0 + 34                  # inference boxes start right of the spine
    IW_TOT = RX - IX

    # on-prem boundary
    c.setStrokeColor(HexColor('#9CC3CE'))
    c.setLineWidth(1.0)
    c.setDash([3, 3], 2)
    c.roundRect(CX0 - 8, b_dat[0] - 12, CW + 16, b_svc[1] - b_dat[0] + 24, 8, stroke=1, fill=0)
    c.setDash()
    bl = "ONE MACHINE  ·  NVIDIA DGX SPARK GB10  ·  128 GB UNIFIED MEMORY  ·  NO EGRESS AT RUNTIME"
    bw = c.stringWidth(bl, FB, 6.4) + 16
    c.setFillColor(WHT)
    c.rect(CX0 + CW / 2 - bw / 2, b_dat[0] - 17, bw, 11, stroke=0, fill=1)
    label(c, CX0 + CW / 2, b_dat[0] - 14, bl, ACC2, 6.4, FB, center=True)

    # ---------- 1. clients ----------
    lane(c, *b_cli, "CLIENTS", "browser only — nothing to install")
    y0, y1 = b_cli
    bw1 = CW * 0.72
    boxed(c, CX0, y0, bw1, y1 - y0, "BROWSER  —  React single-page app",
          ["Knowledge Chat   ·   Live Transcript   ·   Conversation   ·   Document Studio   ·   Boardroom",
           "vertical packs: health · law · meetings · retail · bank  —  each subdomain loads its own theme, persona and knowledge base"],
          FILL2, ACC, INK, 9.4, 6.9)
    boxed(c, CX0 + bw1 + 12, y0, CW - bw1 - 12, y1 - y0, "ADMIN",
          ["users · roles · tenants", "audit log · usage"], FILL, LINE, INK, 9.0, 6.9)

    # ---------- 2. edge ----------
    lane(c, *b_edg, "EDGE", "TLS termination and routing")
    y0, y1 = b_edg
    boxed(c, CX0, y0, CW * 0.62, y1 - y0, "NGINX",
          ["serves the SPA  ·  /api → backend  ·  /voice → voice service (WebSocket)  ·  TLS"],
          FILL, ACC2, INK, 9.2, 6.9)
    boxed(c, CX0 + CW * 0.62 + 12, y0, CW * 0.38 - 12, y1 - y0, "CLOUDFLARE TUNNEL   (optional)",
          ["outbound-only tunnel  ·  identity gate in front of the app  ·  off by default"],
          FILL3, AMBER, INK, 8.6, 6.9)

    # ---------- 3. services ----------
    lane(c, *b_svc, "APPLICATION", "two FastAPI services holding all business logic")
    y0, y1 = b_svc
    bkw = CW * 0.585
    vcw = CW - bkw - 16
    bx, vx = CX0, CX0 + bkw + 16

    rbox(c, bx, y0, bkw, y1 - y0, WHT, ACC2, 1.3)
    label(c, bx + 12, y1 - 15, "BACKEND  —  FastAPI", INK, 9.6, FB)
    label(c, bx + 12, y1 - 25, "documents, retrieval, transcription, generation, identity", MUTE, 6.6)
    chips_b = [
        ("Knowledge Chat API", "route · retrieve · cite"),
        ("Ingestion", "parse · chunk · embed"),
        ("Live Transcript", "WebSocket · streaming ASR"),
        ("Silent Assistant", "fact-check each paragraph"),
        ("Boardroom", "diarise · analyse · report"),
        ("Document Studio", "PDF · PPTX · images"),
        ("Export Gateway", "scan · redact · release"),
        ("Auth · RBAC · Audit", "sessions · tenants · log"),
    ]
    cw2 = (bkw - 24 - 3 * 7) / 4
    for i, (t, s) in enumerate(chips_b):
        cx = bx + 12 + (i % 4) * (cw2 + 7)
        cy = y1 - 66 - (i // 4) * 44
        boxed(c, cx, cy, cw2, 38, "", [], FILL, LINE, INK, 7.4, 6.0)
        label(c, cx + cw2 / 2, cy + 23, t, INK, 6.9, FB, center=True)
        for j, ln in enumerate(simpleSplit(s, F, 5.9, cw2 - 8)):
            label(c, cx + cw2 / 2, cy + 13 - j * 7, ln, MUTE, 5.9, F, center=True)

    rbox(c, vx, y0, vcw, y1 - y0, WHT, PURP, 1.3)
    label(c, vx + 12, y1 - 15, "VOICE  —  FastAPI", INK, 9.6, FB)
    label(c, vx + 12, y1 - 25, "the real-time speech loop", MUTE, 6.6)
    chips_v = [
        ("VAD + endpointing", "when did they stop?"),
        ("Streaming STT", "live partial words"),
        ("Final STT", "accurate transcript"),
        ("Lead phrase", "speaks in < 50 ms"),
        ("Grounded reply", "asks backend RAG"),
        ("TTS + barge-in", "speak · stop on talk"),
    ]
    cw3 = (vcw - 24 - 2 * 7) / 3
    for i, (t, s) in enumerate(chips_v):
        cx = vx + 12 + (i % 3) * (cw3 + 7)
        cy = y1 - 66 - (i // 3) * 44
        boxed(c, cx, cy, cw3, 38, "", [], HexColor('#F4F4FC'), HexColor('#CFD0F0'), INK, 7.4, 6.0)
        label(c, cx + cw3 / 2, cy + 23, t, INK, 6.9, FB, center=True)
        for j, ln in enumerate(simpleSplit(s, F, 5.9, cw3 - 8)):
            label(c, cx + cw3 / 2, cy + 13 - j * 7, ln, MUTE, 5.9, F, center=True)

    # ---------- 4. inference ----------
    lane(c, *b_inf, "INFERENCE", "all model execution, on the local GPU")
    y0, y1 = b_inf
    iw = (IW_TOT - 3 * 12) / 4
    infs = [
        ("TensorRT-LLM", ["Llama-3.3-70B-Instruct", "NVFP4 · OpenAI-compatible", "answers · analysis · drafting"], ACC, HexColor('#EAF6F7')),
        ("Ollama", ["nomic-embed-text", "768-dimension vectors", "used at upload and at query"], GREEN, HexColor('#EDF7F3')),
        ("Speech models", ["Nemotron streaming ASR", "Parakeet-TDT final ASR", "Piper text-to-speech"], PURP, HexColor('#F3F3FC')),
        ("SDXL-Turbo", ["on-device image generation", "Document Studio artwork", "4 steps · up to 1024 px"], AMBER, HexColor('#FBF5EC')),
    ]
    for i, (t, ls, col, fl) in enumerate(infs):
        boxed(c, IX + i * (iw + 12), y0, iw, y1 - y0, t, ls, fl, col, INK, 9.0, 6.5)

    # ---------- 5. data ----------
    lane(c, *b_dat, "DATA", "local disk only")
    y0, y1 = b_dat
    dw = (CW - 3 * 12) / 4
    dats = [
        ("FAISS", ["dense vector index", "documents + transcripts", "flat → IVF at 10k chunks"]),
        ("BM25", ["exact keyword index", "catches names, codes, IDs", "rebuilt with the corpus"]),
        ("SQLite", ["documents · chats · transcripts", "users · tenants · audit log", "boardroom · docgen jobs"]),
        ("Local disk", ["uploaded files", "generated PDF / PPTX", "model weights (baked in)"]),
    ]
    for i, (t, ls) in enumerate(dats):
        boxed(c, CX0 + i * (dw + 12), y0, dw, y1 - y0, t, ls, HexColor('#EFF6F2'), GREEN, INK, 9.0, 6.5)

    # ---------- flow arrows ----------
    mid_b = bx + bkw * 0.32
    mid_v = vx + vcw / 2
    # 1 browser -> nginx
    arrow(c, CX0 + bw1 * 0.30, b_cli[0], CX0 + bw1 * 0.30, b_edg[1], ACC2)
    badge(c, CX0 + bw1 * 0.30 + 13, (b_cli[0] + b_edg[1]) / 2, 1)
    label(c, CX0 + bw1 * 0.30 + 23, (b_cli[0] + b_edg[1]) / 2 - 2.4, "HTTPS", MUTE, 6.2)
    # 2 browser -> nginx (voice ws)
    arrow(c, CX0 + bw1 * 0.72, b_cli[0], CX0 + bw1 * 0.72, b_edg[1], PURP)
    badge(c, CX0 + bw1 * 0.72 + 13, (b_cli[0] + b_edg[1]) / 2, 2, PURP)
    label(c, CX0 + bw1 * 0.72 + 23, (b_cli[0] + b_edg[1]) / 2 - 2.4, "WebSocket", MUTE, 6.2)
    # 3 nginx -> backend
    arrow(c, CX0 + CW * 0.18, b_edg[0], mid_b, b_svc[1], ACC2)
    badge(c, (CX0 + CW * 0.18 + mid_b) / 2 - 13, (b_edg[0] + b_svc[1]) / 2, 3)
    # 4 nginx -> voice
    arrow(c, CX0 + CW * 0.50, b_edg[0], mid_v, b_svc[1], PURP)
    badge(c, (CX0 + CW * 0.50 + mid_v) / 2 + 13, (b_edg[0] + b_svc[1]) / 2, 4, PURP)
    # 5 voice -> backend (grounded answers)
    arrow(c, vx - 1, b_svc[1] - 22, bx + bkw + 1, b_svc[1] - 22, AMBER)
    badge(c, (vx + bx + bkw) / 2, b_svc[1] - 36, 5, AMBER)
    # 6 backend -> LLM + embeddings
    arrow(c, bx + bkw * 0.30, b_svc[0], IX + iw * 0.5, b_inf[1], ACC2)
    arrow(c, bx + bkw * 0.58, b_svc[0], IX + (iw + 12) + iw * 0.5, b_inf[1], GREEN)
    badge(c, IX + iw * 0.5 - 18, (b_svc[0] + b_inf[1]) / 2, 6)
    # backend -> SDXL (dashed: only when generating images)
    arrow(c, bx + bkw * 0.90, b_svc[0], IX + 3 * (iw + 12) + iw * 0.5, b_inf[1], AMBER, dash=[2, 2])
    # 7 voice -> speech models
    arrow(c, mid_v, b_svc[0], IX + 2 * (iw + 12) + iw * 0.5, b_inf[1], PURP)
    badge(c, IX + 2 * (iw + 12) + iw * 0.5 + 20, (b_svc[0] + b_inf[1]) / 2, 7, PURP)
    # 8 backend <-> data, routed down the spine clear of the inference boxes
    c.setStrokeColor(GREEN)
    c.setLineWidth(1.2)
    c.line(bx + 14, b_svc[0], SP, b_svc[0])
    arrow(c, SP, b_svc[0], SP, b_dat[1] + 6, GREEN)
    c.setStrokeColor(GREEN)
    c.setLineWidth(1.2)
    c.line(SP, b_dat[1] + 6, SP, b_dat[1] + 6)
    for i in range(4):
        tx_ = CX0 + i * (dw + 12) + dw * 0.5
        c.setStrokeColor(GREEN)
        c.setLineWidth(1.0)
        c.line(SP, b_dat[1] + 6, tx_, b_dat[1] + 6)
        arrow(c, tx_, b_dat[1] + 6, tx_, b_dat[1] + 1, GREEN, 1.0, 3.4)
    badge(c, SP + 14, (b_svc[0] + b_dat[1]) / 2 + 8, 8, GREEN)

    # ---------- legend ----------
    items = [
        (1, "browser → nginx over HTTPS", ACC2),
        (2, "microphone audio over a WebSocket", PURP),
        (3, "/api routed to the backend", ACC2),
        (4, "/voice routed to the voice service", PURP),
        (5, "voice asks the backend for a grounded, cited answer", AMBER),
        (6, "backend calls the language and embedding models", ACC2),
        (7, "voice calls the local speech models", PURP),
        (8, "indexes and records read from local disk", GREEN),
    ]
    colw = CW / 4
    for i, (n, t, col) in enumerate(items):
        lx = CX0 + (i % 4) * colw
        ly = 50 - (i // 4) * 13
        badge(c, lx + 6, ly + 3, n, col, 5.4)
        label(c, lx + 15, ly + 0.8, t, MUTE, 6.3)
    c.showPage()


# ══════════════════════════ PAGES 3–4 — feature catalogue ══════════════════════════

def p3(c):
    w, h = page(c, "Features  (1 of 2)", "Working with knowledge, meetings and speech")
    x, y = 42, h - 88
    fw = w - 84

    y = feature(c, x, y, fw, "Knowledge Chat", ["RAG", "cited"], (
        "WHAT", "Ask a question in plain English and get an answer built only from your uploaded documents and stored meeting transcripts, with a citation on every claim."), (
        "HOW", "Hybrid retrieval: a meaning search (FAISS) and an exact-keyword search (BM25) run together, are fused by weighted reciprocal-rank fusion, then a cross-encoder re-reads the top 25 passages against your question and keeps the best 15. The model writes the answer from those passages only."), (
        "WHY", "The failure mode of a naive assistant is a confident answer with no source. Here the sources are shown, clickable and traceable to page and section."))

    y = feature(c, x, y, fw, "Conversational routing", ["intent"], (
        "WHAT", "The assistant tells the difference between small talk, a refusal, a follow-up and a real question."), (
        "HOW", "A prototype-embedding classifier compares your message against learned examples of each intent; greetings and refusals never touch the knowledge base, follow-ups inherit the previous question's context. Rules only as a fast path, semantics as the decision."), (
        "WHY", "Without this, saying \"hello\" pulled four random document extracts and the reply read like a machine. Greetings now get a greeting."))

    y = feature(c, x, y, fw, "Live Transcript", ["real-time", "GPU"], (
        "WHAT", "Speak or play a meeting and see the words appear as they are said, then store the finished transcript into the knowledge base."), (
        "HOW", "Audio streams over a WebSocket in 20 ms frames. NVIDIA Nemotron streaming ASR produces partial text continuously; silence boundaries close paragraphs. Transcripts are auto-saved on an interval and become searchable alongside documents."), (
        "WHY", "Meetings are where most decisions actually live. Once stored, yesterday's discussion answers today's question."))

    y = feature(c, x, y, fw, "Silent Assistant", ["fact-check", "live"], (
        "WHAT", "While the meeting runs, each finished paragraph is silently checked against your reference material and labelled: Supported, Contradicted, Unverified, Violating or Risky Statement."), (
        "HOW", "Every completed paragraph is sent through the same retrieval stack, judged by the model, and only surfaced when confidence is at least 60. Domain rule packs sharpen the check per vertical — drug interactions for clinical, missing disclosures for banking, one-sided clauses for legal."), (
        "WHY", "It catches the wrong statement in the room, while it can still be corrected."))
    c.showPage()


def p4(c):
    w, h = page(c, "Features  (2 of 2)", "Producing work, isolating tenants, releasing data safely")
    x, y = 42, h - 88
    fw = w - 84

    y = feature(c, x, y, fw, "Voice Conversation", ["duplex-feel"], (
        "WHAT", "A spoken conversation. You talk, it answers out loud from the same documents the chat uses, and you can interrupt it mid-sentence."), (
        "HOW", "Voice activity detection plus semantic endpointing decide when your turn ends — a complete sentence gets a fast reply, a trailing one waits longer. A short lead phrase starts speaking in under 50 ms while retrieval runs behind it; the grounded answer then streams phrase by phrase into Piper."), (
        "WHY", "Turn-based bots feel dead because of the silence after you stop. Filling that gap is what makes it feel like a conversation."), PURP)

    y = feature(c, x, y, fw, "Document Studio", ["PDF", "PPTX", "images"], (
        "WHAT", "Turn a brief into a finished deliverable: clinical notes, contract review memos, meeting minutes, loan suitability reports, whitepapers, pitch decks, PRDs, case studies, brand books and more — 23 templates across five visual themes."), (
        "HOW", "The template supplies a section blueprint and writing guidance; the model drafts each section grounded in your source or knowledge base; a renderer produces a real PDF or PowerPoint file. Illustrations are generated on-device by SDXL-Turbo. Custom templates can be uploaded."), (
        "WHY", "The output is a file someone can send, not a chat transcript to copy and reformat."), AMBER)

    y = feature(c, x, y, fw, "Boardroom", ["multi-speaker"], (
        "WHAT", "Record a multi-person session, separate who said what, and produce an analysed report you can export."), (
        "HOW", "Audio is uploaded in chunks, concatenated, normalised to 16 kHz mono, diarised into speaker turns, linked to the live transcript, then analysed into decisions, commitments and action items."), (
        "WHY", "Minutes stop being someone's evening job."), GREEN)

    y = feature(c, x, y, fw, "Vertical packs", ["multi-tenant"], (
        "WHAT", "One deployment presents as five products — Health, Law, Meeting Rooms, Retail and Bank — each on its own subdomain with its own knowledge base, persona, terminology and colour theme."), (
        "HOW", "Each pack maps to an isolated namespace. The namespace is applied inside the search itself, not as a filter afterwards, so a query in one pack cannot retrieve another pack's content even in the ranking stage."), (
        "WHY", "Sell the same box to a hospital and a law firm without either seeing the other's documents."), ROSE)

    y = feature(c, x, y, fw, "Secure Export Gateway", ["offline → online"], (
        "WHAT", "Before any content leaves the machine, it is scanned for sensitive data and a redacted version is produced."), (
        "HOW", "Deterministic detectors — regular expressions plus a Luhn check for card numbers — find national IDs, API keys, private keys, tokens, card numbers, emails, phone numbers and IP addresses, rank them by severity and mask them. No GPU, no network, so it runs air-gapped on every export."), (
        "WHY", "\"Nothing leaves unchecked\" has to be a mechanism, not a policy document."), ACC2)
    c.showPage()


# ══════════════════════════ PAGE 5 — ingestion ══════════════════════════

def p5(c):
    w, h = page(c, "Pipeline 1 — Getting Documents In", "From an uploaded file to a searchable, isolated passage")
    x, y = 42, h - 88
    fw = w - 84

    y = para(c, x, y, "Everything the assistant knows arrives through this path. It runs once per document, at upload.", fw, 9.6, 14)
    y -= 14

    steps = [
        ("Upload", "PDF · Word · PowerPoint · text", ACC),
        ("Extract", "text, layout, page numbers", ACC),
        ("Chunk", "~450 words, 120 overlap", ACC2),
        ("Embed", "768-dim vector each", GREEN),
        ("Index", "FAISS + BM25", GREEN),
        ("Tag", "namespace + metadata", ROSE),
    ]
    bw = (fw - 5 * 10) / 6
    for i, (t, s, col) in enumerate(steps):
        bx = x + i * (bw + 10)
        boxed(c, bx, y - 52, bw, 52, t, simpleSplit(s, F, 6.2, bw - 10), FILL, col, INK, 8.6, 6.2)
        if i < 5:
            arrow(c, bx + bw + 1, y - 26, bx + bw + 9, y - 26, col, 1.1, 3.4)
    y -= 74

    y = h2(c, x, y, "What happens at each step")
    y = kv_table(c, x, y, [
        ("Extract", "Text is pulled out with its structure intact — headings, sections and page numbers are preserved so a citation can point at a real place in the file."),
        ("Chunk", "The document is cut into passages of roughly 450 words with 120 words of overlap, so a sentence spanning a boundary is never lost. Structured books use chapter-aware chunking with parent/child passages."),
        ("Coverage guard", "After chunking, the pipeline measures how much of the original text the chunks actually cover. Below 98 %, it discards the clever chunking and falls back to the simple splitter. Content loss is never silent."),
        ("Embed", "Each passage becomes a 768-number vector capturing its meaning, produced locally by nomic-embed-text."),
        ("Index", "Passages are filed twice — into FAISS for meaning search and into BM25 for exact words. Both are needed: meaning search finds paraphrases, keyword search finds part numbers and case codes."),
        ("Tag", "Every passage carries its tenant namespace, document ID, section and page. This tag is what makes isolation and citation possible later."),
    ], fw, kw=88, size=8.9, lead=12.2)
    y -= 6

    y = h2(c, x, y, "Design decisions worth knowing")
    y = bullets(c, x, y, [
        "Two indexes, not one. A single vector index misses exact identifiers; a single keyword index misses paraphrase. Running both and fusing them is measurably better than either.",
        "Overlap costs storage and buys recall. 120 words of overlap on a 450-word window is the setting that stopped answers being cut in half at chunk boundaries.",
        "FAISS starts flat and switches to an approximate index above ten thousand chunks — exact search while the corpus is small, scalable search once it is not.",
        "The namespace is written into the passage at index time. Nothing downstream has to remember to filter.",
    ], fw)
    c.showPage()


# ══════════════════════════ PAGE 6 — answering ══════════════════════════

def p6(c):
    w, h = page(c, "Pipeline 2 — Turning a Question Into an Answer", "Seven stages, every one of them defensible")
    x, y = 42, h - 86
    fw = w - 84

    steps = [
        ("1", "Route", "Is this a real question?",
         "Small talk, refusals and follow-ups are classified by a semantic intent model. Greetings never touch the knowledge base; follow-ups inherit the prior question's context.", ACC),
        ("2", "Search", "Two searches at once",
         "The question is embedded and run against FAISS for meaning, and against BM25 for exact words. Both run inside the tenant namespace.", ACC),
        ("3", "Merge", "Fuse and rank",
         "The two result lists are combined by weighted reciprocal-rank fusion (0.6 meaning / 0.4 keyword), with recency decay and tag boosts applied.", ACC2),
        ("4", "Re-read", "Cross-encoder rerank",
         "A cross-encoder reads each of the top 25 passages together with the question — not as separate vectors — and keeps the best 15. This is the step that removes plausible-but-irrelevant hits.", ACC2),
        ("5", "Gate", "Relevance floor",
         "If nothing clears the relevance bar, the context is dropped rather than forced in, and the assistant says it does not have that information.", ROSE),
        ("6", "Assemble", "Build the prompt",
         "Surviving passages are de-duplicated per document and per section, labelled with their source, capped, and fenced as untrusted data — never as instructions.", PURP),
        ("7", "Answer", "Generate with citations",
         "Llama-3.3-70B writes the answer from that evidence only. Each claim carries a citation to document, section and page. The sources panel shows what was actually used.", GREEN),
    ]
    for n, t, sub, desc, col in steps:
        lines = simpleSplit(desc, F, 8.6, fw - 176)
        bh = max(38, 16 + len(lines) * 11.4)
        rbox(c, x, y - bh, 152, bh, FILL, col, 1.0)
        badge(c, x + 16, y - bh / 2, n, col, 7.4)
        c.setFillColor(INK)
        c.setFont(FB, 9.6)
        c.drawString(x + 32, y - bh / 2 + 3, t)
        c.setFillColor(MUTE)
        c.setFont(F, 6.5)
        c.drawString(x + 32, y - bh / 2 - 7, sub)
        c.setFillColor(BODY)
        c.setFont(F, 8.6)
        yy = y - 16
        for ln in lines:
            c.drawString(x + 168, yy, ln)
            yy -= 11.4
        if n != "7":
            arrow(c, x + 76, y - bh - 1, x + 76, y - bh - 9, col, 1.1, 3.2)
        y -= bh + 11

    y -= 14
    y = h2(c, x, y, "Why the answers can be trusted")
    y = bullets(c, x, y, [
        "Every claim carries a citation — document, section and page — and the passages used are shown to the reader.",
        "Retrieved text is fenced as data, never as instructions, so a document cannot hijack the assistant by containing the words \"ignore your rules\".",
        "If the documents do not contain the answer, the assistant abstains instead of guessing.",
        "Tenant isolation is enforced inside the search, not by filtering results afterwards — the wrong tenant's passage never enters the ranking.",
    ], fw)
    c.showPage()


# ══════════════════════════ PAGE 7 — voice ══════════════════════════

def p7(c):
    w, h = page(c, "Pipeline 3 — The Speech Loop", "What happens between you stopping and it answering")
    x, y = 42, h - 86
    fw = w - 84

    y = para(c, x, y, "The voice service runs a continuous loop. The hard part is not transcription — it is knowing when "
                      "your turn ended, and not leaving silence while it thinks.", fw, 9.6, 14)
    y -= 16

    stages = [
        ("Capture", "20 ms audio frames", ACC),
        ("Detect", "is this speech?", ACC),
        ("Caption", "streaming ASR", PURP),
        ("Endpoint", "have they finished?", ROSE),
        ("Transcribe", "accurate final text", PURP),
        ("Answer", "same RAG as chat", ACC2),
        ("Speak", "phrase-by-phrase TTS", GREEN),
    ]
    bw = (fw - 6 * 8) / 7
    for i, (t, s, col) in enumerate(stages):
        bx = x + i * (bw + 8)
        boxed(c, bx, y - 48, bw, 48, t, simpleSplit(s, F, 6.0, bw - 8), FILL, col, INK, 8.0, 6.0)
        if i < 6:
            arrow(c, bx + bw + 1, y - 24, bx + bw + 7, y - 24, col, 1.0, 3.0)
    # loop-back arrow
    arrow(c, x + fw - bw / 2, y - 50, x + bw / 2, y - 50, MUTE, 0.9, 3.2, dash=[2, 2])
    label(c, x + fw / 2, y - 60, "barge-in: if you start talking, playback stops immediately and the loop returns to listening", MUTE, 6.4, F, center=True)
    y -= 82

    y = h2(c, x, y, "The two parallel tracks")
    tw = (fw - 14) / 2
    rbox(c, x, y - 86, tw, 86, HexColor('#F4F4FC'), PURP, 1.1)
    label(c, x + 14, y - 18, "LEAD  —  starts immediately", INK, 9.2, FB)
    label(c, x + 14, y - 34, "A short, context-appropriate filler phrase is chosen by", BODY, 7.6)
    label(c, x + 14, y - 45, "heuristic — no model call — and spoken in under 50 ms.", BODY, 7.6)
    label(c, x + 14, y - 56, "It covers the retrieval time so there is no dead air.", BODY, 7.6)
    label(c, x + 14, y - 72, "Never carries facts. It cannot be wrong.", PURP, 7.4, FB)

    rbox(c, x + tw + 14, y - 86, tw, 86, FILL, ACC2, 1.1)
    label(c, x + tw + 28, y - 18, "GROUNDED  —  arrives behind it", INK, 9.2, FB)
    label(c, x + tw + 28, y - 34, "The real answer runs the full retrieval pipeline against", BODY, 7.6)
    label(c, x + tw + 28, y - 45, "your documents, then streams into speech phrase by", BODY, 7.6)
    label(c, x + tw + 28, y - 56, "phrase as the sentences complete.", BODY, 7.6)
    label(c, x + tw + 28, y - 72, "Every factual statement comes from here.", ACC2, 7.4, FB)
    y -= 104

    y = h2(c, x, y, "Turn-taking, in settings you can actually tune")
    y = kv_table(c, x, y, [
        ("Complete", "550 ms of silence ends your turn when the sentence sounds finished — a fast, natural reply."),
        ("Default", "700 ms of silence in the ordinary case."),
        ("Incomplete", "1300 ms when you trail off mid-thought, so a pause to think is not mistaken for the end."),
        ("Barge-in", "About 160 ms of your speech stops playback — low enough to interrupt easily, high enough to ignore its own echo."),
        ("Speech out", "Piper synthesises each phrase; edges are faded and controlled pauses inserted (160 ms at sentence ends, 90 ms at clauses) so the joins are inaudible."),
    ], fw, kw=76, size=8.8)
    y -= 4
    y = para(c, x, y, "Two speech-to-text models run in tandem: a streaming model for live captions, and Parakeet-TDT for the "
                      "final transcript that actually feeds the language model. Live captions can be approximate; the text "
                      "the model reasons over cannot.", fw, 8.9, 12.6, MUTE)
    c.showPage()


# ══════════════════════════ PAGE 8 — security ══════════════════════════

def p8(c):
    w, h = page(c, "Security and Trust Model", "What is enforced, and where it is enforced")
    x, y = 42, h - 88
    fw = w - 84

    y = h2(c, x, y, "Five layers")
    layers = [
        ("Perimeter", "The whole system runs on one machine with no runtime egress. Model weights are baked into the images, so it works with the network cable unplugged. Public access, when enabled, is an outbound-only Cloudflare Tunnel behind an identity gate — no inbound ports opened.", ACC),
        ("Identity", "Username/password sessions with signed tokens, roles and per-user tenant assignment. Admins manage users, see the audit log and usage. The voice WebSocket validates the same session signature before accepting audio.", ACC2),
        ("Tenant isolation", "Each vertical pack is a namespace applied inside the retrieval call itself. Both the dense and the keyword path enforce it, including the transcript indexes. The reserved 'default' tenant cannot be used to grant blanket cross-tenant access.", ROSE),
        ("Prompt integrity", "Retrieved passages are wrapped and presented to the model as untrusted data. Instructions embedded in an uploaded document are treated as text to be reported, not commands to be obeyed. Reasoning traces are stripped before display.", PURP),
        ("Egress control", "The export gateway scans any content on its way out for national IDs, card numbers (Luhn-verified), API keys, private keys, tokens, emails, phones and IPs, then returns a severity ranking and a masked version.", GREEN),
    ]
    for t, d, col in layers:
        lines = simpleSplit(d, F, 8.7, fw - 132)
        bh = max(34, 14 + len(lines) * 11.5)
        c.setFillColor(col)
        c.rect(x, y - bh, 3.0, bh, stroke=0, fill=1)
        c.setFillColor(INK)
        c.setFont(FB, 9.2)
        c.drawString(x + 12, y - 14, t)
        c.setFillColor(BODY)
        c.setFont(F, 8.7)
        yy = y - 13
        for ln in lines:
            c.drawString(x + 124, yy, ln)
            yy -= 11.5
        y -= bh + 10

    y -= 4
    y = h2(c, x, y, "Failure behaviour — what it does when it is unsure")
    y = bullets(c, x, y, [
        "No relevant passage clears the gate → the assistant says it does not have that information. It does not fill the gap.",
        "Chunking loses more than 2 % of a document → the clever path is discarded and the proven splitter runs instead.",
        "A speech model hits a fatal GPU fault → the health endpoint returns 503, the container exits and restarts with a fresh context. No silent degradation to CPU.",
        "A subsystem is unavailable → it is reported. The system is built so that a broken component is loud, not quietly wrong.",
    ], fw)
    y -= 6

    y = h2(c, x, y, "Auditability")
    y = para(c, x, y, "Answers carry citations. Uploads, deletions, logins and exports are written to an audit table. "
                      "A 52-question golden evaluation suite runs before a release and reports pass rate per vertical, "
                      "so a retrieval regression is caught by a number rather than by a user.", fw, 9.0, 12.6)
    c.showPage()


# ══════════════════════════ PAGE 9 — code map + ops ══════════════════════════

def p9(c):
    w, h = page(c, "Where the Code Lives, and How It Runs", "A map for anyone opening the repository")
    x, y = 42, h - 88
    fw = w - 84

    y = h2(c, x, y, "Code map")
    rows = [
        ("backend/app/rag/", "The answer engine: retrieval, ranking, prompts, personas, gating."),
        ("  index.py", "FAISS and BM25 indexes; tenant namespace enforcement lives here."),
        ("  advanced.py", "The seven-stage answer pipeline, end to end."),
        ("  chunking/", "Document splitting, structure detection and the coverage guard."),
        ("  intent.py", "Semantic intent classifier — small talk, refusal, real question."),
        ("backend/app/transcribe/", "Live transcription WebSocket and the Silent Assistant fact-checker."),
        ("backend/app/docgen/", "Document Studio: templates, themes, PDF and PPTX renderers, image generation."),
        ("backend/app/boardroom/", "Multi-speaker session capture, diarisation and analysis."),
        ("backend/app/core/", "Config, database, auth and RBAC, audit log, export gateway."),
        ("voice/app/session.py", "The whole speech loop: listening, endpointing, lead phrases, barge-in, speaking."),
        ("voice/app/adapters/", "Speech-to-text, text-to-speech and LLM streaming adapters."),
        ("frontend/", "React app. packs.ts maps each subdomain to its tenant, persona and theme."),
        ("docker-compose.yml", "The five services, their models, ports and GPU assignment."),
        ("eval/", "52-question golden suite plus chunk-coverage and voice end-to-end tests."),
    ]
    for k, v in rows:
        c.setFillColor(ACC2 if not k.startswith("  ") else MUTE)
        c.setFont(FCB if not k.startswith("  ") else FC, 7.4)
        c.drawString(x + (0 if not k.startswith("  ") else 8), y, k.strip())
        c.setFillColor(BODY)
        c.setFont(F, 8.6)
        c.drawString(x + 168, y, v)
        y -= 13.4
    y -= 8

    y = h2(c, x, y, "The five containers")
    svcs = [
        ("trtllm", "TensorRT-LLM serving Llama-3.3-70B-Instruct NVFP4 on an OpenAI-compatible endpoint.", ACC),
        ("backend", "FastAPI — documents, retrieval, transcription, generation, identity."),
        ("voice", "FastAPI — the real-time speech loop."),
        ("ollama", "Local embedding model."),
        ("frontend", "Nginx serving the React app and routing /api and /voice."),
    ]
    bw = (fw - 4 * 8) / 5
    for i, s in enumerate(svcs):
        boxed(c, x + i * (bw + 8), y - 54, bw, 54, s[0],
              simpleSplit(s[1], F, 5.8, bw - 12)[:4], FILL, ACC2 if i else ACC, INK, 8.4, 5.8)
    y -= 72

    y = h2(c, x, y, "Operations")
    y = bullets(c, x, y, [
        "One command starts everything: docker compose up -d. Public access is a separate profile, off by default.",
        "Health checks on the backend and voice services detect fatal GPU faults and restart the container with a clean CUDA context.",
        "Model weights ship inside the images and the runtime is set to offline mode — no download at start-up, no dependency on a registry.",
        "Data lives in named Docker volumes: the knowledge base, the model cache and the embedding store survive a rebuild.",
        "Releases are gated on the golden evaluation suite; a drop in pass rate blocks the change.",
    ], fw)
    y -= 6

    rbox(c, x, y - 64, fw, 64, FILL3, AMBER, 0.9)
    label(c, x + 14, y - 18, "A note on the language model", INK, 9.0, FB)
    for i, ln in enumerate(simpleSplit(
            "EchoMind is model-agnostic. The language model is reached through a single OpenAI-compatible endpoint "
            "setting, so it can be changed without touching application code. This document specifies "
            "nvidia/Llama-3.3-70B-Instruct-FP4, which fits comfortably in the 128 GB unified memory of the GB10.",
            F, 8.0, fw - 28)):
        label(c, x + 14, y - 33 - i * 11.0, ln, BODY, 8.0)
    c.showPage()


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    c = canvas.Canvas(OUT, pagesize=A4)
    c.setTitle("EchoMind Enterprise — Architecture Brief v2")
    c.setAuthor("Ajace AI")
    c.setSubject("Architecture, features and pipelines")
    for fn in (p1, p2, p3, p4, p5, p6, p7, p8, p9):
        fn(c)
    c.save()
    print("wrote", OUT, os.path.getsize(OUT), "bytes")


main()
