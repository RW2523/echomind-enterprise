#!/usr/bin/env python3
"""EchoMind — Architecture & Flow Document (landscape, diagram-led)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emkit import *

OUT = "/home/echomind/Documents/echomind/echomind-enterprise/marketing_doc/EchoMind_Architecture_Flows.pdf"
d = Doc(OUT, "EchoMind — Architecture & Flows", subject="System architecture and end-to-end flowcharts", land=True)
d.footer_text = "EchoMind by Ajace AI   ·   Architecture & Flows   ·   Internal / Confidential"
c = d.c
PW, PH = d.pw, d.ph
X, CW = 42, PW - 84


def caption(y, text, col=MUTE):
    label(c, X, y, text, col, 8.0, FO)
    return y - 12


# ══════════════ 1. SYSTEM CONTEXT ══════════════
y = d.page("Diagram 1", "System Context",
           "Who and what touches EchoMind, and where the trust boundary sits.", ACC, band=82)

# boundary
c.setStrokeColor(tint(ACC, .55)); c.setLineWidth(1.2); c.setDash([4, 3], 2)
c.roundRect(212, 92, 470, 330, 10, stroke=1, fill=0); c.setDash()
label(c, 447, 408, "CUSTOMER-OWNED TRUST BOUNDARY   ·   NO RUNTIME EGRESS", ACC, 7.4, FB, center=True)

# actors left
actors = [("Employees", ["chat, voice,", "documents"]), ("Meeting rooms", ["live audio,", "participants"]),
          ("Administrators", ["users, tenants,", "audit"])]
for i, (t, s) in enumerate(actors):
    node(c, 46, 330 - i * 80, 132, 54, t, s, FILL2, ACC)
    arrow(c, 178, 357 - i * 80, 212, 300, ACC)

# core
node(c, 240, 300, 190, 96, "EchoMind Platform",
     ["five containers, one appliance", "chat · voice · transcription", "generation · governance"], WHT, ACC, INK, 10.4, 7.0, 1.6)
node(c, 240, 190, 190, 82, "Local Knowledge Store",
     ["documents + transcripts", "one index, tenant-tagged", "SQLite metadata + audit"], FILL, GREEN, INK, 9.4, 6.8)
node(c, 240, 108, 190, 62, "On-device Models",
     ["LLM · ASR · TTS · embeddings", "weights baked in"], FILL, PURP, INK, 9.4, 6.8)
arrow(c, 335, 300, 335, 272, GREEN); arrow(c, 335, 190, 335, 170, PURP)

# right side: sources + optional egress
node(c, 470, 330, 180, 62, "Enterprise content", ["PDF · DOCX · PPTX · text", "uploaded by the customer"], FILL, ACC2)
arrow(c, 470, 361, 430, 348, ACC2)
node(c, 470, 240, 180, 62, "Generated output", ["PDF · PPTX reports", "transcripts, minutes"], FILL, AMBER)
arrow(c, 430, 271, 470, 271, AMBER)
node(c, 470, 150, 180, 62, "Secure Export Gateway", ["scan · classify · redact", "runs on every export"], WARM, AMBER, INK, 9.0, 6.8, 1.4)
arrow(c, 560, 240, 560, 212, AMBER)
arrow(c, 650, 181, 700, 181, ROSE)
node(c, 700, 150, 100, 62, "Outside world", ["only redacted", "content, by choice"], FILL, ROSE, INK, 8.6, 6.6)

label(c, 447, 78, "Everything inside the dashed boundary runs with the network cable unplugged. The only path out is a deliberate, scanned export.",
      MUTE, 8.0, FO, center=True)
d.end()

# ══════════════ 2. CONTAINER / RUNTIME ══════════════
y = d.page("Diagram 2", "Runtime Architecture",
           "The five containers, their responsibilities, and every internal call path.", ACC2, band=82)
LX, RX = 34, PW - 34
CX0, CWD = 116, PW - 34 - 116

def lane_label(y0, y1, name, sub):
    label(c, LX, y1 - 11, name, ACC2, 8.0, FB)
    yy = y1 - 21
    for ln in simpleSplit(sub, F, 6.4, CX0 - LX - 12):
        label(c, LX, yy, ln, MUTE, 6.4); yy -= 7.6
    c.setStrokeColor(HexColor('#E1ECF0')); c.setLineWidth(0.7); c.line(LX, y0 - 6, RX, y0 - 6)

b_cli = (392, 440); b_edg = (322, 372); b_app = (168, 302); b_dat = (74, 148)

lane_label(*b_cli, "CLIENTS", "browser only")
node(c, CX0, b_cli[0], CWD * .70, 48, "BROWSER — React SPA",
     ["Knowledge Chat · Live Transcript · Conversation · Document Studio · Boardroom"], FILL2, ACC, INK, 9.4, 6.8)
node(c, CX0 + CWD * .72, b_cli[0], CWD * .28, 48, "ADMIN", ["users · tenants · audit"], FILL, LINE, INK, 9.0, 6.8)

lane_label(*b_edg, "EDGE", "TLS + routing")
node(c, CX0, b_edg[0], CWD * .62, 50, "NGINX  (frontend container)",
     ["serves SPA · /api → backend · /voice → voice (WebSocket)"], FILL, ACC2, INK, 9.2, 6.8)
node(c, CX0 + CWD * .64, b_edg[0], CWD * .36, 50, "CLOUDFLARE TUNNEL  (optional)",
     ["outbound-only · identity gate · off by default"], WARM, AMBER, INK, 8.6, 6.6)

lane_label(*b_app, "APPLICATION", "business logic")
bkw = CWD * .58
bx, vx = CX0, CX0 + bkw + 14
vcw = CWD - bkw - 14
rbox(c, bx, b_app[0], bkw, 134, WHT, ACC2, 1.4)
label(c, bx + 12, b_app[1] - 15, "BACKEND — FastAPI", INK, 9.6, FB)
chips = [("Chat / RAG API", "route·search·cite"), ("Ingestion", "parse·chunk·embed"),
         ("Live Transcript", "WebSocket ASR"), ("Silent Assistant", "para. fact-check"),
         ("Boardroom", "diarize·analyze"), ("Document Studio", "PDF·PPTX·images"),
         ("Export Gateway", "scan·redact"), ("Auth·RBAC·Audit", "tenants·log")]
cw2 = (bkw - 24 - 3 * 6) / 4
for i, (t, s) in enumerate(chips):
    cx = bx + 12 + (i % 4) * (cw2 + 6)
    cy = b_app[1] - 62 - (i // 4) * 40
    node(c, cx, cy, cw2, 34, t, [s], FILL, LINE, INK, 6.9, 5.9)
rbox(c, vx, b_app[0], vcw, 134, WHT, PURP, 1.4)
label(c, vx + 12, b_app[1] - 15, "VOICE — FastAPI", INK, 9.6, FB)
vchips = [("VAD + endpoint", "turn detection"), ("Streaming STT", "live partials"),
          ("Final STT", "accurate text"), ("Lead phrase", "<50 ms"),
          ("Grounded reply", "calls backend"), ("TTS + barge-in", "speak · stop")]
cw3 = (vcw - 24 - 2 * 6) / 3
for i, (t, s) in enumerate(vchips):
    cx = vx + 12 + (i % 3) * (cw3 + 6)
    cy = b_app[1] - 62 - (i // 3) * 40
    node(c, cx, cy, cw3, 34, t, [s], HexColor('#F4F4FC'), HexColor('#CFD0F0'), INK, 6.9, 5.9)

lane_label(*b_dat, "MODELS & DATA", "local only")
iw = (CWD - 4 * 10) / 5
infs = [("TensorRT-LLM", ["Qwen3-30B-A3B", "NVFP4"], ACC),
        ("Ollama", ["nomic-embed-text", "768-dim"], GREEN),
        ("FAISS + BM25", ["dense + lexical", "tenant-tagged"], GREEN),
        ("SQLite + disk", ["metadata · audit", "uploads · outputs"], SLATE),
        ("Speech models", ["Nemotron · Parakeet", "Piper TTS"], PURP)]
for i, (t, ls, col) in enumerate(infs):
    node(c, CX0 + i * (iw + 10), b_dat[0], iw, 74, t, ls, FILL, col, INK, 8.6, 6.4)

# arrows
arrow(c, CX0 + CWD * .22, b_cli[0], CX0 + CWD * .22, b_edg[1], ACC2); badge(c, CX0 + CWD * .22 + 12, (b_cli[0] + b_edg[1]) / 2, 1)
arrow(c, CX0 + CWD * .55, b_cli[0], CX0 + CWD * .55, b_edg[1], PURP); badge(c, CX0 + CWD * .55 + 12, (b_cli[0] + b_edg[1]) / 2, 2, PURP)
arrow(c, CX0 + CWD * .18, b_edg[0], bx + bkw * .3, b_app[1], ACC2); badge(c, CX0 + CWD * .18 - 14, (b_edg[0] + b_app[1]) / 2, 3)
arrow(c, CX0 + CWD * .50, b_edg[0], vx + vcw * .5, b_app[1], PURP); badge(c, CX0 + CWD * .50 + 16, (b_edg[0] + b_app[1]) / 2, 4, PURP)
arrow(c, vx - 1, b_app[1] - 22, bx + bkw + 1, b_app[1] - 22, AMBER); badge(c, (vx + bx + bkw) / 2, b_app[1] - 36, 5, AMBER)
for i, (frac, col) in enumerate([(.15, ACC), (.38, GREEN), (.62, GREEN), (.86, SLATE)]):
    arrow(c, bx + bkw * frac, b_app[0], CX0 + i * (iw + 10) + iw / 2, b_dat[1], col)
arrow(c, vx + vcw * .5, b_app[0], CX0 + 4 * (iw + 10) + iw / 2, b_dat[1], PURP)
badge(c, (vx + vcw * .5 + CX0 + 4 * (iw + 10) + iw / 2) / 2 + 14, (b_app[0] + b_dat[1]) / 2, 6, PURP)

leg = [(1, "browser → nginx (HTTPS)", ACC2), (2, "microphone audio (WebSocket)", PURP),
       (3, "/api → backend", ACC2), (4, "/voice → voice service", PURP),
       (5, "voice asks backend for a grounded, cited answer", AMBER), (6, "local model + data access", PURP)]
lx = CX0
for n, t, col in leg:
    badge(c, lx + 5, 54, n, col, 5.2, 6.0); label(c, lx + 13, 51.6, t, MUTE, 6.3)
    lx += 13 + c.stringWidth(t, F, 6.3) + 18
d.end()

# ══════════════ 3. INGESTION FLOW ══════════════
y = d.page("Diagram 3", "Flow A — Document Ingestion",
           "From an uploaded file to a searchable, tenant-tagged passage. Runs once per document.", GREEN, band=82)
steps = [("Upload", ["PDF · DOCX", "PPTX · text"]), ("Extract", ["text + layout", "headings, pages"]),
         ("Detect type", ["structure", "classification"]), ("Chunk", ["450 tokens", "120 overlap"]),
         ("Coverage check", ["8-word shingles", "vs. source"]), ("Embed", ["768-dim vector", "per passage"]),
         ("Index", ["FAISS + BM25", "written together"]), ("Tag", ["namespace, doc,", "section, page"])]
bw = (CW - 7 * 12) / 8
for i, (t, s) in enumerate(steps):
    bx2 = X + i * (bw + 12)
    col = AMBER if t == "Coverage check" else GREEN
    node(c, bx2, 300, bw, 62, t, s, WARM if col == AMBER else FILL, col, INK, 8.0, 6.2)
    if i < 7:
        arrow(c, bx2 + bw + 1, 331, bx2 + bw + 11, 331, col, 1.1, 3.4)

# coverage guard branch
gx = X + 4 * (bw + 12) + bw / 2
xc = X + 3 * (bw + 12) + bw / 2     # chunk column centre
xe = X + 5 * (bw + 12) + bw / 2     # embed column centre
diamond(c, gx, 228, 118, 48, "coverage >= 0.98 ?", AMBER, WARM, 6.8)
arrow(c, gx, 300, gx, 254, AMBER)
# yes -> continue to Embed
c.setStrokeColor(GREEN); c.setLineWidth(1.1); c.line(gx + 59, 228, xe, 228)
arrow(c, xe, 228, xe, 298, GREEN)
label(c, (gx + 59 + xe) / 2, 234, "yes", GREEN, 7.0, FB, center=True)
# no -> back to Chunk with the flat splitter
c.setStrokeColor(ROSE); c.setLineWidth(1.1); c.line(gx - 59, 228, xc, 228)
arrow(c, xc, 228, xc, 298, ROSE)
label(c, (gx - 59 + xc) / 2, 234, "no", ROSE, 7.0, FB, center=True)
label(c, gx, 194, "Below 0.98 the structured result is discarded and the proven flat splitter runs instead — content loss is never silent.",
      MUTE, 7.2, FO, center=True)

y = 172
y = d.h2(X, y, "What each stage guarantees", GREEN, 10.5)
rows = [("Extract", "Headings, sections and page numbers survive, so a citation can point at a real place in the file."),
        ("Chunk", "450-token windows with 120-token overlap; a sentence spanning a boundary is never lost."),
        ("Coverage check", "Measures how much of the original text the chunks actually cover; below 98% the structured path is discarded."),
        ("Embed / Index", "Filed twice — vector index for meaning, lexical index for exact terms. Both are needed."),
        ("Tag", "Namespace written at index time, so nothing downstream has to remember to filter.")]
for k, v in rows:
    label(c, X, y, k, GREEN, 8.0, FB); label(c, X + 110, y, v, BODY, 8.4); y -= 13
d.end()

# ══════════════ 4. QUERY FLOW ══════════════
y = d.page("Diagram 4", "Flow B — Question to Cited Answer",
           "The governed retrieval path, including the abstention branch.", TEAL, band=82)
# row 1 — main path (fits inside X .. X+758)
node(c, X, 322, 96, 48, "Question", ["typed or spoken"], FILL2, ACC, INK, 8.4, 6.2)
arrow(c, X + 97, 346, X + 152, 346, ACC)
diamond(c, X + 213, 346, 118, 54, "substantive question?", TEAL, FILL, 6.6)
label(c, X + 213, 304, "no → small talk / refusal", MUTE, 6.4, F, center=True)
node(c, X + 149, 250, 128, 42, "Direct reply", ["never touches the KB"], FILL, MUTE, INK, 8.2, 6.1)
arrow(c, X + 213, 319, X + 213, 292, MUTE)
arrow(c, X + 272, 346, X + 288, 346, TEAL)
label(c, X + 280, 352, "yes", GREEN, 6.2, FB)

# parallel search
node(c, X + 288, 356, 118, 38, "Dense search", ["FAISS · meaning"], FILL, GREEN, INK, 8.0, 6.1)
node(c, X + 288, 310, 118, 38, "Lexical search", ["BM25 · exact"], FILL, GREEN, INK, 8.0, 6.1)
label(c, X + 347, 298, "both scoped to the tenant namespace INSIDE the scan", ROSE, 6.2, FB, center=True)
arrow(c, X + 407, 375, X + 422, 358, ACC2); arrow(c, X + 407, 329, X + 422, 340, ACC2)
node(c, X + 422, 326, 100, 42, "Fuse", ["weighted RRF", "0.6 / 0.4"], FILL, ACC2, INK, 8.3, 6.1)
arrow(c, X + 523, 347, X + 536, 347, ACC2)
node(c, X + 536, 326, 108, 42, "Re-rank", ["cross-encoder", "25 → best 15"], FILL, ACC2, INK, 8.3, 6.1)
arrow(c, X + 645, 347, X + 660, 347, ACC2)
diamond(c, X + 706, 347, 92, 54, "clears relevance gate?", ROSE, FILL, 6.2)

# gate branches
label(c, X + 706, 306, "no", ROSE, 6.4, FB, center=True)
node(c, X + 636, 250, 140, 42, "Abstain", ["\"not in the documents\"", "no citation invented"], FILL, ROSE, INK, 8.3, 6.1)
arrow(c, X + 706, 320, X + 706, 292, ROSE)
# yes -> route down and back to the assemble row
label(c, X + 718, 382, "yes", GREEN, 6.4, FB)
c.setStrokeColor(GREEN); c.setLineWidth(1.1)
c.line(X + 706, 374, X + 706, 400); c.line(X + 706, 400, X + 262, 400)
arrow(c, X + 262, 400, X + 262, 212, GREEN)

node(c, X + 198, 168, 128, 42, "Assemble", ["dedupe · envelopes", "cap 24k chars"], FILL, PURP, INK, 8.3, 6.1)
arrow(c, X + 327, 189, X + 344, 189, PURP)
node(c, X + 344, 168, 128, 42, "Generate", ["LLM, evidence only"], FILL, ACC, INK, 8.3, 6.1)
arrow(c, X + 473, 189, X + 490, 189, ACC)
node(c, X + 490, 168, 146, 42, "Cited answer", ["claim + document,", "section, page"], FILL2, ACC, INK, 8.5, 6.1)

y = 138
label(c, X, y, "Two structural properties worth noting:", INK, 9.0, FB); y -= 14
y = d.bullets(X, y, [
    "The tenant predicate runs inside the candidate scan on all five retrieval paths — not as a filter on ranked results — so another tenant's passage never enters the ranking at all.",
    "The relevance gate has a real 'no' branch. If nothing clears it the context is dropped and the assistant abstains, rather than being handed weak evidence and asked to do its best.",
], CW, TEAL, 8.6, 11.4, 3.0)
d.end()

# ══════════════ 5. VOICE FLOW ══════════════
y = d.page("Diagram 5", "Flow C — The Dual-Loop Speech Cycle",
           "How the assistant stays responsive without asserting anything it has not retrieved.", PURP, band=82)
lane(c, X, 250, CW, 120, "Interaction loop", PURP)
lane(c, X, 116, CW, 120, "Governed loop", ACC2)

label(c, X + 210, 358, "bounded latency — owns the audio channel", PURP, 6.6, FO)
seq = [("Mic", ["16 kHz frames"]), ("VAD", ["speech?"]), ("Partial ASR", ["live captions"]),
       ("Endpoint", ["turn ended?"]), ("Lead phrase", ["<50 ms, no facts"]), ("Speak", ["phrase-by-phrase"])]
bw = (CW - 5 * 14) / 6
for i, (t, s) in enumerate(seq):
    bx2 = X + 20 + i * (bw + 12)
    node(c, bx2, 276, bw - 8, 52, t, s, WHT, PURP, INK, 8.2, 6.2)
    if i < 5:
        arrow(c, bx2 + bw - 7, 302, bx2 + bw + 3, 302, PURP, 1.0, 3.2)

label(c, X + 20, 226, "unbounded — retrieval and generation", ACC2, 6.6, FO)
gseq = [("Final ASR", ["accurate text"]), ("Retrieve", ["same RAG as chat"]),
        ("Generate", ["grounded answer"]), ("Increment", ["+ citation set"])]
gbw = (CW - 200) / 4
for i, (t, s) in enumerate(gseq):
    bx2 = X + 20 + i * (gbw + 14)
    node(c, bx2, 142, gbw - 8, 52, t, s, WHT, ACC2, INK, 8.2, 6.2)
    if i < 3:
        arrow(c, bx2 + gbw - 7, 168, bx2 + gbw + 7, 168, ACC2, 1.0, 3.2)

# cross-loop arrows
arrow(c, X + 20 + 3 * (bw + 12) + bw / 2, 276, X + 20 + gbw / 2, 194, ACC2)
label(c, X + 150, 241, "turn ends → hand off to the governed loop", ACC2, 6.8, FB)
arrow(c, X + 20 + 3 * (gbw + 14) + gbw / 2 - 4, 194, X + 20 + 5 * (bw + 12) + bw / 2, 276, GREEN)
label(c, X + 470, 241, "grounded increment returns → spoken with its citations", GREEN, 6.8, FB)

# barge-in loop
mic_cx = X + 20 + (bw - 8) / 2
speak_cx = X + 20 + 5 * (bw + 12) + (bw - 8) / 2
c.setDash([3, 2], 2); c.setStrokeColor(ROSE); c.setLineWidth(1.0)
c.line(speak_cx, 330, speak_cx, 386); c.line(speak_cx, 386, mic_cx, 386); c.setDash()
arrow(c, mic_cx, 386, mic_cx, 330, ROSE)
label(c, X + CW / 2, 398, "BARGE-IN — user speaks → playback stops within ~160 ms, in-flight work cancelled, loop returns to listening",
      ROSE, 7.0, FB, center=True)

y = 100
cols = [("Why two loops", "A single loop must stay silent until the grounded answer is ready. That silence is what makes voice assistants feel broken."),
        ("The safety rule", "The interaction loop may acknowledge, but may never assert a fact retrieval has not returned. Lead phrases carry no content."),
        ("Measured effect", "636 ms to first audio with the dual loop vs 866 ms single-loop. Removing the safety rule produced ungrounded claims on 10 of 10 turns.")]
colw = (CW - 32) / 3
for i, (t, txt) in enumerate(cols):
    px = X + i * (colw + 16)
    label(c, px, y, t, PURP, 8.8, FB)
    d.p(px, y - 13, txt, colw, 8.2, 10.8)
d.end()

# ══════════════ 6. ASSURANCE FLOWS ══════════════
y = d.page("Diagram 6", "Flow D — Live Assurance, Generation and Egress",
           "Three supporting flows: the Silent Assistant, Document Studio, and the export gate.", AMBER, band=82)

# --- Silent assistant ---
label(c, X, 380, "SILENT ASSISTANT — runs alongside every meeting", BLUE, 9.0, FB)
sa = [("Speech", ["live audio"]), ("Paragraph", ["closed on 2 s silence"]), ("Retrieve", ["vs. knowledge base"]),
      ("Judge", ["LLM + domain rules"])]
bw, gap = 104, 14
for i, (t, s) in enumerate(sa):
    bx2 = X + i * (bw + gap)
    node(c, bx2, 314, bw, 46, t, s, FILL, BLUE, INK, 8.2, 6.2)
    if i < 3: arrow(c, bx2 + bw + 1, 337, bx2 + bw + gap - 3, 337, BLUE, 1.0, 3.2)
dcx = X + 4 * (bw + gap) + 53
diamond(c, dcx, 337, 106, 48, "confidence >= 60 ?", AMBER, WARM, 6.6)
arrow(c, X + 3 * (bw + gap) + bw + 1, 337, dcx - 55, 337, BLUE)
arrow(c, dcx + 53, 337, dcx + 69, 337, GREEN)
node(c, dcx + 69, 314, 158, 46, "Surface to the room",
     ["Supported · Contradicted · Unverified", "Violating · Risky Statement"], FILL, GREEN, INK, 8.2, 5.9)
label(c, dcx, 300, "below → stay silent", MUTE, 6.4, F, center=True)

# --- Document studio ---
label(c, X, 268, "DOCUMENT STUDIO — brief to finished file", AMBER, 9.0, FB)
ds = [("Brief", ["+ template choice"]), ("Blueprint", ["sections + guidance"]), ("Draft", ["LLM per section,", "grounded"]),
      ("Illustrate", ["SDXL-Turbo,", "on-device"]), ("Render", ["real PDF / PPTX"])]
bw2 = (CW - 4 * 14) / 5
for i, (t, s) in enumerate(ds):
    bx2 = X + i * (bw2 + 14)
    node(c, bx2, 200, bw2, 48, t, s, WARM, AMBER, INK, 8.2, 6.0)
    if i < 4: arrow(c, bx2 + bw2 + 1, 224, bx2 + bw2 + 11, 224, AMBER, 1.0, 3.2)
label(c, X, 186, "23 templates · 5 visual themes · customer templates can be uploaded", MUTE, 7.2, FO)

# --- Export gateway ---
label(c, X, 158, "SECURE EXPORT GATEWAY — the only way content leaves", ROSE, 9.0, FB)
node(c, X, 90, 128, 48, "Content to export", ["report, summary,", "transcript"], FILL, SLATE, INK, 8.2, 6.0)
arrow(c, X + 129, 114, X + 145, 114, SLATE)
node(c, X + 145, 90, 150, 48, "Deterministic scan",
     ["IDs · cards (Luhn) · keys", "tokens · emails · phones · IPs"], FILL, ROSE, INK, 8.2, 5.9)
arrow(c, X + 296, 114, X + 312, 114, ROSE)
diamond(c, X + 372, 114, 104, 48, "sensitive data found?", ROSE, FILL, 6.6)
arrow(c, X + 425, 114, X + 452, 114, AMBER)
node(c, X + 452, 90, 150, 48, "Rank + mask", ["severity high / medium / low", "redacted copy produced"], WARM, AMBER, INK, 8.2, 5.9)
arrow(c, X + 603, 114, X + 622, 114, GREEN)
node(c, X + 622, 90, 138, 48, "Released", ["risk summary +", "safe version"], FILL, GREEN, INK, 8.4, 6.0)
label(c, X + 372, 76, "no → released unchanged, still logged", MUTE, 6.4, F, center=True)
label(c, X, 66, "Regex + Luhn only — no GPU, no network — so it runs on every export, including fully air-gapped deployments.",
      MUTE, 7.4, FO)
d.end()

# ══════════════ 7. TENANT ISOLATION + DEPLOY ══════════════
y = d.page("Diagram 7", "Multi-Tenant Isolation and Deployment",
           "How one appliance serves several products without their data meeting.", ROSE, band=82)

label(c, X, 384, "TENANT ISOLATION — enforced inside the query, not after it", ROSE, 9.4, FB)
packs = [("Health", "health", GREEN), ("Law", "law", PURP), ("Meetings", "meetings", ACC),
         ("Retail", "retail", AMBER), ("Bank", "bank", BLUE)]
pw2 = (CW - 4 * 12) / 5
for i, (t, ns, col) in enumerate(packs):
    px = X + i * (pw2 + 12)
    node(c, px, 300, pw2, 62, "EchoMind " + t, ["subdomain · persona · theme", "namespace: " + ns], FILL, col, INK, 8.6, 6.2)
    arrow(c, px + pw2 / 2, 300, px + pw2 / 2, 268, col)
rbox(c, X, 214, CW, 54, FILL2, ROSE, 1.3)
label(c, X + CW / 2, 246, "SHARED RETRIEVAL ENGINE", INK, 9.4, FB, center=True)
label(c, X + CW / 2, 232, "every candidate scan evaluates the caller's namespace predicate before a passage can be ranked",
      BODY, 7.6, F, center=True)
label(c, X + CW / 2, 221, "measured: 0 out-of-namespace hits across 359 inspected candidates, on all four active paths",
      GREEN, 7.2, FB, center=True)

label(c, X, 200, "DEPLOYMENT MODELS", SLATE, 9.4, FB)
dep = [("On-premises", ["appliance in the", "customer's rack"], SLATE),
       ("Edge / field", ["remote site, clinic,", "vessel — no connectivity"], GREEN),
       ("Private cloud", ["customer-controlled", "tenancy"], BLUE),
       ("Air-gapped", ["fully disconnected,", "weights baked in"], ROSE)]
dw = (CW - 3 * 14) / 4
for i, (t, ls, col) in enumerate(dep):
    node(c, X + i * (dw + 14), 134, dw, 58, t, ls, FILL, col, INK, 8.8, 6.4)

label(c, X, 114, "OPERATIONAL PROPERTIES", ACC2, 9.4, FB)
ops = ["One command starts all five containers; public access is a separate profile, off by default.",
       "Model weights ship inside the images — no download at startup, no registry dependency.",
       "Health checks detect fatal GPU faults and restart with a clean CUDA context rather than degrading silently.",
       "Knowledge base, model cache and embedding store live in named volumes and survive rebuilds.",
       "A 52-question golden suite gates every release; a drop in pass rate blocks the change."]
yy = 100
colw = (CW - 20) / 2
for i, t in enumerate(ops):
    px = X + (i % 2) * (colw + 20)
    py = yy - (i // 2) * 22
    c.setFillColor(ACC2); c.rect(px, py + 1.6, 3.2, 3.2, stroke=0, fill=1)
    label(c, px + 10, py, t, BODY, 7.8)
d.end()

d.save()
print("wrote", OUT, os.path.getsize(OUT), "bytes,", d.n, "pages")
