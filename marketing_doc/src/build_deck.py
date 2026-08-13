#!/usr/bin/env python3
"""EchoMind — Pitch Deck (landscape slides)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emkit import *

OUT = "/home/echomind/Documents/echomind/echomind-enterprise/marketing_doc/EchoMind_Pitch_Deck.pdf"
d = Doc(OUT, "EchoMind — Pitch Deck", subject="The private AI assistant that can sit in the room", land=True)
c = d.c
PW, PH = d.pw, d.ph
X, CW = 54, PW - 108


def slide(title, kicker="", accent=ACC, sub=""):
    """Light slide with accent rule. Returns content top y."""
    d.n += 1
    c.setFillColor(WHT); c.rect(0, 0, PW, PH, stroke=0, fill=1)
    c.setFillColor(accent); c.rect(0, PH - 8, PW, 8, stroke=0, fill=1)
    y = PH - 46
    if kicker:
        label(c, X, y, kicker.upper(), accent, 8.6, FB); y -= 22
    c.setFillColor(INK)
    ts = 27.0
    while ts > 16 and c.stringWidth(title, FB, ts) > CW:
        ts -= 0.5
    c.setFont(FB, ts); c.drawString(X, y - ts * 0.78, title)
    y -= ts * 0.78 + 22
    if sub:
        c.setFillColor(MUTE); c.setFont(F, 11.5)
        for ln in simpleSplit(sub, F, 11.5, CW):
            c.drawString(X, y, ln); y -= 15
        y -= 4
    # footer
    label(c, X, 26, "EchoMind by Ajace AI", MUTE, 7.6)
    label(c, PW - X, 26, str(d.n), MUTE, 7.6, right=True)
    return y - 8


def persona_slide(kicker, title, sub, accent, who, pains, does, moment, quote):
    y = slide(title, kicker, accent, sub)
    colw = (CW - 26) / 2
    ytop = y - 4
    # left: who + pain
    label(c, X, ytop, "WHO", accent, 9.0, FB)
    yy = d.p(X, ytop - 18, who, colw, 11.6, 15.8)
    yy -= 14
    label(c, X, yy, "WHAT SLOWS THEM DOWN TODAY", accent, 9.0, FB)
    yy = d.bullets(X, yy - 20, pains, colw, accent, 11.0, 14.8, 6.0)
    # right: what it does
    rx = X + colw + 26
    label(c, rx, ytop, "WHAT ECHOMIND DOES", accent, 9.0, FB)
    ry = d.bullets(rx, ytop - 18, does, colw, accent, 11.0, 14.8, 6.0)
    ry -= 10
    ry = d.callout(rx, ry, colw, "The moment it earns its keep", moment, accent, 10.4, 13.8)
    # quote strip
    qy = min(yy, ry) - 22
    if qy < 78: qy = 78
    c.setFillColor(tint(accent, .09)); c.rect(X, qy - 44, CW, 44, stroke=0, fill=1)
    c.setFillColor(accent); c.rect(X, qy - 44, 4.0, 44, stroke=0, fill=1)
    label(c, X + 18, qy - 27, quote, INK, 12.6, FO)
    d.end()


# ═════════ 1 TITLE ═════════
d.n += 1
c.setFillColor(INK); c.rect(0, 0, PW, PH, stroke=0, fill=1)
c.setFillColor(ACC); c.rect(0, 0, PW, 10, stroke=0, fill=1)
label(c, X, PH - 92, "AJACE AI", ACC, 11, FB)
c.setFillColor(WHT); c.setFont(FB, 54); c.drawString(X, PH - 168, "EchoMind")
c.setFillColor(SOFT); c.setFont(F, 17)
c.drawString(X, PH - 200, "The private AI assistant that can sit in the room.")
c.setFillColor(HexColor('#7FA6B2')); c.setFont(F, 12)
for i, ln in enumerate(simpleSplit(
        "Answers from your own documents. Listens to your meetings. Speaks when spoken to. "
        "Runs entirely on one machine you own — with the network cable unplugged.", F, 12, CW * .72)):
    c.drawString(X, PH - 232 - i * 17, ln)
chips = ["On-premises", "Cited answers", "Full-duplex voice", "Multi-tenant", "Peer-reviewed"]
cx = X
for t in chips:
    cx += tag(c, cx, 96, t, ACC, 8.4, 8, 17) + 8
label(c, X, 56, "Internal / Confidential", HexColor('#5A7280'), 8)
d.end()

# ═════════ 2 PROBLEM ═════════
y = slide("AI stopped at the office door", "The problem", ROSE,
          "The organizations with the most to gain from AI are the ones least able to use it.")
items = [
    ("Confidential work is off-limits", "Legal, clinical, financial and government teams cannot paste real material into a cloud chatbot. So the AI everyone talks about is used for drafting emails, not for the work that matters."),
    ("Knowledge is scattered and silent", "Policies sit in one system, contracts in another — and the most valuable context of all, what was actually said in a meeting, is never written down."),
    ("Generic AI invents answers", "A confident, unsourced answer is not a smaller version of a correct one. In a regulated workflow it is a liability."),
    ("Expertise cannot be everywhere", "One compliance officer, one senior partner, one experienced clinician cannot sit in every room at once."),
]
colw = (CW - 30) / 2
for i, (t, txt) in enumerate(items):
    px = X + (i % 2) * (colw + 30)
    py = y - (i // 2) * 104
    c.setFillColor(ROSE); c.rect(px, py - 2, 26, 3, stroke=0, fill=1)
    label(c, px, py - 24, t, INK, 13.5, FB)
    d.p(px, py - 42, txt, colw, 10.0, 13.6)
y = y - 104 * 2 - 6
c.setFillColor(tint(ROSE, .08)); c.rect(X, y - 34, CW, 34, stroke=0, fill=1)
label(c, X + 16, y - 21, "The result: the more sensitive and valuable your knowledge, the less AI you are allowed to use on it.",
      INK, 11.5, FB)
d.end()

# ═════════ 3 SOLUTION ═════════
y = slide("Bring the AI to the data — not the data to the AI", "The solution", ACC,
          "EchoMind is a complete AI assistant on a single appliance inside the customer's own walls.")
cards = [
    ("Reads your documents", "Upload policies, contracts, manuals. Ask in plain language. Every answer cites its source — document, section and page.", TEAL),
    ("Hears your meetings", "Live transcription with a background check against your own policy, flagging what is unsupported or non-compliant as it is said.", BLUE),
    ("Talks with you", "A real spoken conversation — responds in about half a second and stops the moment you interrupt.", PURP),
    ("Writes your documents", "Turns a brief into a finished PDF or PowerPoint in your template — minutes, memos, reports, decks.", AMBER),
    ("Keeps tenants apart", "One box can serve legal, clinical and finance teams with knowledge that cannot cross between them.", ROSE),
    ("Never phones home", "No cloud, no third-party processor, no training on your data. It works with the cable unplugged.", GREEN),
]
cw3 = (CW - 2 * 22) / 3
for i, (t, txt, col) in enumerate(cards):
    px = X + (i % 3) * (cw3 + 22)
    py = y - (i // 3) * 116
    rbox(c, px, py - 100, cw3, 100, FILL, tint(col, .45), 1.0, 6)
    c.setFillColor(col); c.rect(px, py - 100, cw3, 3.5, stroke=0, fill=1)
    label(c, px + 14, py - 26, t, INK, 12.5, FB)
    d.p(px + 14, py - 44, txt, cw3 - 28, 9.4, 12.4)
d.end()

# ═════════ 4 THE BIG IDEA ═════════
y = slide("An assistant that can stay in the room", "The idea", PURP,
          "Because it runs on-premises and speaks naturally, EchoMind is not confined to a browser tab. "
          "It can be installed where the work actually happens.")
places = [
    ("IN THE MEETING ROOM", "A permanent participant", ["hears the whole discussion", "flags a wrong statement live", "writes the minutes and actions"], BLUE),
    ("ON THE DESK", "A personal assistant", ["answers from the case or chart", "drafts the document", "never sends data outside"], PURP),
    ("AT THE FRONT LINE", "A support co-pilot", ["listens to the customer call", "surfaces the right answer", "checks what was promised"], AMBER),
    ("IN THE FIELD", "An offline expert", ["works with no connectivity", "hands-free voice", "full manual library on board"], GREEN),
]
pw2 = (CW - 3 * 18) / 4
for i, (kick, t, bullets_, col) in enumerate(places):
    px = X + i * (pw2 + 18)
    rbox(c, px, y - 168, pw2, 168, WHT, col, 1.4, 7)
    c.setFillColor(col); c.rect(px, y - 34, pw2, 34, stroke=0, fill=1)
    label(c, px + pw2 / 2, y - 22, kick, WHT, 8.4, FB, center=True)
    label(c, px + pw2 / 2, y - 58, t, INK, 12.5, FB, center=True)
    yy = y - 82
    for b in bullets_:
        c.setFillColor(col); c.circle(px + 16, yy + 3.2, 2.0, stroke=0, fill=1)
        for ln in simpleSplit(b, F, 9.4, pw2 - 34):
            label(c, px + 24, yy, ln, BODY, 9.4); yy -= 12.4
        yy -= 4
y = y - 168 - 22
label(c, X, y, "What makes this possible", ACC, 9.6, FB)
y -= 16
row = [("No cloud dependency", "it can live in a locked room or a vehicle"),
       ("Speaks and listens", "it works without a keyboard, hands-free"),
       ("Knows when to speak", "silent by default; surfaces only high-confidence findings"),
       ("Separated knowledge", "the room's tenant sees only the room's content")]
cw4 = (CW - 3 * 16) / 4
for i, (t, s) in enumerate(row):
    px = X + i * (cw4 + 16)
    label(c, px, y, t, INK, 9.8, FB)
    d.p(px, y - 13, s, cw4, 8.8, 11.4)
d.end()

# ═════════ 5–8 PERSONAS ═════════
persona_slide("Persona 1", "The Lawyer", "Private legal associate — on the desk and in the client meeting.", PURP,
    "Partners, associates and in-house counsel working across contracts, regulation, precedent and privileged client material.",
    ["Re-reading the same agreements to find one clause",
     "Precedent buried across matters nobody can search",
     "Public AI tools forbidden — privilege and confidentiality",
     "Client calls where a commitment is made and never recorded"],
    ["Ask across the whole contract library and get the clause, with the document, section and page",
     "Contract-review memos drafted to the firm's template",
     "Client meetings transcribed, with commitments and deadlines extracted",
     "Live flagging of one-sided, missing or risky clauses against the firm's playbook"],
    "Mid-negotiation, counsel asks what the liability cap was in the 2023 agreement — and gets the clause and page in seconds, without the file leaving the building.",
    "\"It reads every contract we have ever signed, and it can tell me where it got the answer.\"")

persona_slide("Persona 2", "The Doctor", "Clinical assistant — at the desk, at the bedside, hands-free.", GREEN,
    "Physicians, nurses and clinical governance teams working under protocol, with PHI that cannot leave the institution.",
    ["Guidelines change; recalling the current one takes time",
     "Documentation eats the evening",
     "PHI rules out every cloud AI service",
     "A drug interaction or contraindication can be missed under pressure"],
    ["Ask protocols and formulary in plain language, with the source shown",
     "Hands-free voice for use while examining or scrubbed in",
     "Consultations transcribed and turned into a structured visit note",
     "Live flagging of interactions, contraindications and dosing against the reference set"],
    "During a consultation the assistant quietly flags that the proposed medication interacts with the patient's existing prescription — while the patient is still in the room.",
    "\"It is documentation support and a second pair of eyes — and the data never leaves the hospital.\"")

persona_slide("Persona 3", "The Manager", "Meeting facilitator — a permanent participant in the room.", BLUE,
    "Executives, chiefs of staff, PMO and operations leaders who live in meetings and own the follow-through.",
    ["Decisions are made and then disputed a month later",
     "Minutes are somebody's evening job, and arrive late",
     "Actions have no owner and no due date",
     "Nobody can search what was said last quarter"],
    ["Sits in the room and transcribes the whole discussion",
     "Separates speakers and produces an analyzed report",
     "Extracts decisions, commitments and action items with owners",
     "Flags anything said that contradicts company policy",
     "Makes every past meeting searchable alongside documents"],
    "Someone asks what was agreed about the vendor in March. The answer comes back with the sentence, the speaker and the date — and the discussion moves on.",
    "\"Minutes stop being a chore, and last quarter stops being a black hole.\"")

persona_slide("Persona 4", "The Support Agent", "Customer care co-pilot — on the call, in real time.", AMBER,
    "Contact-center and customer-care teams answering product, policy, billing and warranty questions at speed.",
    ["New agents take months to learn the product",
     "Answers vary between agents — and some are wrong",
     "Hold time while the agent searches a wiki",
     "Promises made on calls that policy does not support"],
    ["Listens to the live call and surfaces the right answer from the real policy",
     "Every answer cited, so the agent can quote with confidence",
     "Flags a promise that contradicts warranty or pricing policy as it is made",
     "Produces the call summary and follow-up automatically",
     "Customer data never leaves the contact center"],
    "A new agent is asked an obscure warranty question. The correct answer, with the clause behind it, appears before the customer finishes the sentence.",
    "\"A first-week agent answers like a five-year veteran — and we can prove what was said.\"")

# ═════════ 9 HOW IT WORKS ═════════
y = slide("Three steps, then it just works", "How it works", TEAL,
          "No integration project, no data migration, no cloud account.")
steps = [("1", "Install", "One appliance, five containers, one command. No connectivity required.", TEAL),
         ("2", "Feed it", "Upload documents. Let it listen to meetings. It indexes both into one searchable knowledge base.", BLUE),
         ("3", "Use it", "Type, talk, or leave it running in the room. Every answer cites its source.", GREEN)]
sw = (CW - 2 * 26) / 3
for i, (n, t, txt, col) in enumerate(steps):
    px = X + i * (sw + 26)
    c.setFillColor(col); c.circle(px + 22, y - 22, 22, stroke=0, fill=1)
    c.setFillColor(WHT); c.setFont(FB, 20); c.drawCentredString(px + 22, y - 29, n)
    label(c, px + 56, y - 18, t, INK, 16, FB)
    d.p(px, y - 62, txt, sw, 10.4, 14.0)
    if i < 2:
        arrow(c, px + sw + 4, y - 22, px + sw + 20, y - 22, LINE, 2.0, 5.0)
y -= 128
label(c, X, y, "Under the hood — for the technical buyer", ACC2, 10.4, FB)
y -= 20
under = [("Hybrid retrieval", "Meaning search and exact-keyword search run together, then a second model re-reads the best passages against the question."),
         ("Grounded generation", "The answer model sees only the retrieved evidence, fenced as data — so a document cannot give it instructions."),
         ("Honest refusal", "If nothing relevant clears the gate, it says the answer is not in the documents rather than guessing."),
         ("Isolation in the query", "The tenant check runs inside the search itself, so another tenant's content never enters the ranking.")]
cw2 = (CW - 24) / 2
for i, (t, s) in enumerate(under):
    px = X + (i % 2) * (cw2 + 24)
    py = y - (i // 2) * 44
    c.setFillColor(ACC2); c.rect(px, py + 1.8, 3.4, 3.4, stroke=0, fill=1)
    label(c, px + 12, py, t, INK, 10.0, FB)
    d.p(px + 12, py - 13, s, cw2 - 12, 9.0, 11.6)
d.end()

# ═════════ 10 PROOF ═════════
y = slide("Measured, not claimed", "Proof", GREEN,
          "An instrumented evaluation of this system has been accepted for publication at an international "
          "peer-reviewed conference (QASC 2026).")
stats = [("0.98", "citation precision"), ("78%", "correct refusals on unanswerable questions"),
         ("0 / 50", "cross-tenant leaks"), ("0.6 s", "to first spoken response"),
         ("2×", "reduction in prompt-injection success")]
sw = CW / 5
for i, (big, lab) in enumerate(stats):
    px = X + i * sw
    c.setFillColor(GREEN); c.setFont(FB, 30); c.drawString(px, y - 30, big)
    c.setFillColor(MUTE); c.setFont(F, 9.4)
    for j, ln in enumerate(simpleSplit(lab, F, 9.4, sw - 14)):
        c.drawString(px, y - 48 - j * 12, ln)
y -= 96
colw = (CW - 26) / 2
label(c, X, y, "WHAT WE PROVED", GREEN, 8.6, FB)
d.bullets(X, y - 18, [
    "Answers point at the right source, and it refuses when it should.",
    "Tenant isolation held on every retrieval path audited.",
    "A structural defense halved prompt-injection success versus merely instructing the model.",
    "The dual-loop design is faster AND safer — removing the safety rule produced ungrounded claims on every turn.",
], colw, GREEN, 9.6, 12.8, 4.0)
label(c, X + colw + 26, y, "WHAT WE OPENLY REPORT AS OPEN", ROSE, 8.6, FB)
d.bullets(X + colw + 26, y - 18, [
    "Response timing can still reveal that a restricted document exists.",
    "Blocked injection attempts are contained but not yet surfaced to the operator.",
    "Human-panel evaluation of answer quality is not yet complete.",
], colw, ROSE, 9.6, 12.8, 4.0)
y -= 108
c.setFillColor(tint(GREEN, .08)); c.rect(X, y - 36, CW, 36, stroke=0, fill=1)
label(c, X + 16, y - 23, "Publishing the limitations is the point. Technical buyers verify claims — and almost no vendor in this category has any.",
      INK, 11.0, FB)
d.end()

# ═════════ 11 WHY US ═════════
y = slide("Why EchoMind wins", "Competition", ACC2,
          "Every alternative forces a trade-off between capability and control. We remove it.")
hdr = ["", "Runs offline", "Cites sources", "Refuses when unsure", "Live meeting check", "Tenant isolation"]
rows = [["Public AI chatbots", "No", "No", "No", "No", "No"],
        ["Cloud enterprise AI", "No", "Partial", "Partial", "No", "Partial"],
        ["Enterprise search", "Yes", "Links only", "n/a", "No", "Partial"],
        ["Meeting transcription", "No", "No", "n/a", "Transcript only", "No"],
        ["EchoMind", "Yes", "Yes", "Yes", "Yes", "Yes, in-query"]]
wds = [168, 108, 108, 122, 128, 124]
tw = sum(wds)
hh = 26
c.setFillColor(ACC2); c.rect(X, y - hh, tw, hh, stroke=0, fill=1)
cx = X
for h, wd in zip(hdr, wds):
    c.setFillColor(WHT); c.setFont(FB, 9.4); c.drawString(cx + 9, y - hh + 9, h)
    cx += wd
yy = y - hh
for ri, row in enumerate(rows):
    last = (ri == len(rows) - 1)
    rh = 30
    c.setFillColor(tint(ACC, .12) if last else (FILL if ri % 2 == 0 else WHT))
    c.rect(X, yy - rh, tw, rh, stroke=0, fill=1)
    cx = X
    for ci, (v, wd) in enumerate(zip(row, wds)):
        if ci == 0:
            c.setFillColor(INK); c.setFont(FB if last else F, 10.4)
        else:
            good = v in ("Yes", "Yes, in-query")
            c.setFillColor(GREEN if good else (AMBER if v.startswith("Partial") else MUTE))
            c.setFont(FB if good else F, 9.8)
        c.drawString(cx + 9, yy - rh + 11, v)
        cx += wd
    c.setStrokeColor(LINE); c.setLineWidth(0.5); c.line(X, yy - rh, X + tw, yy - rh)
    yy -= rh
yy -= 22
label(c, X, yy, "And one more thing nobody else has", ACC, 11.0, FB)
d.p(X, yy - 18, "A peer-reviewed measurement of the architecture, with the limitations published. In a market of "
    "unverifiable claims, evidence is the differentiator.", CW * .8, 10.4, 13.6)
d.end()

# ═════════ 12 BUSINESS ═════════
y = slide("How it is sold and deployed", "Commercial", SLATE,
          "One platform, five packaged products, four deployment shapes.")
colw = (CW - 30) / 2
label(c, X, y, "PACKAGED PRODUCTS", SLATE, 8.6, FB)
prods = [("EchoMind Health", "hospitals, clinics, clinical governance"),
         ("EchoMind Law", "firms and in-house legal teams"),
         ("EchoMind Bank", "advice, compliance, KYC/AML"),
         ("EchoMind Meeting Rooms", "boards, PMO, operations"),
         ("EchoMind Retail", "stores, sales floors, customer care")]
yy = y - 20
for t, s in prods:
    label(c, X, yy, t, INK, 10.4, FB)
    label(c, X + 172, yy, s, MUTE, 9.4)
    yy -= 19
label(c, X + colw + 30, y, "DEPLOYMENT SHAPES", SLATE, 8.6, FB)
deps = [("On-premises", "the standard sale — an appliance in their rack"),
        ("Edge / field", "remote sites, clinics, vehicles, vessels"),
        ("Private cloud", "their own controlled tenancy"),
        ("Air-gapped", "defense, government, critical infrastructure")]
ry = y - 20
for t, s in deps:
    label(c, X + colw + 30, ry, t, INK, 10.4, FB)
    label(c, X + colw + 30 + 118, ry, s, MUTE, 9.4)
    ry -= 19
y = min(yy, ry) - 18
label(c, X, y, "COMMERCIAL SHAPE", SLATE, 8.6, FB)
y -= 20
com = [("Owned hardware", "One appliance per deployment. No per-token cost that grows with usage."),
       ("No processor agreement", "Nothing is sub-processed, so the procurement path is dramatically shorter."),
       ("Expansion", "More departments = more tenants on the same box; more sites = more boxes."),
       ("Services", "Vertical content packs, template customization, and integration work.")]
cw4 = (CW - 3 * 18) / 4
for i, (t, s) in enumerate(com):
    px = X + i * (cw4 + 18)
    rbox(c, px, y - 74, cw4, 74, FILL, tint(SLATE, .35), 1.0, 5)
    label(c, px + 12, y - 22, t, INK, 10.4, FB)
    d.p(px + 12, y - 38, s, cw4 - 24, 9.0, 11.6)
d.end()

# ═════════ 13 ROADMAP / CLOSE ═════════
y = slide("Where it goes next", "Roadmap and ask", ACC)
label(c, X, y, "SHIPPING TODAY", GREEN, 8.6, FB)
d.bullets(X, y - 18, [
    "Knowledge Chat, Live Transcript, Silent Assistant, Voice, Document Studio, Boardroom",
    "Five vertical products with enforced tenant isolation",
    "Secure Export Gateway; role-based access and audit",
], CW * .46, GREEN, 9.8, 13.0, 4.0)
label(c, X + CW * .52, y, "NEXT", AMBER, 8.6, FB)
d.bullets(X + CW * .52, y - 18, [
    "Surface blocked injection attempts to the operator",
    "Constant-time responses to close the timing channel",
    "Human-panel evaluation and larger-corpus scale study",
    "Deeper meeting-room hardware integration",
], CW * .46, AMBER, 9.8, 13.0, 4.0)
y -= 118
c.setFillColor(INK); c.rect(X, y - 118, CW, 118, stroke=0, fill=1)
c.setFillColor(ACC); c.rect(X, y - 118, 5, 118, stroke=0, fill=1)
label(c, X + 26, y - 34, "The ask", ACC, 10.0, FB)
c.setFillColor(WHT); c.setFont(FB, 19)
c.drawString(X + 26, y - 62, "Give us one room, one department, and one week.")
c.setFillColor(SOFT); c.setFont(F, 11.4)
c.drawString(X + 26, y - 84, "Bring your own documents. We install, index, and demonstrate on your content —")
c.drawString(X + 26, y - 100, "then unplug the network cable and run the whole demo again.")
d.end()

d.save()
print("wrote", OUT, os.path.getsize(OUT), "bytes,", d.n, "slides")
