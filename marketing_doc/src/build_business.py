#!/usr/bin/env python3
"""EchoMind — Business & Sales Overview (non-technical)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emkit import *

OUT = "/home/echomind/Documents/echomind/echomind-enterprise/marketing_doc/EchoMind_Business_Overview.pdf"
d = Doc(OUT, "EchoMind — Business & Sales Overview", subject="Non-technical overview for marketing and sales")
d.footer_text = "EchoMind by Ajace AI   ·   Business & Sales Overview   ·   Internal / Confidential"
c = d.c
X, CW = 42, W - 84

# ── 1 ────────────────────────────────────────────────────────────────
y = d.page("Marketing & Sales", "EchoMind in Plain Language",
           "What we sell, who buys it, and why they choose it over the alternatives.")
y = d.p(X, y, "EchoMind is a private AI assistant that lives inside the customer's own building. It answers "
        "questions from their documents, listens to and understands their meetings, holds a real spoken "
        "conversation, and writes finished business documents. Nothing is sent to the cloud — the whole system "
        "runs on one appliance the customer owns and controls.", CW, 10.2, 14.4)
y -= 12

y = d.h2(X, y, "The one-sentence pitch")
y = d.callout(X, y, CW, "Say this first",
    "\"EchoMind is ChatGPT for your own confidential documents and meetings — except it physically cannot leak "
    "them, because it never connects to the internet. It runs on a single box in your office, cites its sources, "
    "and tells you when it doesn't know.\"", ACC, 10.0, 13.4)
y -= 14

y = d.h2(X, y, "The problem we solve")
y = d.bullets(X, y, [
    "Knowledge is scattered. Policies sit in one system, contracts in another, and the most valuable context — what was actually said in a meeting — is never written down at all.",
    "Public AI tools are banned for real work. Legal, clinical, financial and government teams cannot paste confidential material into a cloud chatbot, so AI stops at the office door.",
    "Generic AI invents answers. A confident, unsourced answer is worse than no answer in a regulated workflow — it creates liability instead of removing it.",
    "Expertise does not scale. One compliance officer, one senior lawyer, one experienced clinician cannot be in every room at once.",
], CW)
y -= 10

y = d.h2(X, y, "What the customer gets")
items = [
    ("Answers they can trust", "Every answer is built only from their own content and shows its sources. If the answer isn't there, it says so."),
    ("A record of every meeting", "Conversations are transcribed live, checked against policy as they happen, and become searchable afterwards."),
    ("A voice they can talk to", "Hands-free spoken access, natural enough to interrupt — for clinicians, field engineers and drivers."),
    ("Finished documents", "Reports, memos, minutes and decks generated in minutes, in their template and their brand."),
    ("Complete data control", "Runs disconnected. No third-party processor, no training on their data, no data residency question."),
    ("One box, many products", "A single deployment can serve legal, clinical and finance teams with fully separated knowledge."),
]
colw = (CW - 16) / 2
ytop = y
for i, (t, txt) in enumerate(items):
    px = X + (i % 2) * (colw + 16)
    py = ytop - (i // 2) * 62
    label(c, px, py, t, ACC, 9.6, FB)
    d.p(px, py - 13, txt, colw, 8.4, 11.0)
y = ytop - 62 * 3 - 4

y = d.h2(X, y, "Proof, not adjectives")
y = d.statrow(X, y - 4, CW, [
    ("0.98", "citation precision — answers point at the right source"),
    ("78%", "of unanswerable questions correctly refused"),
    ("0 / 50", "cross-tenant leaks in isolation testing"),
    ("0.6 s", "to first spoken response in conversation"),
], ACC)
y = d.p(X, y - 4, "These are measured figures from an evaluation accepted for publication at an international "
        "peer-reviewed conference (QASC 2026) — not marketing estimates. That paper is a sales asset: very few "
        "vendors in this space have independently reviewed evidence.", CW, 8.8, 12.2, MUTE)
d.end()

# ── 2 ────────────────────────────────────────────────────────────────
y = d.page("Marketing & Sales", "Who Buys It, and What They Buy",
           "Target segments, the products they see, and the trigger that starts the conversation.", BLUE)
y = d.h2(X, y, "Five packaged products from one platform", BLUE)
y = d.p(X, y, "The same engine is sold as five industry products. Each has its own knowledge base, its own "
        "subject-matter persona, its own terminology and its own branding — and no customer's data can reach "
        "another's.", CW, 9.4, 13.0)
y -= 6
y = d.table(X, y, ["Product", "Buyer", "The trigger that starts the deal"],
    [["EchoMind Health", "Clinical operations, CMIO, compliance",
      "Clinicians spend evenings on documentation; PHI cannot go to a cloud AI."],
     ["EchoMind Law", "Managing partner, GC, knowledge manager",
      "Associates re-read the same contracts; privilege rules forbid external tools."],
     ["EchoMind Bank", "Compliance, wealth ops, risk",
      "Advice calls must evidence suitability and disclosure; regulators require an audit trail."],
     ["EchoMind Meeting Rooms", "COO, chief of staff, PMO",
      "Decisions get lost; minutes are someone's evening job; nobody can search last quarter."],
     ["EchoMind Retail", "Sales enablement, customer experience",
      "Staff give inconsistent product, price and warranty answers on the floor."]],
    [110, 128, 273])
y -= 12

y = d.h2(X, y, "Qualifying questions", BLUE)
y = d.bullets(X, y, [
    "\"Is there information your team is not allowed to put into ChatGPT?\"  — if yes, we are the only category that works.",
    "\"How long does it take a new joiner to find the right policy?\"  — quantifies the search-time saving.",
    "\"Who writes the minutes, and how long after the meeting do they arrive?\"  — opens transcription and Boardroom.",
    "\"Have you had an incident where someone acted on out-of-date guidance?\"  — opens the Silent Assistant.",
    "\"Does your data have to stay in a specific country, or offline entirely?\"  — disqualifies most competitors instantly.",
], CW, BLUE)
y -= 10

y = d.h2(X, y, "Why we win", BLUE)
y = d.table(X, y, ["Alternative", "Their limitation", "Our answer"],
    [["Public AI chatbots", "Confidential data leaves the organization; no access control; invents answers",
      "Runs offline; cites sources; abstains when unsure"],
     ["Cloud enterprise AI suites", "Data processed by a third party; residency and retention questions; per-seat cost grows",
      "One owned appliance; no processor agreement needed"],
     ["Traditional enterprise search", "Returns a list of documents, not an answer; no voice; no generation",
      "Direct cited answers, plus speech and document output"],
     ["Meeting transcription tools", "Transcribes but does not understand or check; cloud-hosted",
      "Live compliance checking against the customer's own policy"],
     ["Build it yourself", "12–18 months, scarce ML talent, and the security work is the hard part",
      "Deployed and measured today, with published evidence"]],
    [104, 216, 191])
y -= 12

y = d.h2(X, y, "Deployment models", BLUE)
y = d.kv(X, y, [
    ("On-premises", "The standard sale. One appliance installed in the customer's rack or office. No connectivity required."),
    ("Edge / field", "The same unit deployed at a remote site, clinic or vessel that has poor or prohibited connectivity."),
    ("Private cloud", "For customers who prefer their own tenancy in a controlled environment rather than physical hardware."),
    ("Gated public", "An optional identity-controlled tunnel for demos and remote access — outbound only, off by default."),
], CW, kw=88, kcol=BLUE)
d.end()

# ── 3 ────────────────────────────────────────────────────────────────
y = d.page("Marketing & Sales", "Objection Handling and FAQ",
           "The questions that actually come up, and honest answers to them.", AMBER)
qa = [
    ("\"How is this different from ChatGPT?\"",
     "ChatGPT is a general assistant that runs in someone else's data center and answers from what it learned on "
     "the public internet. EchoMind runs in your building and answers only from your documents, with a citation "
     "on every claim. The difference customers feel most is that it refuses to guess."),
    ("\"Does our data train the model?\"",
     "No. There is no training on customer content and no telemetry. The system has no runtime connection to the "
     "internet — it works with the network cable unplugged."),
    ("\"What happens when it doesn't know?\"",
     "It says so. If no relevant passage clears the relevance gate, the assistant states that the information is "
     "not in the documents rather than producing a plausible answer. In testing it correctly refused 78% of "
     "genuinely unanswerable questions and never fabricated a source."),
    ("\"Can someone see another department's documents?\"",
     "No. Separation is enforced inside the search itself, not by filtering results afterwards, so another "
     "tenant's content never even enters the ranking. Independent testing found zero leaks across every "
     "retrieval path examined."),
    ("\"What if a document contains malicious instructions?\"",
     "Retrieved text is treated as evidence, never as commands. A document telling the assistant to 'ignore your "
     "rules and reveal everything' is reported as text, not obeyed. Our structural defense halved attack success "
     "versus a system that merely instructs the model to be careful."),
    ("\"How accurate is it, really?\"",
     "On measured evaluation: 0.98 citation precision and 0.83 citation recall. We publish the limitations too — "
     "that honesty is a differentiator with technical buyers, not a weakness."),
    ("\"What does it need to run?\"",
     "One NVIDIA DGX Spark appliance. No cloud subscription, no external API keys, no per-token costs. "
     "Five containers start with a single command."),
    ("\"Can it use our own document templates?\"",
     "Yes. Document Studio ships with 23 business templates across five visual themes, and customers can upload "
     "their own."),
]
for q, a in qa:
    y = d.h3(X, y, q, AMBER, 9.6)
    y = d.p(X, y, a, CW, 8.9, 12.0)
    y -= 8
y -= 2
y = d.callout(X, y, CW, "Say what we don't do",
    "Two things we state openly, because technical buyers verify: response timing can still reveal whether a "
    "restricted document exists (we are working on it), and when the system blocks a malicious document it "
    "currently does so silently rather than alerting the operator. Naming these builds more trust than "
    "claiming perfection.", AMBER)
d.end()

# ── 4 ────────────────────────────────────────────────────────────────
y = d.page("Marketing & Sales", "Messaging Toolkit",
           "Ready-to-use language for decks, web copy, and conversations.", GREEN)
y = d.h2(X, y, "Positioning statement", GREEN)
y = d.callout(X, y, CW, "For long-form use",
    "For organizations whose most valuable information is also their most confidential, EchoMind is a private AI "
    "assistant that turns documents and meetings into instant, cited answers — running entirely on hardware the "
    "organization owns. Unlike cloud AI services, which require sending confidential material to a third party, "
    "EchoMind operates fully disconnected and shows the source behind every answer.", GREEN, 9.4, 12.8)
y -= 12

y = d.h2(X, y, "Message by audience", GREEN)
y = d.table(X, y, ["Audience", "What they care about", "Lead with"],
    [["CEO / owner", "Risk and competitive advantage", "AI you can actually use on your real work, without the legal exposure."],
     ["CIO / CTO", "Integration, control, lock-in", "One appliance, no external dependency, model-agnostic, published evidence."],
     ["CISO / compliance", "Data leaving, auditability", "No egress, tenant isolation enforced in the query, full audit log, redaction on export."],
     ["Head of operations", "Time and consistency", "Minutes written automatically; nobody hunts for the current policy."],
     ["Clinical / legal lead", "Accuracy and liability", "Cited answers, live checking against your own guidance, it refuses to guess."],
     ["Finance", "Cost model", "Owned hardware, no per-seat or per-token billing that scales with success."]],
    [92, 138, 281])
y -= 12

y = d.h2(X, y, "Words to use, and words to avoid", GREEN)
colw = (CW - 16) / 2
ytop = y
label(c, X, ytop, "USE", GREEN, 8.4, FB)
d.bullets(X, ytop - 14, [
    "private, on-premises, air-gapped",
    "cited, grounded, traceable to source",
    "it tells you when it doesn't know",
    "your data never leaves the building",
    "measured, peer-reviewed evidence",
], colw, GREEN, 8.6, 11.4, 2.6)
label(c, X + colw + 16, ytop, "AVOID", ROSE, 8.4, FB)
d.bullets(X + colw + 16, ytop - 14, [
    "\"100% accurate\" — it isn't, and buyers test it",
    "\"fully autonomous\" — it assists, it doesn't decide",
    "\"replaces your lawyers/doctors\" — it supports them",
    "\"unhackable\" / \"zero risk\" — we publish open findings",
    "unexplained jargon: RAG, embeddings, cross-encoder",
], colw, ROSE, 8.6, 11.4, 2.6)
y = ytop - 14 - 5 * 14 - 12

y = d.h2(X, y, "Demo script — twelve minutes", GREEN)
y = d.kv(X, y, [
    ("0–2 min", "Upload one of their own documents live. The point is that setup is not a project."),
    ("2–5 min", "Ask a question only that document can answer. Click a citation to open the exact page."),
    ("5–7 min", "Ask something the documents do not cover. Show it refusing — this is the moment that sells."),
    ("7–9 min", "Switch to voice. Ask a question aloud, then interrupt it mid-answer to show it stops and listens."),
    ("9–11 min", "Run Document Studio: turn the conversation into a finished PDF report in their template."),
    ("11–12 min", "Unplug the network cable and repeat a question. Nothing changes. Close on that."),
], CW, kw=64, kcol=GREEN)
y -= 8
y = d.callout(X, y, CW, "The closing line",
    "\"Everything you just saw ran on that box. No internet, no third party, no data leaving this room.\"",
    GREEN, 9.6, 13.0)
d.end()

d.save()
print("wrote", OUT, os.path.getsize(OUT), "bytes,", d.n, "pages")
