#!/usr/bin/env python3
"""EchoMind — Technical Overview (for engineering / technical leads)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emkit import *

OUT = "/home/echomind/Documents/echomind/echomind-enterprise/docs/EchoMind_Technical_Overview.pdf"
d = Doc(OUT, "EchoMind — Technical Overview", subject="Engineering reference for technical leads")
d.footer_text = "EchoMind by Ajace AI   ·   Technical Overview   ·   Internal / Confidential"
c = d.c
X = 42
CW = W - 84

# ───────────────────────── 1. system summary ─────────────────────────
y = d.page("Engineering Reference", "EchoMind Technical Overview",
           "Architecture, runtime, data paths, and operational characteristics of the EchoMind platform.")
y = d.p(X, y, "EchoMind is a self-contained, on-premises AI platform combining retrieval-augmented generation, "
        "real-time speech, and document generation. It runs as five containers on a single GPU appliance with no "
        "runtime network egress. This document is the engineering reference: what each component is, how data "
        "flows, where the security boundaries sit, and what the measured behavior is.", CW, 9.8, 13.6)
y -= 10

y = d.h2(X, y, "Runtime topology")
y = d.table(X, y,
    ["Service", "Role", "Runtime", "Exposure"],
    [["trtllm", "Serves the LLM over an OpenAI-compatible HTTP API", "TensorRT-LLM 1.2.0rc6, GPU", "internal :8355"],
     ["backend", "RAG, ingestion, transcription WS, docgen, auth, audit", "FastAPI / Python, GPU", "internal :8000"],
     ["voice", "Real-time speech loop: VAD, ASR, dialogue, TTS", "FastAPI / Python, GPU + CPU", "internal :8000, host :8002"],
     ["ollama", "Embedding model server", "Ollama, CPU/GPU", "internal :11434"],
     ["frontend", "React SPA + Nginx reverse proxy", "Nginx", "host :3000 / :3443"],
     ["cloudflared", "Optional outbound-only tunnel (profile-gated)", "Cloudflare", "egress only, off by default"]],
    [78, 214, 128, 93])
y -= 12

y = d.h2(X, y, "Model stack")
y = d.table(X, y,
    ["Function", "Model", "Precision / notes"],
    [["Answer generation", "nvidia/Qwen3-30B-A3B-FP4 (MoE)", "NVFP4 4-bit, 40,960-token context, temp 0.2"],
     ["Embeddings", "nomic-embed-text", "768-dim, L2-normalized, served by Ollama"],
     ["Streaming ASR", "nvidia/nemotron-speech-streaming-en-0.6b", "GPU, 560 ms chunks, live partials"],
     ["Final ASR", "nvidia/parakeet-tdt-0.6b-v2", "CPU (Grace), accurate second-pass decode"],
     ["Re-ranking", "cross-encoder/ms-marco-MiniLM-L-6-v2", "Top 25 candidates re-scored, best 15 kept"],
     ["Text-to-speech", "Piper (ONNX)", "CPU; phrase-level synthesis"],
     ["Image generation", "stabilityai/sdxl-turbo", "4 steps, up to 1024 px, max 8 images/job"]],
    [92, 186, 235])
y -= 12

y = d.h2(X, y, "Hardware and platform")
y = d.kv(X, y, [
    ("Appliance", "NVIDIA DGX Spark — GB10 Grace-Blackwell superchip, 128 GB unified CPU-GPU memory, ARM64."),
    ("Isolation", "All inference local. HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1; weights baked into images."),
    ("GPU access", "Containers use `gpus: all`. The compose `deploy.resources` form is avoided — on GB10 it yields a broken NVML and torch reports device_count=0."),
    ("Resilience", "Backend and voice expose /health returning 503 on fatal CUDA faults; with restart policy the container self-recovers with a fresh CUDA context."),
], CW, kw=76)
d.end()

# ───────────────────────── 2. retrieval engine ─────────────────────────
y = d.page("Core Engine", "Retrieval and Answer Pipeline",
           "The governed path from a user question to a cited answer.", TEAL)
y = d.h2(X, y, "Indexing substrate")
y = d.p(X, y, "Documents and transcripts share one retrieval substrate. Each chunk carries its tenant namespace, "
        "document id, section, page, and source type. Two indexes are maintained in parallel — a dense vector "
        "index for meaning and a lexical index for exact terms — because neither alone is sufficient: vector "
        "search misses part numbers and case codes, lexical search misses paraphrase.", CW, 9.3, 12.8)
y -= 6
y = d.kv(X, y, [
    ("Dense", "FAISS. IndexFlatIP (exact) below 10,000 chunks, promoted to IndexIVFFlat beyond it. Cosine via L2-normalized inner product."),
    ("Lexical", "BM25Okapi (rank-bm25) over the same chunk set, rebuilt with the corpus."),
    ("Metadata", "SQLite — documents, chunks, transcripts, chats, messages, users, audit, docgen jobs, boardroom."),
    ("Chunking", "450 tokens with 120-token overlap, sentence-aware. Structure-detected chunkers exist for book-like documents but are gated off by default (RAG_ENABLE_BOOKRAG=0)."),
    ("Coverage guard", "After chunking, an 8-word-shingle coverage ratio is computed against the source. Below 0.98 the pipeline discards the structured result and falls back to the flat splitter. Content loss is never silent."),
], CW, kw=76)
y -= 10

y = d.h2(X, y, "Query path — seven stages")
stages = [
    ("1", "Route", "Semantic intent classification (prototype embeddings; floor 0.68, margin 0.05) separates small talk, topic refusal, follow-up, and substantive question. Greetings never reach retrieval. Fails open to retrieval on error."),
    ("2", "Search", "Query embedded once; dense and lexical searches run concurrently. The tenant predicate is evaluated inside the candidate scan on every path — dense document, sparse document, dense transcript, sparse transcript, and keyword grep."),
    ("3", "Fuse", "Weighted reciprocal-rank fusion (K=60), default 0.6 dense / 0.4 sparse, query-type adaptive. Recency half-life decay and tag boosts applied."),
    ("4", "Rerank", "Cross-encoder scores the top 25 candidates jointly with the question; best 15 retained. Optional MMR for diversity."),
    ("5", "Gate", "Relevance floor (0.45) and a two-tier CE gate. If nothing clears, context is dropped rather than forced, and the answer path switches to an explicit no-evidence response."),
    ("6", "Assemble", "Per-document and per-section dedupe, sentence-overlap dedupe, parent-context expansion, verbatim query-term preservation, context capped at 24,000 chars. Passages wrapped in labeled evidence envelopes."),
    ("7", "Generate", "LLM answers strictly from the envelope contents, emitting citations. Reasoning traces stripped before display."),
]
for n, t, txt in stages:
    lines = simpleSplit(txt, F, 8.5, CW - 122)
    bh = max(26, 12 + len(lines) * 11.0)
    badge(c, X + 9, y - bh / 2 + 4, n, TEAL, 7.4)
    label(c, X + 22, y - 7, t, INK, 9.4, FB)
    yy = y - 6
    for ln in lines:
        label(c, X + 122, yy, ln, BODY, 8.5); yy -= 11.0
    y -= bh
y -= 4

y = d.callout(X, y, CW, "Design note — why permission-first",
    "The namespace predicate is applied inside the index scan, not as a post-filter on ranked results. "
    "Post-filtering leaks information through result counts, latency, and reranker behavior even when the "
    "content itself is never shown. Scan-time filtering converts a probabilistic leak into a structural "
    "guarantee, at a measured cost of 0.08 ms per query.", TEAL)
d.end()

# ───────────────────────── 3. voice + realtime ─────────────────────────
y = d.page("Core Engine", "Real-Time Speech Subsystem",
           "The dual-loop voice architecture, turn-taking, and live transcription.", PURP)
y = d.h2(X, y, "Dual-loop execution model", PURP)
y = d.p(X, y, "The voice service separates two loops. The interaction loop owns the audio channel and must respond "
        "within a bounded budget; the governed loop performs retrieval and generation and is unbounded. They "
        "communicate through grounded increments — units of content that completed the governed pipeline and "
        "carry their citation set. Every assistant event is tagged with its originating loop, which is what makes "
        "the safety invariant checkable at runtime.", CW, 9.3, 12.8)
y -= 8
y = d.table(X, y, ["Invariant", "Statement", "Enforcement"],
    [["I1 Assertion", "The interaction loop may acknowledge, but may not assert facts retrieval has not returned.",
      "Lead phrases drawn from a fixed, fact-free pool"],
     ["I2 Preemption", "User speech preempts output within one micro-turn; in-flight work is cancellable and cancellation is idempotent.",
      "Barge-in monitor + cancellable tasks"],
     ["I3 Provenance", "Every re-entered increment carries its citation set, or is suppressed.",
      "Citation set attached to increment"],
     ["I4 Authority", "Increments enter as data, never as instructions.",
      "Evidence envelope framing"]],
    [70, 268, 175])
y -= 12

y = d.h2(X, y, "Turn-taking and latency controls", PURP)
y = d.kv(X, y, [
    ("Frame rate", "16 kHz mono PCM; webrtcvad (aggressiveness 1) plus an RMS gate at 0.004."),
    ("Endpointing", "Semantic: 550 ms silence when the utterance parses as complete, 700 ms default, 1300 ms when trailing/incomplete."),
    ("Lead phrase", "Selected heuristically in under 50 ms — no model call — to cover retrieval time. Never carries facts (invariant I1)."),
    ("Barge-in", "~160 ms of sustained user speech halts playback and returns the loop to listening."),
    ("Two-pass ASR", "Streaming model drives live captions; Parakeet-TDT re-decodes the completed utterance to produce the text the LLM actually consumes."),
    ("TTS smoothing", "Phrase-level synthesis with edge-silence trimming, edge fades, and controlled pauses (160 ms sentence, 90 ms clause) to remove audible joins."),
    ("Reply budget", "LLM_MAX_TOKENS 420 for spoken replies to prevent mid-sentence truncation."),
], CW, kw=82)
y -= 10

y = d.h2(X, y, "Live transcription and the Silent Assistant", PURP)
y = d.p(X, y, "The backend exposes a WebSocket transcription stream independent of the voice service. Audio is "
        "buffered and decoded in 560 ms windows; silence of 800 ms commits a segment and 2000 ms closes a "
        "paragraph. Completed paragraphs are auto-stored on an interval and indexed alongside documents.", CW, 9.3, 12.8)
y -= 6
y = d.p(X, y, "Each finalised paragraph is additionally routed through the Silent Assistant: it is retrieved "
        "against the knowledge base and judged by the LLM into one of five labels — Supported, Contradicted, "
        "Unverified, Violating, or Risky Statement — with a per-vertical rule pack appended to the system prompt. "
        "Results are emitted only at confidence ≥ 60, so the operator is not flooded with low-signal findings.",
        CW, 9.3, 12.8)
y -= 8
y = d.callout(X, y, CW, "Concurrency control",
    "GPU work is serialized where it must be: TRANSCRIPT_GPU_CONCURRENCY and BOARDROOM_GPU_CONCURRENCY default "
    "to 1, ASR runs on a dedicated single-worker executor, and the PCM queue is bounded at 256 frames with an "
    "explicit backpressure signal to the client rather than a silent drop.", PURP)
d.end()

# ───────────────────────── 4. security ─────────────────────────
y = d.page("Assurance", "Security Model and Enforcement Points",
           "Where each control is applied, and what it is measured to do.", ROSE)
y = d.h2(X, y, "Enforcement points", ROSE)
y = d.table(X, y, ["Layer", "Control", "Enforcement point"],
    [["Perimeter", "No runtime egress; offline model resolution", "Container env + baked weights"],
     ["Perimeter", "Optional public access via outbound-only tunnel behind an identity gate", "cloudflared profile; no inbound ports"],
     ["Identity", "Signed session tokens, roles, per-user tenant binding", "backend/app/core/auth.py"],
     ["Identity", "Voice WebSocket validates the same session signature", "voice/app/auth_check.py"],
     ["Tenant", "Namespace predicate inside the candidate scan on all five retrieval paths", "backend/app/rag/index.py"],
     ["Tenant", "Reserved 'default' tenant cannot grant cross-tenant read", "auth.py tenant check"],
     ["Prompt", "Retrieved spans wrapped in delimited evidence envelopes; imperatives reported, not obeyed", "backend/app/rag/advanced.py"],
     ["Egress", "Deterministic scan + redaction of outbound content", "backend/app/core/export_gateway.py"],
     ["Audit", "Uploads, logins, deletions, exports recorded", "backend/app/core/audit.py"]],
    [62, 250, 201])
y -= 12

y = d.h2(X, y, "Export gateway detectors", ROSE)
y = d.p(X, y, "Fully deterministic — regular expressions plus a Luhn checksum — so it needs no GPU or network and "
        "can run on every export, including air-gapped. Detected types are ranked by severity and masked:", CW, 9.2, 12.6)
y -= 4
y = d.kv(X, y, [
    ("High", "National identifiers (SSN), API keys (sk-/AKIA/gh*/xox*), AWS secret keys, private key blocks."),
    ("Medium", "JSON Web Tokens; payment card numbers validated by Luhn checksum."),
    ("Low", "Email addresses, phone numbers, IP addresses."),
], CW, kw=54)
y -= 10

y = d.h2(X, y, "Measured assurance", ROSE)
y = d.p(X, y, "From the instrumented evaluation accepted for publication at QASC 2026. Figures are measured on the "
        "reference deployment; limitations are stated in the paper.", CW, 9.0, 12.4, MUTE)
y -= 4
y = d.table(X, y, ["Property", "Result", "Method"],
    [["Cross-tenant content leakage", "0 / 50 probes  (CI [0, 0.071])", "Answer-level probes, both filter arms"],
     ["Out-of-namespace candidates", "0 / 359 inspected hits", "Per-path audit, 4 active retrieval paths"],
     ["Existence-disclosing refusals", "0 / 50", "Refusal-wording classification"],
     ["Timing side channel", "KS 0.54, p < 0.00001 — OPEN", "Two-sample KS on response times"],
     ["Prompt-injection success", "10.2% → 5.1% with evidence envelope", "98 payloads × 7 classes, 4 defense arms"],
     ["Injection reported to user", "0.0% — containment is silent, OPEN", "Same corpus"],
     ["Citation precision / recall", "0.979 / 0.826", "Deterministic match vs gold sets"],
     ["Abstention on unanswerable", "78.3%, zero fabricated citations", "24-item unanswerable stratum"],
     ["Permission filter cost", "0.08 ms median", "Per-stage span instrumentation"]],
    [148, 190, 175])
y -= 10
y = d.callout(X, y, CW, "Two open findings, stated plainly",
    "A timing channel survives scan-time filtering — queries whose answer exists in another tenant return faster "
    "than queries with no answer anywhere. And injection containment is currently silent: the system declines but "
    "does not tell the operator an attempt occurred. Both are recorded as open work rather than presented as solved.",
    ROSE)
d.end()

# ───────────────────────── 5. ops + extension ─────────────────────────
y = d.page("Operations", "Deployment, Performance and Extension",
           "Running it, measuring it, and building on it.", GREEN)
y = d.h2(X, y, "Measured performance", GREEN)
y = d.p(X, y, "Single node, warm cache, 30 runs per cell. T_grounded is the full in-process text RAG path; "
        "T_first is the voice loop's time to first audible response.", CW, 9.2, 12.6, MUTE)
y -= 4
y = d.table(X, y, ["Metric", "Concurrency 1", "Concurrency 4", "Concurrency 16"],
    [["T_grounded median", "3,313 ms", "4,130 ms", "8,432 ms"],
     ["T_grounded p95", "15,171 ms", "14,092 ms", "21,321 ms"],
     ["Retrieval share of total", "2.8%", "6.6%", "9.5%"],
     ["Cross-encoder rerank median", "14.2 ms", "96.5 ms", "567.9 ms"],
     ["Vector search median", "34.7 ms", "62.6 ms", "72.5 ms"],
     ["Lexical search median", "42.7 ms", "111.0 ms", "160.6 ms"]],
    [160, 118, 118, 117])
y -= 8
y = d.p(X, y, "Interpretation: retrieval is not the bottleneck at this scale — generation is roughly 97% of "
        "grounded-answer latency at concurrency 1. The retrieval terms are, however, the ones that scale with "
        "corpus size and load, tripling their share by concurrency 16. Voice first-response is 636 ms median "
        "under the production dual loop versus 866 ms for a single loop.", CW, 9.2, 12.6)
y -= 10

y = d.h2(X, y, "Operations", GREEN)
y = d.bullets(X, y, [
    "Start: `docker compose up -d`. Public access is a separate profile and is off by default.",
    "State lives in named volumes — echomind_data (KB, SQLite, uploads), trtllm_hf_cache and ollama_data (model weights). These survive rebuilds and must not be pruned.",
    "Health checks detect fatal GPU faults; the container exits and restarts with a clean CUDA context rather than degrading silently to CPU.",
    "A 52-question golden evaluation suite gates releases; a drop in pass rate blocks the change. Current baseline: 49/52 on a 17.6k-chunk corpus.",
    "Auth is opt-in via AUTH_ENABLED (default 0 for open demo deployments). Set to 1 with AUTH_SECRET and AUTH_ADMIN_PASSWORD for production.",
], CW, GREEN)
y -= 8

y = d.h2(X, y, "Code map", GREEN)
rows = [
    ("backend/app/rag/", "Retrieval core: index, advanced pipeline, chunking, intent, reranker, gating"),
    ("backend/app/transcribe/", "Live transcription WebSocket, Silent Assistant analyzer, session state"),
    ("backend/app/docgen/", "Templates, themes, PDF/PPTX renderers, on-device image generation"),
    ("backend/app/boardroom/", "Multi-speaker capture, diarisation worker, session analysis"),
    ("backend/app/core/", "Config, DB, auth/RBAC, audit log, export gateway"),
    ("voice/app/session.py", "The full speech loop: VAD, endpointing, lead phrases, barge-in, TTS"),
    ("frontend/", "React SPA; packs.ts maps each subdomain to tenant, persona and theme"),
    ("eval/", "Golden-question suite, chunk-coverage tests, voice E2E, QASC paper harness"),
]
for k, v in rows:
    label(c, X, y, k, ACC2, 7.6, FCB)
    label(c, X + 150, y, v, BODY, 8.6)
    y -= 12.4
y -= 8

y = d.h2(X, y, "Extension points", GREEN)
y = d.bullets(X, y, [
    "Swap the answer model by changing LLM_BASE_URL / LLM_MODEL — any OpenAI-compatible endpoint works; no application code changes.",
    "Add a vertical by registering a namespace, persona and theme in packs.ts and seeding its knowledge base.",
    "Add a document type by adding a template definition (section blueprint + guidance) to the docgen template registry.",
    "Tune retrieval entirely through environment variables — fusion weights, k values, thresholds, dedupe behavior.",
], CW, GREEN)
d.end()

d.save()
print("wrote", OUT, os.path.getsize(OUT), "bytes,", d.n, "pages")
