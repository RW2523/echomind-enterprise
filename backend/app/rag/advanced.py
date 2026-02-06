from __future__ import annotations
from typing import List, Dict, AsyncIterator, Tuple, Optional
import json
import logging
import math
import re
from datetime import datetime, timezone, timedelta
from ..core.config import settings
from ..core.db import get_conn
from .index import index
from .llm import OpenAICompatChat

logger = logging.getLogger(__name__)
chat = OpenAICompatChat(settings.LLM_BASE_URL, settings.LLM_MODEL)

CONTEXT_WINDOW_VALUES = ("24h", "48h", "1w", "all")

def _parse_iso_date(created_at: Optional[str]) -> Optional[datetime]:
    if not created_at:
        return None
    try:
        return datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except Exception:
        return None

def _filter_hits_by_context_window(hits: List[Dict], context_window: str) -> List[Dict]:
    """Keep only hits whose document created_at falls within context_window (24h, 48h, 1w). 'all' = no filter."""
    if not context_window or context_window == "all" or not hits:
        return hits
    now = datetime.now(timezone.utc)
    if context_window == "24h":
        cutoff = now - timedelta(hours=24)
    elif context_window == "48h":
        cutoff = now - timedelta(hours=48)
    elif context_window == "1w":
        cutoff = now - timedelta(days=7)
    else:
        return hits
    filtered = []
    with get_conn() as conn:
        for h in hits:
            doc_id = (h.get("source") or {}).get("doc_id")
            if not doc_id:
                filtered.append(h)
                continue
            row = conn.execute("SELECT created_at FROM documents WHERE id = ?", (doc_id,)).fetchone()
            created_at = row[0] if row else None
            dt = _parse_iso_date(created_at)
            if dt is None:
                filtered.append(h)
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt >= cutoff:
                filtered.append(h)
    return filtered

# Max chars for parent chunk expansion; config overrides to limit context domination (RAG_PARENT_CONTEXT_MAX_CHARS).
def _parent_context_max_chars() -> int:
    return getattr(settings, "RAG_PARENT_CONTEXT_MAX_CHARS", 2400) or 2400

# Greetings / small talk: answer generally, no RAG
_GENERAL_PHRASES = frozenset({
    "hi", "hello", "hey", "hey there", "hi there", "howdy",
    "thanks", "thank you", "thanks!", "thank you!",
    "how are you", "how are you?", "what's up", "whats up", "sup",
    "good morning", "good afternoon", "good evening", "good night",
    "bye", "goodbye", "see you", "ok", "okay", "yes", "no",
    "good", "great", "cool", "nice", "perfect",
})

def _is_general_conversation(question: str) -> bool:
    t = question.strip().lower()
    if not t:
        return True
    if t in _GENERAL_PHRASES:
        return True
    words = t.split()
    if len(words) <= 2 and t not in _GENERAL_PHRASES:
        if "?" not in t and not any(w in t for w in ("what", "which", "when", "where", "who", "how", "why", "can", "does", "is", "are", "do")):
            return True
    return False

def _dedupe_best(items: List[Dict]) -> List[Dict]:
    best={}
    for it in items:
        cid=it["chunk_id"]
        if cid not in best or it["score"]>best[cid]["score"]:
            best[cid]=it
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)

RRF_K = 60  # reciprocal rank fusion constant


def _weighted_rrf(
    dense_hits_per_query: List[List[Dict]],
    sparse_hits_per_query: List[List[Dict]],
    k: int,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[Dict]:
    """Merge dense + sparse with weighted RRF. Higher dense_weight favors semantic similarity; sparse_weight favors keyword match. Improves precision/recall balance."""
    fused: Dict[str, Dict] = {}
    for hit_list in dense_hits_per_query:
        for rank, h in enumerate(hit_list):
            cid = h["chunk_id"]
            rrf = dense_weight * (1.0 / (RRF_K + rank))
            if cid not in fused:
                fused[cid] = {"chunk_id": cid, "rrf": 0.0, "dense_score": 0.0, "text": h["text"], "source": h["source"]}
            fused[cid]["rrf"] += rrf
            fused[cid]["dense_score"] = max(fused[cid]["dense_score"], h["score"])
    for hit_list in sparse_hits_per_query:
        for rank, h in enumerate(hit_list):
            cid = h["chunk_id"]
            rrf = sparse_weight * (1.0 / (RRF_K + rank))
            if cid not in fused:
                fused[cid] = {"chunk_id": cid, "rrf": 0.0, "dense_score": 0.0, "text": h["text"], "source": h["source"]}
            fused[cid]["rrf"] += rrf
    sorted_ids = sorted(fused.keys(), key=lambda cid: fused[cid]["rrf"], reverse=True)[:k]
    out = []
    max_rrf = fused[sorted_ids[0]]["rrf"] if sorted_ids else 1.0
    for cid in sorted_ids:
        f = fused[cid]
        score = f["dense_score"] if f["dense_score"] > 0 else min(1.0, 0.3 + 0.7 * (f["rrf"] / max_rrf))
        out.append({"chunk_id": cid, "score": score, "text": f["text"], "source": f["source"]})
    return out


def _reciprocal_rank_fusion(
    dense_hits_per_query: List[List[Dict]],
    sparse_hits_per_query: List[List[Dict]],
    k: int,
) -> List[Dict]:
    """Legacy equal-weight RRF (used when weighted RRF weights not configured)."""
    return _weighted_rrf(dense_hits_per_query, sparse_hits_per_query, k, dense_weight=0.5, sparse_weight=0.5)


def _get_doc_info_for_hits(hits: List[Dict]) -> Tuple[Dict[str, Optional[str]], Dict[str, dict]]:
    """Fetch created_at and meta_json for each unique doc_id in hits. Returns (doc_id -> created_at, doc_id -> meta)."""
    doc_ids = list({(h.get("source") or {}).get("doc_id") for h in hits if (h.get("source") or {}).get("doc_id")})
    created_map: Dict[str, Optional[str]] = {}
    meta_map: Dict[str, dict] = {}
    if not doc_ids:
        return created_map, meta_map
    with get_conn() as conn:
        for doc_id in doc_ids:
            row = conn.execute("SELECT created_at, meta_json FROM documents WHERE id = ?", (doc_id,)).fetchone()
            if row:
                created_map[doc_id] = row[0]
                try:
                    meta_map[doc_id] = json.loads(row[1]) if isinstance(row[1], str) else (row[1] or {})
                except Exception:
                    meta_map[doc_id] = {}
            else:
                created_map[doc_id] = None
                meta_map[doc_id] = {}
    return created_map, meta_map


def _apply_time_decay(hits: List[Dict], doc_created: Dict[str, Optional[str]], halflife_days: float) -> List[Dict]:
    """Multiply score by exp(-age_days/halflife). Recent docs rank higher; no hard cutoff. halflife_days=0 means no decay."""
    if halflife_days <= 0 or not hits:
        return hits
    now = datetime.now(timezone.utc)
    out = []
    for h in hits:
        doc_id = (h.get("source") or {}).get("doc_id")
        created = doc_created.get(doc_id) if doc_id else None
        dt = _parse_iso_date(created)
        if dt is None:
            out.append(h)
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = (now - dt).total_seconds() / 86400
        decay = 1.0 if age_days <= 0 else max(0.1, math.exp(-age_days / halflife_days))
        out.append({**h, "score": h["score"] * decay})
    return out


def _apply_tag_boost(hits: List[Dict], query: str, doc_meta: Dict[str, dict], factor: float) -> List[Dict]:
    """Boost score for transcript chunks whose tags overlap query terms. Improves recall when tags describe content."""
    if factor <= 0 or not query:
        return hits
    q_lower = query.lower()
    q_words = set(re.findall(r"[a-z0-9]{2,}", q_lower))
    out = []
    for h in hits:
        doc_id = (h.get("source") or {}).get("doc_id")
        meta = doc_meta.get(doc_id, {}) if doc_id else {}
        if meta.get("type") != "transcript":
            out.append(h)
            continue
        tags = meta.get("tags") or []
        if not tags:
            out.append(h)
            continue
        tag_set = set()
        for t in tags:
            tag_set.update(re.findall(r"[a-z0-9]{2,}", (t or "").lower()))
        overlap = len(q_words & tag_set) / max(1, len(q_words))
        boost = 1.0 + factor * overlap
        out.append({**h, "score": min(1.0, h["score"] * boost)})
    return out


def _is_authoritative(source: dict) -> bool:
    """True if chunk is from an uploaded document (PDF/DOCX/PPTX), not a transcript. Used for tie-break preference."""
    ft = (source or {}).get("filetype") or ""
    fn = (source or {}).get("filename") or ""
    if (fn.startswith("transcript_") and ft == "text") or ft == "text":
        return False
    return ft in ("pdf", "docx", "pptx", "txt")


def _prefer_authoritative_sort(hits: List[Dict]) -> List[Dict]:
    """Sort by (score desc, authoritative first). When scores are close, authoritative docs win (reduces transcript noise)."""
    return sorted(hits, key=lambda h: (-h["score"], 0 if _is_authoritative(h.get("source") or {}) else 1))


# --- Deterministic query expansion (pre-LLM): typos, quoted phrases, TOC variants ---
_COMMON_TYPOS = {"mathew": "matthew", "matthew": "matthew", "meriton": "merton", "outler": "outlier", "outlers": "outliers"}


def get_deterministic_query_variants(question: str) -> List[str]:
    """Pre-LLM query variants: fix typos, add quoted phrase for named concepts, add TOC/structure terms when relevant. Return list to prepend to LLM-generated queries."""
    if not (question or "").strip():
        return []
    q = question.strip()
    q_lower = q.lower()
    out: List[str] = []
    # Typo fixes: produce corrected query
    for wrong, right in _COMMON_TYPOS.items():
        if wrong in q_lower:
            out.append(q_lower.replace(wrong, right))
    # Quoted phrase for named concepts (e.g. "matthew effect" -> exact phrase search helps BM25)
    if "matthew" in q_lower and "effect" in q_lower:
        out.append('"matthew effect"')
    if "table of contents" in q_lower or "toc" in q_lower or "contents" in q_lower or "chapters" in q_lower or "chapter list" in q_lower:
        out.extend(["table of contents", "contents", "chapter"])
    return out


def is_toc_chapters_query(question: str) -> bool:
    """True if the user is asking for chapters, table of contents, or section list. Used for TOC guardrail and book retrieval tuning."""
    if not (question or "").strip():
        return False
    t = question.strip().lower()
    toc_phrases = (
        "table of contents", "toc", "list of chapters", "chapter list", "list the chapters",
        "what are the chapters", "chapters in", "contents of the book", "book structure",
        "section list", "list sections", "list of sections", "chapter titles", "section titles",
    )
    if any(p in t for p in toc_phrases):
        return True
    words = set(re.findall(r"[a-z0-9]+", t))
    if "chapters" in words and ("list" in words or "what" in words or "name" in words or "which" in words):
        return True
    if "contents" in words and ("list" in words or "table" in words):
        return True
    return False


def has_toc_signals_in_context(blocks: List[str]) -> bool:
    """True if the combined context blocks contain TOC-like signals (Contents, CHAPTER N, roman numerals list, etc.). Used to avoid hallucinating chapter lists."""
    combined = " ".join((b or "" for b in blocks)).lower()
    if "table of contents" in combined or "contents" in combined and ("chapter" in combined or "part " in combined):
        return True
    if re.search(r"chapter\s+[0-9]+", combined, re.I) or re.search(r"chapter\s+one|two|three|four|five|six|seven|eight|nine|ten", combined, re.I):
        return True
    if re.search(r"\b(i{1,3}|iv|v|vi{0,3}|ix|x|xi|xiv|xv)\s+[a-z]", combined):
        return True
    if "part i" in combined or "part ii" in combined or "part 1" in combined or "part 2" in combined:
        return True
    return False


# Intent classification for query rewriting: reduces noisy expansions by tailoring rewrite to query type.
QUERY_INTENT_SYSTEM = """Classify the user's question into exactly one intent:
- factual: asking for a specific fact, definition, number, or "what is X"
- procedural: how-to, steps, process, "how do I"
- exploratory: broad comparison, list, "what are the options", "tell me about"
- temporal: time-sensitive, "recent", "latest", "when did", "current"
Reply with only one word: factual, procedural, exploratory, or temporal."""

QUERY_REWRITE_BY_INTENT = {
    "factual": "Rewrite into 2 concise search queries that keep the same fact or definition focus. No extra topics. Return only the queries, one per line.",
    "procedural": "Rewrite into 2 search queries that capture steps or process (how-to). Return only the queries, one per line.",
    "exploratory": "Rewrite into 2 search queries that broaden to related concepts or options. Return only the queries, one per line.",
    "temporal": "Rewrite into 2 search queries that keep time/recent/latest focus. Return only the queries, one per line.",
}


async def _classify_intent(question: str) -> str:
    """Classify question intent for intent-aware rewriting. Falls back to 'exploratory' on failure."""
    try:
        out = await chat.chat(
            [{"role": "system", "content": QUERY_INTENT_SYSTEM}, {"role": "user", "content": question[:500]}],
            temperature=0.0,
            max_tokens=20,
        )
        intent = (out or "").strip().lower()
        if intent in QUERY_REWRITE_BY_INTENT:
            return intent
    except Exception:
        pass
    return "exploratory"


async def generate_queries(q: str) -> List[str]:
    """Produce 1–3 alternative search queries. When RAG_INTENT_REWRITE is on, use intent-specific rewrite to reduce noisy expansions."""
    use_intent = getattr(settings, "RAG_INTENT_REWRITE", False)
    if use_intent:
        intent = await _classify_intent(q)
        sys = QUERY_REWRITE_BY_INTENT.get(intent, QUERY_REWRITE_BY_INTENT["exploratory"])
        try:
            txt = await chat.chat([{"role": "system", "content": sys}, {"role": "user", "content": q[:600]}], temperature=0.2, max_tokens=120)
            variants = []
            for line in txt.splitlines():
                line = re.sub(r"^\s*[-\d\).]+\s*", "", line).strip()
                if line and line.lower() != q.lower():
                    variants.append(line)
            out = [q] + variants[:3]
            return out[:4]
        except Exception:
            pass
    # Fallback: original behavior (fewer variants to reduce noise).
    sys = "Rewrite the question into 2 alternative search queries (synonyms or key terms only). Return only the list, one per line."
    txt = await chat.chat([{"role": "system", "content": sys}, {"role": "user", "content": q}], temperature=0.2, max_tokens=120)
    variants = []
    for line in txt.splitlines():
        line = re.sub(r"^\s*[-\d\).]+\s*", "", line).strip()
        if line:
            variants.append(line)
    out = [q] + [v for v in variants if v.lower() != q.lower()]
    return out[:4]


async def _rerank_hits(question: str, hits: List[Dict], top_n: int) -> List[Dict]:
    """Optional LLM rerank: score each chunk 0-10 for relevance to the question, then reorder. Keeps top_n. Adds latency."""
    if not hits or top_n <= 0:
        return hits
    # Batch prompt: list chunks and ask for scores (one line per score, same order). Reduces round-trips.
    prompt_lines = [f"Question: {question}\n"]
    for i, h in enumerate(hits):
        snippet = (h.get("text") or "")[:400].replace("\n", " ")
        prompt_lines.append(f"[{i}] {snippet}")
    prompt_lines.append("\nScore each [i] 0-10 for relevance. Reply with one integer per line, in order 0..n-1.")
    try:
        reply = await chat.chat(
            [{"role": "system", "content": "You score relevance of each excerpt to the question. Reply with only the scores, one per line, same order as the excerpts."}, {"role": "user", "content": "\n".join(prompt_lines)}],
            temperature=0.0,
            max_tokens=min(200, len(hits) * 4),
        )
        scores_list = []
        for line in (reply or "").splitlines():
            line = line.strip()
            m = re.search(r"\b(\d+)\b", line)
            if m:
                scores_list.append(min(10, max(0, int(m.group(1)))))
        if len(scores_list) >= len(hits):
            scored = [(hits[i], scores_list[i] / 10.0) for i in range(len(hits))]
            scored.sort(key=lambda x: -x[1])
            return [{"chunk_id": h["chunk_id"], "score": s, "text": h["text"], "source": h["source"]} for h, s in scored[:top_n]]
    except Exception as e:
        logger.warning("Rerank failed: %s", e)
    return hits[:top_n]


async def retrieve(question: str, k: int, context_window: str = "all") -> List[Dict]:
    """Hybrid retrieve: deterministic + LLM query expansion, dense + sparse, weighted RRF, optional time-decay and tag boost, optional rerank. context_window still applied as hard filter when not 'all'."""
    llm_qs = await generate_queries(question)
    det_qs = get_deterministic_query_variants(question)
    seen_lower: set = set()
    qs: List[str] = []
    for q in det_qs + llm_qs:
        key = (q or "").strip().lower()
        if key and key not in seen_lower:
            seen_lower.add(key)
            qs.append(q.strip())
    if not qs:
        qs = [question.strip() or " "]

    is_toc = is_toc_chapters_query(question)
    if is_toc:
        k_per_query = getattr(settings, "RAG_BOOK_K_PER_QUERY", 20)
        sparse_w = getattr(settings, "RAG_BOOK_SPARSE_WEIGHT", 0.5)
        dense_w = 1.0 - sparse_w
    else:
        k_per_query = max(k, 4)
        dense_w = getattr(settings, "RAG_DENSE_RRF_WEIGHT", 0.5)
        sparse_w = getattr(settings, "RAG_SPARSE_RRF_WEIGHT", 0.5)

    dense_hits_per_query: List[List[Dict]] = []
    sparse_hits_per_query: List[List[Dict]] = []
    for q in qs:
        dense_hits_per_query.append(await index.search(q, k_per_query))
        sparse_hits_per_query.append(index.sparse.search(q, k_per_query))
    candidates_k = max(k, getattr(settings, "RAG_RERANK_CANDIDATES", k)) if getattr(settings, "RAG_RERANK_ENABLED", False) else k
    hits = _weighted_rrf(dense_hits_per_query, sparse_hits_per_query, candidates_k, dense_weight=dense_w, sparse_weight=sparse_w)
    # Hard filter by context_window when not 'all'
    hits = _filter_hits_by_context_window(hits, context_window or "all")
    # Time-decay scoring (soft; keeps older docs but down-weights)
    doc_created, doc_meta = _get_doc_info_for_hits(hits)
    halflife = getattr(settings, "RAG_TIME_DECAY_HALFLIFE_DAYS", 0) or 0
    if halflife > 0:
        hits = _apply_time_decay(hits, doc_created, halflife)
    # Tag boost for transcripts
    if getattr(settings, "RAG_TAG_BOOST_ENABLED", False):
        tag_factor = getattr(settings, "RAG_TAG_BOOST_FACTOR", 0.08)
        hits = _apply_tag_boost(hits, question, doc_meta, tag_factor)
    # Prefer authoritative docs when scores are close
    if getattr(settings, "RAG_PREFER_AUTHORITATIVE", False):
        hits = _prefer_authoritative_sort(hits)
    # Optional rerank
    if getattr(settings, "RAG_RERANK_ENABLED", False):
        rerank_n = getattr(settings, "RAG_RERANK_TOP_N", k)
        hits = await _rerank_hits(question, hits[: getattr(settings, "RAG_RERANK_CANDIDATES", 12)], rerank_n)
    else:
        hits = hits[:k]
    return hits


def _get_chunk_text(chunk_id: str) -> str | None:
    """Fetch chunk text by id from DB (for parent expansion). Returns None if not found."""
    if not chunk_id:
        return None
    with get_conn() as conn:
        row = conn.execute("SELECT text FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        return row[0] if row else None


def _sentences(text: str) -> List[str]:
    """Split text into sentences (simple: by ., !, ?)."""
    if not (text or "").strip():
        return []
    s = re.sub(r"[.!?]+\s*", "\n", text)
    return [x.strip() for x in s.splitlines() if x.strip()]


def _word_set(s: str) -> set:
    """Normalized word set for overlap check."""
    return set(re.findall(r"[a-z0-9]{2,}", (s or "").lower()))


# Stopwords to exclude when deciding "chunk contains key query terms" for verbatim inclusion.
_QUERY_TERM_STOP = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "can", "about", "into", "through",
    "what", "which", "when", "where", "who", "how", "why", "does", "say", "said", "anything", "something",
})


def _key_query_terms(question: str, min_len: int = 3) -> set:
    """Extract significant query terms (e.g. 'matthew', 'effect') for verbatim-chunk bypass and evidence grounding."""
    words = set(re.findall(r"[a-z0-9]+", (question or "").lower()))
    return {w for w in words if len(w) >= min_len and w not in _QUERY_TERM_STOP}


def _dedupe_overlapping_sentences(blocks: List[str], overlap_ratio: float) -> List[str]:
    """Remove sentences from later blocks that overlap too much with already-seen content. Reduces redundancy in context."""
    if overlap_ratio <= 0 or not blocks:
        return blocks
    seen_word_sets: List[set] = []
    out = []
    for block in blocks:
        sentences = _sentences(block)
        kept = []
        for sent in sentences:
            ws = _word_set(sent)
            if len(ws) < 3:
                kept.append(sent)
                seen_word_sets.append(ws)
                continue
            overlap_any = False
            for prev in seen_word_sets:
                if len(prev) < 3:
                    continue
                inter = len(ws & prev)
                jaccard = inter / max(len(ws | prev), 1)
                if jaccard >= overlap_ratio:
                    overlap_any = True
                    break
            if not overlap_any:
                kept.append(sent)
                seen_word_sets.append(ws)
        out.append(" ".join(kept) if kept else block)
    return out


# Compression prompt: extract only answer-critical sentences; label partial/conflicting to improve faithfulness.
COMPRESS_SYSTEM = """Extract only sentences that are directly needed to answer the question. Copy verbatim; do not paraphrase or add interpretation.
- If the excerpt is only partially relevant, prefix with "[Partial]: " and keep only the relevant part.
- If the excerpt contradicts or differs from something you already extracted, include it and note "[Conflicting]: ".
- Omit sentences that do not help answer the question. Keep the result short (at most a few sentences)."""


async def compress(question: str, chunk_text: str, src: dict) -> str:
    """Extract only answer-critical sentences; label partial/conflicting. Reduces token use and improves grounding."""
    usr = f"Question: {question}\n\nRelevant excerpt:\n{chunk_text[:2000]}"
    try:
        return await chat.chat([{"role": "system", "content": COMPRESS_SYSTEM}, {"role": "user", "content": usr}], temperature=0.0, max_tokens=180)
    except Exception:
        return chunk_text


# RAG generation rules: faithfulness, grounding, and explicit "insufficient context" when needed.
def _rag_system_prompt(persona: Optional[str] = None) -> str:
    base = """You are EchoMind, a knowledgeable enterprise assistant. Use ONLY the provided context for factual claims.

Rules:
- Answer in a clear, confident, human way. Do not cite sources, mention file names, chunk numbers, or say "according to the document" or "the document says" unless the context explicitly contains that wording.
- Do not add facts not stated in the context. Do not use "likely", "might", or "inferred" for factual claims — either state what the context says or say the information is not in the materials.
- Do not introduce names, examples, or facts that are not in the provided context blocks. Only use named entities (people, places, terms, chapter titles) that appear explicitly in the context.
- If different parts of the context contradict each other, say so instead of picking one silently.
- Do not introduce concepts, terms, or section names that do not appear in the context.
- If the information is not in the context, say so plainly; do not guess or infer. Do not infer or invent a table of contents or chapter list from general knowledge.
- If the context is insufficient to answer the question, say clearly: "The provided context does not contain enough information to answer this." Do not fabricate an answer."""
    if persona:
        base = f"You are EchoMind in the role of: {persona}. Adapt your reasoning style, vocabulary, and tone to this role.\n\n" + base
    return base

_GENERAL_SYSTEM = "You are EchoMind, a friendly enterprise assistant. The user is greeting you or making small talk. Reply briefly and warmly in one or two sentences. Do not mention documents or sources."

def _general_system_prompt(persona: Optional[str] = None) -> str:
    if persona:
        return f"You are EchoMind in the role of: {persona}. The user is greeting you or making small talk. Reply briefly and warmly in one or two sentences, in character. Do not mention documents or sources."
    return _GENERAL_SYSTEM

# Conversation summary: structured, compressed context for RAG (goals, constraints, decisions, key facts)
CONVERSATION_SUMMARY_SYSTEM = """You maintain a compressed, structured summary of an ongoing conversation.
Given the previous summary (or "None" if this is the first exchange) and the new exchange below, output an updated summary.
Capture: goals (what the user is trying to achieve), constraints (limits, preferences, requirements), decisions (conclusions or choices made), and key facts (important information stated or agreed).
Keep it concise: a few short paragraphs or bullet points. Preserve all signal that would help answer follow-up questions. Output only the updated summary, no preamble."""


async def update_conversation_summary(
    previous_summary: Optional[str],
    user_message: str,
    assistant_message: str,
) -> str:
    """Produce an updated conversation summary from the previous summary and the latest exchange. Does not write to DB."""
    prev = (previous_summary or "").strip() or "None"
    user = (user_message or "").strip() or "(no user message)"
    assistant = (assistant_message or "").strip() or "(no assistant message)"
    prompt = f"Previous summary:\n{prev}\n\nNew exchange:\nUser: {user}\nAssistant: {assistant}\n\nUpdated summary:"
    try:
        out = await chat.chat(
            [{"role": "system", "content": CONVERSATION_SUMMARY_SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
        )
        return (out or "").strip() or prev
    except Exception as e:
        logger.warning("Conversation summary update failed: %s", e)
        return prev

async def _build_rag_context(question: str, hits: List[Dict]) -> Tuple[List[str], List[Dict], List[str]]:
    """Build numbered context blocks [1], [2], ... with parent expansion. No SOURCE lines (internal grounding only).
    Returns (block_texts, enriched_ctx_list for citations/audit, chunk_ids_used for audit).
    When RAG_VERBATIM_QUERY_TERMS is on, chunks that contain key query terms are included verbatim (truncated) instead of compressed."""
    seen_parent_ids: set = set()
    blocks: List[str] = []
    enriched: List[Dict] = []
    chunk_ids_used: List[str] = []
    key_terms = _key_query_terms(question)
    use_verbatim = getattr(settings, "RAG_VERBATIM_QUERY_TERMS", True)
    verbatim_max = getattr(settings, "RAG_VERBATIM_MAX_CHARS", 1200)

    for h in hits:
        src = h.get("source") or {}
        parent_chunk_id = src.get("parent_chunk_id") if isinstance(src.get("parent_chunk_id"), str) else None

        if parent_chunk_id and parent_chunk_id not in seen_parent_ids:
            seen_parent_ids.add(parent_chunk_id)
            parent_text = _get_chunk_text(parent_chunk_id)
            if parent_text:
                max_chars = _parent_context_max_chars()
                truncated = parent_text if len(parent_text) <= max_chars else parent_text[:max_chars].rsplit(" ", 1)[0] + "…"
                blocks.append(truncated)
                chunk_ids_used.append(parent_chunk_id)

        chunk_text = h.get("text") or ""
        chunk_lower = chunk_text.lower()
        use_verbatim_this = use_verbatim and key_terms and any(term in chunk_lower for term in key_terms)
        if use_verbatim_this:
            compressed = chunk_text if len(chunk_text) <= verbatim_max else chunk_text[:verbatim_max].rsplit(" ", 1)[0] + "…"
            compressed = compressed.strip()
        else:
            compressed = (await compress(question, chunk_text, src)).strip()
        blocks.append(compressed)
        enriched.append({**h, "compressed": compressed})
        chunk_ids_used.append(h["chunk_id"])

    # Optional: deduplicate overlapping sentences across blocks to reduce token use and repetition.
    if getattr(settings, "RAG_DEDUPE_SENTENCES", False):
        ratio = getattr(settings, "RAG_DEDUPE_OVERLAP_RATIO", 0.6)
        blocks = _dedupe_overlapping_sentences(blocks, ratio)
    return blocks, enriched, chunk_ids_used

def _rag_context_block(blocks: List[str]) -> str:
    """Format blocks as [1] ... [2] ... (no SOURCE lines)."""
    return "\n\n".join([f"[{i+1}] {b}" for i, b in enumerate(blocks)])

def _build_user_content_with_summary(conversation_summary: Optional[str], question: str, context_block: Optional[str] = None) -> str:
    """Single user message: conversation summary (if any) + current question + optional RAG context."""
    parts = []
    if conversation_summary and conversation_summary.strip():
        parts.append(f"Conversation summary (goals, constraints, decisions, key facts):\n{conversation_summary.strip()}")
    parts.append(f"Current question: {question}")
    if context_block and context_block.strip():
        parts.append(f"Context (use only for factual claims):\n{context_block.strip()}")
    return "\n\n".join(parts)


async def _answer_general(
    question: str,
    history: List[Dict],
    persona: Optional[str] = None,
    conversation_summary: Optional[str] = None,
) -> Dict:
    sys = _general_system_prompt(persona)
    if conversation_summary and conversation_summary.strip():
        user_content = _build_user_content_with_summary(conversation_summary, question, context_block=None)
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user_content}]
    else:
        msgs = [{"role": "system", "content": sys}] + history[-10:] + [{"role": "user", "content": question}]
    ans = await chat.chat(msgs, temperature=settings.LLM_TEMPERATURE, max_tokens=settings.LLM_MAX_TOKENS)
    return {"answer": ans, "citations": []}


async def answer(
    question: str,
    history: List[Dict],
    persona: Optional[str] = None,
    context_window: str = "all",
    conversation_summary: Optional[str] = None,
) -> Dict:
    if _is_general_conversation(question):
        return await _answer_general(question, history, persona, conversation_summary)

    hits = await retrieve(question, settings.TOP_K, context_window=context_window or "all")
    best_score = hits[0]["score"] if hits else 0.0
    if not hits or best_score < settings.RAG_RELEVANCE_THRESHOLD:
        return await _answer_general(question, history, persona, conversation_summary)

    blocks, enriched, chunk_ids_used = await _build_rag_context(question, hits)
    if getattr(settings, "RAG_TOC_GUARDRAIL", True) and is_toc_chapters_query(question) and not has_toc_signals_in_context(blocks):
        return {"answer": "I couldn't find the table of contents/chapter list in the retrieved excerpts from the uploaded text.", "citations": []}

    ctx_block = _rag_context_block(blocks)
    doc_ids = list({(e.get("source") or {}).get("doc_id") for e in enriched if (e.get("source") or {}).get("doc_id")})
    logger.info("RAG answer", extra={"chunk_ids": chunk_ids_used, "doc_ids": doc_ids})

    if conversation_summary and conversation_summary.strip():
        user_content = _build_user_content_with_summary(conversation_summary, question, context_block=ctx_block)
        msgs = [{"role": "system", "content": _rag_system_prompt(persona)}, {"role": "user", "content": user_content}]
    else:
        msgs = [{"role": "system", "content": _rag_system_prompt(persona)}] + history[-10:] + [{"role": "user", "content": f"Question: {question}\n\nContext:\n{ctx_block}"}]
    ans = await chat.chat(msgs, temperature=settings.LLM_TEMPERATURE, max_tokens=settings.LLM_MAX_TOKENS)

    citations = []
    if getattr(settings, "RAG_EXPOSE_SOURCES", False):
        citations = [{"filename": c["source"].get("filename"), "chunk_index": c["source"].get("chunk_index"), "score": c["score"], "snippet": (c.get("compressed") or "")[:360]} for c in enriched]
    return {"answer": ans, "citations": citations}


async def _answer_general_stream(
    question: str,
    history: List[Dict],
    persona: Optional[str] = None,
    conversation_summary: Optional[str] = None,
) -> AsyncIterator[Tuple[str, str | None, List[Dict] | None]]:
    sys = _general_system_prompt(persona)
    if conversation_summary and conversation_summary.strip():
        user_content = _build_user_content_with_summary(conversation_summary, question, context_block=None)
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user_content}]
    else:
        msgs = [{"role": "system", "content": sys}] + history[-10:] + [{"role": "user", "content": question}]
    full = []
    async for delta in chat.chat_stream(msgs, temperature=settings.LLM_TEMPERATURE, max_tokens=settings.LLM_MAX_TOKENS):
        full.append(delta)
        yield ("chunk", delta, None)
    yield ("done", "".join(full).strip(), [])


async def answer_stream(
    question: str,
    history: List[Dict],
    persona: Optional[str] = None,
    context_window: str = "all",
    conversation_summary: Optional[str] = None,
) -> AsyncIterator[Tuple[str, str | None, List[Dict] | None]]:
    """
    Same RAG as answer(), but stream the final LLM response. Yields ("chunk", delta, None) then ("done", full_answer, citations).
    Uses conversation summary when provided (goals, constraints, decisions, key facts) instead of raw last 10 messages.
    """
    if _is_general_conversation(question):
        async for ev in _answer_general_stream(question, history, persona, conversation_summary):
            yield ev
        return

    hits = await retrieve(question, settings.TOP_K, context_window=context_window or "all")
    best_score = hits[0]["score"] if hits else 0.0
    if not hits or best_score < settings.RAG_RELEVANCE_THRESHOLD:
        async for ev in _answer_general_stream(question, history, persona, conversation_summary):
            yield ev
        return

    blocks, enriched, chunk_ids_used = await _build_rag_context(question, hits)
    if getattr(settings, "RAG_TOC_GUARDRAIL", True) and is_toc_chapters_query(question) and not has_toc_signals_in_context(blocks):
        yield ("chunk", "I couldn't find the table of contents/chapter list in the retrieved excerpts from the uploaded text.", None)
        yield ("done", "I couldn't find the table of contents/chapter list in the retrieved excerpts from the uploaded text.", [])
        return

    ctx_block = _rag_context_block(blocks)
    doc_ids = list({(e.get("source") or {}).get("doc_id") for e in enriched if (e.get("source") or {}).get("doc_id")})
    logger.info("RAG answer", extra={"chunk_ids": chunk_ids_used, "doc_ids": doc_ids})

    if conversation_summary and conversation_summary.strip():
        user_content = _build_user_content_with_summary(conversation_summary, question, context_block=ctx_block)
        msgs = [{"role": "system", "content": _rag_system_prompt(persona)}, {"role": "user", "content": user_content}]
    else:
        msgs = [{"role": "system", "content": _rag_system_prompt(persona)}] + history[-10:] + [{"role": "user", "content": f"Question: {question}\n\nContext:\n{ctx_block}"}]
    citations = []
    if getattr(settings, "RAG_EXPOSE_SOURCES", False):
        citations = [{"filename": c["source"].get("filename"), "chunk_index": c["source"].get("chunk_index"), "score": c["score"], "snippet": (c.get("compressed") or "")[:360]} for c in enriched]
    full = []
    async for delta in chat.chat_stream(msgs, temperature=settings.LLM_TEMPERATURE, max_tokens=settings.LLM_MAX_TOKENS):
        full.append(delta)
        yield ("chunk", delta, None)
    answer = "".join(full).strip()
    yield ("done", answer, citations)
