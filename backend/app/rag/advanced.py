from __future__ import annotations
from typing import List, Dict, AsyncIterator, Tuple
import re
import logging
from ..core.config import settings
from ..core.db import get_conn
from .index import index
from .llm import OpenAICompatChat

logger = logging.getLogger(__name__)
chat = OpenAICompatChat(settings.LLM_BASE_URL, settings.LLM_MODEL)

# Max chars of parent chunk to include when expanding context (avoid blowing context window)
_PARENT_CONTEXT_MAX_CHARS = 2400

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

def _reciprocal_rank_fusion(
    dense_hits_per_query: List[List[Dict]],
    sparse_hits_per_query: List[List[Dict]],
    k: int,
) -> List[Dict]:
    """Merge dense + sparse results with RRF. Each list contributes 1/(RRF_K+rank) per chunk. Returns top-k."""
    fused: Dict[str, Dict] = {}
    for hit_list in dense_hits_per_query:
        for rank, h in enumerate(hit_list):
            cid = h["chunk_id"]
            rrf = 1.0 / (RRF_K + rank)
            if cid not in fused:
                fused[cid] = {"chunk_id": cid, "rrf": 0.0, "dense_score": 0.0, "text": h["text"], "source": h["source"]}
            fused[cid]["rrf"] += rrf
            fused[cid]["dense_score"] = max(fused[cid]["dense_score"], h["score"])
    for hit_list in sparse_hits_per_query:
        for rank, h in enumerate(hit_list):
            cid = h["chunk_id"]
            rrf = 1.0 / (RRF_K + rank)
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

async def generate_queries(q: str) -> List[str]:
    sys="Rewrite the question into 3 alternative search queries (synonyms/acronyms/sore thumb keywords). Return only the list."
    txt=await chat.chat([{"role":"system","content":sys},{"role":"user","content":q}], temperature=0.2, max_tokens=180)
    variants=[]
    for line in txt.splitlines():
        line=re.sub(r"^\s*[-\d\).]+\s*", "", line).strip()
        if line: variants.append(line)
    out=[q] + [v for v in variants if v.lower()!=q.lower()]
    return out[:4]

async def retrieve(question: str, k: int) -> List[Dict]:
    """Hybrid retrieve: dense (FAISS) + sparse (BM25), merged with RRF."""
    qs = await generate_queries(question)
    k_per_query = max(k, 4)
    dense_hits_per_query: List[List[Dict]] = []
    sparse_hits_per_query: List[List[Dict]] = []
    for q in qs:
        dense_hits_per_query.append(await index.search(q, k_per_query))
        sparse_hits = index.sparse.search(q, k_per_query)
        sparse_hits_per_query.append(sparse_hits)
    return _reciprocal_rank_fusion(dense_hits_per_query, sparse_hits_per_query, k)


def _get_chunk_text(chunk_id: str) -> str | None:
    """Fetch chunk text by id from DB (for parent expansion). Returns None if not found."""
    if not chunk_id:
        return None
    with get_conn() as conn:
        row = conn.execute("SELECT text FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        return row[0] if row else None


async def compress(question: str, chunk_text: str, src: dict) -> str:
    """Extract only sentences that directly help answer; copy verbatim where possible (audit: reduce citation laundering)."""
    sys = (
        "Extract only sentences that directly help answer the question. "
        "Copy sentences verbatim where possible; do not paraphrase or add interpretation. Keep it short."
    )
    usr = f"Question: {question}\n\nRelevant excerpt:\n{chunk_text}"
    try:
        return await chat.chat([{"role": "system", "content": sys}, {"role": "user", "content": usr}], temperature=0.0, max_tokens=180)
    except Exception:
        return chunk_text

# RAG generation rules (audit: conversational, grounded, no source exposure)
_RAG_SYSTEM_PROMPT = """You are EchoMind, a knowledgeable enterprise assistant. Use ONLY the provided context for factual claims.

Rules:
- Answer in a clear, confident, human way. Do not cite sources, mention file names, chunk numbers, or say "according to the document" or "the document says" unless the context explicitly contains that wording.
- Do not add facts not stated in the context. Do not use "likely", "might", or "inferred" for factual claims — either state what the context says or say the information is not in the materials.
- If different parts of the context contradict each other, say so instead of picking one silently.
- Do not introduce concepts, terms, or section names that do not appear in the context.
- If the information is not in the context, say so plainly; do not guess or infer."""

async def _build_rag_context(question: str, hits: List[Dict]) -> Tuple[List[str], List[Dict], List[str]]:
    """Build numbered context blocks [1], [2], ... with parent expansion. No SOURCE lines (internal grounding only).
    Returns (block_texts, enriched_ctx_list for citations/audit, chunk_ids_used for audit)."""
    seen_parent_ids: set = set()
    blocks: List[str] = []
    enriched: List[Dict] = []
    chunk_ids_used: List[str] = []

    for h in hits:
        src = h.get("source") or {}
        parent_chunk_id = src.get("parent_chunk_id") if isinstance(src.get("parent_chunk_id"), str) else None

        if parent_chunk_id and parent_chunk_id not in seen_parent_ids:
            seen_parent_ids.add(parent_chunk_id)
            parent_text = _get_chunk_text(parent_chunk_id)
            if parent_text:
                truncated = parent_text if len(parent_text) <= _PARENT_CONTEXT_MAX_CHARS else parent_text[:_PARENT_CONTEXT_MAX_CHARS].rsplit(" ", 1)[0] + "…"
                blocks.append(truncated)
                chunk_ids_used.append(parent_chunk_id)

        compressed = (await compress(question, h["text"], src)).strip()
        blocks.append(compressed)
        enriched.append({**h, "compressed": compressed})
        chunk_ids_used.append(h["chunk_id"])

    return blocks, enriched, chunk_ids_used

def _rag_context_block(blocks: List[str]) -> str:
    """Format blocks as [1] ... [2] ... (no SOURCE lines)."""
    return "\n\n".join([f"[{i+1}] {b}" for i, b in enumerate(blocks)])

async def _answer_general(question: str, history: List[Dict]) -> Dict:
    sys = (
        "You are EchoMind, a friendly enterprise assistant. The user is greeting you or making small talk. "
        "Reply briefly and warmly in one or two sentences. Do not mention documents or sources."
    )
    msgs = [{"role": "system", "content": sys}] + history[-10:] + [{"role": "user", "content": question}]
    ans = await chat.chat(msgs, temperature=settings.LLM_TEMPERATURE, max_tokens=settings.LLM_MAX_TOKENS)
    return {"answer": ans, "citations": []}

async def answer(question: str, history: List[Dict]) -> Dict:
    if _is_general_conversation(question):
        return await _answer_general(question, history)

    hits = await retrieve(question, settings.TOP_K)
    best_score = hits[0]["score"] if hits else 0.0
    if not hits or best_score < settings.RAG_RELEVANCE_THRESHOLD:
        return await _answer_general(question, history)

    blocks, enriched, chunk_ids_used = await _build_rag_context(question, hits)
    ctx_block = _rag_context_block(blocks)
    doc_ids = list({(e.get("source") or {}).get("doc_id") for e in enriched if (e.get("source") or {}).get("doc_id")})
    logger.info("RAG answer", extra={"chunk_ids": chunk_ids_used, "doc_ids": doc_ids})

    msgs = [{"role": "system", "content": _RAG_SYSTEM_PROMPT}] + history[-10:] + [{"role": "user", "content": f"Question: {question}\n\nContext:\n{ctx_block}"}]
    ans = await chat.chat(msgs, temperature=settings.LLM_TEMPERATURE, max_tokens=settings.LLM_MAX_TOKENS)

    citations = []
    if getattr(settings, "RAG_EXPOSE_SOURCES", False):
        citations = [{"filename": c["source"].get("filename"), "chunk_index": c["source"].get("chunk_index"), "score": c["score"], "snippet": (c.get("compressed") or "")[:360]} for c in enriched]
    return {"answer": ans, "citations": citations}


async def _answer_general_stream(question: str, history: List[Dict]) -> AsyncIterator[Tuple[str, str | None, List[Dict] | None]]:
    sys = (
        "You are EchoMind, a friendly enterprise assistant. The user is greeting you or making small talk. "
        "Reply briefly and warmly in one or two sentences. Do not mention documents or sources."
    )
    msgs = [{"role": "system", "content": sys}] + history[-10:] + [{"role": "user", "content": question}]
    full = []
    async for delta in chat.chat_stream(msgs, temperature=settings.LLM_TEMPERATURE, max_tokens=settings.LLM_MAX_TOKENS):
        full.append(delta)
        yield ("chunk", delta, None)
    yield ("done", "".join(full).strip(), [])

async def answer_stream(question: str, history: List[Dict]) -> AsyncIterator[Tuple[str, str | None, List[Dict] | None]]:
    """
    Same RAG as answer(), but stream the final LLM response. Yields ("chunk", delta, None) then ("done", full_answer, citations).
    General conversation or low-relevance → no RAG, no citations. Citations only returned when RAG_EXPOSE_SOURCES.
    """
    if _is_general_conversation(question):
        async for ev in _answer_general_stream(question, history):
            yield ev
        return

    hits = await retrieve(question, settings.TOP_K)
    best_score = hits[0]["score"] if hits else 0.0
    if not hits or best_score < settings.RAG_RELEVANCE_THRESHOLD:
        async for ev in _answer_general_stream(question, history):
            yield ev
        return

    blocks, enriched, chunk_ids_used = await _build_rag_context(question, hits)
    ctx_block = _rag_context_block(blocks)
    doc_ids = list({(e.get("source") or {}).get("doc_id") for e in enriched if (e.get("source") or {}).get("doc_id")})
    logger.info("RAG answer", extra={"chunk_ids": chunk_ids_used, "doc_ids": doc_ids})

    msgs = [{"role": "system", "content": _RAG_SYSTEM_PROMPT}] + history[-10:] + [{"role": "user", "content": f"Question: {question}\n\nContext:\n{ctx_block}"}]
    citations = []
    if getattr(settings, "RAG_EXPOSE_SOURCES", False):
        citations = [{"filename": c["source"].get("filename"), "chunk_index": c["source"].get("chunk_index"), "score": c["score"], "snippet": (c.get("compressed") or "")[:360]} for c in enriched]
    full = []
    async for delta in chat.chat_stream(msgs, temperature=settings.LLM_TEMPERATURE, max_tokens=settings.LLM_MAX_TOKENS):
        full.append(delta)
        yield ("chunk", delta, None)
    answer = "".join(full).strip()
    yield ("done", answer, citations)
