from __future__ import annotations
from typing import List, Dict, AsyncIterator, Tuple
import re
from ..core.config import settings
from .index import index
from .llm import OpenAICompatChat

chat = OpenAICompatChat(settings.LLM_BASE_URL, settings.LLM_MODEL)

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

async def generate_queries(q: str) -> List[str]:
    sys="Rewrite the question into 3 alternative search queries (synonyms/acronyms/sore thumb keywords). Return only the list."
    txt=await chat.chat([{"role":"system","content":sys},{"role":"user","content":q}], temperature=0.2, max_tokens=180)
    variants=[]
    for line in txt.splitlines():
        line=re.sub(r"^\s*[-\d\).]+\s*", "", line).strip()
        if line: variants.append(line)
    out=[q] + [v for v in variants if v.lower()!=q.lower()]
    return out[:4]

async def retrieve(question: str, k:int) -> List[Dict]:
    qs = await generate_queries(question)
    hits=[]
    for q in qs:
        hits.extend(await index.search(q, k))
    return _dedupe_best(hits)[:k]

async def compress(question:str, chunk_text:str, src:dict) -> str:
    sys="Extract only sentences that directly help answer the question. Keep it short."
    usr=f"Question: {question}\n\nChunk from {src.get('filename')} (chunk {src.get('chunk_index')}):\n{chunk_text}"
    try:
        return await chat.chat([{"role":"system","content":sys},{"role":"user","content":usr}], temperature=0.0, max_tokens=180)
    except Exception:
        return chunk_text

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

    ctx = []
    for h in hits:
        c = (await compress(question, h["text"], h["source"])).strip()
        ctx.append({**h, "compressed": c})
    ctx_block = "\n\n".join([f"[{i+1}] {c['compressed']}\nSOURCE: {c['source']['filename']}#chunk{c['source']['chunk_index']}" for i, c in enumerate(ctx)])
    sys = (
        "You are EchoMind, an enterprise assistant. Use ONLY the provided context for factual claims. "
        "If context is insufficient, say what's missing. Provide the answer, then a Sources list referencing [1],[2]…."
    )
    msgs = [{"role": "system", "content": sys}] + history[-10:] + [{"role": "user", "content": f"Question: {question}\n\nContext:\n{ctx_block}"}]
    ans = await chat.chat(msgs, temperature=settings.LLM_TEMPERATURE, max_tokens=settings.LLM_MAX_TOKENS)
    citations = [{"filename": c["source"]["filename"], "chunk_index": c["source"]["chunk_index"], "score": c["score"], "snippet": c["compressed"][:360]} for c in ctx]
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
    General conversation or low-relevance → no RAG, no citations.
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

    ctx = []
    for h in hits:
        c = (await compress(question, h["text"], h["source"])).strip()
        ctx.append({**h, "compressed": c})
    ctx_block = "\n\n".join([f"[{i+1}] {c['compressed']}\nSOURCE: {c['source']['filename']}#chunk{c['source']['chunk_index']}" for i, c in enumerate(ctx)])
    sys = (
        "You are EchoMind, an enterprise assistant. Use ONLY the provided context for factual claims. "
        "If context is insufficient, say what's missing. Provide the answer, then a Sources list referencing [1],[2]…."
    )
    msgs = [{"role": "system", "content": sys}] + history[-10:] + [{"role": "user", "content": f"Question: {question}\n\nContext:\n{ctx_block}"}]
    citations = [{"filename": c["source"]["filename"], "chunk_index": c["source"]["chunk_index"], "score": c["score"], "snippet": c["compressed"][:360]} for c in ctx]
    full = []
    async for delta in chat.chat_stream(msgs, temperature=settings.LLM_TEMPERATURE, max_tokens=settings.LLM_MAX_TOKENS):
        full.append(delta)
        yield ("chunk", delta, None)
    answer = "".join(full).strip()
    yield ("done", answer, citations)
