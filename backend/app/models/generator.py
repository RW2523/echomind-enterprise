"""
Answer generation for the RAG platform.
Integrates retrieved chunks into context and generates answer; supports fallback when sources are empty.
Uses Ollama chat API by default; can be extended for local GPU (e.g. Qwen-14B) with bfloat16.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from ..rag.llm import OpenAICompatChat
from ..core.config import settings

# Singleton chat client
_generator: Optional[OpenAICompatChat] = None


def get_generator() -> OpenAICompatChat:
    global _generator
    if _generator is None:
        _generator = OpenAICompatChat(settings.LLM_BASE_URL, settings.LLM_MODEL)
    return _generator


async def generate_answer(
    chunks: List[Dict[str, Any]],
    query: str,
    max_context_chunks: int = 10,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Generate an answer from retrieved chunks and query.
    - chunks: list of {"text", "source", ...}
    - If chunks is empty, returns a short fallback message.
    """
    gen = get_generator()
    temp = temperature if temperature is not None else getattr(settings, "RAG_LLM_TEMPERATURE", settings.LLM_TEMPERATURE)
    max_tok = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS
    context = "\n\n---\n\n".join(
        (c.get("text") or c.get("content") or "")[:4000] for c in chunks[:max_context_chunks]
    )
    if not context.strip():
        return "I couldn't find relevant information to answer that question."
    system = "Answer based only on the following context. If the context does not contain the answer, say so briefly."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]
    return await gen.chat(messages, temperature=temp, max_tokens=max_tok)
