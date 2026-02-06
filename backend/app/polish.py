"""
Polish raw transcript into clear notes. LLM if available, else heuristic placeholder.
"""
from __future__ import annotations
import re
from typing import Optional

from .core.config import settings

# Optional LLM (Ollama)
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        try:
            from .rag.llm import OpenAICompatChat
            _llm = OpenAICompatChat(settings.LLM_BASE_URL, settings.LLM_MODEL)
        except Exception:
            _llm = False
    return _llm if _llm else None


def _heuristic_polish(text: str) -> str:
    """Placeholder: fix spacing, add punctuation heuristics, format into short paragraphs."""
    if not text or not text.strip():
        return ""
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([.,?!:;])", r"\1", t)
    t = re.sub(r"([.!?])\s*", r"\1\n\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    return "\n\n".join(lines)


class Polisher:
    """Polish raw transcript. Uses LLM when available, else heuristic."""

    async def polish_text(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        llm = _get_llm()
        if llm:
            try:
                sys_prompt = (
                    "Polish the transcript into clear, well-structured notes. "
                    "Use headings and bullet points where appropriate. Keep the meaning. Fix grammar and punctuation."
                )
                out = await llm.chat(
                    [{"role": "system", "content": sys_prompt}, {"role": "user", "content": text[:8000]}],
                    temperature=0.2,
                    max_tokens=1024,
                )
                return (out or "").strip() or _heuristic_polish(text)
            except Exception:
                return _heuristic_polish(text)
        return _heuristic_polish(text)


def get_polisher() -> Polisher:
    return Polisher()
