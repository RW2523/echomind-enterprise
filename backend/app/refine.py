"""
Refine feature: LLM-based rewrite of transcript text into clear, structured notes.
Interface: refine_text(text) -> refined_text.
Uses existing Ollama LLM when available; otherwise placeholder (spacing + simple formatting).
"""
from __future__ import annotations
import re
import asyncio
from typing import Optional

from .core.config import settings
from .rag.llm import OpenAICompatChat

_llm: Optional[OpenAICompatChat] = None


def _get_llm() -> Optional[OpenAICompatChat]:
    global _llm
    if _llm is None:
        try:
            _llm = OpenAICompatChat(settings.LLM_BASE_URL, settings.LLM_MODEL)
        except Exception:
            pass
    return _llm


async def refine_text(text: str) -> str:
    """
    Refine transcript text: clear structure, headings, bullet points, fix spacing.
    Falls back to heuristic placeholder if LLM unavailable.
    """
    if not text or not text.strip():
        return ""
    llm = _get_llm()
    if llm is not None:
        try:
            sys_prompt = (
                "Refine the transcript into clear, well-structured notes with headings and bullet points. "
                "Keep meaning. Fix obvious typos and spacing. Output only the refined text."
            )
            refined = await llm.chat(
                [{"role": "system", "content": sys_prompt}, {"role": "user", "content": text[:8000]}],
                temperature=0.2,
                max_tokens=1024,
            )
            return (refined or "").strip()
        except Exception:
            pass
    return _placeholder_refine(text)


def _placeholder_refine(text: str) -> str:
    """Heuristic refine: fix spacing, ensure space after punctuation, short paragraphs."""
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r"\s*([.,?!:;])\s*", r"\1 ", t)
    t = re.sub(r"\s+", " ", t).strip()
    lines = []
    for sent in re.split(r"(?<=[.!?])\s+", t):
        sent = sent.strip()
        if sent:
            lines.append(sent)
    return "\n\n".join(lines) if lines else t
