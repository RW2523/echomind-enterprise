"""
Polish feature: LLM-based rewrite of transcript text.
Interface: polish_text(text) -> polished_text.
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


async def polish_text(text: str) -> str:
    """
    Polish transcript text: clear structure, headings, bullet points, fix spacing.
    Falls back to heuristic placeholder if LLM unavailable.
    """
    if not text or not text.strip():
        return ""
    llm = _get_llm()
    if llm is not None:
        try:
            sys_prompt = (
                "Polish the transcript into clear, well-structured notes with headings and bullet points. "
                "Keep meaning. Fix obvious typos and spacing. Output only the polished text."
            )
            polished = await llm.chat(
                [{"role": "system", "content": sys_prompt}, {"role": "user", "content": text[:8000]}],
                temperature=0.2,
                max_tokens=1024,
            )
            return (polished or "").strip()
        except Exception:
            pass
    return _placeholder_polish(text)


def _placeholder_polish(text: str) -> str:
    """Heuristic polish: fix spacing, ensure space after punctuation, short paragraphs."""
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
