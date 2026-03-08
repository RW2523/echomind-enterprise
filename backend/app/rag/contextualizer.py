"""
Contextual retrieval: generate context headers for BOOK child chunks.

Improves section ranking and concept retrieval by contextualizing every child chunk
before indexing. Uses contextualized_text for embeddings and BM25; raw_text for
evidence extraction and answer generation.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from ..core.config import settings

logger = logging.getLogger(__name__)

SECTION_SUMMARY_SYSTEM = """You are summarizing a section from a financial or regulatory document (e.g. DoD FMR, government regulations).
Output a single short paragraph (2-4 sentences) covering:
- What this section is about
- What responsibilities, rules, procedures, or compliance requirements it covers
- Major policy terms, financial concepts, or regulatory definitions
Be concise. Use regulatory and financial language. No preamble."""

CHUNK_ROLE_SYSTEM = """You are describing the role of a paragraph within a regulatory or financial section (e.g. DoD FMR).
Output exactly one sentence starting with "This paragraph" that explains what this specific paragraph covers.
Examples: "This paragraph explains responsibilities for certifying claims and liability for improper payments."
"This paragraph defines administrative control of funds and outlines the certifying officer's duties."
"This paragraph specifies disbursing officer steps when a payment is issued incorrectly."
Be specific to the content. No preamble."""


def _parse_volume_chapter(section_path: str) -> tuple[Optional[int], Optional[int]]:
    """Extract volume and chapter from section_path (e.g. 'Volume 5 > Chapter 5 > Section 0505')."""
    vol, ch = None, None
    if not section_path:
        return (vol, ch)
    vol_m = re.search(r"Volume\s+(\d+)", section_path, re.I)
    if vol_m:
        try:
            vol = int(vol_m.group(1))
        except ValueError:
            pass
    ch_m = re.search(r"Chapter\s+(\d+)", section_path, re.I)
    if ch_m:
        try:
            ch = int(ch_m.group(1))
        except ValueError:
            pass
    return (vol, ch)


def build_context_header(
    doc_title: str,
    volume: Optional[int],
    chapter: Optional[int],
    section_id: Optional[str],
    section_title: Optional[str],
    section_path: Optional[str],
    parent_summary: Optional[str],
    chunk_role: Optional[str],
) -> str:
    """Build structured context header for a BOOK child chunk.

    Example output:
    [Document: DoD Financial Management Regulation]
    [Volume 5 | Chapter 5 | Section 0505 | Certifying Officers]
    [Section path: Volume 5 > Chapter 5 > Section 0505]
    [Parent summary: This section governs the appointment, accountability, and payment certification duties of certifying officers.]
    [Chunk role: This paragraph explains responsibilities for certifying claims and liability for improper payments.]
    """
    parts: List[str] = []
    if doc_title:
        parts.append(f"[Document: {doc_title}]")
    meta_parts = []
    if volume is not None:
        meta_parts.append(f"Volume {volume}")
    if chapter is not None:
        meta_parts.append(f"Chapter {chapter}")
    if section_id:
        meta_parts.append(f"Section {section_id}")
    if section_title:
        meta_parts.append(section_title.strip())
    if meta_parts:
        parts.append(f"[{' | '.join(meta_parts)}]")
    if section_path:
        parts.append(f"[Section path: {section_path}]")
    if parent_summary:
        parts.append(f"[Parent summary: {parent_summary.strip()}]")
    if chunk_role:
        parts.append(f"[Chunk role: {chunk_role.strip()}]")
    return "\n".join(parts) if parts else ""


def build_contextualized_text(header: str, raw_text: str, max_chars: int | None = None) -> str:
    """Append raw chunk text to context header, truncating body to stay within embedding limits."""
    if max_chars is None:
        from ..core.config import settings
        max_chars = settings.EMBED_MAX_CHARS
    if not header:
        return (raw_text or "")[:max_chars]
    if not raw_text:
        return header[:max_chars]
    body_budget = max(200, max_chars - len(header) - 2)
    body = raw_text[:body_budget]
    return f"{header}\n\n{body}"


async def generate_section_summary(section_text: str, section_path: str) -> Optional[str]:
    """Generate LLM summary for a section (parent chunk text)."""
    if not section_text or not getattr(settings, "RAG_USE_CONTEXTUAL_RETRIEVAL", True):
        return None
    try:
        from .llm import OpenAICompatChat
        chat = OpenAICompatChat(settings.LLM_BASE_URL, settings.LLM_MODEL)
        user = f"Section path: {section_path or 'N/A'}\n\nSection text (excerpt):\n{section_text[:4000]}"
        out = await chat.chat(
            [{"role": "system", "content": SECTION_SUMMARY_SYSTEM}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=200,
        )
        return (out or "").strip() or None
    except Exception as e:
        logger.warning("contextualizer: section summary failed for %s: %s", section_path[:50], e)
        return None


async def generate_chunk_role(chunk_text: str) -> Optional[str]:
    """Generate one-sentence chunk role description."""
    if not chunk_text or not getattr(settings, "RAG_USE_CONTEXTUAL_RETRIEVAL", True):
        return None
    try:
        from .llm import OpenAICompatChat
        chat = OpenAICompatChat(settings.LLM_BASE_URL, settings.LLM_MODEL)
        user = f"Paragraph:\n{chunk_text[:1500]}"
        out = await chat.chat(
            [{"role": "system", "content": CHUNK_ROLE_SYSTEM}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=80,
        )
        return (out or "").strip() or None
    except Exception as e:
        logger.warning("contextualizer: chunk role failed: %s", e)
        return None


async def generate_section_summaries_batch(
    sections: List[Dict],
    max_concurrent: int = 3,
) -> Dict[str, str]:
    """Generate summaries for multiple sections. Returns section_path -> summary."""
    import asyncio
    result: Dict[str, str] = {}

    async def _summarize(s: Dict) -> None:
        sp = s.get("section_path") or ""
        text = s.get("full_section_text") or ""
        if not sp or not text:
            return
        summary = await generate_section_summary(text, sp)
        if summary:
            result[sp] = summary

    sem = asyncio.Semaphore(max_concurrent)
    async def limited(s):
        async with sem:
            await _summarize(s)
    await asyncio.gather(*[limited(s) for s in sections])
    return result


async def generate_chunk_roles_batch(
    chunks: List[tuple],
    max_concurrent: int = 5,
) -> Dict[str, str]:
    """Generate chunk roles for multiple chunks. chunks: [(chunk_id, text), ...]. Returns chunk_id -> role."""
    import asyncio
    result: Dict[str, str] = {}

    async def _role(cid: str, text: str) -> None:
        role = await generate_chunk_role(text)
        if role:
            result[cid] = role

    sem = asyncio.Semaphore(max_concurrent)
    async def limited(item):
        async with sem:
            await _role(item[0], item[1])
    await asyncio.gather(*[limited(c) for c in chunks])
    return result


def build_context_header_from_chunk(
    chunk,
    doc_title: str,
    parent_summary: Optional[str],
    chunk_role: Optional[str],
) -> str:
    """Build context header from Chunk metadata."""
    vol, ch = _parse_volume_chapter(chunk.section_path or "")
    return build_context_header(
        doc_title=doc_title,
        volume=vol,
        chapter=ch,
        section_id=chunk.section_id,
        section_title=chunk.section_title,
        section_path=chunk.section_path,
        parent_summary=parent_summary,
        chunk_role=chunk_role,
    )
