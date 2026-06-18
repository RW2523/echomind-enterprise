"""
Real-time transcript segment analysis using RAG + LLM.
Evaluates each finalized paragraph against the knowledge base and assigns a factual label.
Only emits a result if confidence > 60 and the label is meaningful (not "None").
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from ..core.config import settings
from ..core.db import get_conn
from ..rag.index import index as faiss_index
from ..rag.llm import OpenAICompatChat
from ..utils.ids import new_id, now_iso

logger = logging.getLogger(__name__)

ANALYSIS_LABELS = {"Supported", "Contradicted", "Unverified", "Violating", "Risky Statement"}
CONFIDENCE_THRESHOLD = 60

_llm: Optional[OpenAICompatChat] = None


def _get_llm() -> OpenAICompatChat:
    global _llm
    if _llm is None:
        _llm = OpenAICompatChat(settings.LLM_BASE_URL, settings.LLM_MODEL)
    return _llm


@dataclass
class SourceChunk:
    chunk_id: str
    text: str
    doc_title: str = ""
    doc_id: str = ""


@dataclass
class AnalysisResult:
    id: str
    segment_id: str
    segment_text: str
    label: str
    confidence: float
    explanation: str
    source_chunks: list = field(default_factory=list)
    transcript_id: Optional[str] = None
    session_id: Optional[str] = None

    def to_ws_payload(self) -> dict:
        return {
            "type": "analysis",
            "id": self.id,
            "segment_id": self.segment_id,
            "segment_text": self.segment_text,
            "label": self.label,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "source_chunks": [
                {
                    "chunk_id": s.chunk_id,
                    "text": s.text,
                    "doc_title": s.doc_title,
                    "doc_id": s.doc_id,
                }
                for s in self.source_chunks
            ],
        }


_SYSTEM_PROMPT = (
    "You are a real-time fact-checking assistant embedded in a live meeting. "
    "You receive spoken statements and must evaluate them against provided reference documents.\n\n"
    "Return ONLY a JSON object with no markdown fences or extra text:\n"
    '{"label":"<label>","confidence":<0-100>,"explanation":"<1-2 sentence explanation>",'
    '"relevant_chunk_ids":["<id1>","<id2>"]}\n\n'
    "Label definitions:\n"
    '- "Supported": Statement is directly confirmed or strongly backed by the reference content\n'
    '- "Contradicted": Statement conflicts with or contradicts the reference content\n'
    '- "Unverified": Statement relates to the topic but cannot be verified with available references\n'
    '- "Violating": Statement appears to violate rules, regulations, policies, or guidelines in the references\n'
    '- "Risky Statement": Statement contains potentially dangerous, misleading, or risky claims per the references\n'
    '- "None": General conversation, filler, greetings, or statement not meaningfully evaluable\n\n'
    "Rules:\n"
    "- Only use labels other than None when you have clear evidence from the reference content\n"
    "- Set confidence >= 60 only for unambiguous evaluations with clear textual evidence\n"
    "- Prefer None for casual, general, or off-topic statements\n"
    "- Keep explanation factual and concise (max 2 sentences)"
)


async def analyze_segment(
    text: str,
    segment_id: str,
    session_id: Optional[str] = None,
    transcript_id: Optional[str] = None,
) -> Optional[AnalysisResult]:
    """
    Analyze a finalized transcript segment against the RAG knowledge base.
    Returns None when nothing meaningful to flag (confidence <= 60 or label is None).
    This runs as a background task; errors are logged but never propagated.
    """
    text = (text or "").strip()
    word_count = len(text.split())

    # Skip very short utterances (less than 5 words)
    if not text or word_count < 5:
        logger.debug("Silent Assistant: skipping short segment (%d words): %.80s", word_count, text)
        return None

    logger.info("Silent Assistant: analyzing segment [%s] (%d words): %.80s…", segment_id, word_count, text)

    try:
        # Search uploaded documents only (not transcript chunks) so doc recall isn't diluted
        # in transcript-heavy knowledge bases.
        hits = await faiss_index.search_document_only(text, k=8)
    except Exception as e:
        logger.warning("Silent Assistant: RAG search error for [%s]: %s", segment_id, e)
        return None

    if not hits:
        logger.info("Silent Assistant: no RAG hits for [%s] — knowledge base may be empty", segment_id)
        return None

    # Only evaluate against uploaded documents (not transcript chunks)
    doc_hits = [
        h for h in hits
        if not (h.get("source") or {}).get("filename", "").startswith("transcript_")
    ]
    if not doc_hits:
        logger.info(
            "Silent Assistant: all %d hits are transcript chunks (no uploaded documents) for [%s]",
            len(hits), segment_id,
        )
        return None

    logger.info(
        "Silent Assistant: %d doc hits (of %d total) for [%s]; top score=%.3f",
        len(doc_hits), len(hits), segment_id,
        float(doc_hits[0].get("score") or 0),
    )

    context_parts: list[str] = []
    source_chunks: list[SourceChunk] = []
    skipped_low_score = 0

    for i, hit in enumerate(doc_hits[:5]):
        chunk_text = (hit.get("text") or "").strip()
        if not chunk_text:
            continue
        score = float(hit.get("score") or 0)
        if score < 0.1:   # lowered from 0.3 — inner-product similarity for loosely related docs sits 0.1–0.3
            skipped_low_score += 1
            continue
        chunk_id = hit.get("chunk_id", f"chunk_{i}")
        src = hit.get("source") or {}
        doc_name = src.get("filename") or src.get("doc_name") or ""
        doc_id = src.get("doc_id") or ""
        context_parts.append(f"[Ref {i + 1}] (id={chunk_id}, doc={doc_name!r}): {chunk_text[:600]}")
        source_chunks.append(SourceChunk(
            chunk_id=chunk_id,
            text=chunk_text[:600],
            doc_title=doc_name,
            doc_id=doc_id,
        ))

    if not context_parts:
        logger.info(
            "Silent Assistant: all %d doc hits below score threshold for [%s] (skipped=%d)",
            len(doc_hits), segment_id, skipped_low_score,
        )
        return None

    logger.info("Silent Assistant: calling LLM with %d context chunks for [%s]", len(context_parts), segment_id)

    context = "\n\n".join(context_parts)
    user_msg = (
        f'Spoken statement to evaluate:\n"{text}"\n\n'
        f"Reference document excerpts:\n{context}"
    )

    try:
        llm = _get_llm()
        raw = await asyncio.wait_for(
            llm.chat(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=300,
            ),
            timeout=25.0,
        )
    except asyncio.TimeoutError:
        logger.warning("Silent Assistant: LLM timeout for [%s]", segment_id)
        return None
    except Exception as e:
        logger.warning("Silent Assistant: LLM error for [%s]: %s", segment_id, e)
        return None

    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            logger.warning("Silent Assistant: LLM returned no JSON for [%s]: %.120s", segment_id, raw)
            return None
        parsed = json.loads(match.group())
    except Exception as e:
        logger.warning("Silent Assistant: JSON parse error for [%s]: %s", segment_id, e)
        return None

    label = (parsed.get("label") or "None").strip()
    # Coerce confidence defensively — LLMs sometimes return "85%", "high", or null.
    try:
        confidence = float(re.sub(r"[^0-9.]", "", str(parsed.get("confidence", 0))) or 0)
    except (ValueError, TypeError):
        confidence = 0.0
    confidence = max(0.0, min(100.0, confidence))
    explanation = (parsed.get("explanation") or "").strip()
    relevant_ids: list[str] = parsed.get("relevant_chunk_ids") or []

    logger.info(
        "Silent Assistant: LLM result for [%s] — label=%r confidence=%.0f",
        segment_id, label, confidence,
    )

    if label == "None" or label not in ANALYSIS_LABELS:
        logger.info("Silent Assistant: label %r not actionable for [%s] — skipping", label, segment_id)
        return None
    if confidence < CONFIDENCE_THRESHOLD:
        logger.info(
            "Silent Assistant: confidence %.0f < %d for [%s] — skipping",
            confidence, CONFIDENCE_THRESHOLD, segment_id,
        )
        return None

    # Narrow source_chunks to only what the LLM cited
    if relevant_ids:
        cited = [s for s in source_chunks if s.chunk_id in relevant_ids]
        if cited:
            source_chunks = cited

    analysis_id = new_id("ana")
    source_refs_json = json.dumps(
        [
            {
                "chunk_id": s.chunk_id,
                "text": s.text,
                "doc_title": s.doc_title,
                "doc_id": s.doc_id,
            }
            for s in source_chunks
        ]
    )

    def _store() -> None:
        try:
            with get_conn() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO transcript_analysis
                       (id, session_id, transcript_id, segment_id, segment_text,
                        label, confidence, explanation, source_refs, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        analysis_id, session_id, transcript_id, segment_id, text,
                        label, confidence, explanation, source_refs_json, now_iso(),
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("Failed to persist analysis record: %s", exc)

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _store)
    except Exception as exc:
        logger.warning("Analysis executor error: %s", exc)

    return AnalysisResult(
        id=analysis_id,
        segment_id=segment_id,
        segment_text=text,
        label=label,
        confidence=confidence,
        explanation=explanation,
        source_chunks=source_chunks,
        transcript_id=transcript_id,
        session_id=session_id,
    )
