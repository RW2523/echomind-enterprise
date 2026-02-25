"""
Query orchestrator: intent -> search (transcript/doc) with fallback logic.
Exact fallback:
- Search transcript first: top_k=15, score_threshold=T1.
- If results < 15 OR best_score < T_low -> search documents.
- Search documents: top_k=15, score_threshold=T2.
- If empty/low -> general.
- If both 0 good results -> general answer (marked "not from your sources").
"""
from __future__ import annotations
import logging
from typing import Any, List, Optional

from app.core.config import settings
from app.core.timeutils import now_epoch, parse_relative_time
from app.models.embedder import get_embedder
from app.router.intent import classify_intent, Intent
from app.qdrant.search import search_transcripts, search_documents, get_most_recent_transcript
from app.router.prompts import build_rag_prompt, format_evidence_block
from app.models.generator import get_generator

logger = logging.getLogger(__name__)

TOP_K = settings.TOP_K
T1 = settings.T1_TRANSCRIPT
T2 = settings.T2_DOCUMENT
T_LOW = settings.T_LOW
MAX_CONTEXT_CHUNKS = settings.MAX_CONTEXT_CHUNKS

NOT_FROM_SOURCES_MSG = "I don't have enough information from your documents or transcripts to answer that. This answer is not from your sources."


def _get_time_filter(query: str) -> tuple[Optional[int], Optional[int]]:
    """Return (start_ts_min, end_ts_max) for transcript filter from query."""
    rel = parse_relative_time(query)
    if rel:
        return rel[0], rel[1]
    return None, None


def _retrieve(
    query: str,
    intent: Intent,
    doc_id: Optional[str] = None,
) -> tuple[List[dict], List[dict], str]:
    """
    Run retrieval; returns (transcript_hits, document_hits, source_used).
    source_used: "transcript" | "document" | "both" | "general"
    """
    embedder = get_embedder()
    query_vec = embedder.encode_single(query, is_query=True)
    start_ts_min, end_ts_max = _get_time_filter(query)

    transcript_hits = []
    document_hits = []
    if intent == "TRANSCRIPT_FIRST":
        transcript_hits = search_transcripts(
            query_vec,
            top_k=TOP_K,
            score_threshold=T1,
            start_ts_min=start_ts_min,
            end_ts_max=end_ts_max,
        )
        if len(transcript_hits) < TOP_K or (transcript_hits and transcript_hits[0].get("score", 0) < T_LOW):
            document_hits = search_documents(
                query_vec,
                top_k=TOP_K,
                score_threshold=T2,
                doc_id=doc_id,
            )
        if transcript_hits:
            return transcript_hits, document_hits, "transcript" if not document_hits else "both"
        if document_hits:
            return transcript_hits, document_hits, "document"
        return transcript_hits, document_hits, "general"

    # DOCUMENT_FIRST or SUMMARIZE_DOC
    document_hits = search_documents(
        query_vec,
        top_k=TOP_K,
        score_threshold=T2,
        doc_id=doc_id,
    )
    if not document_hits or (document_hits and document_hits[0].get("score", 0) < T_LOW):
        transcript_hits = search_transcripts(
            query_vec,
            top_k=TOP_K,
            score_threshold=T1,
            start_ts_min=start_ts_min,
            end_ts_max=end_ts_max,
        )
    if document_hits:
        return transcript_hits, document_hits, "document" if not transcript_hits else "both"
    if transcript_hits:
        return transcript_hits, document_hits, "transcript"
    return transcript_hits, document_hits, "general"


def _build_context_and_evidence(
    transcript_hits: List[dict],
    document_hits: List[dict],
    source_used: str,
    max_chunks: int,
) -> tuple[str, List[dict]]:
    """Build context string for generator and evidence list for citation block."""
    evidence = []
    parts = []
    count = 0
    if source_used in ("transcript", "both") and transcript_hits:
        for h in transcript_hits[:max_chunks]:
            if count >= max_chunks:
                break
            p = h.get("payload") or {}
            text = h.get("text") or p.get("text_preview", "")
            parts.append(f"[Transcript {p.get('transcript_id', '')} {p.get('start_ts')}-{p.get('end_ts')} {p.get('location', '')}]\n{text}")
            evidence.append({
                "source_type": "transcript",
                "transcript_id": p.get("transcript_id"),
                "start_ts": p.get("start_ts"),
                "end_ts": p.get("end_ts"),
                "location": p.get("location"),
                "excerpt": text[:200],
            })
            count += 1
    if source_used in ("document", "both") and document_hits:
        for h in document_hits[:max_chunks]:
            if count >= max_chunks:
                break
            p = h.get("payload") or {}
            text = h.get("text") or p.get("text_preview", "")
            parts.append(f"[{p.get('doc_title', '')} p.{p.get('page_start', '')}-{p.get('page_end', '')} {p.get('section_path', '')}]\n{text}")
            evidence.append({
                "source_type": "document",
                "doc_title": p.get("doc_title"),
                "page_start": p.get("page_start"),
                "page_end": p.get("page_end"),
                "section_path": p.get("section_path"),
                "excerpt": text[:200],
            })
            count += 1
    context = "\n\n".join(parts)
    return context, evidence


def answer(
    user_query: str,
    mode: Optional[str] = None,
    doc_id: Optional[str] = None,
    include_evidence_block: bool = True,
) -> dict:
    """
    Main entry: classify intent -> retrieve -> generate answer with citations.
    Returns { "answer", "evidence", "source_used", "from_sources" }.
    """
    intent = classify_intent(user_query)
    if mode == "general":
        intent = "GENERAL"
    if intent == "GENERAL" and not user_query.strip():
        return {
            "answer": "How can I help?",
            "evidence": [],
            "source_used": "general",
            "from_sources": False,
        }

    transcript_hits, document_hits, source_used = _retrieve(user_query, intent, doc_id)

    if source_used == "general":
        return {
            "answer": NOT_FROM_SOURCES_MSG,
            "evidence": [],
            "source_used": "general",
            "from_sources": False,
        }

    context, evidence = _build_context_and_evidence(
        transcript_hits, document_hits, source_used, MAX_CONTEXT_CHUNKS,
    )
    if not context.strip():
        return {
            "answer": NOT_FROM_SOURCES_MSG,
            "evidence": [],
            "source_used": "general",
            "from_sources": False,
        }

    gen = get_generator()
    messages = build_rag_prompt(context, user_query)
    answer_text = gen.generate(messages)
    if include_evidence_block:
        answer_text += "\n\n" + format_evidence_block(evidence)

    return {
        "answer": answer_text,
        "evidence": evidence,
        "source_used": source_used,
        "from_sources": True,
    }
