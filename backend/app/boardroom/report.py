"""
Board Room Report Generator.
Generates structured, RAG-enhanced reports from multi-speaker meeting transcripts.

Pipeline:
  1. Extract key topics from transcript via LLM (TensorRT-LLM)
  2. For each topic, query RAG for supporting evidence, contradictions, gaps
  3. Identify decisions, action items, and recommendations via LLM
  4. Assemble final JSON report and persist to boardroom_reports table
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from ..core.config import settings
from ..core.db import get_conn
from ..rag.llm import OpenAICompatChat
from ..utils.ids import now_iso

logger = logging.getLogger(__name__)

_llm: Optional[OpenAICompatChat] = None


def _get_llm() -> Optional[OpenAICompatChat]:
    global _llm
    if _llm is None:
        try:
            _llm = OpenAICompatChat(settings.LLM_BASE_URL, settings.LLM_MODEL)
        except Exception as e:
            logger.warning("Board Room: LLM init failed: %s", e)
    return _llm


async def _llm_chat(system: str, user: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
    llm = _get_llm()
    if llm is None:
        return ""
    try:
        return await llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.warning("Board Room LLM call failed: %s", e)
        return ""


async def _query_rag(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """Query the backend RAG system for evidence supporting a topic."""
    try:
        from ..rag.advanced import retrieve_semantic_first
        _source_type, hits = await retrieve_semantic_first(
            query,
            k=top_k,
            context_window="all",
            source_options={"transcript": True, "document": True, "general": False},
        )
        results = []
        for h in hits[:top_k]:
            text = h.get("text") or h.get("chunk_text") or ""
            source_meta = h.get("source") or h.get("meta") or {}
            if isinstance(source_meta, str):
                try:
                    source_meta = json.loads(source_meta)
                except Exception:
                    source_meta = {}
            results.append({
                "text": text.strip(),
                "source": source_meta.get("filename") or source_meta.get("name") or "unknown",
                "section": source_meta.get("section") or source_meta.get("sectionPath") or "",
                "page": source_meta.get("pageNumber") or source_meta.get("page"),
                "score": float(h.get("score", 0.0)),
            })
        return [r for r in results if r["text"]]
    except Exception as e:
        logger.debug("Board Room RAG query failed (non-fatal): %s", e)
        return []


def _truncate_transcript(transcript: str, max_chars: int = 6000) -> str:
    if len(transcript) <= max_chars:
        return transcript
    half = max_chars // 2
    return transcript[:half] + "\n\n[... middle of transcript truncated ...]\n\n" + transcript[-half:]


def _build_speaker_summary(speaker_map: Dict[str, str], segments: List[dict]) -> Dict[str, int]:
    """Count speaking turns per speaker."""
    counts: Dict[str, int] = {}
    for seg in segments:
        name = speaker_map.get(seg.get("speaker_id", ""), seg.get("speaker_id", "Unknown"))
        counts[name] = counts.get(name, 0) + 1
    return counts


async def _extract_topics(transcript: str) -> List[str]:
    """Use LLM to extract the main discussion topics from the transcript."""
    sys_prompt = (
        "You are an expert meeting analyst. Extract the main discussion topics "
        "from this board room meeting transcript. Return a JSON array of topic strings "
        "(max 8 topics, concise, 3–7 words each). Example: "
        '[\"Q3 financial performance\", \"Product roadmap 2026\", \"Hiring plan\"]. '
        "Return ONLY the JSON array, no other text."
    )
    truncated = _truncate_transcript(transcript, max_chars=4000)
    result = await _llm_chat(sys_prompt, truncated, max_tokens=512, temperature=0.1)
    try:
        topics = json.loads(result.strip())
        if isinstance(topics, list):
            return [str(t) for t in topics[:8] if t]
    except Exception:
        # Fallback: extract from LLM text
        lines = [ln.strip().lstrip("-•*123456789. ") for ln in result.split("\n") if ln.strip()]
        return [ln for ln in lines if 3 <= len(ln) <= 100][:8]
    return []


async def _generate_executive_summary(transcript: str, speaker_summary: Dict[str, int]) -> str:
    speakers_str = ", ".join(f"{name} ({count} turns)" for name, count in speaker_summary.items())
    sys_prompt = (
        "You are an executive assistant generating a professional board room report. "
        "Write a concise Executive Summary (3–5 sentences) capturing the essence of this meeting. "
        "Mention key participants, main outcomes, and critical decisions. Be factual and professional."
    )
    user = f"Participants: {speakers_str}\n\nTranscript:\n{_truncate_transcript(transcript, 3000)}"
    return await _llm_chat(sys_prompt, user, max_tokens=512)


async def _extract_decisions(transcript: str) -> List[str]:
    sys_prompt = (
        "Extract all decisions made during this meeting. "
        "Return a JSON array of decision strings. Each decision should be a clear, complete sentence. "
        "If no explicit decisions were made, return an empty array []. "
        "Return ONLY the JSON array."
    )
    result = await _llm_chat(sys_prompt, _truncate_transcript(transcript, 4000), max_tokens=1024)
    try:
        decisions = json.loads(result.strip())
        if isinstance(decisions, list):
            return [str(d) for d in decisions if d]
    except Exception:
        lines = [ln.strip().lstrip("-•*123456789. ") for ln in result.split("\n") if ln.strip()]
        return [ln for ln in lines if len(ln) > 10][:15]
    return []


async def _extract_action_items(transcript: str, speaker_map: Dict[str, str]) -> List[Dict[str, str]]:
    speaker_names = list(set(speaker_map.values()))
    sys_prompt = (
        f"Extract all action items from this meeting. "
        f"Known speakers: {', '.join(speaker_names)}. "
        "Return a JSON array of objects with keys: "
        "{\"item\": \"action description\", \"owner\": \"speaker name or Unknown\", \"priority\": \"High/Medium/Low\"}. "
        "Return ONLY the JSON array."
    )
    result = await _llm_chat(sys_prompt, _truncate_transcript(transcript, 4000), max_tokens=1024)
    try:
        items = json.loads(result.strip())
        if isinstance(items, list):
            return [
                {
                    "item": str(i.get("item", "")),
                    "owner": str(i.get("owner", "Unknown")),
                    "priority": str(i.get("priority", "Medium")),
                }
                for i in items if isinstance(i, dict) and i.get("item")
            ]
    except Exception:
        lines = [ln.strip().lstrip("-•*123456789. ") for ln in result.split("\n") if ln.strip()]
        return [{"item": ln, "owner": "Unknown", "priority": "Medium"} for ln in lines if len(ln) > 10][:15]
    return []


async def _generate_key_points_with_rag(
    topics: List[str],
    transcript: str,
) -> List[Dict[str, Any]]:
    """For each topic, generate discussion summary + query RAG for supporting evidence."""
    key_points = []
    for topic in topics:
        # Topic discussion summary from transcript
        sys_prompt = (
            f"Based on this meeting transcript, summarize what was discussed about: '{topic}'. "
            "Be concise (2–4 bullet points). Use • for bullets."
        )
        summary = await _llm_chat(sys_prompt, _truncate_transcript(transcript, 3000), max_tokens=300)

        # RAG evidence
        rag_chunks = await _query_rag(f"{topic} meeting board discussion", top_k=settings.BOARDROOM_REPORT_RAG_TOP_K // 2)
        evidence = [c for c in rag_chunks if c["score"] > 0.35][:3]

        key_points.append({
            "topic": topic,
            "summary": summary.strip(),
            "rag_evidence": evidence,
        })
    return key_points


async def _identify_contradictions(transcript: str, rag_chunks: List[Dict]) -> List[str]:
    if not rag_chunks:
        return []
    rag_context = "\n\n".join(
        f"[{c['source']}] {c['text'][:400]}" for c in rag_chunks[:5]
    )
    sys_prompt = (
        "You are an expert fact-checker. Compare the meeting transcript with the reference documents. "
        "Identify any contradictions, inconsistencies, or statements that conflict with the reference material. "
        "Return a JSON array of contradiction strings. If none found, return []. "
        "Return ONLY the JSON array."
    )
    user = (
        f"Reference documents:\n{rag_context}\n\n"
        f"Meeting transcript:\n{_truncate_transcript(transcript, 2000)}"
    )
    result = await _llm_chat(sys_prompt, user, max_tokens=512)
    try:
        contradictions = json.loads(result.strip())
        if isinstance(contradictions, list):
            return [str(c) for c in contradictions if c]
    except Exception:
        lines = [ln.strip().lstrip("-•*123456789. ") for ln in result.split("\n") if ln.strip()]
        return [ln for ln in lines if len(ln) > 15][:8]
    return []


async def _generate_recommendations(transcript: str, topics: List[str], rag_chunks: List[Dict]) -> List[str]:
    rag_context = "\n\n".join(f"[{c['source']}] {c['text'][:300]}" for c in rag_chunks[:4]) if rag_chunks else ""
    sys_prompt = (
        "Based on this meeting transcript and relevant reference documents, "
        "provide 3–6 actionable intelligent recommendations or next steps. "
        "Return a JSON array of recommendation strings. Return ONLY the JSON array."
    )
    user = (
        f"Topics discussed: {', '.join(topics)}\n\n"
        f"Reference context:\n{rag_context}\n\n"
        f"Transcript:\n{_truncate_transcript(transcript, 2000)}"
    )
    result = await _llm_chat(sys_prompt, user, max_tokens=512)
    try:
        recs = json.loads(result.strip())
        if isinstance(recs, list):
            return [str(r) for r in recs if r]
    except Exception:
        lines = [ln.strip().lstrip("-•*123456789. ") for ln in result.split("\n") if ln.strip()]
        return [ln for ln in lines if len(ln) > 15][:6]
    return []


async def generate_report_async(
    session_id: str,
    transcript: str,
    segments: List[dict],
    speaker_map: Dict[str, str],
) -> str:
    """
    Generate a comprehensive board room report.
    Returns report_id (stored in boardroom_reports table).
    """
    report_id = str(uuid.uuid4())
    created_at = now_iso()

    # Store pending report row immediately
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO boardroom_reports (id, session_id, status, created_at, updated_at) VALUES (?,?,?,?,?)",
            (report_id, session_id, "generating", created_at, created_at),
        )
        conn.commit()

    try:
        import time as _time
        speaker_summary = _build_speaker_summary(speaker_map, segments)

        # Run analysis pipeline
        logger.info("boardroom.report.generation.start session_id=%s speakers=%d transcript_chars=%d",
                    session_id, len(speaker_summary), len(transcript))
        _t0 = _time.monotonic()

        topics = await _extract_topics(transcript)
        if not topics:
            topics = ["General Discussion"]

        executive_summary = await _generate_executive_summary(transcript, speaker_summary)
        decisions = await _extract_decisions(transcript)
        action_items = await _extract_action_items(transcript, speaker_map)
        key_points = await _generate_key_points_with_rag(topics, transcript)

        # Gather all RAG evidence for contradictions + recommendations
        all_rag = []
        for kp in key_points:
            all_rag.extend(kp.get("rag_evidence", []))
        # Deduplicate by text
        seen = set()
        unique_rag = []
        for c in all_rag:
            key = c["text"][:100]
            if key not in seen:
                seen.add(key)
                unique_rag.append(c)

        contradictions = await _identify_contradictions(transcript, unique_rag)
        recommendations = await _generate_recommendations(transcript, topics, unique_rag)

        report = {
            "report_id": report_id,
            "session_id": session_id,
            "generated_at": now_iso(),
            "executive_summary": executive_summary,
            "participants": [
                {"name": name, "speaking_turns": turns}
                for name, turns in speaker_summary.items()
            ],
            "key_discussion_points": key_points,
            "decisions": decisions,
            "action_items": action_items,
            "rag_evidence": unique_rag,
            "contradictions": contradictions,
            "recommendations": recommendations,
            "topics": topics,
        }

        # Persist final report
        with get_conn() as conn:
            conn.execute(
                """UPDATE boardroom_reports
                   SET status=?, report_json=?, rag_evidence_json=?, updated_at=?
                   WHERE id=?""",
                (
                    "ready",
                    json.dumps(report, ensure_ascii=False),
                    json.dumps(unique_rag, ensure_ascii=False),
                    now_iso(),
                    report_id,
                ),
            )
            conn.commit()

        _elapsed = _time.monotonic() - _t0
        logger.info(
            "boardroom.report.generation.completed session_id=%s report_id=%s "
            "elapsed_sec=%.1f topics=%d key_points=%d action_items=%d",
            session_id, report_id, _elapsed, len(topics), len(key_points), len(action_items),
        )
        return report_id

    except Exception as e:
        logger.error("Board Room: report generation failed: %s", e)
        with get_conn() as conn:
            conn.execute(
                "UPDATE boardroom_reports SET status=?, updated_at=? WHERE id=?",
                ("failed", now_iso(), report_id),
            )
            conn.commit()
        raise


def get_report(report_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a report from DB by ID."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, session_id, status, report_json, created_at, updated_at FROM boardroom_reports WHERE id=?",
            (report_id,),
        ).fetchone()
    if not row:
        return None
    report_json = row[3]
    return {
        "report_id": row[0],
        "session_id": row[1],
        "status": row[2],
        "report": json.loads(report_json) if report_json else None,
        "created_at": row[4],
        "updated_at": row[5],
    }


def get_report_by_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Fetch the latest report for a session."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, session_id, status, report_json, created_at, updated_at "
            "FROM boardroom_reports WHERE session_id=? ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    if not row:
        return None
    report_json = row[3]
    return {
        "report_id": row[0],
        "session_id": row[1],
        "status": row[2],
        "report": json.loads(report_json) if report_json else None,
        "created_at": row[4],
        "updated_at": row[5],
    }
