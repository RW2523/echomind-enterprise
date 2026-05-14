from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional

from ..assistant.live_analysis import (
    _evidence_from_hit,
    _llm_classify_insights,
    _retrieve_hits,
    normalize_transcript_text,
)
from ..core.config import settings
from ..rag.llm import OpenAICompatChat

chat = OpenAICompatChat(settings.LLM_BASE_URL, settings.LLM_MODEL)

_reports: Dict[str, Dict[str, Any]] = {}


def _split_segments(text: str, max_chars: int = 1400) -> List[str]:
    t = normalize_transcript_text(text)
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\s+", t)
    segments: List[str] = []
    buf: List[str] = []
    size = 0
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if size + len(p) + 1 > max_chars and buf:
            segments.append(" ".join(buf))
            buf = [p]
            size = len(p)
        else:
            buf.append(p)
            size += len(p) + 1
    if buf:
        segments.append(" ".join(buf))
    return segments[:24]


async def _polish_transcript(text: str, title: str) -> str:
    if not text.strip():
        return ""
    sys_prompt = (
        "You are an executive meeting scribe. Rewrite the transcript into polished meeting minutes: "
        "clear headings, short paragraphs, bullet points for decisions and action items. "
        "Preserve factual content; do not invent attendees or facts. Output markdown only."
    )
    user = f"Meeting title: {title or 'Board session'}\n\nTranscript:\n{text[:12000]}"
    try:
        out = await chat.chat(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=min(2200, settings.LLM_MAX_TOKENS),
        )
        return (out or "").strip() or text
    except Exception:
        return text


async def _executive_summary(polished: str, title: str) -> str:
    if not polished.strip():
        return ""
    sys_prompt = (
        "Write a concise executive summary (4-8 sentences) for leadership. "
        "Highlight outcomes, risks, and follow-ups. Use plain prose, no markdown."
    )
    user = f"Title: {title or 'Board session'}\n\nMinutes:\n{polished[:8000]}"
    try:
        out = await chat.chat(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=700,
        )
        return (out or "").strip()
    except Exception:
        return ""


async def _knowledge_checks(
    segments: List[str],
    scope: Dict[str, bool],
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    rolling = ""
    for seg in segments:
        hits = await _retrieve_hits(seg, scope, k_each=8)
        if not hits:
            continue
        evidence_blocks = [(f"E{i}", h) for i, h in enumerate(hits[:8])]
        raw_list = await _llm_classify_insights(seg, rolling, evidence_blocks, "silent_assistant")
        rolling = (rolling + " " + seg)[-4000:]
        for raw in raw_list:
            conf = float(raw.get("confidence") or 0)
            if conf < 0.70:
                continue
            eids = raw.get("evidence_ids") or []
            evidence = []
            id_map = {f"E{i}": h for i, h in enumerate(hits[:8])}
            for eid in eids if isinstance(eids, list) else []:
                hit = id_map.get(str(eid).strip()) or id_map.get(f"E{str(eid).strip()}")
                if hit:
                    evidence.append(_evidence_from_hit(hit))
            checks.append(
                {
                    "claim": str(raw.get("transcript_text") or seg)[:500],
                    "classification": str(raw.get("classification") or "related"),
                    "confidence": conf,
                    "interpretation": str(raw.get("assistant_interpretation") or ""),
                    "suggested_action": str(raw.get("suggested_action") or ""),
                    "evidence": evidence[:4],
                }
            )
            if len(checks) >= 20:
                return checks
    return checks


def _markdown_report(
    title: str,
    polished: str,
    summary: str,
    checks: List[Dict[str, Any]],
    session_name: str,
    session_location: str,
) -> str:
    lines = [f"# {title or 'Board Room Report'}", ""]
    if session_name or session_location:
        meta = []
        if session_name:
            meta.append(f"Session: {session_name}")
        if session_location:
            meta.append(f"Location: {session_location}")
        lines.append(" · ".join(meta))
        lines.append("")
    if summary:
        lines.extend(["## Executive summary", "", summary, ""])
    lines.extend(["## Polished minutes", "", polished or "_No transcript content._", ""])
    if checks:
        lines.extend(["## Knowledge validation", ""])
        for i, c in enumerate(checks, 1):
            lines.append(f"### {i}. {c.get('classification', 'related').replace('_', ' ').title()}")
            lines.append(f"**Claim:** {c.get('claim', '')}")
            if c.get("interpretation"):
                lines.append(f"**Interpretation:** {c.get('interpretation')}")
            if c.get("suggested_action"):
                lines.append(f"**Suggested action:** {c.get('suggested_action')}")
            ev = c.get("evidence") or []
            if ev:
                lines.append("**Sources:**")
                for e in ev[:3]:
                    lines.append(f"- {e.get('source_name')} — {str(e.get('matched_text') or '')[:240]}")
            lines.append("")
    return "\n".join(lines).strip()


async def generate_board_room_report(
    *,
    session_id: str,
    title: str,
    transcript: str,
    session_name: str = "",
    session_location: str = "",
    include_rag_validation: bool = True,
    analysis_scope: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    scope = analysis_scope or {
        "documents": True,
        "transcripts": False,
        "books": True,
        "faqs": True,
    }
    polished = await _polish_transcript(transcript, title)
    summary = await _executive_summary(polished, title)
    segments = _split_segments(transcript)
    checks: List[Dict[str, Any]] = []
    if include_rag_validation and segments:
        checks = await _knowledge_checks(segments, scope)
    markdown = _markdown_report(title, polished, summary, checks, session_name, session_location)
    report_id = str(uuid.uuid4())
    payload = {
        "report_id": report_id,
        "session_id": session_id,
        "title": title,
        "session_name": session_name,
        "session_location": session_location,
        "polished_transcript": polished,
        "executive_summary": summary,
        "knowledge_checks": checks,
        "markdown": markdown,
        "include_rag_validation": include_rag_validation,
        "analysis_scope": scope,
    }
    _reports[report_id] = payload
    return payload


def get_report(report_id: str) -> Optional[Dict[str, Any]]:
    return _reports.get(report_id)
