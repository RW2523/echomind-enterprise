"""
Assistant Mode: conservative, knowledge-base-only suggestion generation.

Transcript heuristics decide when to query local RAG; a row is emitted only when
retrieval returns citable chunks. At most one suggestion per call; no auto speech.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..core.config import settings
from ..rag.advanced import retrieve_for_kb_probe
from ..rag.llm import OpenAICompatChat
from ..schemas.assistant_suggestion import GenerateSuggestionsOut, SuggestionOut
from . import suggestion_store

logger = logging.getLogger(__name__)

# Minimum confidence (0–1) before persisting a hand-raise suggestion (product spec: 70%+).
ASSISTANT_MIN_SUGGESTION_CONFIDENCE = 0.7

_UNCERTAIN = re.compile(
    r"\b(not sure|unsure|i think|i guess|maybe|probably|might be|could be|seems like)\b",
    re.I,
)
_VERIFY = re.compile(
    r"\b(is that true|does that match|does this match|what about|can you verify|fact[- ]?check|"
    r"is that accurate|am i right about)\b",
    re.I,
)
_CONTRADICT = re.compile(r"\b(but you said|contradict|that conflicts|doesn't match earlier)\b", re.I)
_KB_HINT = re.compile(
    r"\b(fmr|dod|paragraph|section\s+\d|document|uploaded|policy|regulation|manual|handbook)\b",
    re.I,
)
_REMIND = re.compile(r"\b(remind|don't forget|do not forget|remember to)\b", re.I)
_SUMMARY = re.compile(r"\b(in summary|to recap|tl;dr|summarize)\b", re.I)
_MISSING_CTX = re.compile(
    r"\b(we don't have enough|insufficient (information|context)|missing context|can't tell without|"
    r"need more (detail|info|context)|not enough (data|information) to|without more (data|context)|"
    r"hard to say without)\b",
    re.I,
)


@dataclass
class _Trigger:
    category: str
    confidence: float
    reason: str
    rag_query: str


def _tail(text: str, n: int = 1400) -> str:
    t = (text or "").strip()
    return t[-n:] if len(t) > n else t


def _last_sentence_or_line(text: str, max_len: int = 320) -> str:
    t = _tail(text, 1800).strip()
    if not t:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", t)
    last = (parts[-1] if parts else t).strip()
    if len(last) < 12 and len(parts) > 1:
        last = (parts[-2] + " " + parts[-1]).strip()
    return last[:max_len]


def detect_trigger(text: str) -> Optional[_Trigger]:
    """
    Return a trigger only when transcript signals warrant a suggestion.
    Conservative: requires combined cues or strong single cue.
    """
    window = _tail(text, 1600)
    if len(window) < 40:
        return None

    score = 0
    has_q = "?" in window[-500:]
    if has_q:
        score += 2
    if _UNCERTAIN.search(window):
        score += 2
    if _VERIFY.search(window):
        score += 3
    if _CONTRADICT.search(window):
        score += 3
    if _KB_HINT.search(window):
        score += 2
    if _REMIND.search(window):
        score += 2
    if len(text) > 800 and _SUMMARY.search(window):
        score += 2
    if _MISSING_CTX.search(window):
        score += 3

    if score < 2:
        return None

    rag_q = _last_sentence_or_line(text, 400)
    if not rag_q:
        rag_q = window[-200:].strip()

    if _MISSING_CTX.search(window) and score >= 4:
        return _Trigger(
            "missing_context",
            0.58,
            "The conversation signals missing context or insufficient information—gather specifics before deciding.",
            rag_q,
        )
    if _CONTRADICT.search(window):
        return _Trigger("contradiction", 0.62, "Possible conflict with earlier context.", rag_q)
    if _VERIFY.search(window):
        return _Trigger("fact_check", 0.68, "Transcript asks for verification against sources.", rag_q)
    if has_q and score >= 2:
        return _Trigger("follow_up_question", 0.55, "Question detected; a targeted follow-up may help.", rag_q)
    if _UNCERTAIN.search(window):
        return _Trigger("clarification", 0.52, "Uncertainty phrasing; clarifying may reduce risk.", rag_q)
    if _REMIND.search(window):
        return _Trigger("action_reminder", 0.58, "Reminder-style intent detected.", rag_q)
    if len(text) > 800 and _SUMMARY.search(window):
        return _Trigger("summary_help", 0.5, "Summary or recap cue in a longer transcript.", rag_q)
    if _KB_HINT.search(window) and score >= 2:
        return _Trigger("relevant_knowledge", 0.54, "Topic may match indexed documents or transcripts.", rag_q)
    return None


def _hits_to_citations(hits: List[Dict[str, Any]]) -> List[dict]:
    out: List[dict] = []
    for h in hits[:5]:
        src = h.get("source") or {}
        snippet = (h.get("compressed") or h.get("text") or "")[:360]
        c = {
            "filename": src.get("filename") or "Unknown document",
            "snippet": snippet,
            "doc_id": src.get("doc_id"),
            "section_path": src.get("section_path"),
            "section": src.get("section"),
            "page_number": src.get("page_number"),
            "score": h.get("score"),
            "doc_type": src.get("doc_type"),
            "chunk_index": src.get("chunk_index"),
        }
        out.append({k: v for k, v in c.items() if v is not None})
    return out


def _evidence_from_hits(hits: List[Dict[str, Any]]) -> Tuple[str, float]:
    if not hits:
        return "none", 0.45
    top = float(hits[0].get("score") or 0.0)
    if top >= 0.72:
        return "grounded", min(0.9, 0.55 + top * 0.35)
    if top >= 0.48:
        return "partial", min(0.75, 0.45 + top * 0.4)
    return "weak", min(0.65, 0.4 + top * 0.5)


async def _maybe_refine_llm(
    title: str,
    short_text: str,
    speak_text: str,
    category: str,
    transcript_tail: str,
    extra_context: str = "",
) -> Tuple[str, str, str]:
    if os.getenv("ECHOMIND_ASSISTANT_SUGGESTION_LLM", "0").lower() not in ("1", "true", "yes"):
        return title, short_text, speak_text
    base = (settings.LLM_BASE_URL or "").strip()
    if not base:
        return title, short_text, speak_text
    chat = OpenAICompatChat(base, settings.LLM_MODEL)
    sys = (
        "You refine assistant hand-raise cards. Output ONLY compact JSON with keys: "
        "title (<=8 words), short_text (<=220 chars), speak_text (<=500 chars, conversational for TTS). "
        "Do not include markdown. Category is fixed: "
        f"{category}. Ground only in the transcript tail and optional extra_context; do not invent facts."
    )
    user = json.dumps(
        {
            "transcript_tail": transcript_tail[:900],
            "extra_context": (extra_context or "")[:1400],
            "draft_title": title,
            "draft_short": short_text,
        }
    )
    try:
        raw = await chat.chat(
            [{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.15,
            max_tokens=220,
        )
    except Exception as e:
        logger.warning("Assistant suggestion LLM refine skipped: %s", e)
        return title, short_text, speak_text
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start < 0 or end <= start:
            return title, short_text, speak_text
        j = json.loads(raw[start:end])
        t = str(j.get("title") or title).strip()[:500]
        s = str(j.get("short_text") or short_text).strip()[:2000]
        sp = str(j.get("speak_text") or speak_text).strip()[:8000]
        if t and s and sp:
            return t, s, sp
    except Exception:
        pass
    return title, short_text, speak_text


async def generate_suggestions(
    session_id: str,
    recent_transcript: str,
    use_knowledge_base: bool = False,
    context_window: str = "all",
) -> GenerateSuggestionsOut:
    """
    KB-only hand-raises: emit at most one suggestion when local retrieval returns citable chunks.
    Rules, notes, and transcript-only cards are out of product scope.
    """
    text = (recent_transcript or "").strip()
    if len(text) < 48:
        return GenerateSuggestionsOut(suggestions=[], skipped_reason="too_short")

    if suggestion_store.is_within_cooldown(session_id):
        return GenerateSuggestionsOut(suggestions=[], skipped_reason="cooldown")

    if suggestion_store.count_pending(session_id) >= suggestion_store.ASSISTANT_MAX_PENDING_PER_SESSION:
        return GenerateSuggestionsOut(suggestions=[], skipped_reason="max_pending")

    if not use_knowledge_base:
        return GenerateSuggestionsOut(suggestions=[], skipped_reason="kb_disabled")

    trig = detect_trigger(text)
    if not trig:
        return GenerateSuggestionsOut(suggestions=[], skipped_reason="no_signal")

    tail = _tail(text, 900)
    excerpt = _last_sentence_or_line(text, 280)
    rag_q = (trig.rag_query or "").strip()
    if len(rag_q) < 8:
        rag_q = excerpt.strip() if len(excerpt.strip()) >= 8 else tail.strip()[-400:].strip()
    if len(rag_q) < 8:
        return GenerateSuggestionsOut(suggestions=[], skipped_reason="no_signal")

    citations: List[dict] = []
    evidence_status = "none"
    confidence = trig.confidence
    try:
        hits, _probe_mode = await retrieve_for_kb_probe(
            rag_q, k=8, context_window=context_window or "all"
        )
        citations = _hits_to_citations(hits)
        ev, conf_boost = _evidence_from_hits(hits)
        evidence_status = ev
        confidence = min(0.92, max(confidence, conf_boost))
    except Exception as e:
        logger.warning("Assistant suggestion RAG skipped: %s", e)
        citations = []

    if not citations:
        return GenerateSuggestionsOut(suggestions=[], skipped_reason="no_signal")

    source_origin = "transcript_plus_rag"
    cat = trig.category
    title = {
        "fact_check": "Verify this claim",
        "contradiction": "Possible contradiction",
        "relevant_knowledge": "Related sources",
        "action_reminder": "Reminder",
        "follow_up_question": "Follow-up angle",
        "clarification": "Clarify wording",
        "summary_help": "Summary help",
        "missing_context": "More context may help",
    }.get(cat, "Knowledge base match")

    short_text = f"{trig.reason} Latest bit: {excerpt}"[:1900]
    speak_text = (
        f"{trig.reason} For example, you could look at this part of the conversation: {excerpt}"
    )[:7900]

    title, short_text, speak_text = await _maybe_refine_llm(title, short_text, speak_text, cat, tail, "")

    if evidence_status == "weak":
        confidence = min(confidence, 0.62)
    elif evidence_status == "none":
        confidence = min(confidence, 0.58)

    if confidence < ASSISTANT_MIN_SUGGESTION_CONFIDENCE:
        return GenerateSuggestionsOut(suggestions=[], skipped_reason="low_confidence")

    fp = " ".join(short_text.lower().split())[:120]
    for existing in suggestion_store.recent_pending_fingerprints(session_id, 14):
        if existing and fp and (fp in existing or existing in fp):
            return GenerateSuggestionsOut(suggestions=[], skipped_reason="dedupe")

    row = suggestion_store.insert_suggestion(
        session_id=session_id,
        mode="ASSISTANT",
        title=title,
        short_text=short_text,
        speak_text=speak_text,
        reason=trig.reason,
        category=cat,
        confidence=float(confidence),
        source_origin=source_origin,
        evidence_status=evidence_status,
        citations=citations,
        status="pending",
        trigger_excerpt=(excerpt[:2000] if excerpt else None),
    )
    return GenerateSuggestionsOut(suggestions=[row], skipped_reason=None)


def generate_suggestions_sync(
    session_id: str,
    recent_transcript: str,
    use_knowledge_base: bool = False,
    context_window: str = "all",
) -> GenerateSuggestionsOut:
    """Test helper: run async generator from sync code."""
    import asyncio

    return asyncio.run(
        generate_suggestions(session_id, recent_transcript, use_knowledge_base, context_window)
    )
