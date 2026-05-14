"""
Live Silent Assistant analysis: retrieval-first, optional LLM classification using only retrieved chunks.
In-memory dedupe (5 minutes). No DB persistence of insights.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ..core.config import settings
from ..rag.index import index
from ..rag.llm import OpenAICompatChat

logger = logging.getLogger(__name__)

# Retrieval: minimum cosine IP score to consider evidence "present" (below = no LLM / no insights)
_MIN_RETRIEVAL_SCORE = float(getattr(settings, "ECHOMIND_LIVE_ASSISTANT_MIN_SCORE", "0.32"))

# Dedupe: (session_id, norm_snippet, classification, doc_id, chunk_id) -> expiry monotonic
_dedupe_entries: Dict[str, float] = {}
_DEDUPE_TTL_SEC = 300.0

_CLASSIFICATIONS = frozenset({"supported", "contradicted", "related", "missing_context", "warning"})

chat = OpenAICompatChat(settings.LLM_BASE_URL, settings.LLM_MODEL)


def _purge_dedupe() -> None:
    now = time.monotonic()
    dead = [k for k, exp in _dedupe_entries.items() if exp <= now]
    for k in dead:
        del _dedupe_entries[k]


def _dedupe_key(session_id: str, norm_text: str, classification: str, doc_id: str, chunk_id: str) -> str:
    return f"{session_id}|{classification}|{doc_id}|{chunk_id}|{hash(norm_text)}"


def _should_dedupe(session_id: str, norm_text: str, classification: str, doc_id: str, chunk_id: str) -> bool:
    _purge_dedupe()
    key = _dedupe_key(session_id, norm_text[:500], classification, doc_id, chunk_id)
    now = time.monotonic()
    if key in _dedupe_entries and _dedupe_entries[key] > now:
        return True
    _dedupe_entries[key] = now + _DEDUPE_TTL_SEC
    return False


def normalize_transcript_text(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text.strip())
    return t


def _build_retrieval_query(transcript_window: str, rolling_context: str) -> str:
    w = normalize_transcript_text(transcript_window)
    if len(w) > 1800:
        w = w[-1800:]
    c = normalize_transcript_text(rolling_context)
    if c and len(c) > 1200:
        c = c[-1200:]
    if w and c:
        return f"{w}\n\nContext:\n{c}"
    return w or c or ""


async def _retrieve_hits(
    query: str,
    scope: Dict[str, bool],
    k_each: int = 10,
) -> List[Dict[str, Any]]:
    """Merge document and transcript retrieval based on analysis_scope."""
    if not query.strip():
        return []
    hits: Dict[str, Dict[str, Any]] = {}
    doc_on = scope.get("documents", True) or scope.get("books", True) or scope.get("faqs", True)
    tr_on = scope.get("transcripts", False)

    if doc_on:
        try:
            for h in await index.search_document_only(query, k_each):
                cid = h.get("chunk_id")
                if cid and cid not in hits:
                    hits[cid] = h
                elif cid and h.get("score", 0) > hits[cid].get("score", 0):
                    hits[cid] = h
        except Exception as e:
            logger.warning("live_analysis: search_document_only failed: %s", e)
    if tr_on:
        try:
            for h in await index.search_transcript_only(query, k_each):
                cid = h.get("chunk_id")
                if cid and cid not in hits:
                    hits[cid] = h
                elif cid and h.get("score", 0) > hits[cid].get("score", 0):
                    hits[cid] = h
        except Exception as e:
            logger.warning("live_analysis: search_transcript_only failed: %s", e)

    # If scope wants docs but index empty for document_only, fallback to full search filtered
    if not hits and query.strip():
        try:
            for h in await index.search(query, k_each):
                cid = h.get("chunk_id")
                if not cid:
                    continue
                src = h.get("source") or {}
                fn = (src.get("filename") or "")
                is_tr = fn.startswith("transcript_") or src.get("type") == "transcript"
                if is_tr and not tr_on:
                    continue
                if not is_tr and not doc_on:
                    continue
                hits[cid] = h
        except Exception as e:
            logger.warning("live_analysis: fallback search failed: %s", e)

    out = sorted(hits.values(), key=lambda x: float(x.get("score") or 0), reverse=True)
    return out[:18]


def _hit_passes_evidence_floor(h: Dict[str, Any]) -> bool:
    return float(h.get("score") or 0) >= _MIN_RETRIEVAL_SCORE


def _source_type_from_hit(source: Dict[str, Any]) -> str:
    fn = (source.get("filename") or "")
    meta = source.get("docType") or source.get("filetype") or ""
    low = (fn + str(meta)).lower()
    if fn.startswith("transcript_") or source.get("type") == "transcript":
        return "transcript"
    if "faq" in low:
        return "faq"
    if "book" in low or source.get("is_book"):
        return "book"
    if fn.endswith(".pdf") or "pdf" in low:
        return "document"
    return "document"


def _evidence_from_hit(h: Dict[str, Any]) -> Dict[str, Any]:
    src = h.get("source") or {}
    page = src.get("pageNumber") or src.get("page")
    section = src.get("section") or src.get("section_path")
    name = src.get("filename") or src.get("doc_name") or "source"
    return {
        "source_name": name if isinstance(name, str) else str(name),
        "source_type": _source_type_from_hit(src),
        "doc_id": src.get("doc_id"),
        "chunk_id": h.get("chunk_id"),
        "page": page,
        "section": section if isinstance(section, str) else (str(section) if section is not None else None),
        "matched_text": (h.get("text") or "")[:1200],
    }


def _priority_for_classification(c: str) -> int:
    if c in ("contradicted", "warning"):
        return 3
    if c == "missing_context":
        return 1
    if c == "supported":
        return 2
    return 2  # related


def _hand_raise_queue_rank(classification: str) -> int:
    """Ordering for Personal Assistant: warning → contradicted → related → supported."""
    return {"warning": 4, "contradicted": 3, "related": 2, "supported": 1}.get(classification, 0)


def _eligible_hand_raise(insight: Dict[str, Any], mode: str) -> bool:
    if mode != "personal_assistant":
        return False
    conf = float(insight.get("confidence") or 0)
    if conf < 0.75:
        return False
    ev = insight.get("evidence") or []
    if not ev:
        return False
    cls = str(insight.get("classification") or "")
    if cls == "missing_context":
        return False
    if cls not in ("warning", "contradicted", "related", "supported"):
        return False
    if cls == "supported" and conf < 0.88:
        return False
    return True


def _apply_mode_post_process(insight: Dict[str, Any], mode: str) -> Dict[str, Any]:
    if mode == "silent_assistant":
        insight["show_hand_raise"] = False
        insight["suggested_response"] = None
        return insight
    sr = insight.get("suggested_response")
    if sr and str(sr).strip():
        insight["suggested_response"] = str(sr).strip()[:500]
    else:
        insight["suggested_response"] = None
    insight["show_hand_raise"] = _eligible_hand_raise(insight, mode)
    return insight


def _confidence_to_ui(conf: float) -> Tuple[bool, str]:
    """Returns (show_highlight, priority level string)."""
    if conf < 0.70:
        return False, "low"
    if conf >= 0.85:
        return True, "high"
    return True, "medium"


def _filter_insight_confidence(insight: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    conf = float(insight.get("confidence") or 0)
    if conf < 0.50:
        return None
    if 0.50 <= conf < 0.70:
        return None  # internal only — do not return to UI for MVP
    show_hl, pri = _confidence_to_ui(conf)
    insight["show_highlight"] = show_hl
    insight["priority"] = pri if conf >= 0.85 else ("medium" if conf >= 0.70 else "low")
    if not show_hl:
        return None
    return insight


async def _llm_classify_insights(
    transcript_window: str,
    rolling_context: str,
    evidence_blocks: List[Tuple[str, Dict[str, Any]]],
    mode: str,
) -> List[Dict[str, Any]]:
    """
    evidence_blocks: list of (EVIDENCE_ID, hit_dict)
    Returns list of raw dicts from LLM: transcript_text, classification, confidence, evidence_ids, ...
    """
    lines = []
    for eid, h in evidence_blocks:
        lines.append(f"[{eid}] score={h.get('score', 0):.3f}\n{(h.get('text') or '')[:900]}")
    block = "\n\n".join(lines)
    extra = ""
    if mode == "personal_assistant":
        extra = (
            " Also include suggested_response: one short user-facing phrase or talking point (max 25 words) "
            "the user could say next, grounded only in the evidence. If none fits, use null or omit."
        )
    sys = (
        "You are a compliance assistant. You ONLY use the numbered evidence excerpts below. "
        "Do not invent sources or facts not in the excerpts. "
        "Output a single JSON object with key \"insights\" — an array of 0 to 3 items. "
        "Each item: "
        "transcript_text (exact substring copied from TRANSCRIPT_WINDOW), "
        "classification (one of: supported, contradicted, related, missing_context, warning), "
        "confidence (number 0.0-1.0), "
        "evidence_ids (array of evidence id strings like E0, E1 — only from the list), "
        "assistant_interpretation (short), "
        "suggested_action (short)."
        f"{extra} "
        "If nothing in TRANSCRIPT_WINDOW is meaningfully checkable against evidence, return {\"insights\": []}. "
        "Use \"warning\" for risk flags; \"contradicted\" only if evidence clearly conflicts with a specific claim in transcript_text."
    )
    user = (
        f"TRANSCRIPT_WINDOW:\n{transcript_window}\n\n"
        f"ROLLING_CONTEXT (background only):\n{rolling_context[:2000]}\n\n"
        f"NUMBERED_EVIDENCE:\n{block}\n\n"
        "Respond with JSON only, format: {\"insights\": [...]}"
    )
    raw = await chat.chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        temperature=0.1,
        max_tokens=1200 if mode == "personal_assistant" else 1000,
    )
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("live_analysis: LLM JSON parse failed, raw=%s", raw[:400])
        return []
    arr = data.get("insights")
    if not isinstance(arr, list):
        return []
    return [x for x in arr if isinstance(x, dict)]


def _map_raw_llm_insight(
    raw: Dict[str, Any],
    id_by_eid: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    ttext = normalize_transcript_text(str(raw.get("transcript_text") or ""))
    if not ttext:
        return None
    cls = str(raw.get("classification") or "").strip().lower()
    if cls not in _CLASSIFICATIONS:
        cls = "related"
    try:
        conf = float(raw.get("confidence", 0.7))
    except (TypeError, ValueError):
        conf = 0.7
    eids = raw.get("evidence_ids") or raw.get("evidence_chunk_ids") or []
    if not isinstance(eids, list):
        eids = []
    evidence_list: List[Dict[str, Any]] = []
    primary_doc = ""
    primary_chunk = ""
    for eid in eids:
        key = str(eid).strip()
        hit = id_by_eid.get(key)
        if not hit and key.isdigit():
            hit = id_by_eid.get(f"E{key}")
        if not hit and not key.upper().startswith("E"):
            hit = id_by_eid.get(f"E{key}")
        if not hit:
            continue
        ev = _evidence_from_hit(hit)
        evidence_list.append(ev)
        if not primary_doc and ev.get("doc_id"):
            primary_doc = str(ev.get("doc_id"))
        if not primary_chunk and ev.get("chunk_id"):
            primary_chunk = str(ev.get("chunk_id"))
    if not evidence_list:
        return None

    sr_raw = raw.get("suggested_response")
    suggested_response = None
    if sr_raw is not None and str(sr_raw).strip():
        suggested_response = str(sr_raw).strip()[:500]

    insight = {
        "id": f"ins_{uuid.uuid4().hex[:12]}",
        "transcript_text": ttext,
        "classification": cls,
        "confidence": max(0.0, min(1.0, conf)),
        "start_char": None,
        "end_char": None,
        "paragraph_id": None,
        "show_hand_raise": False,
        "evidence": evidence_list,
        "assistant_interpretation": str(raw.get("assistant_interpretation") or "")[:2000],
        "suggested_action": str(raw.get("suggested_action") or "")[:1200],
        "suggested_response": suggested_response,
    }
    # Priority ordering helper stored on object for sort
    insight["_prio"] = _priority_for_classification(cls)
    return insight


async def analyze_window(
    session_id: str,
    mode: str,
    transcript_window: str,
    rolling_context: str,
    analysis_scope: Dict[str, bool],
) -> Dict[str, Any]:
    tw = normalize_transcript_text(transcript_window)
    rc = normalize_transcript_text(rolling_context)
    if not tw and not rc:
        return {"session_id": session_id, "mode": mode, "insights": []}

    query = _build_retrieval_query(tw, rc)
    hits = await _retrieve_hits(query, analysis_scope or {})
    hits = [h for h in hits if _hit_passes_evidence_floor(h)]
    if not hits:
        return {"session_id": session_id, "mode": mode, "insights": []}

    evidence_blocks: List[Tuple[str, Dict[str, Any]]] = []
    id_by_eid: Dict[str, Dict[str, Any]] = {}
    for i, h in enumerate(hits[:14]):
        eid = f"E{i}"
        evidence_blocks.append((eid, h))
        id_by_eid[eid] = h

    raw_insights: List[Dict[str, Any]] = []
    try:
        raw_insights = await _llm_classify_insights(tw or query[:800], rc, evidence_blocks, mode)
    except Exception as e:
        logger.warning("live_analysis: LLM classify failed: %s", e)
        raw_insights = []

    built: List[Dict[str, Any]] = []
    for raw in raw_insights:
        ins = _map_raw_llm_insight(raw, id_by_eid)
        if not ins:
            continue
        filtered = _filter_insight_confidence(ins)
        if not filtered:
            continue
        # Dedupe
        ev0 = (filtered.get("evidence") or [{}])[0]
        doc_id = str(ev0.get("doc_id") or "")
        chunk_id = str(ev0.get("chunk_id") or "")
        norm = normalize_transcript_text(filtered["transcript_text"])[:500]
        cls = filtered["classification"]
        if _should_dedupe(session_id, norm, cls, doc_id, chunk_id):
            continue
        filtered = _apply_mode_post_process(filtered, mode)
        built.append(filtered)

    # Sort: Personal Assistant — Hand Raise order (warning → contradicted → related → supported), then confidence.
    # Silent Assistant — highlight priority (_prio) then confidence.
    if mode == "personal_assistant":
        built.sort(
            key=lambda x: (
                -_hand_raise_queue_rank(str(x.get("classification") or "")),
                -float(x.get("confidence") or 0),
            )
        )
    else:
        built.sort(
            key=lambda x: (
                -int(x.get("_prio", 0)),
                -float(x.get("confidence") or 0),
            )
        )
    for b in built:
        b.pop("_prio", None)

    return {"session_id": session_id, "mode": mode, "insights": built}
