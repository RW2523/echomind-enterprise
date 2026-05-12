"""
Silent Assistant: conservative transcript analysis → CorrectionFinding rows.

Display-only pipeline: no TTS, no audio, no voice imports.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..rag.advanced import retrieve_for_kb_probe
from ..schemas.silent_finding import (
    CorrectionFindingOut,
    FindingCategory,
    SilentAnalyzeIn,
    SilentAnalyzeOut,
    StatusLabel,
)
from . import finding_store

# Policy strings (task-specified wording by evidence tier).
EVIDENCE_REASON: Dict[str, str] = {
    "grounded": (
        "This statement appears inconsistent based on indexed knowledge or active rules."
    ),
    "partial": (
        "This statement may be inconsistent with indexed knowledge or active rules."
    ),
    "weak": "I found a possible issue, but there is not enough evidence to confirm it.",
    "none": "No supporting evidence was found in the local knowledge base.",
}

SUPPORT_REASON: Dict[str, str] = {
    "grounded": "Retrieved sources substantially overlap with this segment and read as supportive.",
    "partial": "Retrieved sources partially align with this segment; confirm nuances if stakes are high.",
    "weak": "A related passage was retrieved; treat as weak corroboration only.",
    "none": "A related passage was retrieved; treat as weak corroboration only.",
}

AMBIGUOUS_ALIGN_REASON = (
    "Indexed passages relate to this topic but agreement or disagreement with this exact wording is unclear."
)

UNRELATED_REASON = (
    "Retrieval returned nearby material that may not directly substantiate or refute this exact claim."
)

_STRONG_CLAIM = re.compile(
    r"\b(always|never|must|guaranteed|impossible|definitely|certainly|all\s+\w+\s+are|none\s+of)\b",
    re.I,
)
_NUMERIC = re.compile(r"\b\d{3,}\b|\$\s*\d+|\d+\s*%")
_VERIFY = re.compile(r"\b(according to|per the|section|paragraph|fmr|policy requires)\b", re.I)
_UNCERTAIN = re.compile(r"\b(i think|maybe|probably|not sure|might)\b", re.I)


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


def _evidence_tier(hits: List[Dict[str, Any]]) -> Tuple[str, float]:
    if not hits:
        return "none", 0.0
    top = float(hits[0].get("score") or 0.0)
    if top >= 0.72:
        return "grounded", top
    if top >= 0.48:
        return "partial", top
    if top >= 0.28:
        return "weak", top
    return "none", top


def _reason_for(evidence_status: str) -> str:
    return EVIDENCE_REASON.get(evidence_status, EVIDENCE_REASON["none"])


def _kb_probe_worthy(text: str) -> bool:
    t = text.strip()
    if len(t) < 36:
        return False
    score = 0
    if _STRONG_CLAIM.search(t):
        score += 2
    if _NUMERIC.search(t):
        score += 1
    if _VERIFY.search(t):
        score += 2
    if "?" in t[-400:]:
        score += 1
    if _UNCERTAIN.search(t):
        score += 1
    return score >= 3


def _fingerprint(category: str, text: str) -> str:
    norm = " ".join(text.lower().split())[:120]
    return f"{category}|{norm}"


def _tok(s: str) -> set:
    return set(re.findall(r"[a-zA-Z]{3,}", (s or "").lower()))


_ALIGNMENT_RANK = {"contradicts": 4, "supports": 3, "ambiguous": 2, "unrelated": 1}


def _alignment_one(segment: str, chunk: str, score: float) -> str:
    """Stance for one retrieved chunk vs transcript segment (same heuristics as before)."""
    chunk = (chunk or "")[:1600]
    seg_w, chunk_w = _tok(segment), _tok(chunk)
    inter = len(seg_w & chunk_w)
    union = len(seg_w | chunk_w) or 1
    jacc = inter / union

    seg_nums = set(re.findall(r"\b\d{3,}\b", segment))
    chunk_nums = set(re.findall(r"\b\d{3,}\b", chunk))
    numeric_clash = bool(seg_nums and chunk_nums and not (seg_nums & chunk_nums))

    conflict_lex = re.search(
        r"\b(however|incorrect|not accurate|inaccurate|false|contrary to|does not apply|"
        r"exception|unless|except when|shall not|must not|prohibited)\b",
        chunk,
        re.I,
    )

    strong_overlap = inter >= 6 or jacc >= 0.11
    moderate_overlap = inter >= 3 or jacc >= 0.035

    if numeric_clash and score >= 0.42:
        return "contradicts"
    if conflict_lex and score >= 0.48 and moderate_overlap:
        return "contradicts"
    if strong_overlap and score >= 0.52:
        return "supports"
    if moderate_overlap and score >= 0.42:
        return "ambiguous"
    if score >= 0.65 and not moderate_overlap:
        return "unrelated"
    if score >= 0.58:
        return "ambiguous"
    return "unrelated"


def _rag_alignment(segment: str, hits: List[Dict[str, Any]]) -> str:
    """
    Stance between transcript segment and retrieved chunks — evaluates top several hits
    (semantic-first RAG returns diverse chunks) and keeps the strongest signal; contradicts
    wins over supports when mixed.
    """
    if not hits:
        return "unrelated"
    best = "unrelated"
    for h in hits[:6]:
        chunk = (h.get("compressed") or h.get("text") or "")[:1600]
        score = float(h.get("score") or 0.0)
        one = _alignment_one(segment, chunk, score)
        if _ALIGNMENT_RANK.get(one, 0) > _ALIGNMENT_RANK.get(best, 0):
            best = one
    return best


async def analyze_segment(session_id: str, inp: SilentAnalyzeIn) -> SilentAnalyzeOut:
    text = (inp.text or "").strip()
    if len(text) < 28:
        return SilentAnalyzeOut(findings=[], skipped_reason="too_short")

    if finding_store.is_within_cooldown(session_id):
        return SilentAnalyzeOut(findings=[], skipped_reason="cooldown")

    if finding_store.count_pending(session_id) >= finding_store.SILENT_MAX_PENDING_FINDINGS:
        return SilentAnalyzeOut(findings=[], skipped_reason="max_pending")

    if not inp.use_knowledge_base:
        return SilentAnalyzeOut(findings=[], skipped_reason="kb_disabled")

    if not _kb_probe_worthy(text):
        return SilentAnalyzeOut(findings=[], skipped_reason="no_signal")

    citations: List[dict] = []
    evidence_status = "none"
    source_origin = "transcript"
    top_score = 0.0
    hits: List[Dict[str, Any]] = []
    if inp.use_knowledge_base:
        try:
            # Query string for embedding/BM25: use segment tail on long text so retrieval stays focused.
            q = text.strip()
            if len(q) > 900:
                q = q[-900:].strip()
            hits, _probe_mode = await retrieve_for_kb_probe(
                q, k=8, context_window=inp.context_window or "all"
            )
            citations = _hits_to_citations(hits)
            evidence_status, top_score = _evidence_tier(hits)
            if citations:
                source_origin = "transcript_plus_rag"
            else:
                source_origin = "transcript"
        except Exception:
            hits = []
            citations = []
            evidence_status = "none"
            source_origin = "transcript"
            top_score = 0.0

    align = _rag_alignment(text, hits) if citations else "unrelated"

    category = FindingCategory.unsupported_claim.value
    status_label = StatusLabel.unsupported.value
    confidence = 0.5
    suggested = "No close match in the local index; confirm externally if needed."
    reason = _reason_for(evidence_status)

    if citations:
        if align == "supports":
            category = FindingCategory.useful_suggestion.value
            if evidence_status == "grounded":
                status_label = StatusLabel.likely_correct.value
                confidence = min(0.9, 0.62 + float(top_score or 0) * 0.22)
            else:
                status_label = StatusLabel.suggestion_available.value
                confidence = min(0.82, 0.54 + float(top_score or 0) * 0.2)
            reason = SUPPORT_REASON.get(evidence_status, SUPPORT_REASON["weak"])
            suggested = (
                "Sources appear to line up with this wording; keep citations for audit or stakeholder review."
            )
        elif align == "contradicts":
            if evidence_status == "grounded":
                category = FindingCategory.contradiction_with_indexed_knowledge.value
                status_label = StatusLabel.contradicted.value
                confidence = min(0.88, 0.55 + float(top_score or citations[0].get("score") or 0) * 0.25)
                suggested = "Compare the spoken claim with the cited excerpt; revise if the document governs this case."
                reason = _reason_for(evidence_status)
            elif evidence_status == "partial":
                category = FindingCategory.factual_inconsistency.value
                status_label = StatusLabel.possibly_wrong.value
                confidence = 0.62
                suggested = "Cross-check this segment against the cited sources before relying on it."
                reason = _reason_for(evidence_status)
            else:
                category = FindingCategory.needs_verification.value
                status_label = StatusLabel.needs_verification.value
                confidence = 0.54
                suggested = "Signals conflict with indexed text are tentative; gather more context before changing the claim."
                reason = _reason_for(evidence_status)
        elif align == "ambiguous":
            category = FindingCategory.possible_misinterpretation.value
            status_label = StatusLabel.needs_verification.value
            confidence = 0.57
            reason = AMBIGUOUS_ALIGN_REASON
            suggested = "Re-read the cited passage against this phrasing; the match is plausible but not definitive."
        elif align == "unrelated" and evidence_status in ("grounded", "partial"):
            category = FindingCategory.unsupported_claim.value
            status_label = StatusLabel.needs_verification.value
            confidence = 0.52
            reason = UNRELATED_REASON
            suggested = "Retrieved text may be off-topic; verify this claim against the right document or section."
        elif evidence_status == "weak":
            category = FindingCategory.needs_verification.value
            status_label = StatusLabel.needs_verification.value
            confidence = 0.52
            suggested = "Treat this as unverified; gather stronger sources if the claim matters."
            reason = _reason_for(evidence_status)
        else:
            category = FindingCategory.unsupported_claim.value
            status_label = StatusLabel.unsupported.value
            confidence = 0.5
            suggested = "No close match in the local index; confirm externally if needed."
            reason = _reason_for(evidence_status)
    else:
        if evidence_status == "weak":
            category = FindingCategory.needs_verification.value
            status_label = StatusLabel.needs_verification.value
            confidence = 0.52
            suggested = "Treat this as unverified; gather stronger sources if the claim matters."
            reason = _reason_for(evidence_status)
        else:
            category = FindingCategory.unsupported_claim.value
            status_label = StatusLabel.unsupported.value
            confidence = 0.5
            suggested = "No close match in the local index; confirm externally if needed."
            reason = _reason_for(evidence_status)

    fp = _fingerprint(category, text)
    if fp in finding_store.recent_fingerprints(session_id, 20):
        return SilentAnalyzeOut(findings=[], skipped_reason="dedupe")

    row = finding_store.insert_finding(
        session_id=session_id,
        transcript_segment_id=inp.transcript_segment_id,
        turn_id=inp.turn_id,
        original_text=text,
        span_start=0,
        span_end=min(len(text), 4000),
        category=category,
        status_label=status_label,
        suggested_correction=suggested,
        reason=reason,
        evidence_status=evidence_status,
        confidence=float(confidence),
        source_origin=source_origin,
        citations=citations,
    )
    return SilentAnalyzeOut(findings=[row], skipped_reason=None)


def analyze_segment_sync(session_id: str, inp: SilentAnalyzeIn) -> SilentAnalyzeOut:
    import asyncio

    return asyncio.run(analyze_segment(session_id, inp))
