"""
Knowledge-base-only transcript analysis for Assistant and Silent Assistant modes.

Uses the same hybrid retrieval path as chat RAG (``retrieve_for_kb_probe``). Classification
and confidence are derived from retrieval scores and lexical overlap — no LLM is used to
decide factual correctness.
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import List, Optional, Tuple

from ..rag.advanced import retrieve_for_kb_probe
from ..schemas.transcript_analyze import (
    AnalyzeTranscriptIn,
    AnalyzeTranscriptOut,
    AssistantAnalysisItemOut,
    AssistantSourceOut,
    KbFindingLabel,
)
from ..silent_assistant import finding_store
from ..silent_assistant.silent_analyzer import _evidence_tier, _hits_to_citations, _rag_alignment
from . import suggestion_store

logger = logging.getLogger(__name__)

ASSISTANT_HAND_RAISE_MIN = 0.70
SILENT_HIGHLIGHT_MIN_EXCLUSIVE = 0.70  # spec: highlight when confidence *above* 70%

_MAX_SPANS_PER_CALL = 6
_MIN_SPAN_CHARS = 36


def _split_spans(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    chunks = re.split(r"(?<=[.!?])\s+", t)
    out: List[str] = []
    for c in chunks:
        s = c.strip()
        if len(s) >= _MIN_SPAN_CHARS:
            out.append(s)
    if not out and len(t) >= _MIN_SPAN_CHARS:
        out = [t[-2000:].strip()] if len(t) > 2000 else [t]
    return out[:_MAX_SPANS_PER_CALL]


def _sources_from_citations(cites: List[dict]) -> List[AssistantSourceOut]:
    out: List[AssistantSourceOut] = []
    for c in cites[:5]:
        out.append(
            AssistantSourceOut(
                document_id=str(c.get("doc_id")) if c.get("doc_id") is not None else None,
                document_name=str(c.get("filename") or "Unknown document"),
                page=int(c["page_number"]) if c.get("page_number") is not None else None,
                snippet=str(c.get("snippet") or "")[:480],
                score=float(c.get("score") or 0.0),
            )
        )
    return out


def _label_confidence(align: str, tier: str, top_score: float) -> Tuple[KbFindingLabel, float]:
    ts = float(top_score or 0.0)

    if align == "contradicts":
        if tier == "grounded":
            return KbFindingLabel.contradicted, min(0.92, 0.68 + ts * 0.22)
        if tier == "partial":
            return KbFindingLabel.contradicted, min(0.86, 0.62 + ts * 0.2)
        if tier == "weak":
            return KbFindingLabel.needs_review, min(0.74, 0.55 + ts * 0.18)
        return KbFindingLabel.unverified, min(0.62, 0.48 + ts * 0.12)

    if align == "supports":
        if tier == "grounded":
            return KbFindingLabel.supported, min(0.92, 0.72 + ts * 0.2)
        if tier == "partial":
            return KbFindingLabel.supported, min(0.84, 0.66 + ts * 0.2)
        if tier == "weak":
            return KbFindingLabel.related, min(0.72, 0.55 + ts * 0.22)
        return KbFindingLabel.unverified, min(0.58, 0.45 + ts * 0.1)

    if align == "ambiguous":
        if tier in ("grounded", "partial"):
            return KbFindingLabel.related, min(0.78, 0.58 + ts * 0.2)
        if tier == "weak":
            return KbFindingLabel.unverified, min(0.65, 0.5 + ts * 0.15)
        return KbFindingLabel.unverified, min(0.55, 0.48 + ts * 0.1)

    if tier in ("grounded", "partial") and ts >= 0.38:
        return KbFindingLabel.related, min(0.72, 0.52 + ts * 0.22)
    if tier == "weak":
        return KbFindingLabel.unverified, min(0.62, 0.48 + ts * 0.14)
    return KbFindingLabel.unverified, min(0.55, 0.42 + ts * 0.1)


def _span_char_range(span: str, full: str, offset_hint: int) -> Tuple[int, int]:
    span_stripped = span.strip()
    if not span_stripped or not full:
        return max(0, offset_hint), max(0, offset_hint) + max(1, len(span_stripped))
    start = full.find(span_stripped, max(0, offset_hint - 200))
    if start < 0:
        probe = span_stripped[: min(80, len(span_stripped))]
        start = full.rfind(probe)
    if start < 0:
        return offset_hint, min(len(full), offset_hint + len(span_stripped))
    end = min(len(full), start + len(span_stripped))
    return start, end


def _category_for_assistant(label: KbFindingLabel) -> str:
    if label == KbFindingLabel.contradicted:
        return "contradiction"
    if label in (KbFindingLabel.supported, KbFindingLabel.related):
        return "relevant_knowledge"
    if label == KbFindingLabel.needs_review:
        return "clarification"
    return "missing_context"


def _silent_category_status(label: KbFindingLabel) -> Tuple[str, str]:
    if label == KbFindingLabel.contradicted:
        return "contradiction_with_indexed_knowledge", "contradicted"
    if label == KbFindingLabel.supported:
        return "useful_suggestion", "likely_correct"
    if label == KbFindingLabel.related:
        return "possible_misinterpretation", "needs_verification"
    if label == KbFindingLabel.needs_review:
        return "needs_verification", "needs_verification"
    return "unsupported_claim", "unsupported"


def _cites_for_store(sources: List[AssistantSourceOut]) -> List[dict]:
    out: List[dict] = []
    for s in sources:
        out.append(
            {
                "filename": s.document_name,
                "snippet": s.snippet,
                "doc_id": s.document_id,
                "page_number": s.page,
                "score": s.score,
            }
        )
    return out


async def analyze_transcript(inp: AnalyzeTranscriptIn) -> AnalyzeTranscriptOut:
    if not inp.knowledge_base_enabled:
        return AnalyzeTranscriptOut(items=[], skipped_reason="kb_disabled")

    delta = (inp.transcript_text or "").strip()
    if len(delta) < 8:
        return AnalyzeTranscriptOut(items=[], skipped_reason="too_short")

    full = (inp.full_transcript or delta).strip()
    base_offset = max(0, inp.transcript_offset)
    if inp.full_transcript and delta and base_offset == 0 and delta not in full:
        tail = delta[-min(len(delta), 400) :]
        idx = full.rfind(tail)
        if idx >= 0:
            base_offset = idx

    spans = _split_spans(delta)
    if not spans:
        return AnalyzeTranscriptOut(items=[], skipped_reason="no_signal")

    cw = inp.context_window or "all"
    staged: List[AssistantAnalysisItemOut] = []

    for span in spans:
        try:
            hits, _probe = await retrieve_for_kb_probe(span[:900], k=8, context_window=cw)
        except Exception as e:
            logger.warning("KB transcript analyze retrieve failed: %s", e)
            hits = []

        cites = _hits_to_citations(hits) if hits else []
        tier, top_score = _evidence_tier(hits)
        align = _rag_alignment(span, hits) if hits else "unrelated"
        label, conf = _label_confidence(align, tier, top_score)

        if not hits and label == KbFindingLabel.unverified and conf < 0.52:
            continue

        start, end = _span_char_range(span, full, base_offset)
        if end <= start:
            end = min(len(full), start + len(span.strip()))

        expl = (
            f"Knowledge-base retrieval ({tier}) compared this span to indexed documents and transcripts. "
            "Only local index overlap and scores were used—no general LLM factual judgment."
        )
        if label == KbFindingLabel.unverified:
            expl += " Evidence was insufficient to confirm or refute."
        elif label == KbFindingLabel.needs_review:
            expl += " Treat as needs review before high-stakes use."

        feedback = {
            KbFindingLabel.supported: "Indexed sources appear to support this wording.",
            KbFindingLabel.contradicted: "Indexed sources may conflict with this wording—compare citations.",
            KbFindingLabel.related: "Sources are related; alignment with this exact claim is unclear.",
            KbFindingLabel.unverified: "No strong knowledge-base evidence for this span.",
            KbFindingLabel.needs_review: "Weak or mixed signals—review sources before deciding.",
        }[label]

        speak = (
            f"{feedback} Review the cited passage in your documents when you can."
            if label != KbFindingLabel.unverified
            else "I could not find strong local evidence for that part. Double-check your documents if it matters."
        )

        sources = _sources_from_citations(cites)
        item = AssistantAnalysisItemOut(
            id=str(uuid.uuid4()),
            text=span.strip(),
            start_char=start,
            end_char=end,
            label=label,
            confidence=float(conf),
            evidence_status=tier,  # type: ignore[arg-type]
            explanation=expl[:3900],
            feedback=feedback,
            speak_text=speak[:7900],
            sources=sources,
        )
        staged.append(item)

    staged.sort(key=lambda x: -x.confidence)
    staged = staged[:8]
    items = list(staged)

    if inp.persist_results:
        if inp.mode == "assistant" and not suggestion_store.is_within_cooldown(inp.session_id):
            for it in staged:
                if it.confidence < ASSISTANT_HAND_RAISE_MIN:
                    continue
                if it.label not in (
                    KbFindingLabel.supported,
                    KbFindingLabel.contradicted,
                    KbFindingLabel.related,
                    KbFindingLabel.needs_review,
                ):
                    continue
                fp = " ".join(it.text.lower().split())[:120]
                if any(fp and (fp in e or e in fp) for e in suggestion_store.recent_pending_fingerprints(inp.session_id, 14)):
                    continue
                if suggestion_store.count_pending(inp.session_id) >= suggestion_store.ASSISTANT_MAX_PENDING_PER_SESSION:
                    break
                title = f"{it.label.value}: {it.text.strip()[:72]}{'…' if len(it.text) > 72 else ''}"
                cites_json = _cites_for_store(it.sources)
                row = suggestion_store.insert_suggestion(
                    inp.session_id,
                    "ASSISTANT",
                    title,
                    f"{it.feedback}\n\nLatest bit: {it.text.strip()}"[:1900],
                    it.speak_text,
                    it.explanation[:3900],
                    _category_for_assistant(it.label),
                    float(it.confidence),
                    "transcript_plus_rag",
                    it.evidence_status,
                    cites_json,
                    status="pending",
                    trigger_excerpt=it.text.strip()[:2000],
                )
                it.persisted_id = row.id

        elif inp.mode == "silent_assistant" and not finding_store.is_within_cooldown(inp.session_id):
            for it in staged:
                if it.confidence <= SILENT_HIGHLIGHT_MIN_EXCLUSIVE:
                    continue
                fp = f"{it.label.value}|{' '.join(it.text.lower().split())[:100]}"
                if fp in finding_store.recent_fingerprints(inp.session_id, 20):
                    continue
                if finding_store.count_pending(inp.session_id) >= finding_store.SILENT_MAX_PENDING_FINDINGS:
                    break
                cat, stlab = _silent_category_status(it.label)
                cites_json = _cites_for_store(it.sources)
                row = finding_store.insert_finding(
                    inp.session_id,
                    None,
                    None,
                    it.text.strip(),
                    it.start_char,
                    it.end_char,
                    cat,
                    stlab,
                    it.feedback[:1900],
                    it.explanation[:3900],
                    it.evidence_status,
                    float(it.confidence),
                    "transcript_plus_rag",
                    cites_json,
                )
                it.persisted_id = row.id

    if not items:
        return AnalyzeTranscriptOut(items=[], skipped_reason="no_signal")

    return AnalyzeTranscriptOut(items=items, skipped_reason=None)
