"""
Evidence-First Retrieval: extract structured evidence sentences from reranked hits.

Sentences are grouped by section_id; only top sections are used to prevent cross-section
contamination. Each sentence is scored by query overlap, section match, policy language,
and chunk rerank score.

Only evidence sentences are sent to the LLM — NOT full chunks.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..core.config import settings

logger = logging.getLogger(__name__)

POLICY_WORDS = frozenset({
    "must", "shall", "required", "responsible", "procedure",
    "approval", "certify", "authorize", "obligate", "prohibit",
    "will", "ensure", "comply", "mandatory", "directed",
})

_OBLIGATION_RE = re.compile(
    r"\b(shall|must|required|will\s+be|must\s+be|shall\s+not|must\s+not|responsible\s+for|procedure|approval|certify)\b",
    re.I,
)

_ABBREV_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Jr|Sr|Gen|Col|Lt|Sgt|Capt|Maj|Cmdr|Adm|Vol|Ch|Sec|Para|No|Fig|e\.g|i\.e|vs|etc|dept|govt)\.",
    re.I,
)


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", (text or "").lower()))


def _split_sentences(text: str) -> List[str]:
    if not (text or "").strip():
        return []
    # Protect abbreviations from splitting
    protected = _ABBREV_RE.sub(lambda m: m.group(0).replace(".", "\x00"), text)
    parts = re.split(r"(?<=[.!?])\s+", protected.strip())
    return [p.replace("\x00", ".").strip() for p in parts if p.strip()]


@dataclass
class EvidenceSentence:
    """A single evidence sentence with provenance metadata."""
    sentence: str
    volume: Optional[int] = None
    chapter: Optional[int] = None
    section: Optional[str] = None
    section_path: Optional[str] = None
    page: Optional[int] = None
    score: float = 0.0
    keyword_hits: int = 0
    has_policy_word: bool = False
    has_section_id: bool = False
    chunk_rerank_score: float = 0.0


def _get_section_id_for_hit(h: dict) -> str:
    """Extract canonical section_id from hit source."""
    src = (h.get("source") or {})
    sid = src.get("section_id") or src.get("section")
    if sid:
        return str(sid).strip()
    sp = (src.get("section_path") or "").strip()
    if not sp:
        return "__no_section__"
    try:
        from .book.section_id import section_id_from_path
        resolved = section_id_from_path(sp)
        return resolved or sp
    except Exception:
        return sp


def _extract_metadata(source: dict) -> dict:
    """Pull volume/chapter/section/page from chunk source metadata."""
    src = source or {}
    sp = (src.get("section_path") or "").strip()
    volume = src.get("volume")
    chapter = src.get("chapter")
    section = src.get("section") or ""
    page = src.get("page_number")

    if sp and volume is None:
        vol_m = re.search(r"Volume\s+(\d+)", sp, re.I)
        if vol_m:
            try:
                volume = int(vol_m.group(1))
            except ValueError:
                pass
    if sp and chapter is None:
        ch_m = re.search(r"Chapter\s+(\d+)", sp, re.I)
        if ch_m:
            try:
                chapter = int(ch_m.group(1))
            except ValueError:
                pass
    if not section and sp:
        sec_m = re.search(r"(\d{4,6})", sp)
        if sec_m:
            section = sec_m.group(1)

    return {"volume": volume, "chapter": chapter, "section": section, "section_path": sp, "page": page}


def extract_evidence_sentences(
    question: str,
    hits: List[Dict],
    explicit_section_ids: Optional[List[str]] = None,
    min_sentences: int = 8,
    max_sentences: int = 12,
) -> List[EvidenceSentence]:
    """Extract evidence sentences from reranked hits, grouped by section.

    Flow: group sentences by section_id → score sections → select top sections →
    extract evidence only from those sections. Capped at MAX_EVIDENCE_SENTENCES.
    """
    if not hits or not question:
        return []

    max_ev = getattr(settings, "MAX_EVIDENCE_SENTENCES", 10)
    max_sentences = min(max_sentences, max_ev)
    max_sections = getattr(settings, "MAX_SECTIONS_PER_ANSWER", 2)
    q_tokens = _tokenize(question)
    explicit_ids = set(explicit_section_ids or [])

    # 1. Group candidate sentences by section_id
    by_section: Dict[str, List[Tuple[EvidenceSentence, float]]] = {}
    seen: Set[str] = set()

    for h in hits:
        src = h.get("source") or {}
        # Skip graph-expanded chunks that are not metadata_valid (cannot be primary evidence)
        if src.get("_graph_ref") and src.get("metadata_valid") is False:
            continue
        text = (h.get("text") or "").strip()
        if not text:
            continue
        meta = _extract_metadata(src)
        chunk_score = float(h.get("score") or 0)
        # Post-rerank chunk scores are CE logits (~[-11,+11]); the per-sentence bonuses
        # below are small (0.5-3.0). Adding a raw logit made the chunk score dominate
        # sentence ranking entirely (and negative logits suppressed every sentence in a
        # chunk). Normalize to 0..1 and scale to the same magnitude as the other bonuses.
        if h.get("score_kind") == "ce_logit" or chunk_score < 0.0 or chunk_score > 1.0:
            chunk_rel = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, chunk_score))))
        else:
            chunk_rel = chunk_score
        sid = _get_section_id_for_hit(h)

        for sent in _split_sentences(text):
            if not sent or len(sent) < 20:
                continue
            sent_norm = sent.strip()
            if sent_norm in seen:
                continue

            sent_lower = sent_norm.lower()
            sent_tokens = _tokenize(sent_norm)
            keyword_hits = len(q_tokens & sent_tokens)
            has_section = bool(
                explicit_ids
                and any(
                    eid in sent_lower or eid in (meta.get("section_path") or "").lower()
                    for eid in explicit_ids
                )
            )
            has_policy = bool(_OBLIGATION_RE.search(sent_norm))
            policy_token_hits = sum(1 for pw in POLICY_WORDS if pw in sent_lower)

            if keyword_hits == 0 and not has_section and not has_policy:
                continue

            score = (
                keyword_hits * 2.0
                + (3.0 if has_section else 0.0)
                + (2.0 if has_policy else 0.0)
                + policy_token_hits * 0.5
                + 2.0 * chunk_rel
            )

            seen.add(sent_norm)
            ev = EvidenceSentence(
                sentence=sent_norm,
                volume=meta.get("volume"),
                chapter=meta.get("chapter"),
                section=meta.get("section"),
                section_path=meta.get("section_path"),
                page=meta.get("page"),
                score=score,
                keyword_hits=keyword_hits,
                has_policy_word=has_policy,
                has_section_id=has_section,
                chunk_rerank_score=chunk_score,
            )
            if sid not in by_section:
                by_section[sid] = []
            by_section[sid].append((ev, chunk_score))

    if not by_section:
        return []

    # 2. Score sections: max(chunk_score) + average(chunk_score) + keyword_match_bonus
    def section_score(sid: str, items: List[Tuple[EvidenceSentence, float]]) -> float:
        scores = [s for _, s in items]
        if not scores:
            return 0.0
        section_match_bonus = 0.5 if sid in explicit_ids else 0.0
        keyword_bonus = sum(1 for ev, _ in items if ev.keyword_hits > 0) * 0.3
        return max(scores) + (sum(scores) / len(scores)) + section_match_bonus + keyword_bonus

    scored_sections = [(sid, items, section_score(sid, items)) for sid, items in by_section.items()]
    scored_sections.sort(key=lambda x: -x[2])
    top_section_ids = {x[0] for x in scored_sections[:max_sections]}

    # 3. Extract evidence only from top sections. Preserve original sentence order within sections.
    collected: List[EvidenceSentence] = []
    for sid, items, _ in scored_sections:
        if sid not in top_section_ids:
            continue
        # Preserve order: sort by score for selection, but output in chunk order (items order)
        for ev, _ in items:
            collected.append(ev)
    # Sort by score for final selection, cap at MAX_EVIDENCE_SENTENCES
    collected.sort(key=lambda e: -e.score)
    result = collected[:max_sentences]

    # Backfill from top sections if below minimum
    if len(result) < min_sentences:
        for h in hits:
            if len(result) >= min_sentences:
                break
            if _get_section_id_for_hit(h) not in top_section_ids:
                continue
            text = (h.get("text") or "").strip()
            src = h.get("source") or {}
            meta = _extract_metadata(src)
            for sent in _split_sentences(text):
                if sent and sent not in seen and len(sent) >= 20:
                    seen.add(sent)
                    result.append(EvidenceSentence(
                        sentence=sent,
                        volume=meta.get("volume"),
                        chapter=meta.get("chapter"),
                        section=meta.get("section"),
                        section_path=meta.get("section_path"),
                        page=meta.get("page"),
                        score=0.0,
                    ))
                    if len(result) >= min_sentences:
                        break

    return result[:max_sentences]


def build_evidence_block(
    question: str,
    hits: List[Dict],
    explicit_section_ids: Optional[List[str]] = None,
) -> str:
    """Build evidence block as 8–12 bullet snippets for prepending to context."""
    evidence = extract_evidence_sentences(
        question, hits,
        explicit_section_ids=explicit_section_ids,
        min_sentences=8,
        max_sentences=12,
    )
    if not evidence:
        return ""
    lines = []
    for e in evidence:
        cite_parts = []
        if e.volume is not None:
            cite_parts.append(f"Vol {e.volume}")
        if e.chapter is not None:
            cite_parts.append(f"Ch {e.chapter}")
        if e.section:
            cite_parts.append(f"Sec {e.section}")
        if e.page is not None:
            cite_parts.append(f"p.{e.page}")
        cite = f" [{', '.join(cite_parts)}]" if cite_parts else ""
        lines.append(f"• {e.sentence}{cite}")
    return "[EVIDENCE]\n" + "\n".join(lines)


def build_evidence_only_context(
    question: str,
    hits: List[Dict],
    explicit_section_ids: Optional[List[str]] = None,
) -> str:
    """Build LLM context from ONLY evidence sentences (no full chunks).

    This is the evidence-first approach: the LLM sees only the strongest
    supporting sentences, each with source metadata.
    """
    evidence = extract_evidence_sentences(
        question, hits,
        explicit_section_ids=explicit_section_ids,
        min_sentences=8,
        max_sentences=12,
    )
    if not evidence:
        return ""
    blocks = []
    for i, e in enumerate(evidence, 1):
        meta_parts = []
        if e.section_path:
            meta_parts.append(f"path: {e.section_path}")
        else:
            if e.volume is not None:
                meta_parts.append(f"Volume {e.volume}")
            if e.chapter is not None:
                meta_parts.append(f"Chapter {e.chapter}")
            if e.section:
                meta_parts.append(f"Section {e.section}")
        if e.page is not None:
            meta_parts.append(f"page: {e.page}")
        meta_line = f"({', '.join(meta_parts)})" if meta_parts else ""
        block = f"[{i}] {meta_line}\n{e.sentence}" if meta_line else f"[{i}] {e.sentence}"
        blocks.append(block)
    return "\n\n".join(blocks)
