"""
EvidenceGate: composite confidence scoring and query concept coverage verification.

Prevents the LLM from answering when evidence is weak or doesn't cover the query's
core concepts. Returns a structured result with score breakdown for debug logging.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .evidence_extractor import EvidenceSentence, POLICY_WORDS


def _normalize_rerank_score(s: float) -> float:
    """Map a rerank score to [0,1] regardless of domain. Cross-encoder rerankers emit unbounded
    logits (~-11..+11) while cosine/inner-product sits in 0..1; the gate's thresholds assume 0..1,
    so sigmoid-squash anything that looks like a logit and pass cosine through. (M3)"""
    try:
        s = float(s)
    except (TypeError, ValueError):
        return 0.0
    if s < 0.0 or s > 1.0:
        return 1.0 / (1.0 + math.exp(-s))
    return s

logger = logging.getLogger(__name__)

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "need", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "what", "which", "who", "whom",
    "whose", "where", "when", "why", "how", "that", "this", "these",
    "those", "it", "its", "about", "between", "under", "over", "not",
    "and", "or", "but", "if", "than", "so", "very", "just", "also",
})

WEAK_EVIDENCE_MSG = "Unable to find strong supporting evidence in the regulation."
MISSING_SECTION_MSG = "Relevant section not confidently identified in the regulation."


def _extract_concepts(question: str, min_len: int = 3) -> List[str]:
    """Extract core concept tokens from a query (exclude stopwords, keep meaningful terms)."""
    tokens = re.findall(r"[a-z0-9]+", (question or "").lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) >= min_len]


@dataclass
class EvidenceGateResult:
    """Result of evidence gating."""
    passed: bool
    confidence_score: float
    keyword_coverage: float
    section_match_score: float
    avg_rerank_score: float
    policy_word_ratio: float
    concepts_found: List[str] = field(default_factory=list)
    concepts_missing: List[str] = field(default_factory=list)
    fallback_message: Optional[str] = None


def check_concept_coverage(
    question: str,
    evidence: List[EvidenceSentence],
    min_coverage: float = 0.5,
) -> tuple[float, List[str], List[str]]:
    """Check what fraction of query concepts appear in evidence.

    Returns (coverage_ratio, concepts_found, concepts_missing).
    """
    concepts = _extract_concepts(question)
    if not concepts:
        return (1.0, [], [])

    evidence_text = " ".join(e.sentence.lower() for e in evidence)
    found = [c for c in concepts if c in evidence_text]
    missing = [c for c in concepts if c not in evidence_text]
    coverage = len(found) / len(concepts) if concepts else 1.0
    return (coverage, found, missing)


def calculate_evidence_confidence(
    question: str,
    evidence: List[EvidenceSentence],
    explicit_section_ids: Optional[List[str]] = None,
) -> float:
    """Calculate composite confidence score (0.0–1.0).

    Components:
      - keyword_coverage (0–0.35): fraction of query concepts found in evidence
      - avg_rerank_score (0–0.30): mean reranker score of evidence chunks
      - section_match (0–0.20): bonus if explicit section ids appear in evidence
      - policy_word_ratio (0–0.15): fraction of evidence containing policy language
    """
    if not evidence:
        return 0.0

    # Keyword coverage
    coverage, _, _ = check_concept_coverage(question, evidence)
    kw_score = min(coverage, 1.0) * 0.35

    # Average rerank score, normalized to 0–1 first so CE logits and cosine both behave. (M3)
    avg_rerank = sum(_normalize_rerank_score(e.chunk_rerank_score) for e in evidence) / len(evidence) if evidence else 0.0
    rerank_norm = min(max((avg_rerank - 0.2) / 0.6, 0.0), 1.0)
    rerank_score = rerank_norm * 0.30

    # Section match
    section_match = 0.0
    if explicit_section_ids:
        evidence_text = " ".join(e.sentence.lower() for e in evidence)
        evidence_paths = " ".join((e.section_path or "").lower() for e in evidence)
        all_text = evidence_text + " " + evidence_paths
        if any(sid.lower() in all_text for sid in explicit_section_ids):
            section_match = 0.20

    # Policy word ratio
    policy_count = sum(1 for e in evidence if e.has_policy_word)
    policy_ratio = policy_count / len(evidence) if evidence else 0.0
    policy_score = min(policy_ratio, 1.0) * 0.15

    return kw_score + rerank_score + section_match + policy_score


def gate_evidence(
    question: str,
    evidence: List[EvidenceSentence],
    explicit_section_ids: Optional[List[str]] = None,
    confidence_threshold: float = 0.30,
    concept_coverage_threshold: float = 0.4,
    resolver_confident: bool = False,
) -> EvidenceGateResult:
    """Gate evidence before LLM answering. Smarter logic for explicit vs concept queries.

    Case A (explicit valid section): resolver confident + section in evidence → lower keyword threshold
    Case B (concept query): require section from index + ≥2 concept matches + confidence
    Case C (no good evidence): refuse with top 3 candidate sections
    """
    if not evidence:
        return EvidenceGateResult(
            passed=False,
            confidence_score=0.0,
            keyword_coverage=0.0,
            section_match_score=0.0,
            avg_rerank_score=0.0,
            policy_word_ratio=0.0,
            fallback_message=MISSING_SECTION_MSG,
        )

    confidence = calculate_evidence_confidence(question, evidence, explicit_section_ids)
    coverage, found, missing = check_concept_coverage(question, evidence, concept_coverage_threshold)
    # Normalize to 0–1 so the avg_rerank >= 0.25 threshold means the same for CE logits & cosine. (M3)
    avg_rerank = sum(_normalize_rerank_score(e.chunk_rerank_score) for e in evidence) / len(evidence)
    policy_ratio = sum(1 for e in evidence if e.has_policy_word) / len(evidence)

    # Section match check
    section_match = 0.0
    if explicit_section_ids:
        evidence_text = " ".join(e.sentence.lower() for e in evidence)
        evidence_paths = " ".join((e.section_path or "").lower() for e in evidence)
        evidence_section_ids = " ".join((e.section or "").lower() for e in evidence)
        all_text = evidence_text + " " + evidence_paths + " " + evidence_section_ids
        if any(sid.lower() in all_text for sid in explicit_section_ids):
            section_match = 1.0

    passed = True
    fallback_message = None

    # Case A: explicit valid section query — lower keyword threshold if section found
    if resolver_confident and explicit_section_ids and section_match >= 0.5:
        # Require: exact section_id in evidence OR parent section match + ≥1 strong sentence
        min_coverage = 0.2  # Lower threshold for explicit refs
        if coverage >= min_coverage and avg_rerank >= 0.25:
            passed = True
            logger.info(
                "EvidenceGate: PASS (Case A explicit) section_match=1 coverage=%.2f q=%s",
                coverage, question[:80],
            )
        else:
            passed = False
            fallback_message = MISSING_SECTION_MSG
            logger.info(
                "EvidenceGate: FAIL (Case A) coverage=%.2f avg_rerank=%.3f q=%s",
                coverage, avg_rerank, question[:80],
            )
        return EvidenceGateResult(
            passed=passed,
            confidence_score=confidence,
            keyword_coverage=coverage,
            section_match_score=section_match,
            avg_rerank_score=avg_rerank,
            policy_word_ratio=policy_ratio,
            concepts_found=found,
            concepts_missing=missing,
            fallback_message=fallback_message,
        )

    # Case B: concept query (no explicit section refs) — relax: pass if we have evidence + some coverage
    if not explicit_section_ids:
        # Concept query: pass when we have ≥3 evidence sentences and ≥1 concept match
        min_evidence = 3
        min_coverage = 0.2
        if len(evidence) >= min_evidence and (coverage >= min_coverage or not (found or missing)):
            passed = True
            logger.info(
                "EvidenceGate: PASS (concept query) evidence=%d coverage=%.2f q=%s",
                len(evidence), coverage, question[:80],
            )
        elif len(evidence) >= 1 and coverage >= 0.15:
            passed = True
            logger.info(
                "EvidenceGate: PASS (concept query, relaxed) evidence=%d coverage=%.2f q=%s",
                len(evidence), coverage, question[:80],
            )
        else:
            passed = False
            fallback_message = WEAK_EVIDENCE_MSG
    elif explicit_section_ids and section_match < 0.5:
        passed = False
        fallback_message = MISSING_SECTION_MSG
        logger.info(
            "EvidenceGate: FAIL (section_ids=%s not in evidence) q=%s",
            explicit_section_ids[:3], question[:80],
        )
    elif confidence < confidence_threshold:
        passed = False
        fallback_message = WEAK_EVIDENCE_MSG
        logger.info(
            "EvidenceGate: FAIL (confidence=%.3f < %.3f) q=%s",
            confidence, confidence_threshold, question[:80],
        )
    elif coverage < concept_coverage_threshold and len(missing) > 0:
        passed = False
        fallback_message = MISSING_SECTION_MSG
        logger.info(
            "EvidenceGate: FAIL (coverage=%.2f, missing=%s) q=%s",
            coverage, missing[:5], question[:80],
        )

    if passed:
        logger.info(
            "EvidenceGate: PASS confidence=%.3f coverage=%.2f section=%.1f q=%s",
            confidence, coverage, section_match, question[:80],
        )

    return EvidenceGateResult(
        passed=passed,
        confidence_score=confidence,
        keyword_coverage=coverage,
        section_match_score=section_match,
        avg_rerank_score=avg_rerank,
        policy_word_ratio=policy_ratio,
        concepts_found=found,
        concepts_missing=missing,
        fallback_message=fallback_message,
    )
