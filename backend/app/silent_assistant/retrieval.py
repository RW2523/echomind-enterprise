"""Retrieval for the Silent Assistant:
  - retrieve_for_claim(): hybrid + CE-reranked evidence sentences for one spoken sentence
  - lookup_records(): pull the records that mention a person / identifier (exact-first)
Both reuse the chat RAG stack (advanced.retrieve_reranked, evidence_extractor) — no
separate index, no LLM."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.config import settings
from ..rag import advanced as _adv
from ..rag.evidence_extractor import EvidenceSentence, extract_evidence_sentences, _split_sentences
from ..rag.index import index as faiss_index, set_active_namespace
from ..utils.ids import new_id
from .entities import Entity
from .state import RecordHit, SessionAssistantState, Sentence

logger = logging.getLogger(__name__)


@dataclass
class ClaimEvidence:
    result: _adv.RetrievalResult
    evidence: List[EvidenceSentence] = field(default_factory=list)
    timing: Dict[str, float] = field(default_factory=dict)

    @property
    def searched_docs(self) -> List[str]:
        seen, out = set(), []
        for h in self.result.hits:
            t = (h.get("source") or {}).get("filename") or ""
            if t and t not in seen:
                seen.add(t); out.append(t)
            if len(out) >= 3:
                break
        return out


async def retrieve_for_claim(claim: str, state: SessionAssistantState) -> ClaimEvidence:
    t0 = time.monotonic()
    q = state.resolve_reference(claim)
    res = await _adv.retrieve_reranked(
        q,
        k_candidates=int(getattr(settings, "ASSISTANT_CE_CANDIDATES", 12)),
        final_n=int(getattr(settings, "ASSISTANT_CE_FINAL", 6)),
        source_options={"document": True, "transcript": bool(state.profile.include_transcripts), "general": False},
        namespace=state.namespace or None,
        allow_default_fallback=True,
    )
    ev: List[EvidenceSentence] = []
    if res.hits and not res.gated:
        # The CE already ranked chunks by relevance to THIS claim; pick proof sentences in that
        # order (best-overlap sentences per chunk, top chunks first). The chat evidence extractor
        # re-ranks by section/keyword bonuses tuned for the FMR book corpus and demoted the
        # rank-1 policy clause below an unrelated regulation sentence.
        ev = _ranked_sentences(claim, res.hits, limit=8)
        if len(ev) < 3:
            try:
                extra = extract_evidence_sentences(claim, res.hits, min_sentences=4, max_sentences=8)
                seen = {e.sentence for e in ev}
                ev.extend(e for e in extra if e.sentence not in seen)
            except Exception as e:  # never let evidence extraction kill the check
                logger.warning("SA evidence extraction failed: %s", e)
    return ClaimEvidence(result=res, evidence=ev, timing={"retrieve_ms": (time.monotonic() - t0) * 1000})


def _stem(t: str) -> str:
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def _ranked_sentences(claim: str, hits: List[dict], limit: int = 8, per_chunk: int = 3) -> List[EvidenceSentence]:
    """Proof sentences chosen globally: score = token/number overlap with the claim (light
    plural stemming) + a bonus for the chunk's CE rank, capped per chunk. The best sentence of
    the top-ranked chunk is always included so the LLM sees what the top document says."""
    q_toks = {_stem(t) for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", claim.lower()) if len(t) >= 3 or t.isdigit()}
    q_nums = set(re.findall(r"\d+(?:\.\d+)?", claim))
    rank_bonus = [3.0, 2.0, 1.2, 0.8, 0.5, 0.3]
    scored = []      # (score, rank, sent, src, h)
    per_chunk_count: Dict[int, int] = {}
    seen: set = set()
    for rank, h in enumerate(hits[:6]):
        src = h.get("source") or {}
        cands = []
        for sent in _split_sentences(h.get("text") or "") or [h.get("text") or ""]:
            sent = sent.strip()
            if len(sent) < 20 or sent in seen:
                continue
            seen.add(sent)
            toks = {_stem(t) for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", sent.lower())}
            ov = len(q_toks & toks)
            nums = set(re.findall(r"\d+(?:\.\d+)?", sent))
            score = ov + 2.0 * len(q_nums & nums) + (0.3 if (q_nums and re.search(r"\d", sent)) else 0.0)
            cands.append((score, sent))
        cands.sort(key=lambda x: -x[0])
        for j, (sc, sent) in enumerate(cands[:per_chunk]):
            if sc <= 0 and not (rank == 0 and j == 0):
                continue
            scored.append((sc + rank_bonus[rank] if rank < len(rank_bonus) else sc, rank, sent, src, h))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out: List[EvidenceSentence] = []
    for sc, rank, sent, src, h in scored[:limit]:
        out.append(EvidenceSentence(
            sentence=sent, page=src.get("page_number"), section_path=src.get("section_path"),
            score=float(round(sc, 2)), chunk_rerank_score=float(h.get("score") or 0),
            chunk_id=str(h.get("chunk_id") or ""), doc_id=str(src.get("doc_id") or ""),
            doc_title=str(src.get("filename") or src.get("name") or ""),
        ))
    return out


def _fallback_sentences(claim: str, hits: List[dict], limit: int = 6) -> List[EvidenceSentence]:
    q_toks = set(re.findall(r"[a-z0-9]{3,}", claim.lower()))
    scored = []
    for h in hits[:4]:
        src = h.get("source") or {}
        for sent in _split_sentences(h.get("text") or ""):
            if len(sent) < 25:
                continue
            toks = set(re.findall(r"[a-z0-9]{3,}", sent.lower()))
            ov = len(q_toks & toks)
            if ov == 0:
                continue
            scored.append((ov, EvidenceSentence(
                sentence=sent.strip(), page=src.get("page_number"), section_path=src.get("section_path"),
                score=float(ov), chunk_rerank_score=float(h.get("score") or 0),
                chunk_id=str(h.get("chunk_id") or ""), doc_id=str(src.get("doc_id") or ""),
                doc_title=str(src.get("filename") or src.get("name") or ""),
            )))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:limit]]


# ── Record pull ────────────────────────────────────────────────────────────────
_KIND_HINTS = [
    ("kyc", r"kyc|know your customer|customer profile"),
    ("account", r"account|statement|balance|deposit"),
    ("product", r"product|terms|fund|rate|fee schedule|disclosure"),
    ("policy", r"policy|regulation|conduct|sop|procedure|guideline"),
    ("contract", r"contract|agreement|terms and conditions|lease|clause"),
    ("ticket", r"ticket|complaint|incident|support history|case log"),
    ("customer_file", r"customer|client record|member|subscriber|profile|crm"),
    ("matter", r"matter|brief|case file|pleading|filing"),
    ("related_case", r"precedent|digest|judgment|ruling|v\.|vs\.?|case law"),
    ("previous_call", r"transcript"),
]


def _record_kind(doc_title: str, is_transcript: bool, profile_kinds: List[str]) -> str:
    if is_transcript:
        return "previous_call"
    t = (doc_title or "").lower()
    for kind, pat in _KIND_HINTS:
        if re.search(pat, t) and (kind in profile_kinds or kind == "document"):
            return kind
    for kind, pat in _KIND_HINTS:
        if re.search(pat, t):
            return kind
    return "document"


def _quotes_containing(text: str, terms: List[str], limit: int = 3) -> List[str]:
    """Sentences of the chunk that literally contain one of the lookup terms (case-insensitive,
    also matching a normalized digits/letters form). These are the proof for 'record-found'."""
    if not text:
        return []
    norm_terms = {re.sub(r"[\s\-./]", "", t.lower()) for t in terms if t}
    out = []
    for sent in _split_sentences(text) or [text]:
        low = sent.lower()
        norm = re.sub(r"[\s\-./]", "", low)
        if any(t.lower() in low for t in terms if len(t) >= 3) or any(nt and len(nt) >= 4 and nt in norm for nt in norm_terms):
            s = sent.strip()
            if s and s not in out:
                out.append(s[:400])
        if len(out) >= limit:
            break
    return out


async def lookup_records(entities: List[Entity], sentence: Sentence, state: SessionAssistantState) -> List[RecordHit]:
    """Exact-first record pull. Returns NEW records only (deduped against state)."""
    if not entities:
        return []
    terms = state.lookup_terms_safe(entities)
    if not terms:
        return []
    # exact-name matches must be the FULL name (>=2 tokens); a lone token is only used for BM25
    exact_terms = [t for t in terms if not (t.replace("'", "").replace("-", "").isalpha() and len(t.split()) < 2)]
    t0 = time.monotonic()
    profile = state.profile
    ns = state.namespace or None
    set_active_namespace(ns)
    prof_kinds = [rt.kind for rt in profile.record_targets]
    include_tx = bool(profile.include_transcripts)

    hits: List[dict] = []
    seen_chunks: set = set()

    def _take(lst: List[dict], match: str, cap: int):
        n = 0
        for h in lst:
            cid = h.get("chunk_id")
            if not cid or cid in seen_chunks:
                continue
            seen_chunks.add(cid)
            hits.append({**h, "_match": match})
            n += 1
            if n >= cap:
                break

    # 1) exact identifier / name substring in SQLite (namespace-safe)
    try:
        exact = await asyncio.get_running_loop().run_in_executor(
            None, lambda: faiss_index.search_chunks_exact(exact_terms, k=8, include_transcripts=include_tx)) if exact_terms else []
        _take(exact, "exact", 12)
    except Exception as e:
        logger.warning("SA exact lookup failed: %s", e)
    # 2) BM25 on "name id" (numbers get compound-tokenized by sparse.py)
    q = " ".join(dict.fromkeys(terms[:4]))
    try:
        _take(faiss_index.search_document_only_sparse(q, 8), "fuzzy", 6)
        if include_tx:
            _take(faiss_index.search_transcript_only_sparse(q, 4), "fuzzy", 4)
    except Exception as e:
        logger.debug("SA sparse lookup failed: %s", e)
    # 3) semantic per record target (only when we have a name — semantic search on bare IDs is noise)
    person = next((e for e in entities if e.kind == "person"), None)
    subj = state.active_subject()
    name = person.value if person else (subj.display_name if subj else "")
    if name:
        ids = " ".join(v for e in entities for v in [e.value] if e.kind != "person")
        for rt in profile.record_targets[:3]:
            qq = rt.query_template.format(name=name, ids=ids).strip()
            try:
                _take(await faiss_index.search_document_only(qq, 4), "semantic", 3)
            except Exception:
                pass
    if ns and not hits:
        # namespace fallback once (records may live in default KB)
        set_active_namespace(None)
        try:
            if exact_terms:
                _take(await asyncio.get_running_loop().run_in_executor(
                    None, lambda: faiss_index.search_chunks_exact(exact_terms, k=8, include_transcripts=include_tx)), "exact", 8)
        finally:
            set_active_namespace(ns)
    if not hits:
        return []

    # 4) rerank the pool with the CE against a natural query, keep 6
    query = f"records for {name} {' '.join(terms[:3])}".strip() if name else f"record {' '.join(terms[:3])}"
    try:
        reranked = await _adv._apply_reranker(query, hits, top_k_candidates=min(len(hits), 12), final_n=8)
    except Exception:
        reranked = hits[:8]
    # exact matches must survive rerank; put them first
    exact_first = [h for h in hits if h.get("_match") == "exact"][:6]
    ordered = exact_first + [h for h in reranked if h.get("_match") != "exact"]

    out: List[RecordHit] = []
    reused: List[RecordHit] = []
    for h in ordered:
        src = h.get("source") or {}
        is_tx = (src.get("filename") or "").startswith("transcript_")
        doc_title = src.get("name") or src.get("filename") or "Document"
        quotes = _quotes_containing(h.get("text") or "", terms)
        match = h.get("_match") or "semantic"
        if match != "exact" and not quotes:
            continue   # semantic hits without a literal mention are not 'records' — proof rule
        if not quotes:
            quotes = [(h.get("text") or "")[:300]]
        key = (src.get("doc_id"), src.get("page_number"), hashlib.md5(quotes[0].encode()).hexdigest()[:10])
        if key in state.emitted_record_keys:
            prev_id = state.record_key_ids.get(key)
            if prev_id and prev_id in state.records:
                reused.append(state.records[prev_id])
            continue
        state.emitted_record_keys.add(key)
        sc = h.get("score")
        score = 1.0 if match == "exact" else (_adv._display_score(h) or 0.5)
        rh = RecordHit(
            id=new_id("rec"),
            kind=_record_kind(doc_title, is_tx, prof_kinds),
            title=(quotes[0][:90] + ("…" if len(quotes[0]) > 90 else "")) if quotes else doc_title,
            doc_id=str(src.get("doc_id") or ""),
            doc_title=str(doc_title),
            page=src.get("page_number"),
            section_path=src.get("section_path"),
            quotes=[{"text": qtext, "chunk_id": h.get("chunk_id")} for qtext in quotes],
            score=float(score),
            match=match,
            namespace=str(src.get("namespace") or "default"),
            sentence_id=sentence.sentence_id,
            subject_id=subj.id if subj else None,
            entity_id=next((e.id for e in entities if e.id), None),
            source_transcript_id=(src.get("transcript_id") or src.get("session_id")) if is_tx else None,
            chunk_ids=[h.get("chunk_id")],
        )
        out.append(rh)
        state.records[rh.id] = rh
        state.record_key_ids[key] = rh.id
        if len(out) >= 6:
            break
    if subj:
        subj.records_count += len(out)
    state.last_reused_records = reused
    logger.info("SA records: %d new, %d reused (terms=%s) in %.0fms", len(out), len(reused), terms[:3], (time.monotonic() - t0) * 1000)
    return out


async def related_case_hop(records: List[RecordHit], sentence: Sentence, state: SessionAssistantState) -> List[RecordHit]:
    """Legal: from matter chunks, find case names/numbers and pull those (no LLM)."""
    if state.profile.id != "legal" or not records:
        return []
    refs: List[str] = []
    pat = re.compile(r"\b([A-Z][A-Za-z.&' ]{2,40}\s+v\.?\s+[A-Z][A-Za-z.&' ]{2,40})|\b(\d{1,5}\s*(?:/|of)\s*\d{4})\b")
    for r in records:
        for q in r.quotes:
            for m in pat.finditer(q.get("text") or ""):
                ref = (m.group(1) or m.group(2) or "").strip()
                if ref and ref not in refs and len(refs) < 4:
                    refs.append(ref)
    if not refs:
        return []
    fake = [Entity(kind="case", value=r, normalized=re.sub(r"[\s\-./]", "", r).upper(), span=(0, 0), confidence=0.6, cue="hop", variants=[r]) for r in refs]
    for f in fake:
        f.id = new_id("ent")
    hops = await lookup_records(fake, sentence, state)
    for h in hops:
        h.kind = "related_case"
    return hops
