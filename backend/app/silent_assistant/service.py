"""Silent Assistant orchestration — the tiers ws.py drives per committed sentence:

  T0  filler gate                (regex, <1 ms)          -> analysis_done{skipped}
  T1  entities + record pull     (regex + SQLite/BM25/CE) -> entity / subject / record messages
  T2  claim retrieval            (hybrid + CE rerank)    -> gated sentences finish here (no LLM)
  T3  batched verify             (ONE LLM call / 1-4 sentences) -> analysis per sentence

`emit` is an async callable(dict) that ws.py binds to its socket. Boardroom calls
check_turns() with no emit and gets the results back."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Awaitable, Callable, Dict, List, Optional

from ..core.config import settings
from ..core.db import get_conn
from ..utils.ids import new_id, now_iso
from .entities import PERSONAL_DETAIL_KINDS, LOOKUP_KINDS, Entity, extract_entities_fast, normalize_spoken_numbers
from .profiles import ScenarioProfile, profile_for, suggest_scenario
from .retrieval import ClaimEvidence, lookup_records, related_case_hop, retrieve_for_claim
from .state import RecordHit, SessionAssistantState, Sentence, Subject
from .verifier import SentenceCheck, phrase_for, verify_batch, _has_claim_signal, _rule_hits

logger = logging.getLogger(__name__)

Emit = Callable[[dict], Awaitable[None]]

_FILLER_RE = re.compile(
    r"^\s*(?:(?:um+|uh+|hmm+|okay|ok|yes|yeah|yep|no|nope|right|sure|thanks|thank you|hello|hi|hey|bye|goodbye|"
    r"good (?:morning|afternoon|evening)|please|alright|got it|i see|great|perfect|fine|one (?:moment|second)|hold on|"
    r"let me (?:check|see)|how are you|i'm (?:fine|good|well))[\s,.!?]*){1,4}$",
    re.IGNORECASE,
)

_LLM_SEM: Optional[asyncio.Semaphore] = None


def llm_sem() -> asyncio.Semaphore:
    global _LLM_SEM
    if _LLM_SEM is None:
        _LLM_SEM = asyncio.Semaphore(max(1, int(getattr(settings, "ASSISTANT_LLM_CONCURRENCY", 3))))
    return _LLM_SEM


async def _noop(_: dict) -> None:
    return None


# ── T0 / T1 ────────────────────────────────────────────────────────────────────
async def on_sentence_fast(state: SessionAssistantState, s: Sentence, emit: Emit = _noop) -> dict:
    """Runs immediately when a sentence commits. Returns {'checkable': bool, 'entities': [...], 'records': [...]}."""
    if not state.push_sentence(s):
        return {"checkable": False, "entities": [], "records": [], "dup": True}
    words = len(s.text.split())
    if _FILLER_RE.match(s.text) or words < 2:
        await emit({"type": "analysis_done", "segment_id": s.paragraph_id, "sentence_id": s.sentence_id,
                    "status": "skipped", "result": None})
        return {"checkable": False, "entities": [], "records": []}

    profile = state.profile
    if not s.norm_text:
        s.norm_text = normalize_spoken_numbers(s.text)
    ents = extract_entities_fast(s.norm_text, profile.entity_kinds)
    for e in ents:
        state.register_entity(e, s)
    # subject formation only from lookup roles (or unknown role) and personal-detail kinds
    lookup_ok = (s.role is None) or (s.role in profile.lookup_roles)
    personal = [e for e in ents if e.kind in PERSONAL_DETAIL_KINDS]
    subj: Optional[Subject] = None
    if lookup_ok and personal:
        subj = state.attach_to_subject(personal, s.role)
    for e in ents:
        await emit({"type": "entity", **e.public(), "sentence_id": s.sentence_id, "segment_id": s.paragraph_id, "role": s.role})
    if subj:
        await emit({"type": "subject", **subj.public()})

    records: List[RecordHit] = []
    lookup_ents = [e for e in ents if e.kind in LOOKUP_KINDS]
    if lookup_ents and state.analysis_mode == "flags_and_records" and (lookup_ok or any(e.kind != "person" for e in lookup_ents)):
        reused: List[RecordHit] = []
        try:
            state.last_reused_records = []
            records = await lookup_records(lookup_ents, s, state)
            reused = list(state.last_reused_records)
            if records:
                hops = await related_case_hop(records, s, state)
                records.extend(hops)
        except Exception as e:
            logger.warning("SA record lookup failed: %s", e)
        for r in records:
            await emit({"type": "record", **r.public()})
        if reused:
            # same identifier mentioned again: link the existing records to THIS sentence too
            for r in reused:
                await emit({"type": "record", **{**r.public(), "sentence_id": s.sentence_id, "reused": True}})
            records = records + reused
        if subj:
            await emit({"type": "subject", **subj.public()})
    checkable = bool(
        (s.role is None or s.role in profile.verify_roles)
        and (_has_claim_signal(s.query_text) or _rule_hits(s, profile) or ents or words >= 6)
    )
    if not checkable and not ents:
        await emit({"type": "analysis_done", "segment_id": s.paragraph_id, "sentence_id": s.sentence_id,
                    "status": "skipped", "result": None})
    return {"checkable": checkable, "entities": ents, "records": records}


# ── T2 / T3 ────────────────────────────────────────────────────────────────────
async def run_batch(state: SessionAssistantState, sentences: List[Sentence], emit: Emit = _noop,
                    records_by_sid: Optional[Dict[str, List[RecordHit]]] = None,
                    entities_by_sid: Optional[Dict[str, List[Entity]]] = None) -> List[SentenceCheck]:
    if not sentences:
        return []
    t0 = time.monotonic()
    records_by_sid = records_by_sid or {}
    entities_by_sid = entities_by_sid or {}
    for s in sentences:
        await emit({"type": "analysis_start", "segment_id": s.paragraph_id, "sentence_id": s.sentence_id})

    # T2: retrieval per sentence concurrently
    async def _ret(s: Sentence) -> ClaimEvidence:
        try:
            return await retrieve_for_claim(s.query_text, state)
        except Exception as e:
            logger.warning("SA retrieval failed for %s: %s", s.sentence_id, e)
            from ..rag.advanced import RetrievalResult
            return ClaimEvidence(result=RetrievalResult(source_type="insufficient", hits=[]))
    evs = await asyncio.gather(*[_ret(s) for s in sentences])
    ev_by_sid = {s.sentence_id: ev for s, ev in zip(sentences, evs)}

    to_llm: List[Sentence] = []
    checks: List[SentenceCheck] = []
    for s in sentences:
        ce = ev_by_sid[s.sentence_id]
        recs = records_by_sid.get(s.sentence_id, [])
        rules = _rule_hits(s, state.profile)
        ents = entities_by_sid.get(s.sentence_id, [])
        speech_act = bool(re.search(r"\b(i will|i'll|we will|we'll|let me|i can|i'm going to|make sure|by (?:monday|tuesday|wednesday|thursday|friday|tomorrow|next|end of)|action item|decided|decision|agreed)\b", s.text, re.IGNORECASE)) or s.text.strip().endswith("?")
        if not s.norm_text:
            s.norm_text = normalize_spoken_numbers(s.text)
        if ce.result.gated and not recs and not rules and not ents and not speech_act:
            # nothing in the KB relates to it and nothing programmatic — checked, nothing to say
            await emit({"type": "analysis_done", "segment_id": s.paragraph_id, "sentence_id": s.sentence_id,
                        "status": "checked", "searched_docs": ce.searched_docs, "result": None})
            continue
        to_llm.append(s)
    if to_llm:
        async with llm_sem():
            batch_checks = await verify_batch(
                to_llm, ev_by_sid, records_by_sid, state,
                entities_by_sid={sid: [e.public() for e in ents] for sid, ents in entities_by_sid.items()},
            )
        checks.extend(batch_checks)

    for chk in checks:
        s = chk.sentence
        if not chk.has_content():
            await emit({"type": "analysis_done", "segment_id": s.paragraph_id, "sentence_id": s.sentence_id,
                        "status": "no_tags", "searched_docs": chk.searched_docs, "result": None})
            continue
        chk.id = new_id("ana")
        payload = to_payload(chk, state)
        state.checks_by_sentence[s.sentence_id] = payload
        if any(t["tag"] in ("action-item", "commitment", "decision") for t in chk.tags):
            state.action_items.append({"sentence_id": s.sentence_id, "text": s.text, "role": s.role,
                                       "tags": [t["tag"] for t in chk.tags if t["tag"] in ("action-item", "commitment", "decision")]})
        await emit(payload)
        asyncio.get_running_loop().run_in_executor(None, persist_check, chk, payload, state)
    logger.info("SA batch: %d sentences, %d to LLM, %d checks, %.0fms", len(sentences), len(to_llm), len(checks), (time.monotonic() - t0) * 1000)
    return checks


def to_payload(chk: SentenceCheck, state: SessionAssistantState) -> dict:
    s = chk.sentence
    return {
        "type": "analysis",
        "id": chk.id or new_id("ana"),
        "session_id": state.session_id,
        "segment_id": s.paragraph_id,
        "sentence_id": s.sentence_id,
        "sentence_text": s.text,
        "segment_text": s.text,                # legacy
        "char_start": s.char_start,
        "char_end": s.char_end,
        "role": s.role,
        "kind": chk.kind,
        "label": chk.legacy_label(),           # legacy
        "verdict": chk.verdict,
        "confidence": chk.confidence,
        "explanation": chk.explanation,
        "tags": chk.tags,
        "evidence": [e.public() for e in chk.evidence],
        "record_ids": chk.record_ids,
        "searched_docs": chk.searched_docs,
        "entities": chk.entities,
        "llm_skipped": chk.llm_skipped,
        "latency_ms": chk.latency_ms,
        "phrase": phrase_for(chk),
        "source_chunks": chk.source_chunks,    # legacy
        "scenario": state.profile.id,
        "created_at": now_iso(),
    }


# ── persistence ────────────────────────────────────────────────────────────────
def persist_check(chk: SentenceCheck, payload: dict, state: SessionAssistantState) -> None:
    try:
        s = chk.sentence
        with get_conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO transcript_analysis
                   (id, session_id, transcript_id, segment_id, segment_text, label, confidence, explanation, source_refs, created_at,
                    sentence_id, paragraph_id, char_start, char_end, role, speaker, scenario, namespace, kind, verdict,
                    tags_json, evidence_json, entities_json, record_ids_json, retrieval_meta_json, latency_ms, model)
                   VALUES (?,?,?,?,?,?,?,?,?,?, ?,?,?,?,?,?,?,?,?,?, ?,?,?,?,?,?,?)""",
                (payload["id"], state.session_id, state.transcript_id, s.paragraph_id, s.text, payload["label"], chk.confidence,
                 chk.explanation, json.dumps(chk.source_chunks), payload["created_at"],
                 s.sentence_id, s.paragraph_id, s.char_start, s.char_end, s.role, s.speaker, state.profile.id, state.namespace,
                 chk.kind, chk.verdict, json.dumps(chk.tags), json.dumps(payload["evidence"]), json.dumps(chk.entities),
                 json.dumps(chk.record_ids), json.dumps(chk.retrieval_meta), chk.latency_ms, settings.LLM_MODEL),
            )
            conn.commit()
    except Exception as e:
        logger.warning("SA persist_check failed: %s", e)


def persist_entities_records(state: SessionAssistantState, ents: List[Entity], records: List[RecordHit], subj: Optional[Subject]) -> None:
    try:
        with get_conn() as conn:
            for e in ents:
                conn.execute(
                    "INSERT OR IGNORE INTO assistant_entities (id, session_id, transcript_id, sentence_id, kind, value, normalized, role, confidence, subject_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (e.id or new_id("ent"), state.session_id, state.transcript_id, "", e.kind, e.value, e.normalized, e.role, e.confidence, e.subject_id, now_iso()))
            for r in records:
                conn.execute(
                    "INSERT OR IGNORE INTO assistant_records (id, session_id, transcript_id, sentence_id, subject_id, entity_id, kind, title, doc_id, doc_title, page, section_path, quotes_json, score, match, namespace, source_transcript_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (r.id, state.session_id, state.transcript_id, r.sentence_id, r.subject_id, r.entity_id, r.kind, r.title, r.doc_id, r.doc_title, r.page, r.section_path, json.dumps(r.quotes), r.score, r.match, r.namespace, r.source_transcript_id, now_iso()))
            if subj:
                conn.execute(
                    "INSERT OR REPLACE INTO assistant_subjects (id, session_id, transcript_id, kind, display_name, matched_fields_json, entity_ids_json, confidence, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (subj.id, state.session_id, state.transcript_id, subj.kind, subj.display_name, json.dumps(subj.matched_fields), json.dumps(subj.entity_ids), subj.confidence, subj.status, now_iso(), now_iso()))
            conn.commit()
    except Exception as e:
        logger.warning("SA persist_entities_records failed: %s", e)


def persist_segment(state: SessionAssistantState, paragraph_id: str, idx: int, text: str, role: Optional[str],
                    start_ms: Optional[int], end_ms: Optional[int], sentences: List[dict]) -> None:
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO transcript_segments (id, session_id, transcript_id, idx, text, role, speaker, start_ms, end_ms, sentences_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (paragraph_id, state.session_id, state.transcript_id, idx, text, role, None, start_ms, end_ms, json.dumps(sentences), now_iso()))
            conn.commit()
    except Exception as e:
        logger.warning("SA persist_segment failed: %s", e)


def backfill_transcript_id(session_id: str, transcript_id: str) -> None:
    try:
        with get_conn() as conn:
            for tbl in ("transcript_analysis", "transcript_segments", "assistant_entities", "assistant_subjects", "assistant_records"):
                conn.execute(f"UPDATE {tbl} SET transcript_id=? WHERE session_id=? AND (transcript_id IS NULL OR transcript_id='')", (transcript_id, session_id))
            conn.commit()
    except Exception as e:
        logger.warning("SA backfill_transcript_id failed: %s", e)


def on_subject_action(state: SessionAssistantState, subject_id: str, action: str) -> Optional[Subject]:
    status = {"confirm": "confirmed", "reject": "rejected"}.get(action)
    if not status:
        return None
    subj = state.set_subject_status(subject_id, status)
    if subj:
        persist_entities_records(state, [], [], subj)
    return subj


def maybe_suggest_scenario(state: SessionAssistantState) -> Optional[dict]:
    """Detector cadence: after ~40 words then every 60 s until the operator confirms."""
    if state.scenario_confirmed:
        return None
    words = sum(len(s.text.split()) for s in state.window)
    now = time.time()
    if words < 40 or (state.last_detect_at and now - state.last_detect_at < 60):
        return None
    state.last_detect_at = now
    sid, conf, reason = suggest_scenario(" ".join(s.text for s in state.window), state.namespace)
    if sid and conf >= 0.7 and sid != state.profile.id and sid != state.suggested:
        state.suggested = sid
        return {"type": "scenario_suggest", "scenario": sid, "confidence": conf, "reason": reason}
    return None


# ── Boardroom / offline ────────────────────────────────────────────────────────
async def check_turns(turns: List[dict], profile: ScenarioProfile, namespace: str = "", session_id: str = "",
                      speaker_map: Optional[dict] = None) -> dict:
    """Offline pass over diarized turns [{speaker, text, start?, end?}]. Returns
    {checks:[payloads], records:[...], entities:[...], subjects:[...], action_items:[...]}."""
    state = SessionAssistantState(session_id or new_id("br"), profile, namespace, "flags_and_records")
    speaker_map = speaker_map or {}
    all_checks: List[dict] = []
    pending: List[Sentence] = []
    rec_by, ent_by = {}, {}
    from ..rag.evidence_extractor import _split_sentences
    n = 0
    for ti, t in enumerate(turns):
        role = speaker_map.get(str(t.get("speaker"))) or None
        text = (t.get("text") or "").strip()
        for si, sent in enumerate(_split_sentences(text) or ([text] if text else [])):
            n += 1
            s = Sentence(sentence_id=f"br-{ti}-s{si}", paragraph_id=f"br-{ti}", text=sent, char_start=0, char_end=len(sent),
                         role=role, speaker=str(t.get("speaker")))
            fast = await on_sentence_fast(state, s)
            if fast.get("dup"):
                continue
            rec_by[s.sentence_id] = fast["records"]; ent_by[s.sentence_id] = fast["entities"]
            if fast["checkable"]:
                pending.append(s)
            if len(pending) >= 4:
                all_checks.extend(to_payload(c, state) for c in await run_batch(state, pending, _noop, rec_by, ent_by) if c.has_content())
                pending = []
    if pending:
        all_checks.extend(to_payload(c, state) for c in await run_batch(state, pending, _noop, rec_by, ent_by) if c.has_content())
    return {
        "checks": all_checks,
        "records": [r.public() for r in state.records.values()],
        "entities": [e.public() for e in state.entities.values()],
        "subjects": [s.public() for s in state.subjects.values()],
        "action_items": state.action_items,
    }
