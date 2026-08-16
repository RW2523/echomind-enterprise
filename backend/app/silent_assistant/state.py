"""Per-session conversation state for the Silent Assistant: sentence window, entities,
subjects (the person the call is about), pulled records, action items."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from ..utils.ids import new_id
from .entities import Entity, LOOKUP_KINDS
from .profiles import ScenarioProfile


@dataclass
class Sentence:
    sentence_id: str
    paragraph_id: str
    text: str
    char_start: int
    char_end: int
    ts_ms: int = 0
    wall_ms: float = 0.0
    role: Optional[str] = None
    speaker: Optional[str] = None
    norm_text: str = ""            # spoken numbers -> digits (used for extraction / retrieval / verify)

    @property
    def query_text(self) -> str:
        return self.norm_text or self.text


@dataclass
class Subject:
    id: str
    kind: str                       # customer|client|account_holder|counterparty|person
    display_name: str
    matched_fields: List[str] = field(default_factory=list)
    entity_ids: List[str] = field(default_factory=list)
    confidence: float = 0.5
    status: str = "candidate"       # candidate|confirmed|rejected
    records_count: int = 0
    identifiers: Dict[str, str] = field(default_factory=dict)   # kind -> normalized

    def public(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "display_name": self.display_name,
            "matched_fields": list(self.matched_fields), "entity_ids": list(self.entity_ids),
            "confidence": round(self.confidence, 2), "status": self.status, "records_count": self.records_count,
        }


@dataclass
class RecordHit:
    id: str
    kind: str
    title: str
    doc_id: str
    doc_title: str
    page: Optional[int]
    section_path: Optional[str]
    quotes: List[dict]              # [{text, chunk_id}]
    score: float                    # 0-1
    match: str                      # exact|fuzzy|semantic
    namespace: str
    sentence_id: str
    subject_id: Optional[str] = None
    entity_id: Optional[str] = None
    source_transcript_id: Optional[str] = None
    chunk_ids: List[str] = field(default_factory=list)

    def public(self) -> dict:
        return {
            "id": self.id, "subject_id": self.subject_id, "entity_id": self.entity_id, "sentence_id": self.sentence_id,
            "kind": self.kind, "title": self.title, "doc_id": self.doc_id, "doc_title": self.doc_title,
            "page": self.page, "section_path": self.section_path, "quotes": list(self.quotes),
            "score": round(float(self.score), 3), "match": self.match, "namespace": self.namespace,
            "source_transcript_id": self.source_transcript_id,
        }


class SessionAssistantState:
    def __init__(self, session_id: str, profile: ScenarioProfile, namespace: str, analysis_mode: str,
                 transcript_id: Optional[str] = None, subject_hint: Optional[dict] = None,
                 participants: Optional[List[dict]] = None):
        self.session_id = session_id
        self.transcript_id = transcript_id
        self.profile = profile
        self.namespace = (namespace or "").strip()
        self.analysis_mode = analysis_mode or profile.analysis_mode_default
        self.participants = participants or []
        self.window: Deque[Sentence] = deque(maxlen=12)
        self.entities: Dict[str, Entity] = {}            # key = kind|normalized
        self.subjects: Dict[str, Subject] = {}
        self.records: Dict[str, RecordHit] = {}
        self.emitted_record_keys: set = set()             # (kind, doc_id, page, first quote hash)
        self.record_key_ids: Dict[tuple, str] = {}        # key -> record id (to link later mentions)
        self.last_reused_records: list = []
        self.sentence_hashes: set = set()
        self.action_items: List[dict] = []
        self.checks_by_sentence: Dict[str, dict] = {}
        self.started_at = time.time()
        self.scenario_confirmed = profile.id != "general" or False
        self.last_detect_words = 0
        self.last_detect_at = 0.0
        self.suggested: Optional[str] = None
        if subject_hint:
            self._seed_subject(subject_hint)

    # ── subject handling ────────────────────────────────────────────────────
    def _seed_subject(self, hint: dict) -> None:
        name = (hint.get("name") or "").strip()
        ids = [str(i).strip() for i in (hint.get("ids") or []) if str(i).strip()]
        if not name and not ids:
            return
        s = Subject(id=new_id("sub"), kind=self._subject_kind(), display_name=name or ids[0],
                    matched_fields=(["name"] if name else []) + (["id"] * len(ids)),
                    confidence=0.6, status="candidate")
        if name:
            s.identifiers["person"] = name.lower()
        for i in ids:
            s.identifiers.setdefault("account", i)
        self.subjects[s.id] = s

    def _subject_kind(self) -> str:
        return {"customer_care": "customer", "legal": "client", "banking": "account_holder", "general": "person"}.get(self.profile.id, "person")

    def register_entity(self, e: Entity, sentence: Sentence) -> Entity:
        key = f"{e.kind}|{e.normalized}"
        existing = self.entities.get(key)
        if existing:
            existing.confidence = max(existing.confidence, e.confidence)
            return existing
        e.id = e.id or new_id("ent")
        e.role = sentence.role
        self.entities[key] = e
        return e

    def attach_to_subject(self, ents: List[Entity], role: Optional[str]) -> Optional[Subject]:
        """Personal details spoken by a lookup role become (or extend) the active subject."""
        if not ents:
            return None
        person = next((e for e in ents if e.kind == "person"), None)
        ids = [e for e in ents if e.kind in ("account", "policy", "order", "ticket", "case", "card", "phone", "email", "dob", "contract")]
        if not person and not ids:
            return None
        subj = self.active_subject()
        # A new person accompanied by an identifier supersedes a weak name-only candidate
        # (STT heard "this is Mark speaking" before the real caller introduced herself).
        supersede = bool(
            subj is not None and subj.status != "confirmed" and person and ids
            and subj.identifiers.get("person") not in (None, person.normalized)
            and not any(k for k in subj.identifiers if k != "person")
        )
        if supersede:
            subj.status = "rejected"
            subj = None
        if subj is None or (person and subj.identifiers.get("person") and subj.identifiers["person"] != person.normalized and subj.status != "confirmed"):
            subj = Subject(id=new_id("sub"), kind=self._subject_kind(),
                           display_name=person.value if person else ids[0].value, confidence=0.5)
            self.subjects[subj.id] = subj
        if person and not subj.identifiers.get("person"):
            subj.identifiers["person"] = person.normalized
            subj.display_name = person.value
            subj.matched_fields.append("name")
            subj.confidence = min(1.0, subj.confidence + 0.2)
        for e in ids:
            if e.kind not in subj.identifiers:
                subj.identifiers[e.kind] = e.normalized
                subj.matched_fields.append(e.kind)
                subj.confidence = min(1.0, subj.confidence + 0.15)
        for e in ents:
            if e.id and e.id not in subj.entity_ids:
                subj.entity_ids.append(e.id)
            e.subject_id = subj.id
        return subj

    def active_subject(self) -> Optional[Subject]:
        confirmed = [s for s in self.subjects.values() if s.status == "confirmed"]
        if confirmed:
            return confirmed[-1]
        cands = [s for s in self.subjects.values() if s.status == "candidate"]
        return cands[-1] if cands else None

    def set_subject_status(self, subject_id: str, status: str) -> Optional[Subject]:
        s = self.subjects.get(subject_id)
        if s:
            s.status = status
        return s

    # ── context helpers ─────────────────────────────────────────────────────
    def resolve_reference(self, text: str) -> str:
        """Prepend the active subject's identity to pronoun-heavy sentences so retrieval has
        the right keys ('her contract ends next month' -> 'Priya Nair AT44821: her contract...')."""
        subj = self.active_subject()
        if not subj:
            return text
        low = f" {text.lower()} "
        if any(p in low for p in (" his ", " her ", " their ", " they ", " she ", " he ", " the client", " the customer", " the account", " this account", " your ", " you ", " my ")):
            ident = " ".join(v for k, v in subj.identifiers.items() if k != "person")
            return f"{subj.display_name} {ident}: {text}".strip()
        return text

    def context_line(self) -> str:
        parts = []
        subj = self.active_subject()
        if subj:
            ident = ", ".join(f"{k} {v}" for k, v in subj.identifiers.items() if k != "person")
            parts.append(f"Subject: {subj.display_name}" + (f" ({ident})" if ident else "") + f" [{subj.status}]")
        recent = [s.text for s in list(self.window)[-4:]]
        if recent:
            parts.append("Recent: " + " | ".join(r[:120] for r in recent))
        return " ".join(parts)

    def lookup_terms(self, ents: List[Entity]) -> List[str]:
        terms: List[str] = []
        for e in ents:
            if e.kind not in LOOKUP_KINDS:
                continue
            for v in [e.value] + list(e.variants):
                v = v.strip()
                if len(v) >= 3 and v not in terms:
                    terms.append(v)
            if e.kind == "person":
                parts = e.normalized.split()
                if len(parts) >= 2 and e.value not in terms:
                    terms.append(e.value)  # full name as spoken (surname alone is too noisy: 'Nair' hit 'v. Nair')
        return terms[:12]

    def lookup_terms_safe(self, ents: List[Entity]) -> List[str]:
        """lookup_terms() minus lone single-token person names when no identifier accompanies
        them (STT mishears names; 'Pioneer' must not pull every doc containing that word)."""
        has_id = any(e.kind in LOOKUP_KINDS and e.kind != "person" for e in ents)
        filtered = [e for e in ents if not (e.kind == "person" and len(e.normalized.split()) < 2 and not has_id)]
        return self.lookup_terms(filtered)

    def push_sentence(self, s: Sentence) -> bool:
        h = hash(" ".join(s.text.lower().split()))
        if h in self.sentence_hashes:
            return False
        self.sentence_hashes.add(h)
        self.window.append(s)
        return True
