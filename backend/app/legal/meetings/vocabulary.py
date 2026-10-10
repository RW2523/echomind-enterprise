"""Matter vocabulary for transcripts (S2.4, LIVE-01; decision D-040).

Speech recognition mishears the names that matter most: "harrow gate systems" for Harrowgate
Systems, "petra fos" for Petra Voss. The matter already knows these names:
- the meeting's participants;
- the companies named in its documents, with their legal form;
- the terms its documents define.

For each segment, words that sound or read close to such a name, but are not it, are offered as a
correction. Each suggestion carries its provenance: the participant, or the document, version and
characters where the name is written.

Only differences in letters or word boundaries are offered ("harrow gate" → "Harrowgate"); case or
a possessive alone ("the provider's") is not. Nothing is applied by itself. A person accepts a suggestion, and it becomes a revision (source
"vocabulary") with that provenance. Only documents the person can see are used.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from ...core.db import get_conn
from ..documents.repo import VISIBLE_TO

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’&.-]*")
_MIN_SCORE = 0.8


@dataclass
class Term:
    text: str                       # as it should be written
    kind: str                       # participant | company | defined_term
    source: dict                    # where it comes from


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def terms(matter_id: str, meeting_id: str, uid: str) -> list[Term]:
    from ..compare import facts
    out: dict[str, Term] = {}
    with get_conn() as conn:
        for pid, name in conn.execute("SELECT id, name FROM legal_meeting_participants WHERE meeting_id=? AND removed_at IS NULL",
                                      (meeting_id,)):
            out.setdefault(_squash(name), Term(name, "participant", {"kind": "participant", "participant_id": pid}))
        docs = conn.execute(f"SELECT d.id, d.title, v.id, v.number, t.text FROM legal_documents d "
                            f"JOIN legal_document_versions v ON v.id = d.current_version_id "
                            f"JOIN legal_version_text t ON t.version_id = v.id WHERE d.matter_id=? AND {VISIBLE_TO}",
                            (matter_id, uid)).fetchall()
        defined = conn.execute(f"SELECT x.term, x.term_start, x.term_end, d.id, d.title, v.id, v.number FROM legal_defined_terms x "
                               f"JOIN legal_document_versions v ON v.id = x.version_id JOIN legal_documents d ON d.id = v.document_id "
                               f"WHERE d.matter_id=? AND v.id = d.current_version_id AND {VISIBLE_TO}", (matter_id, uid)).fetchall()
    for did, title, vid, number, text in docs:
        for f in facts.extract(text, [(0, len(text))]):
            if f.kind == "entity" and f.subject:
                src = {"kind": "document", "document_id": did, "document_title": title, "version_id": vid, "version_number": number,
                       "char_start": f.start, "char_end": f.end}
                base = " ".join(f.shown.split()[:-1]) or f.shown          # the name without its legal form
                out.setdefault(_squash(base), Term(base, "company", src))
    for term, a, b, did, title, vid, number in defined:
        if len(_squash(term)) >= 4:
            out.setdefault(_squash(term), Term(term, "defined_term", {"kind": "document", "document_id": did, "document_title": title,
                                                                     "version_id": vid, "version_number": number,
                                                                     "char_start": a, "char_end": b}))
    return [t for k, t in out.items() if len(k) >= 4]


def suggest(segment: dict, vocabulary: list[Term]) -> list[dict]:
    """Corrections for one segment's (current) text: a run of words close to a known name but not it."""
    text = segment["text"]
    words = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    found: list[dict] = []
    taken: list[tuple[int, int]] = []
    for t in sorted(vocabulary, key=lambda t: -len(t.text)):
        want = _squash(t.text)
        n = len(t.text.split())
        best: Optional[tuple[float, int, int]] = None
        for size in {max(1, n - 1), n, n + 1}:
            for i in range(0, len(words) - size + 1):
                a, b = words[i][0], words[i + size - 1][1]
                heard = text[a:b]
                got = _squash(heard)
                if not got or abs(len(got) - len(want)) > max(2, len(want) // 3):
                    continue
                score = SequenceMatcher(None, got, want).ratio()
                if score >= _MIN_SCORE and (best is None or score > best[0]):
                    best = (score, a, b)
        if not best:
            continue
        score, a, b = best
        heard = text[a:b].rstrip(".,;:")
        b = a + len(heard)
        plain = re.sub(r"['’]s\b", "", heard).lower().split()
        if plain == t.text.lower().split() or any(x < b and a < y for x, y in taken):
            continue                                  # right but for case or a possessive, or overlapping a better suggestion
        taken.append((a, b))
        sid = hashlib.sha256(f"{segment['id']}|{a}|{b}|{t.text}".encode()).hexdigest()[:16]
        found.append({"id": sid, "segment_id": segment["id"], "start": a, "end": b, "heard": heard, "suggested": t.text,
                      "kind": t.kind, "score": round(score, 3), "source": t.source,
                      "text": text[:a] + t.text + text[b:]})
    return sorted(found, key=lambda s: s["start"])
