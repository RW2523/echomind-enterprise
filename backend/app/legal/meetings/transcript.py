"""The transcript as a record (S2.4, LIVE-01; decisions D-017 and D-040).

- **Raw and immutable.** Each segment keeps the speech model's text exactly as produced (no
  lowercasing, punctuation stripping or token drops) with the speaker marked while it was spoken and
  its wall-clock times. A database trigger refuses any change.
- **Revisions.** A correction of the text or of the speaker is a revision: who, when, why, and where
  it came from (a person, or a matter-vocabulary suggestion with its document). What is shown is
  derived: the latest revision, else the raw text. Revisions are never edited.
- **Marks.** Who was speaking from when, where the model fell behind (degraded), and where a
  transcription error left a gap. They are shown in the transcript, so nothing is lost silently.
- **Finalised.** After the meeting, a contributor can finalise the transcript. It then becomes a
  matter document (a new version each time it is finalised again), read like any other: research
  can cite it (R-7), and holds, purges and exports apply. Purging the transcript also purges that
  document.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Optional

from ...core.db import get_conn
from ..policy import now_iso
from . import repo

SPEAKER_UNKNOWN = "Unattributed"


class NotFound(Exception):
    pass


class Conflict(Exception):
    pass


def _participants(conn, meeting_id: str) -> dict[str, dict]:
    return {r[0]: {"id": r[0], "name": r[1], "role": r[2], "removed": r[3] is not None} for r in conn.execute(
        "SELECT id, name, role, removed_at FROM legal_meeting_participants WHERE meeting_id=?", (meeting_id,))}


def view(matter_id: str, meeting_id: str) -> dict:
    """Segments with their derived text and speaker, every revision, and the capture marks."""
    segs = repo.segments(matter_id, meeting_id)
    with get_conn() as conn:
        people = _participants(conn, meeting_id)
        revs: dict[str, list[dict]] = {}
        for r in conn.execute("SELECT r.id, r.segment_id, r.kind, r.text, r.speaker_participant_id, r.reason, r.source, r.provenance_json, "
                              "u.username, r.at FROM legal_transcript_revisions r LEFT JOIN users u ON u.id = r.by "
                              "WHERE r.meeting_id=? AND r.matter_id=? ORDER BY r.at, r.rowid", (meeting_id, matter_id)):
            revs.setdefault(r[1], []).append({"id": r[0], "kind": r[2], "text": r[3], "speaker_participant_id": r[4], "reason": r[5],
                                              "source": r[6], "provenance": json.loads(r[7] or "{}"), "by": r[8], "at": r[9]})
        marks = [{"kind": r[0], "participant_id": r[1], "detail": r[2], "audio_ms": r[3], "at": r[4], "capture_id": r[5]}
                 for r in conn.execute("SELECT kind, participant_id, detail, audio_ms, at, capture_id FROM legal_capture_marks "
                                       "WHERE meeting_id=? ORDER BY at, rowid", (meeting_id,))]
        meeting = conn.execute("SELECT transcript_document_id, transcript_finalised_at, transcript_deleted_at FROM legal_meetings "
                               "WHERE id=?", (meeting_id,)).fetchone()
    out = []
    for s in segs:
        history = revs.get(s["id"], [])
        text_rev = next((r for r in reversed(history) if r["kind"] == "text"), None)
        speaker_rev = next((r for r in reversed(history) if r["kind"] == "speaker"), None)
        speaker = speaker_rev["speaker_participant_id"] if speaker_rev else s["speaker_participant_id"]
        out.append({**s, "raw_text": s["text"], "text": text_rev["text"] if text_rev else s["text"],
                    "speaker_participant_id": speaker, "raw_speaker_participant_id": s["speaker_participant_id"],
                    "speaker": people.get(speaker, {}).get("name") if speaker else None,
                    "revised": bool(history), "revisions": history})
    last_revision = max((r["at"] for h in revs.values() for r in h), default=None)
    finalised_at = meeting[1] if meeting else None
    return {"segments": out, "marks": marks if not (meeting and meeting[2]) else [],
            "participants": list(people.values()),
            "finalised": {"document_id": meeting[0], "at": finalised_at,
                          "stale": bool(last_revision and finalised_at and last_revision > finalised_at)} if meeting and meeting[0] else None}


def revise(matter_id: str, meeting_id: str, segment_id: str, by: str, reason: str, text: Optional[str] = None,
           speaker_participant_id: Optional[str] = None, source: str = "person", provenance: Optional[dict] = None) -> list[dict]:
    """Record a correction of the text and/or the speaker. Raises NotFound or Conflict."""
    with get_conn() as conn:
        row = conn.execute("SELECT s.id, mt.transcript_deleted_at FROM legal_transcript_segments s JOIN legal_meetings mt ON mt.id = s.meeting_id "
                           "WHERE s.id=? AND s.meeting_id=? AND s.matter_id=?", (segment_id, meeting_id, matter_id)).fetchone()
        if not row or row[1]:
            raise NotFound(segment_id)
        if speaker_participant_id is not None and speaker_participant_id not in _participants(conn, meeting_id):
            raise Conflict("That person is not a participant of this meeting.")
        made = []
        for kind, value in (("text", text), ("speaker", speaker_participant_id)):
            if value is None:
                continue
            rid = str(uuid.uuid4())
            conn.execute("INSERT INTO legal_transcript_revisions (id, segment_id, meeting_id, matter_id, kind, text, speaker_participant_id, "
                         "reason, source, provenance_json, by, at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (rid, segment_id, meeting_id, matter_id, kind, value if kind == "text" else None,
                          value if kind == "speaker" else None, reason, source, json.dumps(provenance or {}), by, now_iso()))
            made.append({"id": rid, "kind": kind})
        conn.commit()
    return made


def add_mark(capture_id: str, meeting_id: str, kind: str, participant_id: Optional[str] = None, detail: Optional[str] = None,
             audio_ms: Optional[int] = None, by: Optional[str] = None, at: Optional[str] = None) -> None:
    """`at` is when it happened in the meeting, if not now (a gap found while the model catches up)."""
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_capture_marks (id, capture_id, meeting_id, kind, participant_id, detail, audio_ms, at, by) "
                     "VALUES (?,?,?,?,?,?,?,?,?)", (str(uuid.uuid4()), capture_id, meeting_id, kind, participant_id, detail, audio_ms,
                                                    at or now_iso(), by))
        conn.commit()


# ── Finalising: the transcript as a matter document ──────────────────────────────
GAP_REASONS = {None: "a transcription error at this point.", "transcription_error": "a transcription error at this point.",
               "transcription_behind": "speech recognition fell too far behind here."}


def _clock(iso: Optional[str]) -> str:
    if not iso:
        return "--:--:--"
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%H:%M:%S")
    except ValueError:
        return "--:--:--"


def _before(gap: dict, seg: dict) -> bool:
    """Whether a gap comes before a segment: by where it is in the audio within one capture (exact, even when audio
    arrives faster than real time), otherwise by wall-clock time."""
    if gap.get("capture_id") and gap.get("capture_id") == seg.get("capture_id") and None not in (gap.get("audio_ms"), seg.get("t_start_ms")):
        return gap["audio_ms"] <= seg["t_start_ms"]
    return bool(seg.get("wall_start")) and gap["at"] <= seg["wall_start"]


def as_document(meeting: dict, data: dict) -> str:
    """The transcript as text: a heading, the participants, then one paragraph per segment with its
    wall-clock time and speaker, and the gaps where nothing could be transcribed."""
    lines = [f"# Transcript — {meeting['title']}", "",
             f"Meeting {meeting['number'] if meeting.get('number') else ''} · {meeting.get('kind', '')} · "
             f"transcribed verbatim; corrections are marked in the record.".replace("  ", " "), "",
             "Participants: " + "; ".join(f"{p['name']} ({p['role']})" for p in data["participants"] if not p["removed"]), ""]
    gaps = sorted((m for m in data["marks"] if m["kind"] == "gap"), key=lambda m: (m["at"], m.get("audio_ms") or 0))
    g = 0
    for s in data["segments"]:
        while g < len(gaps) and _before(gaps[g], s):
            lines += [f"[{_clock(gaps[g]['at'])}] [Not transcribed: {GAP_REASONS.get(gaps[g].get('detail'), GAP_REASONS[None])}]", ""]
            g += 1
        lines += [f"[{_clock(s.get('wall_start'))}] {s['speaker'] or SPEAKER_UNKNOWN}: {s['text']}", ""]
    for m in gaps[g:]:
        lines += [f"[{_clock(m['at'])}] [Not transcribed: {GAP_REASONS.get(m.get('detail'), GAP_REASONS[None])}]", ""]
    return "\n".join(lines).rstrip() + "\n"


def finalise(org_id: str, matter_id: str, meeting: dict, by: str) -> dict:
    """Make the transcript (as corrected so far) a matter document, or a new version of it. Raises
    Conflict when there is nothing to finalise or the meeting is still running."""
    import hashlib

    from ..documents import blobs
    from ..documents import repo as docs
    from ..jobs import worker
    if meeting["status"] != "ended":
        raise Conflict("End the meeting before finalising its transcript.")
    data = view(matter_id, meeting["id"])
    if not data["segments"]:
        raise Conflict("There is no transcript to finalise.")
    body = as_document(meeting, data).encode("utf-8")
    sha = hashlib.sha256(body).hexdigest()
    with get_conn() as conn:
        row = conn.execute("SELECT transcript_document_id FROM legal_meetings WHERE id=? AND matter_id=?",
                           (meeting["id"], matter_id)).fetchone()
    doc_id = row[0] if row and row[0] and docs.get_document(matter_id, row[0]) else None
    if doc_id and docs.current_sha(matter_id, doc_id) == sha:
        return {"document_id": doc_id, "changed": False}
    if not doc_id:
        doc_id = docs.create_document(org_id, matter_id, f"Transcript — {meeting['title']}"[:300], "note", "final", by)
    version_id = docs.new_id()
    wrapped = blobs.store(matter_id, version_id, body)
    try:
        v = docs.add_version(version_id, doc_id, matter_id, f"transcript-{meeting['id'][:8]}.md", "text/markdown", len(body), sha,
                             None, wrapped, by, status="final")
    except BaseException:
        blobs.remove(matter_id, version_id)
        raise
    with get_conn() as conn:
        conn.execute("UPDATE legal_meetings SET transcript_document_id=?, transcript_finalised_at=?, transcript_finalised_by=? WHERE id=?",
                     (doc_id, now_iso(), by, meeting["id"]))
        conn.commit()
    worker.wake()
    return {"document_id": doc_id, "version_number": v["number"], "changed": True}
