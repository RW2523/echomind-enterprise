"""The stages a matter moves through (case management, 2026-11-07).

Each matter follows one track; each track's stages are fixed here, in order, with the standard tasks a firm usually
plans at that stage. A person moves a matter between stages and chooses which standard tasks to add; nothing here
sets a date. Firm-defined tracks are a later step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Stage:
    key: str
    label: str
    tasks: tuple[str, ...] = ()


@dataclass(frozen=True)
class Track:
    key: str
    label: str
    stages: tuple[Stage, ...]


TRACKS: dict[str, Track] = {t.key: t for t in (
    Track("disputes", "Disputes", (
        Stage("intake", "Intake & conflicts", ("Run the conflict check", "Send the engagement letter", "Collect the client's documents")),
        Stage("assessment", "Case assessment", ("Review the key documents", "Assess merits and risks", "Advise the client on strategy")),
        Stage("pleadings", "Pleadings", ("Draft the claim or defence", "Serve the claim or defence", "Diary the response deadline")),
        Stage("disclosure", "Disclosure & evidence", ("Agree the disclosure list", "Take witness statements", "Instruct any expert")),
        Stage("settlement", "Settlement / ADR", ("Prepare the settlement position", "Attend mediation or negotiation", "Record the outcome")),
        Stage("hearing", "Hearing / trial", ("Prepare the hearing bundle", "Brief counsel", "Confirm the hearing date with the court")),
        Stage("concluded", "Concluded", ("Report the outcome to the client", "Close the file and send the final bill")),
    )),
    Track("transactions", "Transactions", (
        Stage("instruction", "Instruction", ("Run the conflict check", "Send the engagement letter", "Agree the timetable")),
        Stage("diligence", "Due diligence", ("Open the data room review", "Send the questions to the other side", "Report the key issues")),
        Stage("negotiation", "Drafting & negotiation", ("Prepare the first draft", "Negotiate the open points", "Agree the final form")),
        Stage("signing", "Signing", ("Prepare the signing checklist", "Collect the signatures")),
        Stage("completion", "Completion", ("Satisfy the conditions", "Complete and exchange", "Make the filings")),
        Stage("post_completion", "Post-completion", ("Report to the client", "Close the file")),
    )),
    Track("advisory", "Advisory", (
        Stage("instruction", "Instruction", ("Run the conflict check", "Confirm the question and the scope")),
        Stage("research", "Research & analysis", ("Collect the facts and documents", "Research the law")),
        Stage("drafting", "Advice drafted", ("Draft the advice", "Have the advice reviewed")),
        Stage("advised", "Advice given", ("Send the advice to the client",)),
        Stage("follow_up", "Follow-up", ("Answer the client's questions", "Close the file")),
    )),
)}

_TRACK_WORDS = (
    ("disputes", ("litigat", "disput", "employment", "arbitrat", "insolvenc")),
    ("transactions", ("corporate", "m&a", "commercial", "real estate", "property", "finance", "transaction")),
)


def suggest_track(practice_area: Optional[str]) -> str:
    """The track a practice area usually follows (the same rule migration 0042 used for existing matters)."""
    area = (practice_area or "").casefold()
    for track, words in _TRACK_WORDS:
        if any(w in area for w in words):
            return track
    return "advisory"


def first_stage(track: str) -> str:
    return TRACKS[track].stages[0].key


def stage(track: str, key: Optional[str]) -> Optional[Stage]:
    return next((s for s in TRACKS[track].stages if s.key == key), None) if track in TRACKS else None


def stage_label(track: str, key: Optional[str]) -> str:
    s = stage(track, key)
    return s.label if s else (key or "")


def next_stage(track: str, key: Optional[str]) -> Optional[str]:
    keys = [s.key for s in TRACKS[track].stages]
    return keys[keys.index(key) + 1] if key in keys and keys.index(key) + 1 < len(keys) else None


def public() -> list[dict]:
    """The tracks for the client: each stage with its label and standard tasks."""
    return [{"key": t.key, "label": t.label,
             "stages": [{"key": s.key, "label": s.label, "tasks": list(s.tasks)} for s in t.stages]} for t in TRACKS.values()]
