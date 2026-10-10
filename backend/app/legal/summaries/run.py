"""The summary job (S4.2, D-055): the document's current version, summarised for the person who asked, as a
new draft version of its summary (an AI artifact, reviewed like any other)."""
from __future__ import annotations

import re
from typing import Optional

from ...core.config import settings
from ...core.db import get_conn
from .. import artifacts
from ..documents import repo as documents
from . import summarize

TRANSLATION = {"available": False, "reason": "No local translation model is approved yet."}
_EXT = re.compile(r"\.(txt|md|pdf|docx?|rtf|html?)$", re.I)


def title_of(document_title: str) -> str:
    return "Summary — " + _EXT.sub("", document_title or "document").strip()


async def run(job: dict) -> Optional[str]:
    matter_id, document_id, by = job["matter_id"], job["object_id"], job["requested_by"]
    d = documents.get_document(matter_id, document_id, by)          # still visible to the person who asked
    if not d or not d.get("current_version"):
        return "not_visible"
    v = d["current_version"]
    reader = documents.reader(v["id"])
    if not reader:
        return "not_read"
    with get_conn() as conn:
        clauses = [dict(zip(("number", "start", "end"), r)) for r in conn.execute(
            "SELECT number, char_start, char_end FROM legal_clauses WHERE version_id=? ORDER BY char_start, level", (v["id"],))]
    out = await summarize.summarise(reader["text"], {**reader, "clauses": clauses})
    content = {"title": title_of(d["title"]), "document": {"id": document_id, "title": d["title"], "version_id": v["id"],
                                                         "version_number": v["number"]},
               **out, "translation": TRANSLATION}
    artifacts.create(matter_id, "summary", "document", document_id, content,
                     {"model": settings.LLM_MODEL, "prompt_version": summarize.VERSION, "version_id": v["id"]}, by, "model")
    return None
