"""Machine translation (OPT-02; decision D-070). Off unless LEGAL_TRANSLATION is on (404 otherwise).

- `POST /api/legal/matters/{m}/documents/{d}/translate` {language, from?}: the document's current version, streamed as
  NDJSON: `start` (the label, the paragraph count), one `paragraph` per paragraph (its original and translation), a
  `limit` with where to continue when LEGAL_TRANSLATE_MAX_CHARS is reached, then `done`. Access is re-checked between
  batches, as research does; a revocation ends the stream (`scope_revoked`).
- `POST /api/legal/matters/{m}/summaries/{artifact_id}/translate` {language}: a summary's points, translated.
Anyone who may read the matter may translate it: it is a way of reading. Nothing is stored; the audit trail records
the counts.
"""
from __future__ import annotations

import json
import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...core.config import settings
from .. import artifacts
from ..audit.record import audited, note
from ..documents import repo as documents
from ..matters.routes import principal_of, require, require_law_edition
from ..retrieval import scope as scopemod
from . import translate as tr

logger = logging.getLogger(__name__)


def translation_on() -> None:
    if not settings.LEGAL_TRANSLATION:
        raise HTTPException(status_code=404, detail="Not Found")


router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition), Depends(translation_on)])
Language = Literal["en", "fr", "de", "es", "it", "pt", "nl", "pl", "zh", "ja", "ko", "ar"]


class TranslateIn(BaseModel):
    language: Language
    from_: int = Field(default=0, alias="from", ge=0, le=100000)


def _event(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode()


async def _stream(scope, language: str, paras: list[str], start: int):
    name = tr.LANGUAGES[language]
    yield _event({"type": "start", "language": {"code": language, "name": name}, "paragraphs": len(paras), "from": start,
                  "label": tr.label(settings.LLM_MODEL)})
    budget, end = settings.LEGAL_TRANSLATE_MAX_CHARS, start
    while end < len(paras) and (end == start or budget - len(paras[end]) >= 0):
        budget -= len(paras[end])
        end += 1
    done = 0
    try:
        async for k, out in tr.translate(name, paras[start:end]):
            if k % tr.BATCH_PARAGRAPHS == 0:
                await run_in_threadpool(scopemod.recheck, scope)       # raises ScopeRevoked
            yield _event({"type": "paragraph", "i": start + k, "source": paras[start + k], **out})
            done += 1
        if end < len(paras):
            yield _event({"type": "limit", "next": end, "remaining": len(paras) - end})
        yield _event({"type": "done", "translated": done})
    except scopemod.ScopeRevoked:
        yield _event({"type": "scope_revoked"})
    except Exception as e:
        logger.error("Translation failed (%s)", type(e).__name__)
        yield _event({"type": "error", "message": "The translation could not be completed. Try again."})


@router.post("/matters/{matter_id}/documents/{document_id}/translate")
@audited("translate.document", object_type="document", object_param="document_id")
async def translate_document(matter_id: str, document_id: str, inp: TranslateIn, request: Request):
    p = principal_of(request)
    await run_in_threadpool(require, p, "documents.read", matter_id)
    d = await run_in_threadpool(documents.get_document, matter_id, document_id, p["sub"])
    reader = await run_in_threadpool(documents.reader, d["current_version"]["id"]) if d and d.get("current_version") else None
    if not reader:
        raise HTTPException(status_code=404, detail="Not Found")
    scope = await run_in_threadpool(scopemod.authorize, p, matter_id, "documents.read")
    paras = tr.paragraphs(reader["text"])
    start = min(inp.from_, len(paras))
    note(matter_id=matter_id, details={"paragraphs": len(paras) - start, "from": start, "chars": sum(len(x) for x in paras[start:])})
    return StreamingResponse(_stream(scope, inp.language, paras, start), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


class SummaryIn(BaseModel):
    language: Language


@router.post("/matters/{matter_id}/summaries/{artifact_id}/translate")
@audited("translate.summary", object_type="ai_artifact", object_param="artifact_id")
async def translate_summary(matter_id: str, artifact_id: str, inp: SummaryIn, request: Request):
    p = principal_of(request)
    await run_in_threadpool(require, p, "documents.read", matter_id)
    a = await run_in_threadpool(artifacts.get, matter_id, artifact_id)
    if not a or a["kind"] != "summary" or not await run_in_threadpool(artifacts.visible_to, a, p["sub"]):
        raise HTTPException(status_code=404, detail="Not Found")
    points = a["content"].get("points") or []
    name = tr.LANGUAGES[inp.language]
    out = [{"id": pt["id"], "translation": res["text"], "figures_missing": res["figures_missing"]}
           async for (k, res), pt in _zip_async(tr.translate(name, [pt["text"] for pt in points]), points)]
    note(matter_id=matter_id, details={"points": len(out), "chars": sum(len(pt["text"]) for pt in points)})
    return {"language": {"code": inp.language, "name": name}, "label": tr.label(settings.LLM_MODEL), "points": out}


async def _zip_async(agen, items):
    i = 0
    async for x in agen:
        yield x, items[i]
        i += 1
