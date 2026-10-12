"""The conversation assistant (S5.1; decision D-062). Off unless LEGAL_CONVERSATION is on (404 otherwise).

A lawyer speaks a question about a matter; this server transcribes it with its own speech recognition, the page
asks it through the research pipeline (`POST /api/legal/matters/{m}/research/query`, answered from the matter's
documents with checked quotes, as a typed question is), and the answer is read aloud by this server's own voice:
- `POST /api/legal/matters/{m}/conversation/transcribe?sample_rate=16000`: the clip as 16-bit mono PCM, at most
  LEGAL_CONVERSATION_MAX_SECONDS; returns `{"text", "seconds"}`. The audio is not kept.
- `POST /api/legal/matters/{m}/conversation/speak` `{"text"}`: the answer as WAV, its citation marks left out.

Both need what research on the matter needs, so a matter is as invisible here as anywhere (404). Speech goes
through this signed-in route rather than the voice service's own, which answer without a session and are refused
by the Law frontend. The audit holds lengths only.
"""
from __future__ import annotations

import logging
import re

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ...core.config import settings
from ..audit.record import audited, note
from ..matters.routes import principal_of, require, require_law_edition

logger = logging.getLogger(__name__)


class Unavailable(Exception):
    """Speech recognition or the voice is not available on this server."""


def conversation_on() -> None:
    if not settings.LEGAL_CONVERSATION:
        raise HTTPException(status_code=404, detail="Not Found")


router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition), Depends(conversation_on)])


async def transcribe_clip(pcm: bytes, rate: int) -> str:
    """The server's own speech recognition over one clip (the model meetings use, loaded on first use)."""
    try:
        from ...transcribe import stt_streaming as stt
    except Exception as e:
        raise Unavailable(type(e).__name__) from None
    if not getattr(stt, "NEMOTRON_AVAILABLE", False):
        raise Unavailable("not installed")
    await run_in_threadpool(stt.get_shared_asr_adapter)                 # a first load must not hold up other requests
    return (await stt.process_audio_stream([pcm], rate, lambda piece, ts: None)).strip()


async def synthesise(text: str) -> bytes:
    """The server's own voice (the voice service on the internal network)."""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(settings.VOICE_TTS_URL, json={"text": text})
    except httpx.HTTPError as e:
        raise Unavailable(type(e).__name__) from None
    if r.status_code != 200 or not r.content:
        raise Unavailable(f"status {r.status_code}")
    return r.content


_CITATION = re.compile(r'\s*\[\d+(?:\s*,\s*\d+)*(?::\s*"[^"]*")?\]')
_MARKUP = re.compile(r"\*\*|__|`|^#{1,6}\s+|^\s*[-*]\s+", re.M)


def spoken(text: str) -> str:
    """An answer as it is read aloud: without citation marks or markup."""
    return " ".join(_MARKUP.sub("", _CITATION.sub("", text)).split())


@router.post("/matters/{matter_id}/conversation/transcribe")
@audited("conversation.transcribe", object_type="matter", object_param="matter_id")
async def transcribe(matter_id: str, request: Request, sample_rate: int = 16000):
    p = principal_of(request)
    await run_in_threadpool(require, p, "research.query", matter_id)
    if not 8000 <= sample_rate <= 48000:
        raise HTTPException(status_code=400, detail="The sample rate must be between 8000 and 48000.")
    limit = int(settings.LEGAL_CONVERSATION_MAX_SECONDS * sample_rate * 2)
    if int(request.headers.get("content-length") or 0) > limit:
        raise HTTPException(status_code=413, detail=f"A question can be at most {settings.LEGAL_CONVERSATION_MAX_SECONDS} seconds.")
    pcm = await request.body()
    if not pcm or len(pcm) % 2:
        raise HTTPException(status_code=400, detail="Send the question as 16-bit mono audio.")
    if len(pcm) > limit:
        raise HTTPException(status_code=413, detail=f"A question can be at most {settings.LEGAL_CONVERSATION_MAX_SECONDS} seconds.")
    seconds = round(len(pcm) / 2 / sample_rate, 1)
    try:
        text = await transcribe_clip(pcm, sample_rate)
    except Unavailable:
        raise HTTPException(status_code=503, detail="Speech recognition is not installed on this server. Type your question instead.")
    note(details={"seconds": seconds, "words": len(text.split())})
    return {"text": text, "seconds": seconds}


class SpeakIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@router.post("/matters/{matter_id}/conversation/speak")
@audited("conversation.speak", object_type="matter", object_param="matter_id")
async def speak(matter_id: str, inp: SpeakIn, request: Request):
    p = principal_of(request)
    await run_in_threadpool(require, p, "research.query", matter_id)
    text = spoken(inp.text)
    if not text:
        raise HTTPException(status_code=400, detail="Nothing to read aloud.")
    try:
        audio = await synthesise(text)
    except Unavailable:
        raise HTTPException(status_code=503, detail="Reading aloud is not available on this server.")
    note(details={"chars": len(text)})
    return Response(content=audio, media_type="audio/wav", headers={"Cache-Control": "no-store"})
