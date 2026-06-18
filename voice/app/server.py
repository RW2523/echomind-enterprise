import asyncio
import base64
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, HTTPException

# Root default WARNING; bump LLM adapter to INFO so VOICE_LLM stream/sync timing lines appear in docker logs.
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("app.adapters.llm_openai_stream").setLevel(logging.INFO)
logging.getLogger("app.adapters.stt_nemotron").setLevel(logging.INFO)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .session import OmniSessionA
from .voice_download import list_installed_voices, download_voice

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Nemotron: lazy load on first STT by default (VOICE_NEMOTRON_STARTUP_LOAD=0) so the server accepts WS immediately.
    # Set VOICE_NEMOTRON_STARTUP_LOAD=1 to fail fast if the model cannot load.
    from .adapters.stt_nemotron import ensure_nemotron_loaded_at_startup

    try:
        await asyncio.to_thread(ensure_nemotron_loaded_at_startup)
    except Exception:
        logger.exception("Voice service: Nemotron startup load failed")
        raise
    yield


app = FastAPI(title="(Context + Memory)", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class DownloadVoiceBody(BaseModel):
    voice_id: str


@app.get("/voices/installed")
def get_installed_voices():
    """List Piper voice ids that are already downloaded (have .onnx + .onnx.json)."""
    return {"voice_ids": list_installed_voices()}


@app.post("/voices/download")
async def post_download_voice(body: DownloadVoiceBody):
    """Download a Piper voice by id (e.g. en_US-lessac-medium) to the voices dir. Disabled in offline mode."""
    if os.environ.get("VOICE_OFFLINE", "").strip().lower() in ("1", "true", "yes"):
        raise HTTPException(
            status_code=503,
            detail="Voice download is disabled in offline mode. Use only pre-installed voices in /voices.",
        )
    voice_id = (body.voice_id or "").strip()
    if not voice_id:
        raise HTTPException(status_code=400, detail="voice_id required")
    try:
        await asyncio.to_thread(download_voice, voice_id)
        return {"ok": True, "voice_id": voice_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

class SpeakBody(BaseModel):
    text: str
    voice_id: str | None = None


@app.post("/speak")
async def speak_tts(body: SpeakBody):
    """
    Simple TTS endpoint: synthesize text with Piper and return raw WAV bytes.
    Used internally by the backend /api/transcribe/speak endpoint.
    """
    from fastapi.responses import Response as FastAPIResponse
    import io, wave
    import soundfile as sf
    import numpy as np

    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    from .config import SETTINGS
    from .adapters.tts_piper import PiperTTS  # (H7) was referenced but never imported -> always 500
    try:
        tts = PiperTTS(
            model_path=SETTINGS.PIPER_MODEL,
            speaker_id=SETTINGS.PIPER_SPEAKER,
            noise_scale=SETTINGS.PIPER_NOISE_SCALE,
            length_scale=SETTINGS.PIPER_LENGTH_SCALE,
            use_cuda=SETTINGS.PIPER_USE_CUDA,
        )
        audio = tts.synth(text[:2000])
        buf = io.BytesIO()
        sf.write(buf, audio, tts.sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return FastAPIResponse(content=buf.read(), media_type="audio/wav")
    except Exception as e:
        logger.error("Piper TTS speak error: %s", e)
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


@app.get("/")
def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    sess = OmniSessionA(ws)
    await sess.start(str(uuid.uuid4()))
    try:
        while True:
            msg = await ws.receive()
            if "text" in msg and msg["text"]:
                try:
                    data = json.loads(msg["text"])
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(data, dict):
                    continue
                if data.get("type") == "audio_frame":
                    try:
                        pcm = base64.b64decode(data.get("pcm16_b64") or b"")
                    except Exception:
                        continue
                    await sess.on_audio_frame(float(data.get("ts", 0.0)), pcm)
                else:
                    await sess.on_control(data)
    except Exception:
        pass
    finally:
        await sess.close()
