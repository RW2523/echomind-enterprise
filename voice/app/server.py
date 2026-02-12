import asyncio
import base64
import json
import uuid
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .session import OmniSessionA
from .voice_download import list_installed_voices, download_voice

app = FastAPI(title="(Context + Memory)")
app.mount("/static", StaticFiles(directory="static"), name="static")


class DownloadVoiceBody(BaseModel):
    voice_id: str


@app.get("/voices/installed")
def get_installed_voices():
    """List Piper voice ids that are already downloaded (have .onnx + .onnx.json)."""
    return {"voice_ids": list_installed_voices()}


@app.post("/voices/download")
async def post_download_voice(body: DownloadVoiceBody):
    """Download a Piper voice by id (e.g. en_US-lessac-medium) to the voices dir. Blocks until done."""
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
