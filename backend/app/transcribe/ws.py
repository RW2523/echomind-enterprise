from __future__ import annotations
import base64, time, asyncio
import numpy as np
from fastapi import WebSocket
from ..core.config import settings
import whisper

_model=None

def pcm16_b64_to_float32(b64: str) -> np.ndarray:
    pcm = base64.b64decode(b64)
    return (np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0)

def _samples(sec:int)->int:
    return int(16000*sec)

async def handler(ws: WebSocket):
    await ws.accept()
    global _model
    if _model is None:
        _model = whisper.load_model(settings.WHISPER_MODEL)
    buf=[]
    last=time.time()
    await ws.send_json({"type":"ready","model":settings.WHISPER_MODEL})
    try:
        while True:
            msg = await ws.receive_json()
            t = msg.get("type")
            if t=="audio":
                x = pcm16_b64_to_float32(msg["pcm16_b64"])
                buf.append(x)
                if time.time()-last>0.9:
                    last=time.time()
                    audio=np.concatenate(buf)[-_samples(8):] if buf else np.zeros(0,dtype=np.float32)
                    txt = await asyncio.get_running_loop().run_in_executor(None, lambda: (_model.transcribe(audio, fp16=False).get("text","") or "").strip())
                    if txt:
                        await ws.send_json({"type":"partial","text":txt})
            elif t=="stop":
                break
    except Exception as e:
        await ws.send_json({"type":"error","message":str(e)})
        return
    audio=np.concatenate(buf) if buf else np.zeros(0,dtype=np.float32)
    if audio.size < _samples(1):
        await ws.send_json({"type":"final","text":""})
        return
    try:
        txt = await asyncio.get_running_loop().run_in_executor(None, lambda: (_model.transcribe(audio, fp16=False).get("text","") or "").strip())
        await ws.send_json({"type":"final","text":txt})
    except Exception as e:
        await ws.send_json({"type":"error","message":str(e)})
