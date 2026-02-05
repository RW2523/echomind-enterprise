import base64
import json
import uuid
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .session import OmniSessionA

app = FastAPI(title="Unmute Path A v5.1 (Context + Memory)")
app.mount("/static", StaticFiles(directory="static"), name="static")

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
                data = json.loads(msg["text"])

                if data.get("type") == "audio_frame":
                    pcm = base64.b64decode(data.get("pcm16_b64", ""))
                    await sess.on_audio_frame(float(data.get("ts", 0.0)), pcm)
                else:
                    await sess.on_control(data)
    except Exception:
        pass
    finally:
        await sess.close()
