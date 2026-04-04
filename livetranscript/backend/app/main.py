"""FastAPI app: WebSocket transcript endpoint and health check."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.config import HOST, PORT, WS_PATH
from app.streaming_service import StreamingASRService
from app.websocket_handler import handle_transcribe_ws

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

service: StreamingASRService = StreamingASRService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.ensure_loaded()
    yield
    # shutdown if needed
    pass


app = FastAPI(title="Live Transcript ASR", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    import torch
    gpu = torch.cuda.is_available()
    device_name = ""
    if gpu:
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return {"status": "ok", "gpu": gpu, "device": device_name or ("cuda" if gpu else "cpu")}


@app.websocket(WS_PATH)
async def websocket_transcribe(websocket: WebSocket):
    await handle_transcribe_ws(websocket, service)


def run():
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    run()
