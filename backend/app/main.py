import logging
import sys
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .core.db import init_db
from .api.routes.docs import router as docs_router
from .api.routes.chat import router as chat_router
from .api.routes.transcribe import router as transcribe_router

# So Docker logs (stdout) show app logs including RAG intent debug
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _warm_kyutai_stt():
    """Run in background: download Kyutai model then pre-load one STT instance (cache for first Live Transcript)."""
    try:
        from .transcribe.stt_streaming import KYUTAI_AVAILABLE, download_kyutai_model, preload_kyutai_stt
        if not KYUTAI_AVAILABLE:
            return
        logger.info("Kyutai STT: pre-downloading model (first run may take 2–5 min)...")
        if download_kyutai_model():
            logger.info("Kyutai STT: model cached.")
        logger.info("Kyutai STT: pre-loading model...")
        if preload_kyutai_stt():
            logger.info("Kyutai STT: ready (Live Transcript will connect instantly).")
        else:
            logger.warning("Kyutai STT: pre-load failed (Live Transcript will load on first use).")
    except Exception as e:
        logger.warning("Kyutai STT: warmup failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-download and pre-load Kyutai STT in background so first Live Transcript connection is fast
    t = threading.Thread(target=_warm_kyutai_stt, daemon=True)
    t.start()
    yield
    # shutdown: nothing to clean up (daemon thread exits with process)


init_db()
app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "app": settings.APP_NAME}

app.include_router(docs_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(transcribe_router, prefix="/api")
