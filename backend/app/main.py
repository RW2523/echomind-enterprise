import asyncio
import logging
import sys
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
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
    """Run in background: ensure Kyutai model in cache (no download when HF_HUB_OFFLINE=1), then pre-load one STT instance."""
    try:
        from .transcribe.stt_streaming import KYUTAI_AVAILABLE, download_kyutai_model, preload_kyutai_stt
        if not KYUTAI_AVAILABLE:
            return
        # In offline mode download_kyutai_model only checks local cache (no network).
        if download_kyutai_model():
            logger.info("Kyutai STT: model in cache.")
        logger.info("Kyutai STT: pre-loading model...")
        if preload_kyutai_stt():
            logger.info("Kyutai STT: ready (Live Transcript will connect instantly).")
        else:
            logger.warning("Kyutai STT: pre-load failed (Live Transcript will load on first use or fail if offline and missing).")
    except Exception as e:
        logger.warning("Kyutai STT: warmup failed: %s", e)


async def _warm_llm_and_embeddings():
    """Warm OpenAI-compatible chat (TensorRT-LLM, Ollama /v1, etc.) and Ollama embeddings."""
    base = (settings.LLM_BASE_URL or "").rstrip("/")
    if base:
        chat_url = f"{base}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                logger.info("LLM: warming %s via %s ...", settings.LLM_MODEL, chat_url)
                r = await client.post(
                    chat_url,
                    json={
                        "model": settings.LLM_MODEL,
                        "messages": [{"role": "user", "content": "."}],
                        "max_tokens": 1,
                    },
                )
                if r.is_success:
                    logger.info("LLM: warmup ok.")
                else:
                    logger.warning("LLM: warmup returned %s", r.status_code)
        except Exception as e:
            logger.warning("LLM warmup failed (first chat may be slow): %s", e)
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            logger.info("Embeddings: warming %s ...", settings.OLLAMA_EMBED_MODEL)
            r2 = await client.post(
                settings.OLLAMA_EMBED_URL,
                json={"model": settings.OLLAMA_EMBED_MODEL, "prompt": "."},
            )
            if r2.is_success:
                logger.info("Embeddings: warmup ok.")
            else:
                logger.warning("Embeddings: warmup returned %s", r2.status_code)
    except Exception as e:
        logger.warning("Embedding warmup failed (first RAG may be slow): %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-download and pre-load Kyutai STT in background so first Live Transcript connection is fast
    t = threading.Thread(target=_warm_kyutai_stt, daemon=True)
    t.start()
    # Warm LLM + embed backends so first chat/RAG is responsive
    asyncio.create_task(_warm_llm_and_embeddings())
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
