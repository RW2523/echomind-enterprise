import asyncio
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
from .core.config import settings
from .core.db import init_db
from .api.routes.docs import router as docs_router
from .api.routes.chat import router as chat_router
from .api.routes.transcribe import router as transcribe_router
from .api.routes.boardroom import router as boardroom_router
from .api.routes.docgen import router as docgen_router

# So Docker logs (stdout) show app logs including RAG intent debug
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _warm_nemotron_stt():
    """Background: ensure Nemotron weights in HF cache when online; pre-load shared NeMo ASR model."""
    try:
        from .transcribe.stt_streaming import NEMOTRON_AVAILABLE, download_nemotron_model, preload_nemotron_stt

        if not NEMOTRON_AVAILABLE:
            return
        if download_nemotron_model():
            logger.info("Nemotron ASR: Hugging Face cache check ok.")
        logger.info("Nemotron ASR: loading model...")
        if preload_nemotron_stt():
            logger.info("Nemotron ASR: ready (Live Transcript can connect).")
        else:
            logger.warning(
                "Nemotron ASR: pre-load failed (Live Transcript will retry on first connection or fail if offline and missing)."
            )
    except Exception as e:
        logger.warning("Nemotron ASR: warmup failed: %s", e)


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


async def _stt_fatal_watchdog():
    """Exit the process if a fatal CUDA fault poisons the shared GPU context (Nemotron STT + FAISS-GPU),
    so restart: unless-stopped recreates the container. A poisoned context can't recover in-process."""
    try:
        from .transcribe.stt_streaming import stt_fatal_event
    except Exception:
        return
    await asyncio.to_thread(stt_fatal_event().wait)
    grace = float(os.getenv("ECHOMIND_FATAL_EXIT_GRACE_S", "2"))
    logger.critical("Backend: fatal GPU/CUDA fault detected — exiting in %.0fs for restart", grace)
    await asyncio.sleep(grace)
    if os.getenv("ECHOMIND_EXIT_ON_FATAL_CUDA", "1").strip().lower() in ("1", "true", "yes"):
        os._exit(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-download and pre-load Nemotron ASR in background so first Live Transcript connection is faster
    t = threading.Thread(target=_warm_nemotron_stt, daemon=True)
    t.start()
    # Warm LLM + embed backends so first chat/RAG is responsive
    asyncio.create_task(_warm_llm_and_embeddings())
    watchdog = asyncio.create_task(_stt_fatal_watchdog())
    try:
        yield
    finally:
        watchdog.cancel()


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
    # Report 503 once a fatal GPU/CUDA fault poisons the shared context so the Docker healthcheck
    # flags the container and the watchdog can trigger a restart.
    try:
        from .transcribe.stt_streaming import stt_healthy
        unhealthy = not stt_healthy()
    except Exception:
        unhealthy = False
    if unhealthy:
        raise HTTPException(status_code=503, detail="GPU/CUDA fault — restart required")
    return {"ok": True, "app": settings.APP_NAME}

@app.post("/api/client-error")
async def client_error(request: Request):
    """Receive a front-end crash report (from the React ErrorBoundary) and log it.
    Lets us see browser-side render errors in server logs even on remote deployments."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    where = str(body.get("where", "app"))[:80]
    message = str(body.get("message", ""))[:500]
    url = str(body.get("url", ""))[:300]
    comp = str(body.get("componentStack", ""))[:1500]
    stack = str(body.get("stack", ""))[:1500]
    logger.error(
        "CLIENT UI CRASH [%s] url=%s\n  message: %s\n  componentStack: %s\n  stack: %s",
        where, url, message, comp, stack,
    )
    return {"ok": True}


app.include_router(docs_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(transcribe_router, prefix="/api")
app.include_router(boardroom_router, prefix="/api")
app.include_router(docgen_router, prefix="/api")
