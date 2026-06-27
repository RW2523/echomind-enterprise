import asyncio
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
from .core.config import settings
from .core.db import init_db
from .api.routes.docs import router as docs_router
from .api.routes.chat import router as chat_router
from .api.routes.transcribe import router as transcribe_router
from .api.routes.boardroom import router as boardroom_router
from .api.routes.docgen import router as docgen_router
from .api.routes.auth import router as auth_router
from .core.auth import user_from_request, seed_admin
from .core.audit import record_activity

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
seed_admin()  # ensure an admin login exists when auth is (or will be) enabled; no-op otherwise
app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

_cors_origins = [o.strip() for o in (settings.CORS_ALLOW_ORIGINS or "*").split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=(_cors_origins != ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth guard (Phase 0b) — enforced only when AUTH_ENABLED; pass-through otherwise. ──
# HTTP-only (WebSocket endpoints are not gated here yet). Login/config stay public.
_AUTH_PUBLIC_PATHS = {"/health", "/api/auth/login", "/api/auth/config", "/api/auth/logout", "/api/client-error"}


@app.middleware("http")
async def _auth_guard(request: Request, call_next):
    if settings.AUTH_ENABLED and request.method != "OPTIONS":
        path = request.url.path
        if path.startswith("/api/") and path not in _AUTH_PUBLIC_PATHS:
            if not user_from_request(request):
                return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    return await call_next(request)


# ── Activity log (audit view + usage metering) — best-effort; logs /api mutations + logins. ──
_ACTIVITY_EXCLUDE = {"/api/auth/config", "/api/client-error"}


@app.middleware("http")
async def _activity_log(request: Request, call_next):
    response = await call_next(request)
    try:
        path = request.url.path
        if path.startswith("/api/") and request.method in ("POST", "PUT", "DELETE", "PATCH") and path not in _ACTIVITY_EXCLUDE:
            payload = user_from_request(request)
            ip = request.client.host if request.client else ""
            record_activity((payload or {}).get("username"), (payload or {}).get("role"), request.method, path, response.status_code, ip)
    except Exception:
        pass
    return response


# ── Per-IP rate limit (token bucket) — opt-in via RATE_LIMIT_PER_MIN; 0 = off (default). ──
_RL_BUCKETS: dict = {}  # ip -> [tokens, last_monotonic]


@app.middleware("http")
async def _rate_limit(request: Request, call_next):
    rpm = settings.RATE_LIMIT_PER_MIN
    if rpm > 0 and request.url.path.startswith("/api/") and request.url.path != "/api/auth/config":
        ip = request.client.host if request.client else "?"
        now = time.monotonic()
        tokens, last = _RL_BUCKETS.get(ip, (float(rpm), now))
        tokens = min(float(rpm), tokens + (now - last) * (rpm / 60.0))
        if tokens < 1.0:
            _RL_BUCKETS[ip] = (tokens, now)
            return JSONResponse({"detail": "Rate limit exceeded — please slow down."}, status_code=429)
        _RL_BUCKETS[ip] = (tokens - 1.0, now)
    return await call_next(request)

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
app.include_router(auth_router, prefix="/api")
