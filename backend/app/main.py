import asyncio
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager

if os.getenv("LAW_EGRESS_LOG"):                              # T11 test runs only: record every outbound attempt
    from .legal import egress as _egress                     # stdlib only; before anything else is imported
    _egress.install(os.environ["LAW_EGRESS_LOG"])
from starlette.concurrency import run_in_threadpool
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
from .core.config import settings
from .core.edition import api_docs_enabled, install_log_redaction, is_law, validate_law_posture

# EchoMind Law (D-003): refuse to start with an insecure configuration — before anything below
# imports routers, opens the database or seeds an admin.
validate_law_posture()

from .core.db import init_db
from .api.routes.docs import router as docs_router
from .api.routes.chat import router as chat_router
from .api.routes.transcribe import router as transcribe_router
from .api.routes.boardroom import router as boardroom_router
from .api.routes.docgen import router as docgen_router
from .api.routes.auth import router as auth_router
from .api.routes.export import router as export_router
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
if is_law():
    install_log_redaction()  # keep matter content (questions, titles, quotes) out of the logs


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
    # Warm the semantic small-talk classifier too: cold, its first call outlives the 4 s budget and
    # small talk falls through to retrieval (NC-2026-014).
    try:
        from .rag.intent import _ensure_prototypes
        if await _ensure_prototypes():
            logger.info("Intent classifier: warmup ok.")
    except Exception as e:
        logger.warning("Intent classifier warmup failed: %s", e)


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
    # Pre-download and pre-load Nemotron ASR in background so first Live Transcript connection is faster.
    # ECHOMIND_ASR_PRELOAD=0 leaves it to the first connection (the Law pilot on a shared host, D-060).
    if os.getenv("ECHOMIND_ASR_PRELOAD", "1").strip().lower() not in ("0", "false", "no"):
        t = threading.Thread(target=_warm_nemotron_stt, daemon=True)
        t.start()
    # Warm LLM + embed backends so first chat/RAG is responsive
    asyncio.create_task(_warm_llm_and_embeddings())
    watchdog = asyncio.create_task(_stt_fatal_watchdog())
    if is_law():
        # S1.6: background indexing of matter documents (jobs survive restarts in the database).
        from .legal.jobs import worker as law_jobs
        law_jobs.start()
    try:
        yield
    finally:
        watchdog.cancel()
        if is_law():
            law_jobs.stop()


# Diagnostics (development only): `kill -USR1 <pid>` prints every thread's stack to stderr.
if os.getenv("LAW_DEBUG_STACKS") == "1":
    import faulthandler
    import signal as _signal
    faulthandler.register(_signal.SIGUSR1, all_threads=True)

init_db()
seed_admin()  # ensure an admin login exists when auth is (or will be) enabled; no-op otherwise
_docs_off = {} if api_docs_enabled() else {"docs_url": None, "redoc_url": None, "openapi_url": None}
app = FastAPI(title=settings.APP_NAME, lifespan=lifespan, **_docs_off)


# ── Release identification (ISO 9001 8.5.2 traceability) ──────────────────────
# BUILD_* are baked into the image by backend/Dockerfile (docker compose build
# args, set by scripts/release.sh). They can also be supplied as plain container
# environment variables, so an image built before this mechanism existed can
# still be told which revision it came from. Reported by /health and
# /api/version so an auditor can trace a running instance to its source.
def build_info() -> dict:
    return {
        "version": os.getenv("BUILD_VERSION", "0.0.0-dev"),
        "commit": os.getenv("BUILD_COMMIT", "unknown"),
        "date": os.getenv("BUILD_DATE", "unknown"),
    }

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
_AUTH_PUBLIC_PATHS = {"/health", "/api/version", "/api/auth/login", "/api/auth/config", "/api/auth/logout", "/api/client-error"}


# Law edition (S1.1, SEC-01): deny by default. Only these are reachable without a session; the
# client-error sink is not (it would let anyone write into the logs).
_LAW_PUBLIC_PATHS = {"/health", "/api/version", "/api/auth/login", "/api/auth/config", "/api/auth/logout"}
# S1.7 (decision D-025): only the Law API is served in the Law edition. The existing modules
# (legacy chat, documents, transcripts, Boardroom, document studio, evaluation) are closed for
# everyone — none of them is matter-scoped — and their capabilities return as /api/legal routes.
_LAW_OPEN_PREFIXES = ("/api/auth/", "/api/legal/")
_LAW_LINK_PREFIX = "/api/intake-link/"
_MUTATING = {"POST", "PUT", "PATCH", "DELETE"}


def _law_guard(request: Request):
    """None to let the request through, or the error response to return."""
    from .legal.identity import CSRF_HEADER
    from .legal.identity.principal import PARTIAL_PATHS, PASSWORD_CHANGE_PATHS
    from .legal.identity.sessions import csrf_ok
    path = request.url.path
    if request.method == "OPTIONS" or path in _LAW_PUBLIC_PATHS or not path.startswith("/api/"):
        return None
    if path.startswith(_LAW_LINK_PREFIX):
        # S4.1 (D-054): an enquirer's intake link is its own key, with no session; served only when the firm turned
        # intake links on, and answered 404 otherwise. The routes read nothing but the token.
        from .core.config import settings as _s
        if _s.LEGAL_INTAKE and _s.LEGAL_INTAKE_LINKS:
            return None
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    if not path.startswith(_LAW_OPEN_PREFIXES):
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    principal = user_from_request(request, allow_partial=path in PARTIAL_PATHS)
    if not principal:
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    if principal.get("must_change_password") and path not in PASSWORD_CHANGE_PATHS | PARTIAL_PATHS:
        return JSONResponse({"detail": "password_change_required"}, status_code=403)
    # Cookie sessions must echo the per-session CSRF token on state-changing requests. Bearer
    # (on-behalf-of) callers are services, never driven by a browser, so CSRF does not apply.
    if request.method in _MUTATING and principal["auth"] == "session" \
            and not csrf_ok(principal["session"], request.headers.get(CSRF_HEADER)):
        from .legal.audit.record import guard_refused
        guard_refused(request, 403, "csrf_failed")
        return JSONResponse({"detail": "CSRF check failed"}, status_code=403)
    return None


@app.middleware("http")
async def _auth_guard(request: Request, call_next):
    if is_law():
        # S1.5: every Law request runs inside an audit context; a state-changing request that wrote
        # no event of its own gets a generic one when it finishes (legal/audit/record.py).
        from .legal.audit import record as law_audit
        token = law_audit.begin(request)
        try:
            denied = _law_guard(request)
            if denied is not None:
                return denied
            response = await call_next(request)
            await run_in_threadpool(law_audit.request_finished, request, response.status_code)
            return response
        finally:
            law_audit.end(token)
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
    # Identify the caller before the request runs: logout revokes the token, so resolving it
    # afterwards recorded every logout as anonymous.
    try:
        payload = user_from_request(request)
    except Exception:
        payload = None
    response = await call_next(request)
    try:
        path = request.url.path
        if path.startswith("/api/") and request.method in ("POST", "PUT", "DELETE", "PATCH") and path not in _ACTIVITY_EXCLUDE:
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
    return {"ok": True, "app": settings.APP_NAME, "build": build_info()}


@app.get("/api/version")
def version():
    """Release identification: which source revision this instance was built from.
    Public (unauthenticated) so monitoring and audit checks can read it."""
    return {"app": settings.APP_NAME, **build_info()}

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
app.include_router(export_router, prefix="/api")
# EchoMind Law (S1.2+): /api/legal/*. Every route 404s outside the Law edition.
from .legal.matters.routes import router as legal_matters_router  # noqa: E402
from .legal.identity.routes import router as legal_users_router  # noqa: E402
app.include_router(legal_matters_router, prefix="/api")
app.include_router(legal_users_router, prefix="/api")
from .legal.walls.routes import router as legal_walls_router  # noqa: E402
app.include_router(legal_walls_router, prefix="/api")
from .legal.audit.routes import router as legal_audit_router  # noqa: E402
app.include_router(legal_audit_router, prefix="/api")
from .legal.documents.routes import router as legal_documents_router  # noqa: E402
app.include_router(legal_documents_router, prefix="/api")
from .legal.research.routes import router as legal_research_router  # noqa: E402
app.include_router(legal_research_router, prefix="/api")
from .legal.compare.routes import router as legal_compare_router  # noqa: E402
app.include_router(legal_compare_router, prefix="/api")
from .legal.meetings.routes import router as legal_meetings_router, ws_router as legal_capture_router  # noqa: E402
app.include_router(legal_meetings_router, prefix="/api")
app.include_router(legal_capture_router, prefix="/api")
from .legal.records.routes import router as legal_records_router  # noqa: E402
app.include_router(legal_records_router, prefix="/api")
from .legal.exports.routes import router as legal_exports_router  # noqa: E402
app.include_router(legal_exports_router, prefix="/api")
from .legal.connectors.routes import router as legal_connectors_router  # noqa: E402
app.include_router(legal_connectors_router, prefix="/api")
from .legal.reviews.routes import router as legal_reviews_router  # noqa: E402
app.include_router(legal_reviews_router, prefix="/api")
from .legal.evaluations.routes import router as legal_evaluations_router  # noqa: E402
app.include_router(legal_evaluations_router, prefix="/api")
from .legal.drafting.routes import router as legal_drafting_router  # noqa: E402
app.include_router(legal_drafting_router, prefix="/api")
from .legal.playbooks.routes import router as legal_playbooks_router  # noqa: E402
app.include_router(legal_playbooks_router, prefix="/api")
from .legal.redlines.routes import router as legal_redlines_router  # noqa: E402
app.include_router(legal_redlines_router, prefix="/api")
from .legal.citations.routes import router as legal_citations_router  # noqa: E402
app.include_router(legal_citations_router, prefix="/api")
from .legal.definitions.routes import router as legal_definitions_router  # noqa: E402
app.include_router(legal_definitions_router, prefix="/api")
from .legal.obligations.routes import router as legal_obligations_router  # noqa: E402
app.include_router(legal_obligations_router, prefix="/api")
from .legal.diligence.routes import router as legal_diligence_router  # noqa: E402
app.include_router(legal_diligence_router, prefix="/api")
from .legal.intake.routes import parties_router as legal_parties_router, public_router as legal_intake_link_router, router as legal_intake_router  # noqa: E402
app.include_router(legal_parties_router, prefix="/api")
app.include_router(legal_intake_router, prefix="/api")
app.include_router(legal_intake_link_router, prefix="/api")
from .legal.summaries.routes import router as legal_summaries_router  # noqa: E402
app.include_router(legal_summaries_router, prefix="/api")
from .legal.timeentries.routes import router as legal_time_router  # noqa: E402
app.include_router(legal_time_router, prefix="/api")
from .legal.panel.routes import router as legal_panel_router  # noqa: E402
app.include_router(legal_panel_router, prefix="/api")
from .legal.conversation.routes import router as legal_conversation_router  # noqa: E402
app.include_router(legal_conversation_router, prefix="/api")
from .legal.regulatory.routes import router as legal_regulatory_router  # noqa: E402
app.include_router(legal_regulatory_router, prefix="/api")
