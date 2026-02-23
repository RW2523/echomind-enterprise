import logging
import sys
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

init_db()
app = FastAPI(title=settings.APP_NAME)


@app.on_event("startup")
def _startup_warm_kyutai():
    """Preload Kyutai STT in the background so the first Live Transcript connection is instant."""
    import logging
    log = logging.getLogger(__name__)
    try:
        from .transcribe.stt_streaming import (
            KYUTAI_AVAILABLE,
            _kyutai_import_error,
            warm_kyutai_stt,
        )
        model_dir = getattr(settings, "KYUTAI_MODEL_DIR", None)
        if KYUTAI_AVAILABLE:
            log.info("Kyutai STT: available, ECHOMIND_KYUTAI_MODEL_DIR=%s", model_dir or "(not set)")
            warm_kyutai_stt()
        else:
            log.warning("Kyutai STT: not available (%s). Live Transcript will show an error.", _kyutai_import_error or "import failed")
    except Exception as e:
        log.warning("Kyutai STT startup: %s", e)


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
