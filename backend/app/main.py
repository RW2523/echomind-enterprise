import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .core.db import init_db
from .api.routes.docs import router as docs_router
from .api.routes.chat import router as chat_router
from .api.routes.transcribe import router as transcribe_router
from .api.routes.debug import router as debug_router

# So Docker logs (stdout) show app logs including RAG intent debug
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def _rag_warmup() -> None:
    """Load FAISS index, run embedding sanity check, one dummy search, warm Ollama LLM. Ensures first user query is fast and retrieval is valid."""
    if getattr(settings, "RAG_SANITY_MODE", False):
        logger.info("RAG sanity mode: query rewrite, compression, rerank, decay, gating disabled (core retrieval only)")
    try:
        from .rag.index import index
        from .rag.embeddings import run_embedding_sanity_check
        await run_embedding_sanity_check(index.emb)
        # Dummy search warms FAISS/GPU and embed path.
        _ = await index.search("warmup", 1)
    except Exception as e:
        logger.debug("RAG warmup search: %s", e)
    try:
        from .rag.llm import OpenAICompatChat
        chat = OpenAICompatChat(settings.LLM_BASE_URL, settings.LLM_MODEL)
        await chat.chat(
            [{"role": "user", "content": "Say OK."}],
            temperature=0,
            max_tokens=4,
        )
    except Exception as e:
        logger.debug("RAG warmup LLM: %s", e)
    logger.info("RAG warmup complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _rag_warmup()
    yield


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
app.include_router(debug_router, prefix="/api")