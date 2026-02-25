"""
FastAPI app: GPU singleton load at startup (embedder + generator), Qdrant + catalog init.
"""
from __future__ import annotations
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core import logging as logging_config
import app.models.embedder as embedder_mod
import app.models.generator as generator_mod
from app.qdrant.collections import ensure_all_collections
from app.catalog.db import init_db
from app.api import docs_router, transcript_router, query_router

logging_config.setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models once on GPU (DGX Spark ARM64 compatible)
    logger.info("Loading embedder: %s on %s", settings.EMBED_MODEL_ID, settings.DEVICE)
    embedder_mod.embedder = embedder_mod.HFEmbedder(
        model_id=settings.EMBED_MODEL_ID,
        device=settings.DEVICE,
        use_bf16=settings.USE_BF16,
        low_cpu_mem_usage=settings.LOW_CPU_MEM,
        use_safetensors=settings.USE_SAFETENSORS,
    )
    logger.info("Loading generator: %s on %s", settings.GENERATOR_MODEL_ID, settings.DEVICE)
    generator_mod.generator = generator_mod.HFGenerator(
        model_id=settings.GENERATOR_MODEL_ID,
        device=settings.DEVICE,
        use_bf16=settings.USE_BF16,
        low_cpu_mem_usage=settings.LOW_CPU_MEM,
        use_safetensors=settings.USE_SAFETENSORS,
    )
    init_db()
    ensure_all_collections()
    logger.info("Startup complete: embedder + generator on GPU, Qdrant collections ready")
    yield
    # Shutdown: no explicit unload needed
    logger.info("Shutdown")


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(docs_router)
app.include_router(transcript_router)
app.include_router(query_router)


@app.get("/health")
def health():
    return {"status": "ok"}
