"""
Embedding model interface for the RAG platform.
Uses Ollama embed API by default; can be extended for local GPU (e.g. Qwen3 0.6B) with torch.inference_mode.
"""
from __future__ import annotations
from typing import List
import numpy as np

# Use existing Ollama-based embedder; single interface for pipelines and Qdrant upsert.
from ..rag.embeddings import OllamaEmbeddings as _OllamaEmbeddings
from ..core.config import settings


def get_embedder():
    """Return the configured embedder instance (singleton-style via module)."""
    if get_embedder._instance is None:
        get_embedder._instance = _OllamaEmbeddings()
    return get_embedder._instance


get_embedder._instance = None


async def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts; returns (n, dim) float32 array. Uses Ollama by default."""
    return await get_embedder().embed(texts)
