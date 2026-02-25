# Expose submodules so "app.models.embedder" / "app.models.generator" are the modules, not the singletons
from . import embedder
from . import generator
from .embedder import HFEmbedder, get_embedder
from .generator import HFGenerator, get_generator

__all__ = [
    "embedder",
    "generator",
    "HFEmbedder",
    "get_embedder",
    "HFGenerator",
    "get_generator",
]
