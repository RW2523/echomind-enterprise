"""
Centralized logging configuration for the RAG platform.
Configures format, level, and handlers so Docker logs show app and RAG debug output.
"""
from __future__ import annotations
import logging
import sys


def setup_logging(level: int = logging.INFO, uvicorn_access_level: int = logging.WARNING) -> None:
    """Configure root logging: level, format, and suppress noisy uvicorn access logs."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("uvicorn.access").setLevel(uvicorn_access_level)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name."""
    return logging.getLogger(name)
