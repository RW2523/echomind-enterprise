"""
Catalog DB: SQLite (default) or Postgres for documents_catalog.
"""
from __future__ import annotations
import os
import json
import sqlite3
from contextlib import contextmanager
from typing import Generator, Optional

from app.core.config import settings


def _is_postgres() -> bool:
    return bool(settings.DATABASE_URL and settings.DATABASE_URL.startswith("postgres"))


@contextmanager
def get_conn():
    if _is_postgres():
        try:
            import psycopg2
            conn = psycopg2.connect(settings.DATABASE_URL)
            try:
                yield conn
            finally:
                conn.close()
        except ImportError:
            raise RuntimeError("Postgres requested but psycopg2 not installed")
    else:
        os.makedirs(os.path.dirname(settings.DB_PATH) or ".", exist_ok=True)
        conn = sqlite3.connect(settings.DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()


def init_db() -> None:
    """Create documents_catalog table if not exists."""
    sql = """
    CREATE TABLE IF NOT EXISTS documents_catalog (
        doc_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        doc_type TEXT,
        file_type TEXT,
        uploaded_at INTEGER NOT NULL,
        tags TEXT,
        num_pages INTEGER,
        num_chunks INTEGER,
        source_path TEXT,
        summary_short TEXT,
        summary_chapters TEXT
    )
    """
    with get_conn() as conn:
        conn.execute(sql)
        if hasattr(conn, "commit"):
            conn.commit()


def _adapt_json(v):
    if v is None:
        return None
    if isinstance(v, (list, dict)):
        return json.dumps(v)
    return v


def _convert_json(val):
    if val is None:
        return None
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return val
    return val
