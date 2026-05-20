import os, sqlite3
from contextlib import contextmanager
from typing import List, Tuple
from .config import settings


def _safe_add_columns(conn: sqlite3.Connection, table: str, columns: List[Tuple[str, str]]) -> None:
    """Add columns to an existing table without failing if they already exist."""
    for col_name, col_def in columns:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")
        except Exception:
            pass  # Column already exists — safe to ignore


def init_db():
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    with sqlite3.connect(settings.DB_PATH) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, filename TEXT, filetype TEXT, created_at TEXT, meta_json TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY, doc_id TEXT, chunk_index INTEGER, text TEXT, source_json TEXT)")
        try:
            conn.execute("ALTER TABLE chunks ADD COLUMN contextualized_text TEXT")
        except Exception:
            pass
        conn.execute("CREATE TABLE IF NOT EXISTS chats(id TEXT PRIMARY KEY, title TEXT, created_at TEXT, conversation_summary TEXT)")
        try:
            conn.execute("ALTER TABLE chats ADD COLUMN conversation_summary TEXT")
        except Exception:
            pass
        conn.execute("CREATE TABLE IF NOT EXISTS messages(id TEXT PRIMARY KEY, chat_id TEXT, role TEXT, content TEXT, created_at TEXT)")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS transcripts(id TEXT PRIMARY KEY, title TEXT, raw_text TEXT, polished_text TEXT, tags_json TEXT, echotag TEXT, echodate TEXT, created_at TEXT)"
        )
        try:
            conn.execute("ALTER TABLE transcripts ADD COLUMN title TEXT")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE transcripts ADD COLUMN echotag TEXT")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE transcripts ADD COLUMN echodate TEXT")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE transcripts ADD COLUMN name TEXT")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE transcripts ADD COLUMN location TEXT")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE transcripts ADD COLUMN updated_at TEXT")
        except Exception:
            pass
        # Hierarchical section index: stores section-level metadata for BOOK documents.
        conn.execute(
            """CREATE TABLE IF NOT EXISTS book_sections(
                section_id TEXT PRIMARY KEY,
                doc_id TEXT,
                section_title TEXT,
                section_path TEXT,
                full_section_text TEXT,
                created_at TEXT
            )"""
        )
        try:
            conn.execute("ALTER TABLE book_sections ADD COLUMN section_summary TEXT")
        except Exception:
            pass
        # Cross-reference graph: extracted "See paragraph / Refer to Volume" links between sections.
        conn.execute(
            """CREATE TABLE IF NOT EXISTS section_references(
                id TEXT PRIMARY KEY,
                source_section_path TEXT,
                referenced_section_path TEXT,
                ref_section_id TEXT,
                doc_id TEXT,
                reference_text TEXT
            )"""
        )
        try:
            conn.execute("ALTER TABLE section_references ADD COLUMN ref_section_id TEXT")
        except Exception:
            pass
        # Board Room sessions: multi-speaker meeting capture.
        conn.execute(
            """CREATE TABLE IF NOT EXISTS boardroom_sessions(
                id TEXT PRIMARY KEY,
                title TEXT,
                location TEXT,
                status TEXT DEFAULT 'active',
                started_at TEXT,
                ended_at TEXT,
                duration_sec REAL,
                raw_transcript TEXT,
                speaker_map_json TEXT,
                segments_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )"""
        )
        # Safe migrations — add new columns without breaking existing rows.
        _safe_add_columns(conn, "boardroom_sessions", [
            ("audio_file_path", "TEXT"),
            ("speaker_count", "INTEGER DEFAULT 0"),
            ("cleaned_transcript", "TEXT"),
            ("primary_model_name", "TEXT"),
            ("diarization_model_name", "TEXT"),
            ("cleanup_model_name", "TEXT"),
            ("transcription_source", "TEXT DEFAULT 'boardroom_multitalker'"),
            ("rag_ingested", "INTEGER DEFAULT 0"),
            ("mode", "TEXT DEFAULT 'boardroom'"),
            ("sample_rate", "INTEGER DEFAULT 16000"),
            ("audio_format", "TEXT DEFAULT 'pcm16'"),
            ("error_message", "TEXT"),
            ("report_id", "TEXT"),
        ])
        # Board Room reports: LLM+RAG generated analysis of a session.
        conn.execute(
            """CREATE TABLE IF NOT EXISTS boardroom_reports(
                id TEXT PRIMARY KEY,
                session_id TEXT,
                status TEXT DEFAULT 'pending',
                report_json TEXT,
                rag_evidence_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )"""
        )
        _safe_add_columns(conn, "boardroom_reports", [
            ("report_markdown", "TEXT"),
            ("pdf_path", "TEXT"),
            ("pptx_path", "TEXT"),
            ("transcript_id", "TEXT"),
        ])
        conn.commit()

@contextmanager
def get_conn():
    conn = sqlite3.connect(settings.DB_PATH)
    try:
        yield conn
    finally:
        conn.close()
