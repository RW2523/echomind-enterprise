import os, sqlite3
from contextlib import contextmanager
from .config import settings

def init_db():
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    with sqlite3.connect(settings.DB_PATH) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, filename TEXT, filetype TEXT, created_at TEXT, meta_json TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY, doc_id TEXT, chunk_index INTEGER, text TEXT, source_json TEXT)")
        # Phase 0b auth: local user accounts (used only when AUTH_ENABLED).
        conn.execute("CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'user', created_at TEXT)")
        try:  # Phase 0b #4: per-tenant scoping
            conn.execute("ALTER TABLE users ADD COLUMN tenant TEXT DEFAULT ''")
        except Exception:
            pass
        # Phase 0b: activity log — powers the audit view + usage metering (admin only).
        conn.execute("CREATE TABLE IF NOT EXISTS activity_log(id TEXT PRIMARY KEY, ts TEXT, username TEXT, role TEXT, method TEXT, path TEXT, status INTEGER, ip TEXT)")
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_activity_ts ON activity_log(ts)")
        except Exception:
            pass
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
        # Real-time RAG analysis of live transcript segments.
        conn.execute(
            """CREATE TABLE IF NOT EXISTS transcript_analysis(
                id TEXT PRIMARY KEY,
                session_id TEXT,
                transcript_id TEXT,
                segment_id TEXT,
                segment_text TEXT,
                label TEXT,
                confidence REAL,
                explanation TEXT,
                source_refs TEXT,
                created_at TEXT
            )"""
        )
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ta_session ON transcript_analysis(session_id)")
        except Exception:
            pass
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ta_transcript ON transcript_analysis(transcript_id)")
        except Exception:
            pass
        # session_id on transcripts — lets us always correlate a transcript back to its live session.
        try:
            conn.execute("ALTER TABLE transcripts ADD COLUMN session_id TEXT")
        except Exception:
            pass
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_session ON transcripts(session_id)")
        except Exception:
            pass
        # Boardroom sessions: full-meeting audio capture, diarization, and AI report.
        # If the table exists with the old schema (no chunk_count column), rename it so we
        # can create the new schema alongside without losing legacy rows.
        _brm_cols = [
            row[1]
            for row in conn.execute("PRAGMA table_info(boardroom_sessions)").fetchall()
        ]
        if _brm_cols and "chunk_count" not in _brm_cols:
            try:
                conn.execute(
                    "ALTER TABLE boardroom_sessions RENAME TO boardroom_sessions_legacy"
                )
            except Exception:
                pass
        conn.execute(
            """CREATE TABLE IF NOT EXISTS boardroom_sessions(
                id TEXT PRIMARY KEY,
                transcript_id TEXT,
                status TEXT,
                chunk_count INTEGER DEFAULT 0,
                audio_format TEXT,
                diarized_json TEXT,
                report_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS docgen_jobs(
                id TEXT PRIMARY KEY,
                title TEXT,
                template_id TEXT,
                persona TEXT,
                mode TEXT,
                status TEXT,
                stage TEXT,
                doc_json TEXT,
                error TEXT,
                created_at TEXT,
                updated_at TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS docgen_templates(
                id TEXT PRIMARY KEY,
                name TEXT,
                blueprint_json TEXT,
                source_path TEXT,
                created_at TEXT
            )"""
        )
        conn.commit()

@contextmanager
def get_conn():
    conn = sqlite3.connect(settings.DB_PATH)
    # WAL mode: concurrent readers + writers without blocking each other.
    # NORMAL sync: safe on power loss (WAL checkpoint handles durability).
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
    finally:
        conn.close()
