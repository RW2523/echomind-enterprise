import os, sqlite3
from contextlib import contextmanager
from .config import settings

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
        # Assistant (Silent / Personal): persisted visible insights only — not indexed into RAG.
        conn.execute(
            """CREATE TABLE IF NOT EXISTS assistant_insights(
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                transcript_id TEXT,
                mode TEXT NOT NULL,
                dedupe_key TEXT NOT NULL,
                transcript_text TEXT NOT NULL,
                classification TEXT NOT NULL,
                confidence REAL NOT NULL,
                priority TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                assistant_interpretation TEXT,
                suggested_action TEXT,
                suggested_response TEXT,
                start_char INTEGER,
                end_char INTEGER,
                paragraph_id TEXT,
                action_status TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_assistant_insights_session_dedupe "
            "ON assistant_insights(session_id, dedupe_key)"
        )
        conn.commit()

@contextmanager
def get_conn():
    conn = sqlite3.connect(settings.DB_PATH)
    try:
        yield conn
    finally:
        conn.close()
