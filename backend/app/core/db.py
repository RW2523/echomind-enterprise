import os, sqlite3
from contextlib import contextmanager
from .config import settings


def _configure_conn(conn: sqlite3.Connection) -> None:
    """Apply optimal SQLite performance pragmas to a connection."""
    # WAL mode allows concurrent reads while a write is in progress (huge win for RAG workloads).
    conn.execute("PRAGMA journal_mode=WAL")
    # Don't wait forever if another writer holds the lock; retry for 5 s.
    conn.execute("PRAGMA busy_timeout=5000")
    # Larger page cache (default 2 MB → 32 MB) reduces I/O for repeated chunk reads.
    conn.execute("PRAGMA cache_size=-32768")
    # Synchronous=NORMAL is safe with WAL and much faster than FULL.
    conn.execute("PRAGMA synchronous=NORMAL")
    # Keep temp tables in memory.
    conn.execute("PRAGMA temp_store=MEMORY")


def init_db():
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    with sqlite3.connect(settings.DB_PATH) as conn:
        _configure_conn(conn)
        conn.execute("CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, filename TEXT, filetype TEXT, created_at TEXT, meta_json TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY, doc_id TEXT, chunk_index INTEGER, text TEXT, source_json TEXT)")
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
        # Indexes for the most frequent RAG queries (CREATE INDEX IF NOT EXISTS is idempotent).
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_id ON chunks(id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_created_at ON transcripts(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_updated_at ON transcripts(updated_at)")
        conn.commit()


@contextmanager
def get_conn():
    conn = sqlite3.connect(settings.DB_PATH)
    _configure_conn(conn)
    try:
        yield conn
    finally:
        conn.close()
