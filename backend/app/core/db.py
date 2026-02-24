import os, sqlite3
from contextlib import contextmanager
from .config import settings

def init_db():
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    with sqlite3.connect(settings.DB_PATH) as conn:
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
        # RAG platform: documents catalog for metadata and clarification
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents_catalog (
                doc_id TEXT PRIMARY KEY,
                title TEXT,
                doc_type TEXT,
                file_type TEXT,
                uploaded_at TEXT,
                tags_json TEXT,
                num_pages INTEGER,
                num_chunks INTEGER,
                source_path TEXT,
                summary_short TEXT,
                summary_chapters_json TEXT
            )
        """)
        conn.commit()

@contextmanager
def get_conn():
    conn = sqlite3.connect(settings.DB_PATH)
    try:
        yield conn
    finally:
        conn.close()
