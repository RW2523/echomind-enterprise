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

        # ── BookRAG-lite++: page-level index ─────────────────────────────────
        # One row per PDF page per document; enables accurate page citations,
        # logical page mapping, and table/low-text detection per page.
        conn.execute(
            """CREATE TABLE IF NOT EXISTS page_index(
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                page_number_pdf INTEGER NOT NULL,
                page_number_logical INTEGER,
                page_text TEXT,
                page_char_start INTEGER,
                page_char_end INTEGER,
                section_path TEXT,
                has_table INTEGER DEFAULT 0,
                has_low_text INTEGER DEFAULT 0,
                extraction_source TEXT DEFAULT 'text',
                created_at TEXT
            )"""
        )
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_page_index_doc_id ON page_index(doc_id)")
        except Exception:
            pass
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_page_index_page ON page_index(doc_id, page_number_pdf)")
        except Exception:
            pass

        # ── BookRAG-lite++: extracted tables ──────────────────────────────────
        # One row per detected table region; enables table-aware retrieval and citation.
        conn.execute(
            """CREATE TABLE IF NOT EXISTS doc_tables(
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                page_number INTEGER,
                section_path TEXT,
                caption TEXT,
                raw_rows_text TEXT,
                char_start INTEGER,
                char_end INTEGER,
                created_at TEXT
            )"""
        )
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_tables_doc_id ON doc_tables(doc_id)")
        except Exception:
            pass

        # ── BookRAG-lite++: clause_id and adjacency on chunks ─────────────────
        # Added as optional columns; existing rows get NULL values (safe migration).
        for col_def in [
            "ALTER TABLE chunks ADD COLUMN clause_id TEXT",
            "ALTER TABLE chunks ADD COLUMN prev_chunk_id TEXT",
            "ALTER TABLE chunks ADD COLUMN next_chunk_id TEXT",
            "ALTER TABLE chunks ADD COLUMN retrieval_text TEXT",
            "ALTER TABLE chunks ADD COLUMN canonical_id TEXT",
            "ALTER TABLE chunks ADD COLUMN page_start INTEGER",
            "ALTER TABLE chunks ADD COLUMN page_end INTEGER",
            "ALTER TABLE chunks ADD COLUMN evidence_type TEXT",
            "ALTER TABLE chunks ADD COLUMN has_table INTEGER DEFAULT 0",
        ]:
            try:
                conn.execute(col_def)
            except Exception:
                pass

        conn.commit()

@contextmanager
def get_conn():
    conn = sqlite3.connect(settings.DB_PATH)
    try:
        yield conn
    finally:
        conn.close()
