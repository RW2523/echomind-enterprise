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
        # Assistant Mode: hand-raise suggestions (local only; session_id is client-owned UUID string).
        conn.execute(
            """CREATE TABLE IF NOT EXISTS assistant_suggestions(
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                mode TEXT NOT NULL DEFAULT 'ASSISTANT',
                title TEXT NOT NULL,
                short_text TEXT NOT NULL,
                speak_text TEXT NOT NULL,
                reason TEXT,
                category TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_origin TEXT NOT NULL,
                evidence_status TEXT NOT NULL,
                citations_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                updated_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_assistant_suggestions_session_status "
            "ON assistant_suggestions(session_id, status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_assistant_suggestions_session_created "
            "ON assistant_suggestions(session_id, created_at)"
        )
        # Silent Assistant Mode: correction findings (display-only; never TTS).
        conn.execute(
            """CREATE TABLE IF NOT EXISTS silent_findings(
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                transcript_segment_id TEXT,
                turn_id TEXT,
                original_text TEXT NOT NULL,
                highlighted_span_start INTEGER NOT NULL DEFAULT 0,
                highlighted_span_end INTEGER NOT NULL DEFAULT 0,
                category TEXT NOT NULL,
                status_label TEXT NOT NULL,
                suggested_correction TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL,
                evidence_status TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_origin TEXT NOT NULL,
                citations_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                user_action TEXT NOT NULL DEFAULT 'pending',
                updated_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_silent_findings_session_action ON silent_findings(session_id, user_action)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_silent_findings_session_created ON silent_findings(session_id, created_at)"
        )
        for col in (
            "influencing_rule_set_id TEXT",
            "influencing_rule_set_name TEXT",
            "influencing_rule_id TEXT",
            "influencing_rule_title TEXT",
        ):
            try:
                conn.execute(f"ALTER TABLE silent_findings ADD COLUMN {col}")
            except Exception:
                pass
        # Rules Library (local policy): named sets, rules, per-session enablement.
        conn.execute(
            """CREATE TABLE IF NOT EXISTS rule_sets(
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                version TEXT NOT NULL DEFAULT '1.0.0',
                priority INTEGER NOT NULL DEFAULT 0,
                is_active_default INTEGER NOT NULL DEFAULT 0,
                source_policy_text TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS rules(
                id TEXT PRIMARY KEY,
                rule_set_id TEXT NOT NULL,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'medium',
                category TEXT NOT NULL DEFAULT 'general',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rules_rule_set ON rules(rule_set_id)")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS session_rule_activations(
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                rule_set_id TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                priority_override INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(session_id, rule_set_id)
            )"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_rule_act_session ON session_rule_activations(session_id, enabled)"
        )
        for col in (
            "influencing_rule_set_id TEXT",
            "influencing_rule_set_name TEXT",
            "influencing_rule_id TEXT",
            "influencing_rule_title TEXT",
            "trigger_excerpt TEXT",
        ):
            try:
                conn.execute(f"ALTER TABLE assistant_suggestions ADD COLUMN {col}")
            except Exception:
                pass
        conn.execute(
            """CREATE TABLE IF NOT EXISTS session_notes(
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                citations_json TEXT NOT NULL DEFAULT '[]',
                tags_json TEXT NOT NULL DEFAULT '[]',
                pinned INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(session_id, source_type, source_id)
            )"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_notes_session_updated ON session_notes(session_id, updated_at DESC)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session_notes_session_pinned ON session_notes(session_id, pinned)")
        conn.commit()

@contextmanager
def get_conn():
    conn = sqlite3.connect(settings.DB_PATH)
    try:
        yield conn
    finally:
        conn.close()
