"""Indexing one document version (the `document.index` job; S1.6, reading rebuilt in S2.1).

Runs with the authority of the person who imported the file: their right to import into the
matter is checked before the work starts and again just before anything is written, so if they
lose access mid-job nothing reaches the matter's index ("rights_revoked"). Only the document's
current version is indexed; a newer version replaces the older one's passages in search.

S2.1: the file is read by `extract` (OCR for scanned pages, Word numbering, tracked changes as
accepted, hidden text left out and reported). Its canonical text is stored with page, line and block
maps, and every passage is an exact slice of it. Progress is written as reading goes.
"""
from __future__ import annotations

import json
import logging
import uuid

import numpy as np

from ...core.db import get_conn
from ...rag.chunking.sanitize import find_pii
from ...rag.parse import page_for_offset
from .. import policy
from ..audit.record import record
from ..retrieval import embedder, registry
from . import blobs, chunking, extract, structure

logger = logging.getLogger(__name__)


class IngestFailed(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _version(version_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT v.id, v.document_id, v.matter_id, v.filename, v.wrapped_key, "
                           "CASE WHEN d.deleted_at IS NULL THEN d.current_version_id END, d.org_id "
                           "FROM legal_document_versions v JOIN legal_documents d ON d.id = v.document_id WHERE v.id=?",
                           (version_id,)).fetchone()
    keys = ("id", "document_id", "matter_id", "filename", "wrapped_key", "current_version_id", "org_id")
    return dict(zip(keys, row)) if row else None


def _progress(version_id: str, state: dict) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE legal_document_versions SET progress_json=? WHERE id=?", (json.dumps(state), version_id))
        conn.commit()


def _set_status(version_id: str, status: str, error: str | None = None, **extra) -> None:
    sets = ", ".join(["index_status=?", "index_error=?"] + [f"{k}=?" for k in extra])
    with get_conn() as conn:
        conn.execute(f"UPDATE legal_document_versions SET {sets} WHERE id=?", (status, error, *extra.values(), version_id))
        conn.commit()


async def run(job: dict) -> str | None:
    """Index the job's version; returns None on success, else the failure code."""
    v = _version(job["object_id"])
    if v is None:
        raise IngestFailed("missing")
    principal = {"sub": job["requested_by"], "org_id": job["org_id"]}

    def still_allowed() -> bool:
        return policy.authorize(principal, "documents.import", v["matter_id"]).allowed

    def audit(outcome: str, reason: str | None = None, **details) -> None:
        record("document.index", outcome, actor_user_id=job["requested_by"], object_type="document_version",
               object_id=v["id"], matter_id=v["matter_id"], reason=reason, details=details or None)

    try:
        if v["current_version_id"] != v["id"]:
            _set_status(v["id"], "superseded")
            audit("allow", "superseded")
            return None
        if not still_allowed():
            raise IngestFailed("rights_revoked")
        _set_status(v["id"], "indexing")
        try:
            data = blobs.load(v["matter_id"], v["id"], v["wrapped_key"])
        except blobs.IntegrityError:
            raise IngestFailed("integrity")
        _progress(v["id"], {"stage": "reading"})
        try:
            x = extract.extract(v["filename"], data, progress=lambda state: _progress(v["id"], state))
        except (RuntimeError, OSError) as e:
            logger.error("Documents: reading version %s failed (%s)", v["id"], type(e).__name__)
            raise IngestFailed("unreadable")
        text, offsets = x.text, x.page_offsets()
        pages = len(x.pages) or None
        if not text.strip():
            raise IngestFailed("needs_ocr" if any(p.source == "image" for p in x.pages) else "no_text")
        chunks = chunking.passages(x)
        if not chunks:
            raise IngestFailed("no_text")
        _progress(v["id"], {"stage": "indexing", "passages": len(chunks)})
        vecs = (await embedder.get().embed([c.text for c in chunks], kind="document")).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.where(norms == 0, 1, norms)
        findings = find_pii(text)
        if not still_allowed():
            raise IngestFailed("rights_revoked")      # lost access while we worked: write nothing
        _commit(v, chunks, vecs, findings, pages, offsets, x)
        from ..records import labels
        suggested = labels.suggest_from_text(v["org_id"], v["matter_id"], v["document_id"], text)
        flags = flag_counts(x)
        audit("allow", chunks=len(chunks), pages=pages, findings=len(findings), label_suggestions=suggested,
              **{k: n for k, n in flags.items() if n})
        return None
    except IngestFailed as e:
        _progress(v["id"], {"stage": "failed"})
        _set_status(v["id"], "failed", e.code)
        audit("error", e.code)
        return e.code
    except Exception as e:
        logger.error("Documents: indexing version %s failed (%s)", v["id"], type(e).__name__)
        _set_status(v["id"], "failed", "error")
        audit("error", "error")
        raise


def flag_counts(x: "extract.Extraction") -> dict:
    kinds = [m.kind for m in x.marks]
    return {"hidden_text": kinds.count("hidden_text"), "instructions": kinds.count("instructions"),
            "tracked_changes": kinds.count("tracked_insert") + kinds.count("tracked_delete"), "comments": kinds.count("comment"),
            "ocr_pages": sum(1 for p in x.pages if p.source == "ocr"),
            "ocr_low_confidence": kinds.count("ocr_low_confidence")}


def _commit(v: dict, chunks, vecs: np.ndarray, findings, pages, offsets, x: "extract.Extraction") -> None:
    rows = []
    for i, (c, vec) in enumerate(zip(chunks, vecs)):
        start, end = c.char_start, c.char_end
        p1 = page_for_offset(start, offsets) if offsets and start is not None else c.page_number
        p2 = page_for_offset(max(start, end - 1), offsets) if offsets and start is not None and end is not None else p1
        rows.append((str(uuid.uuid4()), v["id"], v["document_id"], v["matter_id"], i, c.text, start, end, p1, p2,
                     vec.tobytes()))
    idx = registry.for_matter(v["matter_id"])
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            # Every passage of this document now in the index (older versions, or an earlier run of this one).
            old = [r[0] for r in conn.execute("SELECT id FROM legal_chunks WHERE document_id=? AND active=1", (v["document_id"],))]
            conn.execute("UPDATE legal_chunks SET active=0 WHERE document_id=? AND version_id<>?", (v["document_id"], v["id"]))
            conn.execute("UPDATE legal_document_versions SET index_status='superseded' "
                         "WHERE document_id=? AND id<>? AND index_status='indexed'", (v["document_id"], v["id"]))
            conn.execute("DELETE FROM legal_chunks WHERE version_id=?", (v["id"],))       # a re-index starts clean
            conn.execute("DELETE FROM legal_findings WHERE version_id=?", (v["id"],))
            conn.execute("DELETE FROM legal_marks WHERE version_id=?", (v["id"],))
            conn.execute("DELETE FROM legal_version_text WHERE version_id=?", (v["id"],))
            maps = x.as_json()
            conn.execute("INSERT INTO legal_version_text (version_id, matter_id, text, pages_json, blocks_json, lines_json, engines_json, "
                         "created_at) VALUES (?,?,?,?,?,?,?,?)",
                         (v["id"], v["matter_id"], x.text, json.dumps(maps["pages"]), json.dumps(maps["blocks"]),
                          json.dumps(maps["lines"]), json.dumps(maps["engines"]), policy.now_iso()))
            conn.execute("DELETE FROM legal_clauses WHERE version_id=?", (v["id"],))
            conn.execute("DELETE FROM legal_defined_terms WHERE version_id=?", (v["id"],))
            conn.executemany("INSERT INTO legal_clauses (id, version_id, matter_id, number, label, heading, level, parent, char_start, "
                             "char_end, page) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                             [(str(uuid.uuid4()), v["id"], v["matter_id"], c.number, c.label, c.heading, c.level, c.parent, c.start,
                               c.end, c.page) for c in structure.clauses(x)])
            conn.executemany("INSERT INTO legal_defined_terms (id, version_id, matter_id, term, char_start, char_end, term_start, "
                             "term_end) VALUES (?,?,?,?,?,?,?,?)",
                             [(str(uuid.uuid4()), v["id"], v["matter_id"], t.term, t.start, t.end, t.term_start, t.term_end)
                              for t in structure.defined_terms(x)])
            conn.executemany("INSERT INTO legal_marks (id, version_id, matter_id, kind, char_start, char_end, page, meta_json) "
                             "VALUES (?,?,?,?,?,?,?,?)",
                             [(str(uuid.uuid4()), v["id"], v["matter_id"], m.kind, m.start, m.end, m.page, json.dumps(m.meta))
                              for m in x.marks])
            conn.executemany("INSERT INTO legal_chunks (id, version_id, document_id, matter_id, chunk_index, text, char_start, "
                             "char_end, page_start, page_end, vec) VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
            conn.executemany("INSERT INTO legal_findings (id, version_id, matter_id, kind, char_start, char_end) VALUES (?,?,?,?,?,?)",
                             [(str(uuid.uuid4()), v["id"], v["matter_id"], k, a, b) for k, a, b in findings])
            conn.execute("UPDATE legal_document_versions SET index_status='indexed', index_error=NULL, chunk_count=?, pages=COALESCE(pages, ?), "
                         "indexed_at=?, progress_json=? WHERE id=?",
                         (len(rows), pages, policy.now_iso(), json.dumps({"stage": "done"}), v["id"]))
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    # The database is the source of truth; the partition follows it (and is rebuilt from it if
    # these files are ever lost or disagree).
    with idx.lock:
        idx.remove(old)
        idx.add([(r[0], r[5], vec) for r, vec in zip(rows, vecs)])
