"""Indexing one document version (the `document.index` job; S1.6).

Runs with the authority of the person who imported the file: their right to import into the
matter is checked before the work starts and again just before anything is written, so if they
lose access mid-job nothing reaches the matter's index ("rights_revoked"). Only the document's
current version is indexed; a newer version replaces the older one's passages in search.
"""
from __future__ import annotations

import logging
import uuid

import numpy as np

from ...core.db import get_conn
from ...rag.chunking.pipeline import chunk_document
from ...rag.chunking.sanitize import find_pii
from ...rag.parse import page_for_offset, parse_any
from .. import policy
from ..audit.record import record
from ..retrieval import embedder, registry
from . import blobs

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
        filetype, text, pages, offsets = parse_any(v["filename"], data)
        if filetype == "txt":
            text = data.decode("utf-8-sig")
        chunks = [c for c in chunk_document(text, doc_id=v["id"], estimated_pages=pages, page_offsets=offsets,
                                            already_normalized=(filetype == "pdf"), redact_pii=False) if not c.is_parent]
        if not chunks:
            raise IngestFailed("no_text")              # e.g. a scan without a text layer (OCR arrives in S2.1)
        vecs = (await embedder.get().embed([c.text for c in chunks], kind="document")).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.where(norms == 0, 1, norms)
        findings = find_pii(text)
        if not still_allowed():
            raise IngestFailed("rights_revoked")      # lost access while we worked: write nothing
        _commit(v, chunks, vecs, findings, pages or None, offsets)
        from ..records import labels
        suggested = labels.suggest_from_text(v["org_id"], v["matter_id"], v["document_id"], text)
        audit("allow", chunks=len(chunks), pages=pages or None, findings=len(findings), label_suggestions=suggested)
        return None
    except IngestFailed as e:
        _set_status(v["id"], "failed", e.code)
        audit("error", e.code)
        return e.code
    except Exception as e:
        logger.error("Documents: indexing version %s failed (%s)", v["id"], type(e).__name__)
        _set_status(v["id"], "failed", "error")
        audit("error", "error")
        raise


def _commit(v: dict, chunks, vecs: np.ndarray, findings, pages, offsets) -> None:
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
            conn.executemany("INSERT INTO legal_chunks (id, version_id, document_id, matter_id, chunk_index, text, char_start, "
                             "char_end, page_start, page_end, vec) VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
            conn.executemany("INSERT INTO legal_findings (id, version_id, matter_id, kind, char_start, char_end) VALUES (?,?,?,?,?,?)",
                             [(str(uuid.uuid4()), v["id"], v["matter_id"], k, a, b) for k, a, b in findings])
            conn.execute("UPDATE legal_document_versions SET index_status='indexed', index_error=NULL, chunk_count=?, pages=COALESCE(pages, ?), "
                         "indexed_at=? WHERE id=?", (len(rows), pages, policy.now_iso(), v["id"]))
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    # The database is the source of truth; the partition follows it (and is rebuilt from it if
    # these files are ever lost or disagree).
    with idx.lock:
        idx.remove(old)
        idx.add([(r[0], r[5], vec) for r, vec in zip(rows, vecs)])
