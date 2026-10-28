"""Citations the browser holds only by an opaque id (docs/echomind-legal/06 "Citation validation", AC02).

When research finishes an answer, each source it cites becomes one citation record: the document, the exact version,
the character offsets of the best quote found for it and the SHA-256 of those characters — never the words. Opening
the citation later:

1. finds it only for the person it was made for (anyone else: 404, like an id that does not exist);
2. asks the policy engine again (`documents.read` on the matter) and the document's own source permissions;
3. re-reads the version's stored text and recomputes the SHA-256 of the cited characters: if they differ the citation
   is refused (409) instead of showing a passage that no longer matches what was cited;
4. returns the passage with its surroundings, page, clause and line, the version's file hash, whether the quote was
   verified, and whether a newer version of the document exists.

A citation is "verified" only when the model's quote was found in the passage it named (research's verbatim or
normalized match). A source the answer points at without a matching quote is shown, but as not verified.
"""
from __future__ import annotations

import hashlib
from typing import Optional
from urllib.parse import quote as urlquote

from ...core.db import get_conn
from .. import policy
from ..documents import repo as doc_repo
from ..retrieval import search as searchmod
from . import store

VERIFIED = ("verbatim", "normalized")
CONTEXT_CHARS = 500


class NotFound(Exception):
    pass


class Mismatch(Exception):
    """The stored text no longer matches what was cited."""


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _best(n: int, evidence: list[dict]) -> Optional[dict]:
    items = [e for e in evidence if e.get("n") == n]
    for wanted in (VERIFIED, ("not_quoted",)):
        for e in items:
            if e.get("verification") in wanted and e.get("char_start") is not None and e.get("char_end") is not None:
                return e
    return items[0] if items else None


def from_research(turn: dict, done: dict) -> list[dict]:
    """Citation records for every source a finished research answer cites, saved; their public references back."""
    if done.get("abstained"):
        return []
    evidence = done.get("evidence") or []
    texts: dict[str, Optional[str]] = {}
    out, now = [], policy.now_iso()
    with get_conn() as conn:
        for src in done.get("citations") or []:
            n = int(src["n"])
            e = _best(n, evidence)
            vid = (e or {}).get("version_id") or src["version_id"]
            a, b = (e or {}).get("char_start"), (e or {}).get("char_end")
            verification = (e or {}).get("verification") or "not_quoted"
            if verification not in ("verbatim", "normalized", "not_quoted", "failed"):
                verification = "failed"
            if a is None or b is None:                     # no located quote: point at the passage itself
                a, b = src.get("focus_start", src.get("char_start")), src.get("focus_end", src.get("char_end"))
                if verification in VERIFIED:
                    verification = "not_quoted"
            if vid not in texts:
                row = conn.execute("SELECT text FROM legal_version_text WHERE version_id=?", (vid,)).fetchone()
                texts[vid] = row[0] if row else None
            text = texts[vid]
            digest = sha(text[a:b]) if text is not None and a is not None and b is not None and 0 <= a < b <= len(text) else None
            if digest is None:
                a = b = None
                if verification in VERIFIED:
                    verification = "failed"
            cid = store.new_id("cit")
            conn.execute("INSERT INTO legal_counsel_citations (id, turn_id, matter_id, owner_id, n, document_id, version_id, char_start, "
                         "char_end, quote_sha256, verification, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (cid, turn["id"], src.get("matter_id") or turn["matter_id"], turn["owner_id"], n, src["document_id"], vid,
                          a, b, digest, verification, now))
            page = (e or {}).get("page_start") or src.get("page_start")
            out.append({"id": cid, "n": n, "document_id": src["document_id"], "document_title": src.get("document_title") or "",
                        "version_number": src.get("version_number"), "version_status": src.get("version_status"),
                        "page": page if isinstance(page, int) and page >= 1 else None,
                        "clause": ((e or {}).get("clause") or src.get("clause") or None),
                        "verified": verification in VERIFIED,
                        "instructions": bool(src.get("instructions")),
                        **({"matter": src["matter_number"]} if src.get("matter_number") else {})})
        conn.commit()
    return out


def reader_path(matter_number: str, document_id: str, version_id: str, a: Optional[int], b: Optional[int]) -> str:
    at = f"&at={a}-{b}" if a is not None and b is not None and b > a else ""
    return f"/m/{urlquote(matter_number, safe='')}/documents/{document_id}/read?v={version_id}{at}"


def resolve(principal: dict, citation_id: str) -> dict:
    """The cited passage for its owner, re-authorized and re-verified. Raises NotFound or Mismatch."""
    with get_conn() as conn:
        r = conn.execute("SELECT id, matter_id, n, document_id, version_id, char_start, char_end, quote_sha256, verification "
                         "FROM legal_counsel_citations WHERE id=? AND owner_id=?", (citation_id, principal["sub"])).fetchone()
    if not r:
        raise NotFound(citation_id)
    _, matter_id, n, document_id, version_id, a, b, digest, verification = r
    if not policy.authorize(principal, "documents.read", matter_id).allowed:
        raise NotFound(citation_id)
    v = doc_repo.version_row(matter_id, document_id, version_id, principal["sub"])
    doc = doc_repo.get_document(matter_id, document_id, principal["sub"])
    body = doc_repo.reader(version_id) if v and doc else None
    if not v or not doc or body is None:
        raise NotFound(citation_id)
    text = body["text"]
    if a is not None and b is not None:
        if not (0 <= a < b <= len(text)) or sha(text[a:b]) != digest:
            raise Mismatch(citation_id)
        lo = max(0, a - CONTEXT_CHARS)
        hi = min(len(text), b + CONTEXT_CHARS)
        lo = text.rfind(" ", 0, lo) + 1 if lo > 0 and text.rfind(" ", 0, lo) >= 0 else lo
        nxt = text.find(" ", hi)
        hi = nxt if 0 <= nxt < hi + 40 else hi
        with get_conn() as conn:
            where = searchmod.locate(conn, version_id, a, b)
            number = conn.execute("SELECT number FROM matters WHERE id=?", (matter_id,)).fetchone()[0]
        context = {"text": text[lo:hi], "highlight_start": a - lo, "highlight_end": b - lo, "offset": lo,
                   "truncated_before": lo > 0, "truncated_after": hi < len(text)}
        quote = text[a:b]
    else:
        where, context, quote = {}, None, None
        with get_conn() as conn:
            number = conn.execute("SELECT number FROM matters WHERE id=?", (matter_id,)).fetchone()[0]
    current = doc.get("current_version") or {}
    newer = current.get("number") if current.get("id") and current.get("id") != version_id else None
    return {
        "id": citation_id, "n": n, "matter_id": matter_id, "verification": verification, "verified": verification in VERIFIED,
        "document": {"id": document_id, "title": doc["title"], "kind": doc["kind"], "status": doc["status"]},
        "version": {"id": version_id, "number": v["number"], "status": v["status"], "filename": v["filename"], "sha256": v["sha256"]},
        "quote": quote, "context": context, "clause": where.get("clause"), "page": where.get("page_start"),
        "page_end": where.get("page_end"), "line": where.get("line"), "newer_version": newer,
        "reader_path": reader_path(number, document_id, version_id, a, b),
    }


def purge_document(conn, matter_id: str, document_id: str) -> None:
    conn.execute("DELETE FROM legal_counsel_citations WHERE matter_id=? AND document_id=?", (matter_id, document_id))
