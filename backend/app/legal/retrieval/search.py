"""Searching one matter's documents (S1.6).

Three channels over the scope's partition only — dense (embeddings), sparse (BM25 with the
partition's own statistics) and keyword (distinctive terms such as clause numbers, codes and
addresses, matched literally) — fused by reciprocal rank, then re-ranked by the cross-encoder.

The cross-encoder reads about a thousand characters, and an indexed passage can be longer, so each
candidate is scored in sentence-aligned windows of about WINDOW_CHARS and represented by its best
window: the clause that matched, not the start of the passage it sits in. That window is what
the result quotes, with offsets into the document. Windows scoring below LEGAL_RERANK_MIN_SCORE
are dropped; when nothing relevant is left the result says so (`abstained`) rather than offering
the nearest unrelated text.

Every candidate is re-read from the database with the scope's matter id in the WHERE clause, so
even a corrupted index could not surface another matter's passage.
"""
from __future__ import annotations

import bisect
import json
import re
from typing import Iterable

import numpy as np

from ...core.config import settings
from ...core.db import get_conn
from . import embedder, registry
from .scope import AuthorizedScope, recheck

ALL_MODES = ("dense", "sparse", "keyword")
_RRF_K = 60


async def rerank(query: str, hits: list[dict], top_k: int) -> list[dict]:
    from ...rag.reranker import rerank_hits
    return await rerank_hits(query, hits, top_k)


# ── Keyword channel ──────────────────────────────────────────────────────────────
_TERM_RE = re.compile(r'"([^"]{2,120})"|(\S+)')


def keyword_terms(query: str) -> list[str]:
    """Quoted phrases and distinctive tokens (containing a digit, @ or an inner . _ - /, or written
    in capitals); without any, the query's longer words."""
    phrases, distinctive, words = [], [], []
    for quoted, token in _TERM_RE.findall(query or ""):
        if quoted:
            phrases.append(quoted.strip())
            continue
        t = token.strip(".,;:!?()[]{}'\"")
        if len(t) < 2:
            continue
        if re.search(r"\d|@", t) or re.search(r"\w[._/-]\w", t) or (t.isupper() and len(t) >= 2):
            distinctive.append(t)
        elif len(t) >= 4:
            words.append(t)
    terms = phrases + distinctive or words
    return list(dict.fromkeys(t.lower() for t in terms))[:12]


def _keyword(matter_id: str, query: str, k: int) -> list[tuple[str, float]]:
    terms = keyword_terms(query)
    if not terms:
        return []
    score = " + ".join(["(instr(lower(text), ?) > 0)"] * len(terms))
    with get_conn() as conn:
        rows = conn.execute(f"SELECT id, {score} AS s FROM legal_chunks WHERE matter_id = ? AND active = 1 "
                            f"AND ({score}) > 0 ORDER BY s DESC LIMIT ?", (*terms, matter_id, *terms, k)).fetchall()
    return [(r[0], float(r[1])) for r in rows]


# ── Windows ──────────────────────────────────────────────────────────────────────
WINDOW_CHARS = 600
_MAX_WINDOWS = 160
# A sentence ends at . ; ! or ? followed by whitespace, so clause numbers ("8.1") and amounts
# ("1,250.00") never split.
_SENTENCE = re.compile(r".+?(?:[.;!?](?=\s)|$)", re.S)


def windows(text: str, size: int = WINDOW_CHARS) -> list[tuple[int, int]]:
    """Sentence-aligned (start, end) spans of at most about `size` characters covering `text`."""
    spans: list[tuple[int, int]] = []
    start = end = None
    for m in _SENTENCE.finditer(text or ""):
        a, b = m.span()
        if not text[a:b].strip():
            continue
        while b - a > size:                                   # one very long sentence: cut at a space
            cut = text.rfind(" ", a, a + size)
            cut = cut if cut > a else a + size
            if start is not None:
                spans.append((start, end))
                start = end = None
            spans.append((a, cut))
            a = cut
        heading = text[a:b].lstrip().startswith("#")          # a Markdown heading opens a new window
        if start is not None and (b - start > size or heading):
            spans.append((start, end))
            start = None
        if start is None:
            start = a
        end = b
    if start is not None:
        spans.append((start, end))
    out = []
    for a, b in spans:                                        # trim surrounding whitespace
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if b > a:
            out.append((a, b))
    return out


def _windowed(hits: list[dict]) -> list[dict]:
    out = []
    for h in hits:
        for a, b in windows(h["text"]):
            out.append({**h, "text": h["text"][a:b], "window": (a, b)})
            if len(out) >= _MAX_WINDOWS:
                return out
    return out


# ── Fusion and loading ───────────────────────────────────────────────────────────
def _fuse(rankings: Iterable[list[tuple[str, float]]]) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores, key=lambda c: scores[c], reverse=True)


def _load(matter_id: str, chunk_ids: list[str], uid: str) -> list[dict]:
    """Candidates re-read with the matter id AND the document's visibility for this person (source
    permissions and connector freshness, S1.11)."""
    from ..documents.repo import VISIBLE_TO
    if not chunk_ids:
        return []
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT c.id, c.text, c.page_start, c.page_end, c.char_start, c.char_end, c.document_id, c.version_id, "
            "d.title, v.number, v.filename FROM legal_chunks c JOIN legal_documents d ON d.id = c.document_id "
            "JOIN legal_document_versions v ON v.id = c.version_id "
            f"WHERE c.matter_id = ? AND c.active = 1 AND c.id IN ({','.join('?' * len(chunk_ids))}) AND {VISIBLE_TO}",
            (matter_id, *chunk_ids, uid)).fetchall()
    by_id = {r[0]: r for r in rows}
    out = []
    for cid in chunk_ids:
        r = by_id.get(cid)
        if r:
            out.append({"chunk_id": r[0], "text": r[1], "page_start": r[2], "page_end": r[3], "char_start": r[4],
                        "char_end": r[5], "document_id": r[6], "version_id": r[7], "document_title": r[8],
                        "version_number": r[9], "filename": r[10], "matter_id": matter_id})
    return out


def _passage(h: dict) -> dict:
    a, b = h.get("window") or (0, len(h["text"]))
    base = h["char_start"]
    return {
        "chunk_id": h["chunk_id"], "text": h["text"],
        "score": h.get("score"), "score_kind": h.get("score_kind"),
        "citation": {"matter_id": h["matter_id"], "document_id": h["document_id"], "document_title": h["document_title"],
                     "version_id": h["version_id"], "version_number": h["version_number"],
                     "page_start": h["page_start"], "page_end": h["page_end"],
                     "char_start": base + a if base is not None else None,
                     "char_end": base + b if base is not None else None},
    }


_STOP = frozenset("the a an and or of to in on for by with is are be as at that this which shall may any all its it from".split())


def _focus(text: str, query: str) -> tuple[int, int]:
    """The sentence of a quote that shares most words with the question (ties: the earliest)."""
    from ..research.prompt import _spans
    q = {w for w in re.findall(r"[a-z0-9%]{2,}", query.lower()) if w not in _STOP}
    best, score = (0, len(text)), -1
    for a, b in _spans(text):
        n = len(q & set(re.findall(r"[a-z0-9%]{2,}", text[a:b].lower())))
        if n > score:
            best, score = (a, b), n
    return best


def _located(passages: list[dict], query: str = "") -> list[dict]:
    """Where each quote points (S2.1): the sentence that best answers the question (the focus), its
    clause, and the page and line it is on, from the version's stored maps. Versions read before
    S2.1 have no maps and keep the passage's pages."""
    if not passages:
        return passages
    for p in passages:
        c = p["citation"]
        if c["char_start"] is not None:
            a, b = _focus(p["text"], query)
            c["focus_start"], c["focus_end"] = c["char_start"] + a, c["char_start"] + b
    maps: dict[str, tuple] = {}
    with get_conn() as conn:
        for p in passages:
            c = p["citation"]
            vid, off = c["version_id"], c.get("focus_start", c["char_start"])
            if off is None:
                continue
            if vid not in maps:
                row = conn.execute("SELECT pages_json, lines_json FROM legal_version_text WHERE version_id=?", (vid,)).fetchone()
                maps[vid] = (json.loads(row[0]), json.loads(row[1])) if row else None
            clause = conn.execute("SELECT number FROM legal_clauses WHERE version_id=? AND char_start<=? AND ?<char_end "
                                  "ORDER BY level DESC, char_start DESC LIMIT 1", (vid, off, off)).fetchone()
            c["clause"] = clause[0] if clause else None
            if not maps[vid]:
                continue
            pages, lines = maps[vid]
            end = max(off, (c.get("focus_end") or c["char_end"] or off) - 1)
            here = next((pg for pg in pages if pg["start"] <= off <= pg["end"]), None)
            there = next((pg for pg in pages if pg["start"] <= end <= pg["end"]), here)
            base = here["start"] if here else 0
            c["line"] = max(1, bisect.bisect_right(lines, off) - bisect.bisect_left(lines, base))
            if here:
                c["page_start"], c["page_end"] = here["number"], (there or here)["number"]
    return passages


async def search(scope: AuthorizedScope, query: str, *, k: int = 8, modes: Iterable[str] = ALL_MODES) -> dict:
    recheck(scope)                                    # raises TypeError for anything but a real scope
    modes = [m for m in modes if m in ALL_MODES] or list(ALL_MODES)
    idx = registry.for_matter(scope.matter_id)
    n = max(k, settings.LEGAL_SEARCH_CANDIDATES)
    rankings = []
    if "dense" in modes:
        q = (await embedder.get().embed([query], kind="query")).astype(np.float32)
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        rankings.append(idx.search_dense(q / np.where(norm == 0, 1, norm), n))
    if "sparse" in modes:
        rankings.append(idx.search_sparse(query, n))
    if "keyword" in modes:
        rankings.append(_keyword(scope.matter_id, query, n))
    hits = _load(scope.matter_id, _fuse(rankings)[:n], scope.principal_id)
    subs = _windowed(hits)
    scored = await rerank(query, subs, len(subs)) if subs else []
    best: dict[str, dict] = {}
    for h in scored:                                  # best window of each passage, in score order
        best.setdefault(h["chunk_id"], h)
    ranked = list(best.values())
    if ranked and ranked[0].get("score_kind") == "ce_logit":
        ranked = [h for h in ranked if float(h["score"]) >= settings.LEGAL_RERANK_MIN_SCORE]
    recheck(scope)                                    # rights may have changed while this ran
    return {"passages": _located([_passage(h) for h in ranked[:k]], query), "abstained": not ranked}
