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
from typing import Iterable, Optional

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
            out.append({**h, "text": h["text"][a:b], "window": (a, b), "chunk_text": h["text"]})
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


def _load(matter_id: str, chunk_ids: list[str], uid: str, versions: str = "signed",
          documents: Optional[list[str]] = None) -> list[dict]:
    """Candidates re-read with the matter id AND the document's visibility for this person (source
    permissions and connector freshness, S1.11) AND the version policy (S2.2a: signed by default),
    and, when given, only these documents (S2.6: a board pack)."""
    from ..documents.repo import VISIBLE_TO
    from ..documents.versions import policy_sql
    if not chunk_ids:
        return []
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT c.id, c.text, c.page_start, c.page_end, c.char_start, c.char_end, c.document_id, c.version_id, "
            "d.title, v.number, v.filename, v.status FROM legal_chunks c JOIN legal_documents d ON d.id = c.document_id "
            "JOIN legal_document_versions v ON v.id = c.version_id "
            f"WHERE c.matter_id = ? AND c.active = 1 AND c.id IN ({','.join('?' * len(chunk_ids))}) AND {VISIBLE_TO} "
            f"AND {policy_sql(versions)}" + (f" AND c.document_id IN ({','.join('?' * len(documents))})" if documents else ""),
            (matter_id, *chunk_ids, uid, *(documents or []))).fetchall()
    by_id = {r[0]: r for r in rows}
    out = []
    for cid in chunk_ids:
        r = by_id.get(cid)
        if r:
            out.append({"chunk_id": r[0], "text": r[1], "page_start": r[2], "page_end": r[3], "char_start": r[4],
                        "char_end": r[5], "document_id": r[6], "version_id": r[7], "document_title": r[8],
                        "version_number": r[9], "filename": r[10], "version_status": r[11], "matter_id": matter_id})
    return out


# Research reads the whole passage around its best window (2026-11-03 evaluation): a 600-character window of an 896-character
# change of control notice left out "Please direct any questions to our General Counsel, Thomas Reyes" and "Your account
# team will not change", so the model declined questions the document answers. The window still ranks the passage and
# still marks the sentence the reader opens on; callers that check values against exact windows keep them.
WIDE_CHARS = 1800


def _wide(h: dict) -> dict:
    full = h.get("chunk_text") or h["text"]
    a, b = h.get("window") or (0, len(full))
    if len(full) <= WIDE_CHARS:
        return {**h, "text": full, "window": (0, len(full))}
    lo = max(0, min(a - (WIDE_CHARS - (b - a)) // 2, len(full) - WIDE_CHARS))
    return {**h, "text": full[lo:lo + WIDE_CHARS], "window": (lo, lo + WIDE_CHARS)}


# A passage that announces what follows ends before it: "The following four comparable lettings have been reviewed …" was
# one passage and the four lettings the next, so "Which comparables does our surveyor rely on?" named one of four
# (2026-11-03 chat set). Such a passage is read on into the next passage of the same version, within WIDE_CHARS; the gap
# between them is kept as spaces so every character keeps its place in the document.
_ANNOUNCES = re.compile(r"\bthe\s+following\b(?!\s+(?:day|week|month|year|business|working|calendar|quarter|period|morning"
                        r"|afternoon|evening|meeting)\b)|\bas\s+follows\b|\b(?:set\s+out|listed|shown|described)\s+below\b|:\s*$", re.I)


def _continued(scope: AuthorizedScope, h: dict, versions: str, documents: Optional[list[str]]) -> dict:
    full = h.get("chunk_text") or h["text"]
    if h.get("char_start") is None or len(full) >= WIDE_CHARS or not _ANNOUNCES.search(full.rstrip()[-600:]):
        return h
    end = h["char_start"] + len(full)
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM legal_chunks WHERE matter_id = ? AND version_id = ? AND active = 1 AND char_start >= ? "
                           "ORDER BY char_start LIMIT 1", (scope.matter_id, h["version_id"], end)).fetchone()
    nxt = _load(scope.matter_id, [row[0]], scope.principal_id, versions, documents) if row else []
    if not nxt or nxt[0]["char_start"] is None or nxt[0]["char_start"] - h["char_start"] + len(nxt[0]["text"]) > WIDE_CHARS:
        return h
    joined = full + " " * (nxt[0]["char_start"] - end) + nxt[0]["text"]
    return {**h, "chunk_text": joined, "page_end": nxt[0]["page_end"] or h.get("page_end")}


def opening(scope: AuthorizedScope, version_id: str, names: Iterable[str] = (), versions: str = "signed") -> Optional[dict]:
    """The passage where a version's preamble defines its parties (2026-11-03 evaluation): the earliest of its first three
    passages that names one of them, else its first. Read like a search: this matter, what this person may see, the version
    policy."""
    recheck(scope)
    with get_conn() as conn:
        ids = [r[0] for r in conn.execute("SELECT id FROM legal_chunks WHERE matter_id = ? AND version_id = ? AND active = 1 "
                                          "ORDER BY char_start LIMIT 3", (scope.matter_id, version_id)).fetchall()]
    hits = _load(scope.matter_id, ids, scope.principal_id, versions if versions in ("signed", "current") else "signed")
    keys = [n.lower() for n in names if n]
    h = next((h for h in hits if any(k in h["text"].replace("**", "").lower() for k in keys)), hits[0] if hits else None)
    if not h:
        return None
    return _located([_passage(_wide(h))], " ".join(names))[0]


_CLAUSE_REF = re.compile(r"\b(?:clauses?|sections?|cl\.|§)\s*(\d{1,2}(?:\.\d{1,2}){0,2})((?:(?:\s*(?:,|and|&|to|-|–))+\s*\d{1,2}(?:\.\d{1,2}){0,2})*)", re.I)


def clause_numbers(text: str) -> list[str]:
    """The clause numbers a question names: "clause 14", "clauses 4.1, 4.2 and 4.3", "section 9" (2026-11-05)."""
    out = []
    for m in _CLAUSE_REF.finditer(text or ""):
        out.append(m.group(1))
        out += re.findall(r"\d{1,2}(?:\.\d{1,2}){0,2}", m.group(2) or "")
    return list(dict.fromkeys(out))[:6]


def clause_passages(scope: AuthorizedScope, numbers: list[str], versions: str = "signed", documents: Optional[list[str]] = None,
                    kinds: tuple = ("contract",), limit: int = 4) -> list[dict]:
    """The passages holding the clauses a question names, from the matter's documents of these kinds (or only `documents`),
    read like a search: this matter, what this person may see, the version policy (2026-11-05 end-to-end run: "What does clause
    14 actually cover?" was answered from the letters describing it, not from the agreement's clause 14)."""
    if not numbers:
        return []
    numbers = list(dict.fromkeys(x for n in numbers for x in ([n] if "." in n else [n, f"{n}.1"])))   # "clause 14": 14, else 14.1
    recheck(scope)
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT c.version_id, c.number, c.char_start FROM legal_clauses c JOIN legal_document_versions v ON v.id = c.version_id "
            f"JOIN legal_documents d ON d.id = v.document_id WHERE c.matter_id = ? AND d.deleted_at IS NULL "
            f"AND c.number IN ({','.join('?' * len(numbers))}) "
            + (f"AND d.id IN ({','.join('?' * len(documents))}) " if documents else f"AND d.kind IN ({','.join('?' * len(kinds))}) ")
            + "ORDER BY c.char_start", (scope.matter_id, *numbers, *(documents or kinds))).fetchall()
        ids = []
        for vid, _number, start in rows:
            r = conn.execute("SELECT id FROM legal_chunks WHERE matter_id = ? AND version_id = ? AND active = 1 AND char_start <= ? "
                             "AND ? < char_end ORDER BY char_start LIMIT 1", (scope.matter_id, vid, start, start)).fetchone()
            if r and r[0] not in ids:
                ids.append(r[0])
    hits = _load(scope.matter_id, ids, scope.principal_id, versions if versions in ("signed", "current") else "signed", documents)
    order = {cid: i for i, cid in enumerate(ids)}
    hits.sort(key=lambda h: order.get(h["chunk_id"], 99))
    return [_located([_passage(_wide(h))], " ".join(numbers))[0] for h in hits[:limit]]


def _passage(h: dict) -> dict:
    a, b = h.get("window") or (0, len(h["text"]))
    base = h["char_start"]
    return {
        "chunk_id": h["chunk_id"], "text": h["text"],
        "score": h.get("score"), "score_kind": h.get("score_kind"),
        "citation": {"matter_id": h["matter_id"], "document_id": h["document_id"], "document_title": h["document_title"],
                     "version_id": h["version_id"], "version_number": h["version_number"],
                     "version_status": h.get("version_status"), "signed": h.get("version_status") == "executed",
                     "page_start": h["page_start"], "page_end": h["page_end"],
                     "char_start": base + a if base is not None else None,
                     "char_end": base + b if base is not None else None},
    }


_STOP = frozenset("the a an and or of to in on for by with is are be as at that this which shall may any all its it from".split())


def _focus(text: str, query: str) -> tuple[int, int]:
    """The sentence of a quote that shares most words with the question (ties: the earliest)."""
    from ..research.prompt import document_sentences
    q = {w for w in re.findall(r"[a-z0-9%]{2,}", query.lower()) if w not in _STOP}
    best, score = (0, len(text)), -1
    for a, b in document_sentences(text):
        n = len(q & set(re.findall(r"[a-z0-9%]{2,}", text[a:b].lower())))
        if n > score:
            best, score = (a, b), n
    return best


def locate(conn, version_id: str, start: int, end: int, cache: Optional[dict] = None) -> dict:
    """Clause, pages and line of characters [start, end) of a version (S2.1 maps). Empty when the
    version was read before S2.1."""
    cache = cache if cache is not None else {}
    if version_id not in cache:
        row = conn.execute("SELECT pages_json, lines_json FROM legal_version_text WHERE version_id=?", (version_id,)).fetchone()
        cache[version_id] = (json.loads(row[0]), json.loads(row[1])) if row else None
    clause = conn.execute("SELECT number FROM legal_clauses WHERE version_id=? AND char_start<=? AND ?<char_end "
                          "ORDER BY level DESC, char_start DESC LIMIT 1", (version_id, start, start)).fetchone()
    out = {"clause": clause[0] if clause else None}
    if not cache[version_id]:
        return out
    pages, lines = cache[version_id]
    last = max(start, end - 1)
    here = next((pg for pg in pages if pg["start"] <= start <= pg["end"]), None)
    there = next((pg for pg in pages if pg["start"] <= last <= pg["end"]), here)
    base = here["start"] if here else 0
    out["line"] = max(1, bisect.bisect_right(lines, start) - bisect.bisect_left(lines, base))
    if here:
        out["page_start"], out["page_end"] = here["number"], (there or here)["number"]
    return out


def _located(passages: list[dict], query: str = "") -> list[dict]:
    """Where each quote points (S2.1): the sentence that best answers the question (the focus), its
    clause, and the page and line it is on."""
    if not passages:
        return passages
    cache: dict = {}
    with get_conn() as conn:
        for p in passages:
            c = p["citation"]
            if c["char_start"] is None:
                continue
            a, b = _focus(p["text"], query)
            c["focus_start"], c["focus_end"] = c["char_start"] + a, c["char_start"] + b
            c.update(locate(conn, c["version_id"], c["focus_start"], c["focus_end"], cache))
    return passages


async def search(scope: AuthorizedScope, query: str, *, k: int = 8, modes: Iterable[str] = ALL_MODES,
                 versions: str = "signed", documents: Optional[list[str]] = None, floor: bool = True, wide: bool = False) -> dict:
    """floor=False keeps windows below LEGAL_RERANK_MIN_SCORE: for callers that check every value against its
    quote (drafting, s3.1d) rather than abstain, a short document's one window must not be dropped. wide=True gives each
    passage whole (up to WIDE_CHARS around its best window) for a model to read."""
    recheck(scope)                                    # raises TypeError for anything but a real scope
    modes = [m for m in modes if m in ALL_MODES] or list(ALL_MODES)
    idx = registry.for_matter(scope.matter_id)
    n = max(k, settings.LEGAL_SEARCH_CANDIDATES) * (5 if documents else 1)    # more candidates when only some documents count
    rankings = []
    if "dense" in modes:
        q = (await embedder.get().embed([query], kind="query")).astype(np.float32)
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        rankings.append(idx.search_dense(q / np.where(norm == 0, 1, norm), n))
    if "sparse" in modes:
        rankings.append(idx.search_sparse(query, n))
    if "keyword" in modes:
        rankings.append(_keyword(scope.matter_id, query, n))
    hits = _load(scope.matter_id, _fuse(rankings)[:n], scope.principal_id, versions if versions in ("signed", "current") else "signed",
                 documents)
    subs = _windowed(hits)
    scored = await rerank(query, subs, len(subs)) if subs else []
    best: dict[str, dict] = {}
    for h in scored:                                  # best window of each passage, in score order
        best.setdefault(h["chunk_id"], h)
    ranked = list(best.values())
    if floor and ranked and ranked[0].get("score_kind") == "ce_logit":
        ranked = [h for h in ranked if float(h["score"]) >= settings.LEGAL_RERANK_MIN_SCORE]
    top = ranked[:k]
    if wide:
        policy = versions if versions in ("signed", "current") else "signed"
        top = [_continued(scope, h, policy, documents) for h in top]
    recheck(scope)                                    # rights may have changed while this ran
    return {"passages": _located([_passage(_wide(h) if wide else h) for h in top], query), "abstained": not ranked}
