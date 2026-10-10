"""Checking what a document cites (S3.4, DOC-03, spec test T07; decision D-048). No model is involved.

- **References** resolve to the exact clause, in the document itself or in another matter document it
  names that the person can open. A clause that does not exist there is `missing`; a document that is
  not there, or that the person cannot open, is `not_found` (the two read the same, so nothing is learnt
  about documents the person cannot see).
- **Quotations** are compared, word for word, with the passage they quote: `accurate`, `altered` (with
  the changed words), or `not_found`.
- **External authorities** are listed and always `unverified`: no licensed law library is connected
  (P-009) and nothing is looked up. A citation that cannot be real is flagged as malformed.

A stored check keeps where each item points (document, version, offsets), never the target's words:
the passage and the changed words are read again when the check is shown, for a person who can open
that document, so a check never shows a document to someone who cannot see it, and a target's purge
leaves nothing of it behind.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Optional

from ...core.db import get_conn
from ..documents import repo as documents
from ..redlines.word import ops as diff_ops
from . import parse

VERSION = "citations-s3.4b"
NOTICE = ("External authorities are listed but not verified: no licensed law library is connected (decision P-009) and nothing "
          "is looked up. Check each one in a licensed source before relying on it.")
_GENERIC = {"agreement", "contract", "lease", "deed", "letter", "policy", "terms", "memorandum", "licence", "license", "order form"}
_EXT = re.compile(r"\.(docx?|pdf|txt|md|rtf|eml|msg|html?)$", re.I)
_NOISE = re.compile(r"\((?:fictional|demo|draft|executed|final|signed)\)", re.I)


def _norm_title(s: str) -> str:
    s = _NOISE.sub(" ", _EXT.sub("", s or ""))
    s = re.sub(r"^(?:the|this)\s+", "", s.strip(), flags=re.I)
    return " ".join(re.sub(r"[_\-–—]+", " ", s).split()).casefold()


def _clauses(version_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT number, heading, level, char_start, char_end FROM legal_clauses WHERE version_id=? "
                            "ORDER BY char_start, level", (version_id,)).fetchall()
    return [{"number": n, "heading": h, "level": lv, "start": s, "end": e} for n, h, lv, s, e in rows]


def _version(document_id: str, matter_id: str, version_id: Optional[str] = None) -> Optional[tuple[str, str, int]]:
    with get_conn() as conn:
        if version_id:
            r = conn.execute("SELECT v.id, t.text, v.number FROM legal_document_versions v JOIN legal_version_text t ON t.version_id = v.id "
                             "WHERE v.id=? AND v.document_id=? AND v.matter_id=?", (version_id, document_id, matter_id)).fetchone()
        else:
            r = conn.execute("SELECT v.id, t.text, v.number FROM legal_documents d JOIN legal_document_versions v ON v.id = d.current_version_id "
                             "JOIN legal_version_text t ON t.version_id = v.id WHERE d.id=? AND d.matter_id=?", (document_id, matter_id)).fetchone()
    return (r[0], r[1], r[2]) if r else None


class _Doc:
    def __init__(self, document_id: str, title: str, version_id: str, text: str):
        self.id, self.title, self.version_id, self.text = document_id, title, version_id, text
        self.clauses = _clauses(version_id)

    def clause(self, keys: list[str], within: Optional[str] = None) -> Optional[dict]:
        pool = self.clauses
        if within:                                         # "paragraph 1 of Schedule 4": only inside that schedule
            box = next((c for c in self.clauses if c["number"].casefold() == within.casefold()), None)
            if not box:
                return None
            pool = [c for c in self.clauses if box["start"] < c["start"] < max(box["end"], box["start"] + 1)]
        for k in keys:
            hit = next((c for c in pool if c["number"].casefold() == k.casefold()), None)
            if hit:
                return hit
        return None


class Library:
    """The documents a person can open in a matter, read once per check."""

    def __init__(self, matter_id: str, uid: str, own: _Doc):
        self.matter_id, self.own = matter_id, own
        self.listed = [d for d in documents.list_documents(matter_id, uid) if d["id"] != own.id and d.get("current_version")]
        self._open: dict[str, Optional[_Doc]] = {}

    def named(self, name: Optional[str]) -> tuple[Optional[_Doc], bool]:
        """(document, is_self) for a name in the text; (None, False) when nothing the person can open is called that."""
        if not name:
            return self.own, True
        want = _norm_title(name)
        exact = [d for d in self.listed if _norm_title(d["title"]) == want]
        if not exact and (want in _GENERIC or want == _norm_title(self.own.title)):
            return self.own, True                          # "the Agreement" in an agreement is the agreement itself
        near = exact or [d for d in self.listed if re.search(rf"(?:^|\s){re.escape(want)}(?:\s|$)", _norm_title(d["title"]))]
        if not near:
            return None, False
        rank = {"executed": 0, "final": 1, "received": 2, "draft": 3}
        pick = sorted(near, key=lambda d: (rank.get(d["status"], 9), d.get("updated_at") or d.get("created_at") or ""))[0]
        if pick["id"] not in self._open:
            v = _version(pick["id"], self.matter_id)
            self._open[pick["id"]] = _Doc(pick["id"], pick["title"], v[0], v[1]) if v else None
        return self._open[pick["id"]], False


def _keys(f: parse.Found) -> list[str]:
    n, sub = f.data["number"], f.data.get("sub") or ""
    if f.data.get("attachment"):
        return [f"{f.sub.title()} {n}"]
    keys = [n + sub] if sub else []
    if f.sub == "article":
        keys += [f"Article {n}{sub}", f"Article {n}"]
    keys += [n, f"§ {n}"] if not sub else [f"§ {n}{sub}"]
    return keys


def _label(f: parse.Found) -> str:
    n = f.data["number"] + (f.data.get("sub") or "")
    out = f"{f.sub} {n}" if not f.data.get("attachment") else f"{f.sub.title()} {n}"
    return out + (f" in {f.data['within']}" if f.data.get("within") else "")


# ── Quotations ─────────────────────────────────────────────────────────────────────
def _words(text: str, start: int = 0, end: Optional[int] = None) -> list[tuple[str, int, int]]:
    end = len(text) if end is None else end
    return [(m.group(0), m.start() + start, m.end() + start) for m in re.finditer(r"\S+", text[start:end])]


def _w(token: str) -> str:
    return token.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


def _key(token: str) -> str:
    return re.sub(r"[^\w$€£%]", "", _w(token)).casefold()


def _best(quote: str, text: str, spans: list[tuple[int, int]], mask: Optional[tuple[int, int]] = None) -> Optional[dict]:
    """The passage in text (within spans) that best matches the quotation: offsets, and how closely it matches."""
    q = [_w(t) for t, _, _ in _words(quote)]
    qk = [_key(t) for t in q]
    if not q:
        return None
    best = None
    for a, b in spans:
        toks = [t for t in _words(text, a, b) if not (mask and mask[0] <= t[1] < mask[1])]
        if not toks:
            continue
        keys = [_key(t[0]) for t in toks]
        sm = SequenceMatcher(None, keys, qk, autojunk=False)
        blocks = sorted((b for b in sm.get_matching_blocks() if b.size), key=lambda b: -b.size)[:4]
        for anchor in sorted({b.a - b.b for b in blocks}):
            for lo in range(max(0, anchor - 3), min(len(toks), max(0, anchor) + 4)):
                for n in range(max(1, len(q) - 3), len(q) + 4):
                    hi = min(len(toks), lo + n)
                    if hi <= lo:
                        continue
                    m = sum(b.size for b in SequenceMatcher(None, keys[lo:hi], qk, autojunk=False).get_matching_blocks())
                    # fewest words changed (a changed word is one substitution), then the window closest to
                    # the quotation's length
                    cost = _edits(keys[lo:hi], qk) if len(q) <= 150 else max(hi - lo, len(q)) - m
                    rank = (cost, abs(hi - lo - len(q)), toks[lo][1])
                    if not best or rank < best["rank"]:
                        best = {"rank": rank, "ratio": 2 * m / (hi - lo + len(q)), "start": toks[lo][1], "end": toks[hi - 1][2]}
    return best


def _edits(a: list[str], b: list[str]) -> int:
    """Word-level edit distance."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _compare(quote: str, text: str, at: dict) -> tuple[str, list]:
    """accurate (the same words, quotation marks and letter case of a bracketed letter aside) or altered, with the diff."""
    passage = text[at["start"]:at["end"]]
    same = [_w(t).strip(".,;:") for t, _, _ in _words(passage)] == [re.sub(r"\[(\w)\]", r"\1", _w(t)).strip(".,;:") for t, _, _ in _words(quote)]
    if same or [_key(t) for t, _, _ in _words(passage)] == [_key(t) for t, _, _ in _words(quote)] and _bracketed_only(quote, passage):
        return "accurate", []
    return "altered", [list(o) for o in diff_ops(" ".join(t for t, _, _ in _words(passage)), " ".join(_w(t) for t, _, _ in _words(quote)))]


def _bracketed_only(quote: str, passage: str) -> bool:
    """Every difference is a letter put in square brackets (the convention for a changed capital)."""
    qs, ps = _words(quote), _words(passage)
    for (qt, _, _), (pt, _, _) in zip(qs, ps):
        if _w(qt).strip(".,;:") != _w(pt).strip(".,;:") and not re.match(r"^\[\w\]", qt):
            return False
    return True


def _segments(quote: str) -> list[str]:
    return [s.strip() for s in re.split(r"\s*(?:\.\s?\.\s?\.|…|\[\s?\.\.\.\s?\]|\[…\])\s*", quote) if len(s.strip().split()) >= 2]


# ── The check ──────────────────────────────────────────────────────────────────────
def _reference(f: parse.Found, lib: Library, paragraph_doc: Optional[str]) -> dict:
    doc, is_self = lib.named(f.data.get("document"))
    out = {"status": "not_found", "target": None, "note": None}
    if doc is None:
        out["note"] = f"No document you can open in this matter is called the {f.data['document']}."
        return out
    within = f.data.get("within")
    c = doc.clause(_keys(f), within)
    if c is None and is_self and not f.data.get("document") and paragraph_doc:
        other, other_self = lib.named(paragraph_doc)
        if other and not other_self and other.clause(_keys(f), within):
            doc, is_self, c = other, False, other.clause(_keys(f), within)
            out["note"] = f"Read as {_label(f)} of the {paragraph_doc}, named earlier in the paragraph."
    if c is None:
        cited = "This document" if is_self else f"The {f.data.get('document') or paragraph_doc}"
        out.update(status="missing", note=f"{cited} has no {_label(f)}.",
                   target={"document_id": doc.id, "version_id": doc.version_id, "clause": None, "start": 0, "end": 0})
        return out
    out.update(status="resolved", target={"document_id": doc.id, "version_id": doc.version_id, "clause": c["number"],
                                          "start": c["start"], "end": c["end"]})
    return out


def _quote(f: parse.Found, lib: Library) -> dict:
    ref: Optional[parse.Found] = f.data.get("reference")
    tried: list[tuple[_Doc, list[tuple[int, int]], Optional[dict]]] = []
    names = [n for n in (ref.data.get("document") if ref else None, f.data.get("document")) if n]
    docs = []
    for n in names or [None]:
        d, _ = lib.named(n)
        if d and d not in docs:
            docs.append(d)
    if not names and lib.own not in docs:
        docs.append(lib.own)
    if names and lib.own not in docs:
        docs.append(lib.own)                               # a quotation of this document, as a last resort
    if not docs:
        return {"status": "not_found", "target": None, "note": f"No document you can open in this matter is called the {names[0]}."}
    for d in docs:
        c = d.clause(_keys(ref), ref.data.get("within")) if ref else None
        tried.append((d, [(c["start"], c["end"])] if c else [(0, len(d.text))], c))
        if ref and c:
            tried.append((d, [(0, len(d.text))], None))   # the words may be quoted from a neighbouring clause
    found = None
    for d, spans, c in tried:
        mask = (f.start, f.end) if d is lib.own else None
        segs = _segments(f.text) or [f.text]
        hits = [_best(s, d.text, spans, mask) for s in segs]
        if any(h is None for h in hits):
            continue
        ratio = sum(h["ratio"] for h in hits) / len(hits)
        if ratio >= 0.6 and (not found or ratio > found[0] + 1e-9):
            found = (ratio, d, hits, segs)
        if found and found[0] >= 0.999:
            break
    if not found:
        return {"status": "not_found", "target": None, "note": "These words were not found in the documents it points to that you can open."}
    _, d, hits, segs = found
    status = "accurate"
    for s, h in zip(segs, hits):
        st, _ = _compare(s, d.text, h)
        if st == "altered":
            status = "altered"
    start, end = min(h["start"] for h in hits), max(h["end"] for h in hits)
    clause = next((c for c in sorted(d.clauses, key=lambda c: -c["level"]) if c["start"] <= start < max(c["end"], c["start"] + 1)), None)
    note = None
    if ref and clause and ref.data["number"] + (ref.data.get("sub") or "") != clause["number"] and _keys(ref)[0] != clause["number"]:
        note = f"The words are in clause {clause['number']}, not {_label(ref)}."
    return {"status": status, "target": {"document_id": d.id, "version_id": d.version_id, "clause": clause["number"] if clause else None,
                                         "start": start, "end": end, "segments": [[h["start"], h["end"]] for h in hits]}, "note": note,
            "quoted": segs}


def _paragraph_doc(text: str, at: int, short: dict[str, str]) -> Optional[str]:
    return parse.last_name(text, text.rfind("\n", 0, at) + 1, at, short)


def run(matter_id: str, document_id: str, uid: str, version_id: Optional[str] = None) -> tuple[list[dict], dict]:
    v = _version(document_id, matter_id, version_id)
    if not v:
        raise LookupError(document_id)
    title = (documents.get_document(matter_id, document_id) or {}).get("title") or ""
    own = _Doc(document_id, title, v[0], v[1])
    lib = Library(matter_id, uid, own)
    items, n = [], {"reference": 0, "quote": 0, "authority": 0}
    short = parse.aliases(own.text)
    for f in parse.read(own.text):
        n[f.kind] += 1
        it = {"id": f"{f.kind[0]}{n[f.kind]}", "kind": f.kind, "type": f.sub, "text": f.text, "start": f.start, "end": f.end,
              "malformed": f.malformed, "note": None, "target": None}
        if f.kind == "authority":
            it.update(status="unverified", fields={k: v for k, v in f.data.items()})   # the check's notice says why (P-009)
        elif f.kind == "reference":
            it.update(_reference(f, lib, _paragraph_doc(own.text, f.start, short)))
        else:
            if not f.data.get("document") and not (f.data.get("reference") and f.data["reference"].data.get("document")):
                f.data["document"] = _paragraph_doc(own.text, f.start, short)
            ref = f.data.get("reference")
            it["cited_as"] = ref.text if ref else None
            it.update(_quote(f, lib))
        items.append(it)
    return items, {"version_id": own.version_id, "version_number": v[2], "engine": VERSION, "external_lookup": False}


def summary(items: list[dict]) -> dict:
    by = lambda k, s=None: sum(1 for i in items if i["kind"] == k and (s is None or i["status"] == s))  # noqa: E731
    return {"references": by("reference"), "resolved": by("reference", "resolved"), "missing": by("reference", "missing"),
            "quotes": by("quote"), "accurate": by("quote", "accurate"), "altered": by("quote", "altered"),
            "not_found": by("reference", "not_found") + by("quote", "not_found"), "authorities": by("authority"),
            "malformed": sum(1 for i in items if i["kind"] == "authority" and i["malformed"])}


# ── Showing a check to a person ──────────────────────────────────────────────────────
def present(matter_id: str, items: list[dict], uid: str) -> list[dict]:
    """Items as this person may see them: a target they cannot open (or that is gone) reads as not found."""
    seen: dict[str, Optional[tuple[str, str]]] = {}
    out = []
    for it in items:
        it = dict(it)
        t = it.get("target")
        if t:
            did = t["document_id"]
            if did not in seen:
                doc = documents.get_document(matter_id, did, uid)
                seen[did] = (doc["title"], doc["id"]) if doc else None
            text = None
            if seen[did]:
                with get_conn() as conn:
                    r = conn.execute("SELECT text FROM legal_version_text WHERE version_id=?", (t["version_id"],)).fetchone()
                text = r[0] if r else None
            if text is None:
                it.update(status="not_found", target=None, diff=[], note=(
                    "No document you can open in this matter is called that." if it["kind"] == "reference" else
                    "These words were not found in the documents it points to that you can open."))
            else:
                passage = text[t["start"]:t["end"]]
                it["target"] = {**t, "title": seen[did][0], "passage": passage[:1500] if it["status"] != "missing" else ""}
                if it["kind"] == "quote" and it["status"] == "altered":
                    segs = t.get("segments") or [[t["start"], t["end"]]]
                    it["diff"] = []
                    for (a, b), q in zip(segs, it.get("quoted") or [it["text"]]):
                        if it["diff"]:
                            it["diff"].append(["eq", " … "])
                        it["diff"] += [list(o) for o in diff_ops(" ".join(text[a:b].split()), " ".join(_w(x) for x in q.split()))]
        it.setdefault("diff", [])
        it.pop("quoted", None)
        out.append(it)
    return out
