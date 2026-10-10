"""Redlines (S3.3, DOC-02b; decision D-047).

A redline revises the clauses a playbook check found outside the firm's position:
- **The wording is the firm's.** Each revision is the approved standard (or the fallback the lawyer
  picks) for that topic. The model may only fit it to the contract: use the contract's defined terms
  and party names. The fit is kept only when every word it adds appears in the clause it replaces,
  and its figures, periods, percentages and laws are the firm's (the S2.3 fact reader). Otherwise
  the firm's wording is kept as it is, and the redline says so.
- The clause's own number is kept in front of the revision.
- Each change carries the reason from the check, and a word-level diff for the screen.
- A redline is an AI artifact (S2.7): a lawyer edits it (a new version), a second lawyer approves it,
  and it is filed as the next version of the document — a Word file with tracked changes
  (`word.apply`), checked before release (`word.verify`).

Text addressed to an AI is withheld from the model (T08).
"""
from __future__ import annotations

import re
from typing import Optional

from ...core.config import settings
from ...core.db import get_conn
from ..compare import facts as factsmod
from ..research import llm as llm_mod
from ..research import prompt as research_prompt
from . import word

VERSION = "redline-s3.3"
_NUMBER = re.compile(r"^\s*(\(?(?:\d+(?:\.\d+)*|[A-Za-z]|[ivxlcdm]+)[.)]?)\s+")
_KINDS = ("money", "period", "percent", "law")
FIT_SYSTEM = (
    "You adjust a law firm's standard wording so that it uses a contract's own defined terms. Keep every obligation, figure, "
    "period, condition and law of the firm's wording exactly as it is. Only replace a generic word with one of the contract's "
    "defined terms where it means the same thing (for example \"the agreement\" becomes \"this Agreement\", \"the supplier\" "
    "becomes \"the Provider\"). If nothing needs replacing, answer with the firm's wording unchanged. Answer with the wording "
    "only: one paragraph, no quotation marks, no comment."
)
_SENTENCE_START = {"The", "This", "These", "Each", "Either", "Neither", "No", "Any", "All", "If", "In", "On", "For", "Subject", "Without", "A", "An"}


def _figures(text: str) -> dict[str, set]:
    out: dict[str, set] = {}
    for f in factsmod.extract(text, [(0, len(text))]):
        if f.kind in _KINDS:
            out.setdefault(f.kind, set()).add(f.value)
    return out


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:['’][a-z]+)?", (text or "").casefold())


def fit_ok(firm: str, fitted: str, clause: str) -> Optional[str]:
    """None when the fitted wording only borrows the clause's words and keeps the firm's figures;
    otherwise why it was not kept."""
    if not fitted.strip():
        return "the model gave no wording"
    added = set(_words(fitted)) - set(_words(firm))
    foreign = sorted(w for w in added if w not in set(_words(clause)))
    if foreign:
        return "it added words that are neither the firm's nor the contract's (" + ", ".join(foreign[:5]) + ")"
    if _figures(fitted) != _figures(firm):
        return "it changed a figure, period or law"
    return None


def defined_terms(clause: str, document_terms: Optional[list[str]] = None) -> list[str]:
    """The contract's own names for things: its defined terms (S2.1), and capitalised words the clause
    uses mid-sentence ("this Agreement", "the Provider"). The model sees these, never the clause's
    substance, so it has nothing to copy back."""
    found = list(document_terms or [])
    for m in re.finditer(r"(?<=[a-z,;] )([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", clause or ""):
        word_ = m.group(1)
        if word_ not in _SENTENCE_START and word_ not in found:
            found.append(word_)
    return found[:25]


async def fit(firm: str, clause: str, document_terms: Optional[list[str]] = None) -> tuple[str, bool, Optional[str]]:
    """(wording, fitted, note)."""
    terms = defined_terms(clause, document_terms)
    if not terms:
        return firm, False, None
    user = f"FIRM WORDING: {firm}\n\nCONTRACT'S DEFINED TERMS: {', '.join(terms)}"
    try:
        raw = await llm_mod.complete([{"role": "system", "content": FIT_SYSTEM}, {"role": "user", "content": user}],
                                     temperature=0.0, max_tokens=400)
    except Exception:                                     # noqa: BLE001 — the firm's wording stands
        return firm, False, "The firm's wording was kept: the model could not be reached."
    fitted = " ".join(research_prompt.clean(raw or "").strip().strip('"“”').split())
    if fitted == firm:
        return firm, False, None
    why = fit_ok(firm, fitted, clause)
    if why:
        return firm, False, f"The firm's wording was kept: {why}."
    return fitted, True, None


def own_text(number: Optional[str], text: str) -> str:
    """The clause's own paragraphs: a span read from the document can run into the next heading or
    clause ("21.1 …\n\n## 22. Notices"), which a revision must never delete. Stops at a heading, a table
    row, or a paragraph numbered as another clause (sub-items such as "(a)" or "21.1.1" stay)."""
    kept = []
    for i, part in enumerate(word.parts(text)):
        if i and (part.lstrip().startswith("#") or not re.search(r"[A-Za-z0-9]", part) or " | " in part):
            break                                          # a heading, a mark, or a table: not this clause's wording
        m = re.match(r"^\s*(\d+(?:\.\d+)*)[.)]?\s", part)
        if i and m and number and not m.group(1).startswith(number + "."):
            break
        kept.append(part)
    return "\n\n".join(kept)


def _number(clause_text: str) -> str:
    m = _NUMBER.match(clause_text)
    return m.group(1) + " " if m else ""


def _document(matter_id: str, document_id: str, version_id: str) -> dict:
    with get_conn() as conn:
        r = conn.execute("SELECT d.title, v.number, v.filename, v.mime, t.text FROM legal_documents d JOIN legal_document_versions v ON v.document_id = d.id "
                         "JOIN legal_version_text t ON t.version_id = v.id WHERE d.id=? AND v.id=? AND d.matter_id=?",
                         (document_id, version_id, matter_id)).fetchone()
    if not r:
        raise LookupError(document_id)
    return {"id": document_id, "title": r[0], "version_id": version_id, "version_number": r[1], "filename": r[2], "mime": r[3], "text": r[4]}


def ops_of(original: str, revised: str) -> list[list[str]]:
    return [[op, piece] for op, piece in word.ops(original, revised)]


async def propose(check: dict, positions: dict[str, dict], playbook_name: str, choices: dict[str, str],
                  topics: Optional[list[str]], matter_id: str) -> tuple[dict, dict]:
    doc = _document(matter_id, check["document_id"], check["version_id"])
    text = doc.pop("text")
    with get_conn() as conn:
        doc_terms = [r[0] for r in conn.execute("SELECT DISTINCT term FROM legal_defined_terms WHERE version_id=? LIMIT 40", (doc["version_id"],))]
    changes = []
    for line in check["results"]:
        wanted = line["topic"] in topics if topics else line["classification"] == "outside"
        if not wanted or not line.get("clause") or line["topic"] not in positions:
            continue
        pos = positions[line["topic"]]
        choice = (choices or {}).get(line["topic"], "standard")
        m = re.fullmatch(r"fallback\s*(\d+)", choice)
        if m and 1 <= int(m.group(1)) <= len(pos["fallbacks"]):
            firm, frm = pos["fallbacks"][int(m.group(1)) - 1]["text"], {"kind": "fallback", "n": int(m.group(1))}
        else:
            firm, frm = pos["standard"], {"kind": "standard", "n": None}
        c = line["clause"]
        original = own_text(c.get("number"), text[c["start"]:c["end"]])
        wording, fitted, note = await fit(firm, original, doc_terms)
        revised = _number(original) + wording
        if word.fold(revised) == word.fold(original):     # already the firm's wording: nothing to change
            continue
        reason = line.get("reason") or ""
        comment = (f"{line['title']}: revised to the firm's {'standard' if frm['kind'] == 'standard' else 'fallback ' + str(frm['n'])} "
                   f"position ({playbook_name}, version {pos['version']}). {reason}").strip()
        changes.append({"id": f"c{len(changes) + 1}", "topic": line["topic"], "title": line["title"], "clause": c, "original": original,
                        "revised": revised, "ops": ops_of(original, revised), "reason": reason, "comment": comment, "fitted": fitted,
                        "fit_note": note, "from": {**frm, "position_id": pos["id"], "position_version": pos["version"], "text": firm}})
    content = {"title": f"Redline — {doc['title']}", "document": doc, "check_id": check["id"], "playbook": check["playbook"],
               "changes": changes, "summary": {"changes": len(changes), "fitted": sum(c["fitted"] for c in changes)}}
    inputs = {"check_id": check["id"], "model": settings.LLM_MODEL, "prompt_version": VERSION,
              "positions": {k: {"id": v["id"], "version": v["version"]} for k, v in positions.items()}}
    return content, inputs


def source(matter_id: str, content: dict) -> tuple[Optional[bytes], str]:
    """The original version's Word file (or None for text and PDF) and its text."""
    from ..documents import blobs
    with get_conn() as conn:
        r = conn.execute("SELECT v.filename, v.wrapped_key, t.text FROM legal_document_versions v JOIN legal_version_text t ON t.version_id = v.id "
                         "WHERE v.id=? AND v.matter_id=?", (content["document"]["version_id"], matter_id)).fetchone()
    if not r:
        raise LookupError(content["document"]["version_id"])
    filename, wrapped, text = r
    data = blobs.load(matter_id, content["document"]["version_id"], wrapped) if filename.lower().endswith(".docx") else None
    return data, text


class Unverified(Exception):
    def __init__(self, checks: dict):
        super().__init__("the Word file failed its checks")
        self.checks = checks


def to_word(matter_id: str, artifact: dict, author: str) -> tuple[bytes, dict]:
    """The redline as tracked changes in the original Word file (or a Word file of the text), checked."""
    data, text = source(matter_id, artifact["content"])
    changes = [{"id": c["id"], "original": c["original"], "revised": c["revised"], "comment": c["comment"]} for c in artifact["content"]["changes"]]
    when = artifact["created_at"][:19] + "Z"
    out, plan = word.apply(data, text, changes, author, when)
    result = word.verify(data or word.plain_docx(text), out, plan)
    if not result["ok"]:
        raise Unverified(result["checks"])
    return out, {"missed": plan["missed"], "checks": result["checks"]}
