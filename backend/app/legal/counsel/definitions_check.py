"""The definitions check in the chat (2026-11-05 end-to-end run: "Run a definitions check on the Supply Agreement (DEMO): what
drafting problems does it have?" went to research, which read a few passages and named none of the three planted problems).

The check is the document page's own (definitions/check.py, S3.5: rules only, no model): defined terms, numbering and
cross-references of one version. It runs on the document the message names, or the one document chosen in the chat; it is
kept like a check run from the document's page, and its findings are shown as a cited answer: each line opens the exact
words it is about. When no single document is named, the chat asks which one instead of guessing.
"""
from __future__ import annotations

import re
from typing import Optional

from ...core.db import get_conn
from ..definitions import check as checkmod
from ..definitions import routes as defroutes
from ..documents import repo as doc_repo
from ..research import attribution
from ..retrieval import search as searchmod

MAX_FINDINGS = 8
_ORDER = {"high": 0, "medium": 1, "low": 2}
CAUTION = ("Found by rule from the document's text, not by the model. Nothing changes until a lawyer picks a fix on the "
           "document's definitions page and a second lawyer approves the redline.")


def choose(principal: dict, matter: dict, text: str, document_ids: list[str]) -> tuple[Optional[dict], list[dict]]:
    """The one document to check, and the documents the person may read (to ask which, when none is clear)."""
    docs = [d for d in doc_repo.list_documents(matter["id"], principal["sub"]) if (d.get("current_version") or {}).get("id")]
    by_id = {d["id"]: d for d in docs}
    picked = [i for i in document_ids if i in by_id]
    if len(picked) == 1:
        return by_id[picked[0]], docs
    said = " ".join((text or "").casefold().split())          # a full title, the longest one written ("Supply Agreement (DEMO)")
    whole = sorted((d for d in docs if d.get("title") and " ".join(d["title"].casefold().split()) in said),
                   key=lambda d: -len(d["title"]))
    if whole and (len(whole) == 1 or len(whole[0]["title"]) > len(whole[1]["title"])):
        return whole[0], docs
    named = attribution.referenced(text, {d["id"]: d.get("title") or "" for d in docs})
    if len(named) == 1:
        return by_id[next(iter(named))], docs
    contracts = [d for d in docs if d.get("kind") == "contract"]
    if not named and not picked and len(contracts) == 1:
        return contracts[0], docs
    return None, docs


def ask_which(matter: dict, docs: list[dict]) -> str:
    contracts = [d["title"] for d in docs if d.get("kind") == "contract"][:3] or [d["title"] for d in docs][:3]
    if not contracts:
        return f"There is no document you can read in {matter['number']} to check."
    names = "; ".join(f"“{t}”" for t in contracts)
    return (f"Which document should be checked? Name it in the question, for example “Run a definitions check on {contracts[0]}”. "
            f"Documents in {matter['number']}: {names}.")[:800]


def _clauses(places: list[dict]) -> str:
    where = list(dict.fromkeys(p["where"].replace("clause ", "") for p in places if p.get("where", "").startswith("clause ")))
    if not where:
        return ""
    return where[0] if len(where) == 1 else ", ".join(where[:-1]) + " and " + where[-1]


def build(principal: dict, matter: dict, document: dict) -> tuple[str, dict, dict]:
    """(answer text, a research-shaped `done` for counsel/citations.from_research, the stored check)."""
    check = defroutes.save(principal, matter["id"], document["id"])
    vid = check["version_id"]
    body = doc_repo.reader(vid) or {}
    text = body.get("text") or ""
    sentence_of = {d["start"]: d["sentence"] for d in checkmod.definitions(text)}
    findings = sorted(check["findings"], key=lambda f: (_ORDER.get(f.get("severity"), 3), f.get("start") or 0))
    title = document.get("title") or ""
    n_defs = check["summary"].get("definitions", 0)
    lines = [f"Definitions check of {title} (version {check.get('version_number') or 1}): {n_defs} defined "
             f"term{'s' if n_defs != 1 else ''} read, {len(findings)} finding{'s' if len(findings) != 1 else ''}. {CAUTION}"]
    spans: list[tuple[int, int, str]] = []                 # (start, end, the line that cites them)

    def cite(line: str, a: Optional[int], b: Optional[int]) -> None:
        if a is not None and b is not None and 0 <= a < b <= len(text):
            spans.append((a, b, line))
            lines.append(f"{line} [1].")
        else:
            lines.append(f"{line}.")

    for f in findings[:MAX_FINDINGS]:
        msg = f["message"].rstrip(".")
        if f["type"] == "duplicate" and f.get("places"):
            lines.append(f"{msg}.")
            for p in f["places"][:3]:
                a, b = sentence_of.get(p["start"], (p["start"], p["end"]))
                words = re.sub(r"^\d+(?:\.\d+)*\s+", "", " ".join(text[a:b].split()))[:300].rstrip(".")   # not its clause number
                cite(f"In {p.get('where') or 'the document'}: “{words}”", a, b)
        elif f["type"] == "undefined" and f.get("places"):
            where = _clauses(f["places"])
            first = f["places"][0]
            cite(f"{msg}; it is used in clause{'s' if ' and ' in where else ''} {where}" if where else msg, first["start"], first["end"])
        else:
            where = f.get("where")
            cite(f"{msg} ({where})" if where and where not in msg else msg, f.get("start"), f.get("end"))
    if len(findings) > MAX_FINDINGS:
        lines.append(f"{len(findings) - MAX_FINDINGS} more finding{'s are' if len(findings) - MAX_FINDINGS != 1 else ' is'} on the "
                     "document's definitions page.")
    if not findings:
        lines.append("No undefined, unused or doubly defined terms, numbering gaps or broken cross-references were found.")
    evidence, cache = [], {}
    with get_conn() as conn:
        for a, b, line in spans:
            at = searchmod.locate(conn, vid, a, b, cache)
            evidence.append({"n": 1, "version_id": vid, "char_start": a, "char_end": b, "verification": "verbatim",
                             "clause": at.get("clause"), "page_start": at.get("page_start"), "sentences": [line]})
    version = document.get("current_version") or {}
    done = {"answer": "\n\n".join(lines), "status": "answered", "abstained": False, "evidence": evidence,
            "citations": [{"n": 1, "document_id": document["id"], "document_title": title, "version_id": vid,
                           "version_number": check.get("version_number"), "version_status": version.get("status"),
                           "char_start": spans[0][0] if spans else None, "char_end": spans[0][1] if spans else None}] if spans else []}
    return done["answer"], done, check
