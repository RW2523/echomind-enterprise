"""Research notes: an answer sent for review (S2.7, RAG-01; decision D-043).

The person who asked can send an answer for review. The answer is fixed as it was given: the
question, the answer with its numbered citations, its status (answered, partly supported, not
verified, or not answered by the documents, with the reason), the evidence (each quote, where it
is, and whether it was found), the passages it cites, and any version conflicts. A second lawyer
approves exactly that version, and only an approved note is filed as a matter document.
"""
from __future__ import annotations

import re
from typing import Optional

from ...core.config import settings
from . import prompt

KINDS = ("answered", "partial", "unverified", "abstained")       # what can be sent; errors and revoked scopes cannot
_SOURCE_KEYS = ("n", "document_id", "document_title", "version_id", "version_number", "version_status", "signed", "clause",
                "page_start", "page_end", "line", "char_start", "char_end", "text", "instructions")
_STATUS = {"answered": "Answered from the documents", "partial": "Partly supported: some statements are not",
           "unverified": "Not verified: no quote was found in the sources", "abstained": "Not answered by the documents"}


def content(turn: dict, asked_by: str) -> dict:
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", turn.get("answer") or "")} | {e["n"] for e in turn.get("evidence") or []}
    return {"question": turn["question"], "answer": turn.get("answer") or "", "status": turn["status"],
            "abstention_reason": turn.get("abstention_reason"), "versions": turn.get("versions") or "signed",
            "unsupported": turn.get("unsupported") or [], "evidence": turn.get("evidence") or [],
            "sources": [{k: s.get(k) for k in _SOURCE_KEYS} for s in turn.get("citations") or [] if s["n"] in cited],
            "conflicts": turn.get("conflicts") or [], "asked_by": asked_by, "asked_at": turn["created_at"]}


def inputs(turn: dict) -> dict:
    return {"turn_id": turn["id"], "model": settings.LLM_MODEL, "prompt_version": prompt.VERSION,
            "passages": len(turn.get("citations") or [])}


def _where(e: dict) -> str:
    bits = [f"{e.get('document_title') or 'document'}, version {e.get('version_number')}" +
            (f" ({e['version_status']})" if e.get("version_status") else "")]
    if e.get("clause"):
        bits.append(f"clause {e['clause']}")
    if e.get("page_start"):
        bits.append(f"page {e['page_start']}")
    if e.get("line"):
        bits.append(f"line {e['line']}")
    return ", ".join(bits)


def as_text(c: dict, approval: Optional[dict] = None) -> str:
    """The note as a document: the answer with its evidence, and the approval that binds it."""
    out = [f"# Research note — {c['question']}", "", f"Question: {c['question']}", "",
           f"Answer ({'from signed versions' if c['versions'] == 'signed' else 'from the latest drafts'}):", "", c["answer"] or "—", "",
           f"Status: {_STATUS.get(c['status'], c['status'])}" + (f" ({c['abstention_reason'].replace('_', ' ')})" if c.get("abstention_reason") else ""), ""]
    if c["evidence"]:
        out += ["## Evidence", ""]
        for e in c["evidence"]:
            found = "found as written" if e["verification"] == "verbatim" else "found (spacing or quotes differ)" \
                if e["verification"] == "normalized" else "NOT found in the source"
            out.append(f"- [{e['n']}] “{e['quote']}” — {_where(e)} — {found}")
        out.append("")
    if c["unsupported"]:
        out += ["Statements without a source: " + " ".join(f"“{u}”" for u in c["unsupported"]), ""]
    if c["conflicts"]:
        out += ["## Versions that read differently", ""]
        out += [f"- {x.get('document_title') or 'document'}, clause {x.get('clause') or '—'}: version {x['other']['number']} "
                f"({x['other']['status']}) reads “{x['other']['text'][:200]}”" for x in c["conflicts"]]
        out.append("")
    out += [f"Asked by {c['asked_by']} on {c['asked_at'][:10]}.", ""]
    if approval:
        out += [f"Approved by {approval['by']} on {approval['at'][:10]}. Content hash {approval['content_sha256']}.", ""]
    return "\n".join(out).rstrip() + "\n"
