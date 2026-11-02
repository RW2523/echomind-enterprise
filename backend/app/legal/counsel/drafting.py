"""Drafting from the conversation (docs/echomind-legal/03 `guided_questions` and `document_preview`; CLAUDE.md "Document
draft: open right-side editor … never auto-file"; owner's request 2026-11-02).

"Draft a demand letter to Apex about the late delivery" asks at most three questions, each with a default: which of the
firm's approved templates, the draft's title, and what the written passages should say (the request itself). Sending
the answers writes the draft with the Drafting engine, exactly as the Drafting page does (drafting/engine.py: every
fact cited to the matter's documents, anything unsupported left as a [[NEEDS INPUT]] gap), and stores it as a draft
artifact, which puts it in the review queue: a second lawyer approves before anything leaves the matter. Nothing is
filed or sent from here.

The preview card stores only the draft's reference and always shows its newest version the person may see; it opens
beside the conversation, where fields and written passages can be corrected (each save a new version).
"""
from __future__ import annotations

import re
import uuid
from typing import Optional
from urllib.parse import quote as urlquote

from .. import artifacts, policy
from ..audit.record import record
from ..drafting import engine, templates

_HINTS = {"demand_letter": ("demand",), "nda": ("nda", "non-disclosure", "confidentiality agreement"),
          "engagement_letter": ("engagement",), "memo": ("memo", "memorandum"), "contract_review": ("review of", "contract review"),
          "board_minutes": ("board", "minutes")}
_LEAD = re.compile(r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+)?(?:draft|write|prepare|compose)\s+(?:me\s+)?(?:(?:an|a|the)\s+)?", re.I)
NOTE_MAX = 1000
TITLE_MAX = 200


def _title(request: str) -> str:
    t = _LEAD.sub("", " ".join(request.split())).rstrip(" .?!")
    return (t[:1].upper() + t[1:])[:TITLE_MAX] if t else "Draft"


def _guess(options: list[dict], request: str) -> Optional[str]:
    """The template whose kind and name share most words with the request ("letter before claim" is not an engagement
    letter just because both say "letter"); the first one when nothing matches."""
    low = request.lower()

    def score(t: dict) -> int:
        named = [w for w in re.findall(r"[a-z]+", t["name"].lower()) if len(w) > 3]
        return 2 * sum(1 for w in _HINTS.get(t["kind"], ()) if w in low) + sum(1 for w in named if w in low)
    best = max(options, key=score, default=None)
    return (best["id"] if best and score(best) else options[0]["id"]) if options else None


def form(principal: dict, matter: dict, request: str, document_ids: list[str]) -> Optional[tuple[dict, dict]]:
    """(guided_questions props, what to store with them), or None when the firm has no approved template."""
    options = [t for t in templates.listing(principal["org_id"], drafts=False) if t["versions"]][:12]
    if not options:
        return None
    using = (f"Only the {len(document_ids)} document{'s' if len(document_ids) != 1 else ''} you chose." if document_ids
             else f"All the documents of {matter['number']} you may read.")
    props = {"purpose": "draft", "title": "Draft from a firm template",
             "intro": f"Three questions, each already answered with a suggestion. Sources: {using}"[:400],
             "questions": [
                 {"id": "template", "label": "Template", "kind": "choice", "default": _guess(options, request),
                  "options": [{"value": t["id"], "label": t["name"][:200],
                               "detail": f"Approved version {t['versions'][0]['version']}"} for t in options]},
                 {"id": "title", "label": "Title of the draft", "kind": "text", "default": _title(request), "max_length": TITLE_MAX},
                 {"id": "note", "label": "What the written passages should say", "kind": "text", "default": " ".join(request.split())[:NOTE_MAX],
                  "required": False, "max_length": NOTE_MAX}],
             "submit_label": "Write the draft",
             "note": "Every fact is cited to the matter's documents; anything unsupported is left as a gap for you. A second lawyer "
                     "approves the draft before anything leaves the matter."}
    return props, {"purpose": "draft", "document_ids": list(document_ids), "templates": [t["id"] for t in options]}


async def run(principal: dict, matter: dict, values: dict, stored: dict) -> dict:
    """Write the draft (the Drafting engine, under `ai.draft`) and store it as a draft artifact; returns the artifact."""
    from starlette.concurrency import run_in_threadpool
    found = await run_in_threadpool(templates.approved, principal["org_id"], values["template"])
    if not found:
        raise LookupError("template")
    template, version = found
    docs = stored.get("document_ids") or None
    content, inputs = await engine.draft(principal, matter["id"], template, version, values["title"], docs, values.get("note") or None)
    a = await run_in_threadpool(artifacts.create, matter["id"], "legal_draft", "draft", str(uuid.uuid4()), content, inputs, principal["sub"], "model")
    record("draft.create", "allow", principal=principal, object_type="matter", object_id=matter["id"], matter_id=matter["id"],
           details={"artifact": a["id"], "template_version": version["version"], **content["summary"], "gaps": content["gaps"], "mode": "chat"})
    return a


def _status(a: dict) -> str:
    return "filed" if a.get("document_id") else a["status"] if a["status"] in ("draft", "approved", "superseded") else "draft"


def preview(principal: dict, matter_number: str, a: dict) -> dict:
    c = a["content"]
    last = a["decisions"][-1] if a["decisions"] else None
    if a.get("document_id"):
        review = "Approved and filed as a matter document."
    elif a["status"] == "approved":
        review = f"Approved by {last['by'] if last else 'a lawyer'}. Not filed: filing it is a separate step in Drafting."
    elif last and last["decision"] == "changes_requested":
        review = f"Sent back by {last['by']}: {(last.get('reason') or 'see the review')}"[:300]
    elif c.get("gaps"):
        review = f"{c['gaps']} gap{'s' if c['gaps'] != 1 else ''} to fill before a second lawyer can approve it."
    else:
        review = "In the review queue: a second lawyer approves it before anything leaves the matter."
    blocks = [b for b in c.get("blocks") or [] if (b.get("text") or "").strip()]
    sources = {(s.get("document_id"), s.get("char_start")) for item in (c.get("fields") or []) + blocks for s in item.get("sources") or []}
    can_edit = policy.authorize(principal, "ai.draft", a["matter_id"]).allowed and a["status"] == "draft" and not a.get("document_id")
    return {"artifact_id": a["id"], "title": (c.get("title") or "Draft")[:200], "template": ((c.get("template") or {}).get("name") or "")[:200],
            "version": a["version"], "status": _status(a), "gaps": int(c.get("gaps") or 0), "sources": len(sources),
            "fields": [{"label": f["label"][:120], "value": (f.get("value") or ("Needs input" if f["status"] == "needs_input" else ""))[:300],
                        "status": f["status"]} for f in (c.get("fields") or [])[:12]],
            "blocks": [{"type": b["type"] if b.get("type") in ("heading", "paragraph") else "paragraph", "text": b["text"][:1500],
                        "origin": b.get("origin") if b.get("origin") in ("template", "written", "edited") else "template"} for b in blocks[:8]],
            "more_blocks": max(0, len(blocks) - 8), "review": review[:300],
            "path": f"/m/{urlquote(matter_number, safe='')}/drafting/{a['id']}", "can_edit": can_edit}


def newest(principal: dict, matter_id: str, artifact_id: str) -> Optional[dict]:
    """The newest version of the draft this artifact is a version of, if the person may still see it."""
    a = artifacts.get(matter_id, artifact_id)
    if not a or a["kind"] != "legal_draft":
        return None
    latest = next((v for v in artifacts.versions(matter_id, a["subject_type"], a["subject_id"], "legal_draft")
                   if artifacts.visible_to(v, principal["sub"])), None)
    return latest
