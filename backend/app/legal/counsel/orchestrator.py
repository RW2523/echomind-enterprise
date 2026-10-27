"""One chat turn, end to end (docs/echomind-legal/05 "Answer state machine").

    requested → authorized (scope) → intent → existing pipeline → verified components → validated envelope → stored

The turn never retrieves anything itself. It hands an `AuthorizedScope` (minted only by the policy engine for this
person and this matter) to the pipelines that already enforce isolation: research answers (quote-verified, abstaining)
and the conflict scan. Their results are turned into components from the trusted registry (schemas.py), validated as
a whole, stored, and only then sent as `ui.final`. Streamed text (`answer.delta`) is research's own paragraph text,
released only after each paragraph's quotes were checked.

Events, one JSON object per line: turn.started, turn.understood, task.progress, answer.delta, citation.verified,
ui.final, then exactly one of turn.completed / turn.abstained / turn.failed / scope_revoked.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional
from urllib.parse import quote as urlquote

from starlette.concurrency import run_in_threadpool

from ...core.config import settings
from .. import policy
from ..audit.record import record
from ..compare import scan
from ..research import followup
from ..research import routes as research
from ..retrieval import scope as scopemod
from . import briefing, citations, discrepancies, intent as intents, schemas, store

logger = logging.getLogger(__name__)

FAILED = "The answer could not be completed. Try again."
_STATE_EVENT = {"completed": "turn.completed", "abstained": "turn.abstained"}


class TurnFailed(Exception):
    pass


@dataclass
class Built:
    summary: str = ""
    components: list[dict] = field(default_factory=list)
    documents: dict[str, list[str]] = field(default_factory=dict)       # component id -> document ids it names
    actions: list[dict] = field(default_factory=list)
    status: str = "completed"
    understood: Optional[str] = None
    research_turn_id: Optional[str] = None
    stored: dict[str, dict] = field(default_factory=dict)              # component id -> props to store, when not the shown ones

    def add(self, type_: str, props: dict, document_ids: Optional[list[str]] = None, data_ref: bool = False,
            stored: Optional[dict] = None) -> str:
        cid = store.new_id("cmp")
        comp = {"id": cid, "type": type_, "props": props}
        if data_ref:
            comp["data_ref"] = cid
        self.components.append(comp)
        self.documents[cid] = list(document_ids or [])
        if stored is not None:
            self.stored[cid] = stored
        return cid


def event(name: str, data: dict) -> dict:
    return {"event": name, "data": data}


# ── Reading stored components back ───────────────────────────────────────────────
def materialize(principal: dict, comp: dict) -> dict:
    """A stored component as the person may see it now. An evidence comparison is resolved from its discrepancy (its
    state may have changed, or a document may no longer be visible); anything gone becomes `unavailable`."""
    if comp["type"] == "evidence_comparison":
        did = (comp["props"] or {}).get("discrepancy_id")
        row = discrepancies.get(principal, comp["matter_id"], did) if did else None
        if not row:
            return {"id": comp["id"], "type": "unavailable", "props": {"text": "This item is no longer available."}}
        return {"id": comp["id"], "type": "evidence_comparison", "data_ref": comp["id"], "version": row["version"],
                "props": discrepancies.render(principal, row)}
    return {"id": comp["id"], "type": comp["type"], "props": comp["props"]}


def envelope_for(principal: dict, turn: dict) -> dict:
    comps = [materialize(principal, c) for c in store.components(turn["component_ids"], principal["sub"])]
    return schemas.validate_envelope({"message_id": turn["id"], "matter_id": turn["matter_id"], "summary": turn.get("summary") or "",
                                      "components": comps, "suggested_actions": turn.get("actions") or []})


# ── Intents ──────────────────────────────────────────────────────────────────────
def _matter(matter_id: str) -> dict:
    from ...core.db import get_conn
    with get_conn() as conn:
        r = conn.execute("SELECT id, number, title FROM matters WHERE id=?", (matter_id,)).fetchone()
    return {"id": r[0], "number": r[1], "title": r[2]}


def _notes(cites: list[dict], conflicts: list[dict]) -> list[str]:
    notes = []
    for c in conflicts[:2]:
        other = c.get("other") or {}
        notes.append(f"Clause {c.get('clause')} of {c.get('document_title')} reads differently in version {other.get('number')} "
                     f"({other.get('status')}). This answer quotes version {(c.get('version') or {}).get('number')}.")
    if any(c.get("instructions") for c in cites):
        notes.append("A cited document contains text addressed to an AI. It was treated as document text only, never as instructions.")
    return notes[:3]


async def _ask(principal: dict, scope, conv: dict, turn: dict, question: str, document_ids: list[str], built: Built):
    hist = await run_in_threadpool(store.history, conv["id"], principal["sub"], turn["seq"])
    if hist:
        rewritten_q, rewritten = await followup.standalone(question, hist)
        if rewritten:
            question, built.understood = rewritten_q, rewritten_q
            yield event("turn.understood", {"question": rewritten_q})
    rturn = {"id": str(uuid.uuid4()), "org_id": principal["org_id"], "matter_id": conv["matter_id"], "owner_id": principal["sub"],
             "question": question, "auth": principal.get("auth"), "created_at": policy.now_iso()}
    built.research_turn_id = rturn["id"]
    done = None
    async for line in research._answer(scope, principal, question, rturn, "signed", document_ids or None, None):
        e = json.loads(line)
        kind = e.get("type")
        if kind == "sources":
            docs = {c["document_id"] for c in e.get("citations") or []}
            n = len(e.get("citations") or [])
            if n:
                yield event("task.progress", {"label": f"Reading {n} passage{'s' if n != 1 else ''} from {len(docs)} "
                                                       f"document{'s' if len(docs) != 1 else ''}", "done": 0, "total": n, "state": "running"})
        elif kind == "chunk":
            yield event("answer.delta", {"text": e.get("text") or ""})
        elif kind == "scope_revoked":
            raise scopemod.ScopeRevoked()
        elif kind == "error":
            raise TurnFailed(e.get("message") or FAILED)
        elif kind == "done":
            done = e
    if done is None:
        raise TurnFailed(FAILED)
    cites = await run_in_threadpool(citations.from_research, turn, done)
    for c in cites:
        if c["verified"]:
            yield event("citation.verified", {"citation_id": c["id"], "n": c["n"], "document_id": c["document_id"], "page": c.get("page")})
    state = done.get("status") if done.get("status") in ("answered", "partial", "unverified", "abstained") else "unverified"
    props = {"answer": done.get("answer") or "", "state": state,
             "citations": [{k: v for k, v in c.items() if k != "instructions"} for c in cites],
             "unsupported": [s[:500] for s in (done.get("unsupported") or [])][:10],
             "notes": _notes(cites, done.get("conflicts") or [])}
    if built.understood:
        props["understood"] = built.understood[:400]
    built.add("source_answer", props, [c["document_id"] for c in cites])
    if state == "abstained":
        built.status = "abstained"
        built.summary = "No source found in this matter's documents."
    else:
        verified = sum(1 for c in cites if c["verified"])
        built.summary = f"Answered from {len(cites)} source{'s' if len(cites) != 1 else ''}, {verified} with a verified quote."


def _comparison(principal: dict, built: Built, rows: list[dict], scanned: Optional[int]) -> None:
    top, rest = rows[0], rows[1:]
    props = discrepancies.render(principal, top)
    cid = built.add("evidence_comparison", props, [top["document_a_id"], top["document_b_id"]], data_ref=True,
                    stored={"discrepancy_id": top["id"]})
    built.components[-1]["version"] = top["version"]
    if rest:
        items = [{"id": r["id"], "kind": "discrepancy", "title": discrepancies.title(r),
                  "detail": f"{r['detail']['a'].get('document_title')} and {r['detail']['b'].get('document_title')}.",
                  "action": {"id": "review", "label": "Review", "action_type": "open_discrepancy", "target_id": r["id"]}}
                 for r in rest[:2]]
        built.add("priority_list", {"summary": f"{len(rest)} more to review.", "items": items, "more": max(0, len(rest) - 2)},
                  [d for r in rest[:2] for d in (r["document_a_id"], r["document_b_id"])])
    n = len(rows)
    checked = f" across {scanned} documents" if scanned else ""
    built.summary = f"{n} potential inconsistenc{'y' if n == 1 else 'ies'} found{checked}. Nothing is decided until a person reviews it."
    return cid


async def _compare(principal: dict, scope, conv: dict, question: str, focus: Optional[str], built: Built):
    mid, uid = conv["matter_id"], principal["sub"]
    if focus:
        row = await run_in_threadpool(discrepancies.get, principal, mid, focus)
        if not row:
            built.add("unavailable", {"text": "This item is no longer available."})
            built.summary = "This item is no longer available."
            return
        _comparison(principal, built, [row], None)
        built.summary = discrepancies.title(row)
        return
    docs = await run_in_threadpool(scan.documents, mid, uid, "signed")
    if len(docs) < 2:
        built.add("notice", {"tone": "info", "text": f"Checking for conflicting facts needs at least two documents you can read in this "
                                                     f"matter that have finished reading. Found {len(docs)}."})
        built.summary = "Not enough documents to compare."
        return
    sha = discrepancies.versions_sha(docs)
    if not await run_in_threadpool(discrepancies.already_scanned, mid, sha):
        found = await run_in_threadpool(scan.candidates, docs)
        total = len(found)
        label = f"Checking {total} possible difference{'s' if total != 1 else ''} across {len(docs)} documents"
        yield event("task.progress", {"label": label, "done": 0, "total": total, "state": "running"})
        checked = []
        async for c in scan.run_checks(found):
            await run_in_threadpool(scopemod.recheck, scope)
            checked.append(c)
            yield event("task.progress", {"label": label, "done": len(checked), "total": total, "state": "running"})
        ids = await run_in_threadpool(discrepancies.upsert, principal["org_id"], mid, checked, uid)
        await run_in_threadpool(discrepancies.record_scan, mid, sha, uid, len(docs), total, len(ids))
        yield event("task.progress", {"label": label, "done": total, "total": total, "state": "done"})
    rows = await run_in_threadpool(discrepancies.listed, principal, mid, discrepancies.OPEN, question)
    if not rows:
        reviewed = await run_in_threadpool(discrepancies.listed, principal, mid, discrepancies.RESOLVED)
        text = f"No conflicting facts found between the {len(docs)} documents you can read."
        if reviewed:
            text += f" {len(reviewed)} earlier item{'s were' if len(reviewed) != 1 else ' was'} already resolved."
        built.add("notice", {"tone": "info", "text": text})
        built.summary = text
        return
    _comparison(principal, built, rows, len(docs))


def _notice(built: Built, text: str, tone: str = "info", actions: Optional[list[dict]] = None, summary: Optional[str] = None) -> None:
    built.add("notice", {"tone": tone, "text": text})
    built.actions = actions or []
    built.summary = summary or text[:400]


def _other_matters(matter: dict, built: Built) -> None:
    _notice(built, f"This conversation uses only the documents in {matter['number']}. EchoMind never combines matters in one "
                   "answer. To work with another matter's documents, open that matter and start a conversation there.",
            summary=f"Answers here come only from {matter['number']}.")


def _draft(principal: dict, matter: dict, built: Built) -> None:
    base = f"/m/{urlquote(matter['number'], safe='')}"
    if policy.authorize(principal, "ai.draft", matter["id"]).allowed:
        _notice(built, "Drafting from the conversation is not available yet. Drafts are made from your firm's approved templates in "
                       "Drafting: every fact is cited to the matter's documents, anything unsupported is left as a gap, and a "
                       "second lawyer approves before anything leaves the matter.",
                actions=[{"id": "open_drafting", "label": "Open Drafting", "action_type": "open_page", "path": f"{base}/drafting"}],
                summary="Drafts are made in Drafting, from firm templates.")
    else:
        _notice(built, "Your role in this matter does not include drafting.", summary="Drafting is not part of your role here.")


def _deliberate(matter: dict, built: Built) -> None:
    base = f"/m/{urlquote(matter['number'], safe='')}"
    if settings.LEGAL_PANEL:
        _notice(built, "The review panel looks at one question from four perspectives, each citing the matter's documents, and a "
                       "partner approves its summary. It gives options for you to consider, never a decision.",
                actions=[{"id": "open_panel", "label": "Open the review panel", "action_type": "open_page", "path": f"{base}/panel"}],
                summary="Use the review panel to test an argument.")
    else:
        _notice(built, "Testing an argument from several perspectives (the review panel) is not turned on for this firm. You can "
                       "still ask what the documents say for and against a point.",
                summary="The review panel is not turned on.")


_SMALLTALK = ("Ask about this matter's documents, for example: what a clause says, which dates or amounts the documents "
              "disagree on, or “What matters today?”.")


# ── The turn ─────────────────────────────────────────────────────────────────────
async def run(principal: dict, conv: dict, text: str, client_request_id: Optional[str], document_ids: list[str],
              focus: Optional[str]) -> AsyncIterator[dict]:
    kind = "compare" if focus else intents.classify(text)
    turn = await run_in_threadpool(store.start_turn, conv, principal["sub"], text, kind, client_request_id)
    built = Built()
    final_status = "failed"
    yield event("turn.started", {"turn_id": turn["id"], "conversation_id": conv["id"], "matter_id": conv["matter_id"], "intent": kind})
    try:
        scope = await run_in_threadpool(scopemod.authorize, principal, conv["matter_id"])
        matter = await run_in_threadpool(_matter, conv["matter_id"])
        if kind == "ask_fact":
            async for e in _ask(principal, scope, conv, turn, text, document_ids, built):
                yield e
        elif kind == "compare":
            async for e in _compare(principal, scope, conv, text, focus, built):
                yield e
        elif kind == "inspect_case":
            props, named = await run_in_threadpool(briefing.build, principal, matter)
            built.add("priority_list", props, named)
            built.summary = props["summary"]
        elif kind == "other_matters":
            _other_matters(matter, built)
        elif kind == "draft":
            await run_in_threadpool(_draft, principal, matter, built)
        elif kind == "deliberate":
            _deliberate(matter, built)
        else:
            _notice(built, _SMALLTALK)
        await run_in_threadpool(scopemod.recheck, scope)
        envelope = schemas.validate_envelope({"message_id": turn["id"], "matter_id": conv["matter_id"], "summary": built.summary,
                                              "components": built.components, "suggested_actions": built.actions})
        for comp in envelope["components"]:
            await run_in_threadpool(store.save_component, {**comp, "props": built.stored.get(comp["id"], comp["props"])}, turn,
                                    built.documents.get(comp["id"], []))
        final_status = built.status
        await run_in_threadpool(store.finish_turn, turn["id"], final_status, built.summary, [c["id"] for c in envelope["components"]],
                                envelope.get("suggested_actions", []), built.understood, built.research_turn_id)
        await run_in_threadpool(store.touch, conv["id"], text)
        yield event("ui.final", envelope)
        yield event(_STATE_EVENT[final_status], {"turn_id": turn["id"], "status": final_status})
    except (scopemod.ScopeRevoked, scopemod.ScopeDenied):
        final_status = "scope_revoked"
        await run_in_threadpool(store.finish_turn, turn["id"], final_status, "", [], [])
        yield event("scope_revoked", {"turn_id": turn["id"]})
    except BaseException as e:            # includes cancellation: the turn is closed before the error propagates
        cancelled = not isinstance(e, Exception)
        final_status = "canceled" if cancelled else "failed"
        if not cancelled:
            logger.error("Counsel: turn failed (%s)", type(e).__name__)
        try:
            store.finish_turn(turn["id"], final_status, "", [], [])
        except Exception:
            pass
        if cancelled:
            raise
        yield event("turn.failed", {"turn_id": turn["id"], "message": str(e) if isinstance(e, TurnFailed) else FAILED})
    finally:
        try:
            record("counsel.answer", "deny" if final_status == "scope_revoked" else "error" if final_status in ("failed", "canceled") else "allow",
                   principal=principal, object_type="counsel_turn", object_id=turn["id"], matter_id=conv["matter_id"], reason=final_status,
                   details={"kind": kind, "status": final_status, "components": len(built.components),
                            "citations": sum(len(c["props"].get("citations", [])) for c in built.components if c["type"] == "source_answer")})
        except Exception as e:
            logger.error("Counsel: could not record the outcome (%s)", type(e).__name__)
