"""One chat turn, end to end (docs/echomind-legal/05 "Answer state machine").

    requested → authorized (scope) → intent → existing pipeline → verified components → validated envelope → stored

The turn never retrieves anything itself. It hands an `AuthorizedScope` (minted only by the policy engine for this
person and this matter) to the pipelines that already enforce isolation: research answers (quote-verified, abstaining)
and the conflict scan. Their results are turned into components from the trusted registry (schemas.py), validated as
a whole, stored, and only then sent as `ui.final`. Streamed text (`answer.delta`) is research's own paragraph text,
released only after each paragraph's quotes were checked.

Events, one JSON object per line: turn.started, turn.understood, task.progress, answer.delta, citation.verified,
ui.partial (a validated component shown as soon as it is ready), ui.final, then exactly one of turn.completed /
turn.abstained / turn.failed / scope_revoked.

A conversation may cover several of the person's matters (2026-10-28). Each matter gets its own `AuthorizedScope` and is
re-checked at the end; losing any one stops the turn. Research searches each and labels every source with its matter; the
briefing and the chronology are built per matter and put together with labels; the conflict scan runs inside each matter
only (documents of different matters are never compared with each other); a table can show the matters side by side.
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
from . import briefing, cases, chronology, citations, discrepancies, intent as intents, schemas, store

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
    matter_of: dict[str, str] = field(default_factory=dict)             # component id -> its matter, when not the conversation's

    def add(self, type_: str, props: dict, document_ids: Optional[list[str]] = None, data_ref: bool = False,
            stored: Optional[dict] = None, matter_id: Optional[str] = None) -> str:
        cid = store.new_id("cmp")
        comp = {"id": cid, "type": type_, "props": props}
        if data_ref:
            comp["data_ref"] = cid
        if matter_id:
            comp["matter_id"] = matter_id
            self.matter_of[cid] = matter_id
        self.components.append(comp)
        self.documents[cid] = list(document_ids or [])
        if stored is not None:
            self.stored[cid] = stored
        return cid


def event(name: str, data: dict) -> dict:
    return {"event": name, "data": data}


# ── Reading stored components back ───────────────────────────────────────────────
WITHHELD = "This item quoted a document you can no longer open."


def materialize(principal: dict, comp: dict, conv_matter: Optional[str] = None) -> dict:
    """A stored component as the person may see it now. An evidence comparison is resolved from its discrepancy (its
    state may have changed, or a document may no longer be visible); anything gone becomes `unavailable`. Anything else that
    drew on a document the person may no longer read (its access list narrowed after the answer) is withheld whole: its
    wording came from that document (commercial-readiness review F3, 2026-10-28)."""
    from ..documents import repo as doc_repo
    if comp["type"] != "evidence_comparison" and doc_repo.withheld_from(comp.get("document_ids") or [], principal["sub"]):
        out = {"id": comp["id"], "type": "unavailable", "props": {"text": WITHHELD}}
        if conv_matter and comp["matter_id"] != conv_matter:
            out["matter_id"] = comp["matter_id"]
        return out
    if comp["type"] == "evidence_comparison":
        did = (comp["props"] or {}).get("discrepancy_id")
        row = discrepancies.get(principal, comp["matter_id"], did) if did else None
        if not row:
            return {"id": comp["id"], "type": "unavailable", "props": {"text": "This item is no longer available."}}
        out = {"id": comp["id"], "type": "evidence_comparison", "data_ref": comp["id"], "version": row["version"],
               "props": discrepancies.render(principal, row)}
    else:
        out = {"id": comp["id"], "type": comp["type"], "props": comp["props"]}
    if conv_matter and comp["matter_id"] != conv_matter:
        out["matter_id"] = comp["matter_id"]
    return out


def _matter_refs(matter_ids: list[str]) -> list[dict]:
    return [_matter(m) for m in matter_ids] if len(matter_ids) > 1 else []


def envelope_for(principal: dict, turn: dict, conv: Optional[dict] = None) -> dict:
    comps = [materialize(principal, c, turn["matter_id"]) for c in store.components(turn["component_ids"], principal["sub"])]
    return schemas.validate_envelope({"message_id": turn["id"], "matter_id": turn["matter_id"], "summary": turn.get("summary") or "",
                                      "matters": _matter_refs((conv or {}).get("matter_ids") or []),
                                      "components": comps, "suggested_actions": turn.get("actions") or []})


# ── Intents ──────────────────────────────────────────────────────────────────────
def _matter(matter_id: str) -> dict:
    from ...core.db import get_conn
    with get_conn() as conn:
        r = conn.execute("SELECT id, number, title FROM matters WHERE id=?", (matter_id,)).fetchone()
    return {"id": r[0], "number": r[1], "title": r[2]}


def _amendment_notes(changed: list[dict], matter_number: str) -> tuple[list[str], list[dict]]:
    """What a signed (or unsigned) amendment changes in a clause the answer relies on, in the documents' own words, and a way to
    open the amending clause. Deterministic: research found the link in the documents, not the model."""
    notes, actions = [], []
    for a in changed[:2]:
        by = a["by"]
        eff = by.get("effective") or {}
        when = f' It says: “{eff["words"]}”' if eff.get("words") else " It does not say from when it applies."
        if by.get("signed"):
            text = (f"Clause {a['clause']} of {a['document_title']} was {a['how']} by clause {by.get('clause') or '?'} of "
                    f"{by['document_title']} (signed).{when}")
            if a["cited"] and not a["amendment_cited"]:
                text += " This answer relies on the earlier wording: read the amendment before relying on it."
        else:
            text = (f"Clause {a['clause']} of {a['document_title']} would be {a['how']} by {by['document_title']}, "
                    f"which is not signed, so the earlier wording still applies.")
        notes.append(text[:400])
        if not actions and by.get("char_start") is not None:
            actions.append({"id": "open_amendment", "label": "Open the amendment", "action_type": "open_page",
                            "path": citations.reader_path(a.get("matter_number") or matter_number, by["document_id"], by["version_id"],
                                                          by["char_start"], by["char_end"])})
    return notes, actions


def _notes(cites: list[dict], conflicts: list[dict]) -> list[str]:
    notes = []
    for c in conflicts[:2]:
        other = c.get("other") or {}
        notes.append(f"Clause {c.get('clause')} of {c.get('document_title')} reads differently in version {other.get('number')} "
                     f"({other.get('status')}). This answer quotes version {(c.get('version') or {}).get('number')}.")
    if any(c.get("instructions") for c in cites):
        notes.append("A cited document contains text addressed to an AI. It was treated as document text only, never as instructions.")
    return notes[:3]


async def _ask(principal: dict, scope, conv: dict, turn: dict, question: str, document_ids: list[str], built: Built,
               extra_scopes: tuple = ()):
    hist = await run_in_threadpool(store.history, conv["id"], principal["sub"], turn["seq"])
    if hist:
        rewritten_q, rewritten = await followup.standalone(question, hist)
        if rewritten:
            question, built.understood = rewritten_q, rewritten_q
            yield event("turn.understood", {"question": rewritten_q})
    rturn = {"id": str(uuid.uuid4()), "org_id": principal["org_id"], "matter_id": conv["matter_id"], "owner_id": principal["sub"],
             "question": question, "auth": principal.get("auth"), "created_at": policy.now_iso()}
    built.research_turn_id = None if extra_scopes else rturn["id"]          # several matters: kept in the conversation only
    done = None
    async for line in research._answer(scope, principal, question, rturn, "signed", document_ids or None, None, extra_scopes):
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
    changed = done.get("amendments") or []
    amend_notes, amend_actions = await run_in_threadpool(_amendment_notes, changed, (await run_in_threadpool(_matter, conv["matter_id"]))["number"])
    props = {"answer": done.get("answer") or "", "state": state,
             "citations": [{k: v for k, v in c.items() if k != "instructions"} for c in cites],
             "unsupported": [s[:500] for s in (done.get("unsupported") or [])][:10],
             "notes": (([OTHER_DOCUMENTS] if done.get("abstention_reason") == "other_documents" else [])
                       + amend_notes + _notes(cites, done.get("conflicts") or []))[:3]}
    if built.understood:
        props["understood"] = built.understood[:400]
    built.add("source_answer", props, [c["document_id"] for c in cites] + [a["by"]["document_id"] for a in changed])
    built.actions = amend_actions
    if state == "abstained":
        built.status = "abstained"
        built.summary = "No source found in these matters' documents." if extra_scopes else "No source found in this matter's documents."
    else:
        verified = sum(1 for c in cites if c["verified"])
        built.summary = f"Answered from {len(cites)} source{'s' if len(cites) != 1 else ''}, {verified} with a verified quote."


OTHER_DOCUMENTS = ("The passages found are from other documents than the one asked about, so they are not given as its answer: "
                   "that document does not say this.")


def _comparison(principal: dict, built: Built, rows: list[dict], scanned: Optional[int]) -> None:
    """The most important potential inconsistency as a comparison, the next ones listed. Rows of a conversation about
    several matters carry `_matter` ({id, number}): the comparison belongs to its matter and each item names one."""
    top, rest = rows[0], rows[1:]
    props = discrepancies.render(principal, top)
    cid = built.add("evidence_comparison", props, [top["document_a_id"], top["document_b_id"]], data_ref=True,
                    stored={"discrepancy_id": top["id"]}, matter_id=(top.get("_matter") or {}).get("id"))
    built.components[-1]["version"] = top["version"]
    if rest:
        items = [{"id": r["id"], "kind": "discrepancy", "title": discrepancies.title(r),
                  "detail": f"{r['detail']['a'].get('document_title')} and {r['detail']['b'].get('document_title')}.",
                  "action": {"id": "review", "label": "Review", "action_type": "open_discrepancy", "target_id": r["id"]},
                  **({"matter": r["_matter"]["number"]} if r.get("_matter") else {})}
                 for r in rest[:2]]
        built.add("priority_list", {"summary": f"{len(rest)} more to review.", "items": items, "more": max(0, len(rest) - 2)},
                  [d for r in rest[:2] for d in (r["document_a_id"], r["document_b_id"])])
    n = len(rows)
    checked = f" across {scanned} documents" if scanned else ""
    built.summary = f"{n} potential inconsistenc{'y' if n == 1 else 'ies'} found{checked}. Nothing is decided until a person reviews it."
    return cid


async def _compare(principal: dict, scope, conv: dict, question: str, focus: Optional[str], built: Built, label: str = "",
                   collect: Optional[list] = None):
    """The conflict scan inside one matter. With `collect`, the open rows are added to it (a conversation about several
    matters puts them together) instead of being shown here."""
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
        if collect is not None:
            return
        built.add("notice", {"tone": "info", "text": f"Checking for conflicting facts needs at least two documents you can read in this "
                                                     f"matter that have finished reading. Found {len(docs)}."})
        built.summary = "Not enough documents to compare."
        return
    sha = discrepancies.versions_sha(docs)
    if not await run_in_threadpool(discrepancies.already_scanned, mid, sha):
        found = await run_in_threadpool(scan.candidates, docs)
        total = len(found)
        label = f"{label}Checking {total} possible difference{'s' if total != 1 else ''} across {len(docs)} documents"
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
    if collect is not None:
        collect.extend(rows)
        return
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


def _other_matters(matters: list[dict], built: Built) -> None:
    if len(matters) == 1:
        _notice(built, f"This conversation uses only the documents in {matters[0]['number']}. To work across cases, start a new "
                       "conversation and choose up to five of your matters in the question box; every answer then says which "
                       "matter each source comes from.",
                summary=f"Answers here come only from {matters[0]['number']}.")
    else:
        names = ", ".join(m["number"] for m in matters)
        _notice(built, f"This conversation covers {names}. Answers use only these matters' documents you may read and say which "
                       "matter each source comes from. To add another matter, start a new conversation with it.",
                summary=f"Answers here come from {names}.")


def _label_events(props: dict, number: str) -> dict:
    for ev in props["events"]:
        ev["matter"] = number
    return props


def _chronology_several(parts: list[tuple[dict, dict, list[str]]]) -> tuple[dict, list[str]]:
    """Each matter's chronology (matter, props, document ids), put together in date order, every event naming its matter."""
    dated, undated, named, more = [], [], [], 0
    matters = [m for m, _, _ in parts]
    for m, props, ids in parts:
        _label_events(props, m["number"])
        dated += [e for e in props["events"] if e["certainty"] != "undated"]
        undated += [e for e in props["events"] if e["certainty"] == "undated"]
        named += ids
        more += props["more"]
    dated.sort(key=lambda e: (e["date"], e["matter"], e["source"]["document_title"]))
    events = dated + undated
    shown = events[:chronology.MAX_EVENTS]
    for i, ev in enumerate(shown):
        ev["id"] = f"ev{i + 1}"
    docs = {e["source"]["document_id"] for e in shown}
    summary = (f"{len(dated)} dated event{'s' if len(dated) != 1 else ''} from {len(docs)} document{'s' if len(docs) != 1 else ''} "
               f"in {len(matters)} matters" + (f", and {len(undated)} whose date is not stated" if undated else "") + ".")
    return ({"summary": summary[:400], "events": shown, "more": more + max(0, len(events) - len(shown)),
             "caution": "Dates are shown as each matter's documents state them; nothing is inferred and nothing is combined across "
                        "matters. Recollections and uncertain or partial dates are marked."}, sorted(set(named)))


def _briefing_several(principal: dict, matters: list[dict]) -> tuple[dict, list[str]]:
    """Each matter's briefing; the most important item of each first, then the next, at most three shown."""
    per, named, total = [], [], 0
    for m in matters:
        props, ids = briefing.build(principal, m)
        items = [{**it, "matter": m["number"]} for it in props["items"]]
        per.append(items)
        named += ids
        total += len(props["items"]) + props.get("more", 0)
    ordered = [it for rank in range(schemas.MAX_PRIORITY_ITEMS) for items in per if rank < len(items) for it in [items[rank]]]
    shown = ordered[:schemas.MAX_PRIORITY_ITEMS]
    n = len(matters)
    if not shown:
        return {"summary": f"Nothing needs your attention in these {n} matters.", "items": [], "more": 0,
                "empty": f"Nothing needs your attention in these {n} matters."}, []
    return {"summary": f"{total} thing{'s' if total != 1 else ''} across {n} matters need{'s' if total == 1 else ''} you.",
            "items": shown, "more": max(0, total - len(shown))}, sorted(set(named))


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
    ids = conv.get("matter_ids") or [conv["matter_id"]]
    several = len(ids) > 1
    yield event("turn.started", {"turn_id": turn["id"], "conversation_id": conv["id"], "matter_id": conv["matter_id"], "intent": kind,
                                 **({"matter_ids": ids} if several else {})})
    try:
        scopes = [await run_in_threadpool(scopemod.authorize, principal, m) for m in ids]
        scope = scopes[0]
        matters = [await run_in_threadpool(_matter, m) for m in ids]
        matter = matters[0]
        if kind == "ask_fact":
            async for e in _ask(principal, scope, conv, turn, text, document_ids, built, tuple(scopes[1:])):
                yield e
        elif kind == "compare" and several and focus:
            rows = []
            for m in matters:                                  # the item is in exactly one of the conversation's matters
                row = await run_in_threadpool(discrepancies.get, principal, m["id"], focus)
                if row:
                    rows = [{**row, "_matter": {"id": m["id"], "number": m["number"]}}]
                    break
            if not rows:
                built.add("unavailable", {"text": "This item is no longer available."})
                built.summary = "This item is no longer available."
            else:
                _comparison(principal, built, rows, None)
                built.summary = discrepancies.title(rows[0])
        elif kind == "compare" and several:
            rows = []
            for i, (m, sc) in enumerate(zip(matters, scopes)):
                found: list = []
                async for e in _compare(principal, sc, {**conv, "matter_id": m["id"]}, text, None, built,
                                        f"{m['number']} ({i + 1} of {len(matters)}): ", found):
                    yield e
                rows += [{**r, "_matter": {"id": m["id"], "number": m["number"]}} for r in found]
            if not rows:
                text_ = (f"No conflicting facts found inside these {len(matters)} matters. Each matter is checked on its own: "
                         "documents of different matters are never compared with each other.")
                built.add("notice", {"tone": "info", "text": text_})
                built.summary = text_
            else:
                _comparison(principal, built, rows, None)
                n = len(rows)
                built.summary = (f"{n} potential inconsistenc{'y' if n == 1 else 'ies'} in {len({r['_matter']['id'] for r in rows})} "
                                 "of these matters, each inside its own matter. Nothing is decided until a person reviews it.")
        elif kind == "compare":
            async for e in _compare(principal, scope, conv, text, focus, built):
                yield e
        elif kind == "chronology":
            if several:
                parts = []
                for i, m in enumerate(matters):
                    yield event("task.progress", {"label": f"Reading the dates in {m['number']}", "done": i, "total": len(matters),
                                                  "state": "running"})
                    parts.append((m, *await run_in_threadpool(chronology.build, principal, m)))
                props, named = _chronology_several(parts)
            else:
                props, named = await run_in_threadpool(chronology.build, principal, matter)
            built.add("timeline", props, named)
            yield event("ui.partial", schemas.validate_component(built.components[-1]))
            built.summary = props["summary"]
        elif kind == "inspect_case":
            props, named = await run_in_threadpool(_briefing_several if several else briefing.build, principal, matters if several else matter)
            built.add("priority_list", props, named)
            built.summary = props["summary"]
        elif kind == "cases_overview":
            props = await run_in_threadpool(cases.build, principal, ids)
            built.add("case_table", props)
            yield event("ui.partial", schemas.validate_component(built.components[-1]))
            built.summary = props["summary"]
        elif kind == "other_matters":
            _other_matters(matters, built)
        elif kind == "draft":
            await run_in_threadpool(_draft, principal, matter, built)
        elif kind == "deliberate":
            _deliberate(matter, built)
        else:
            _notice(built, _SMALLTALK)
        for sc in scopes:
            await run_in_threadpool(scopemod.recheck, sc)
        envelope = schemas.validate_envelope({"message_id": turn["id"], "matter_id": conv["matter_id"], "summary": built.summary,
                                              "matters": _matter_refs(ids), "components": built.components,
                                              "suggested_actions": built.actions})
        for comp in envelope["components"]:
            await run_in_threadpool(store.save_component, {**comp, "props": built.stored.get(comp["id"], comp["props"])}, turn,
                                    built.documents.get(comp["id"], []), built.matter_of.get(comp["id"]))
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
                            **({"matters": conv.get("matter_ids")} if len(conv.get("matter_ids") or []) > 1 else {}),
                            "citations": sum(len(c["props"].get("citations", [])) for c in built.components if c["type"] == "source_answer")})
        except Exception as e:
            logger.error("Counsel: could not record the outcome (%s)", type(e).__name__)
