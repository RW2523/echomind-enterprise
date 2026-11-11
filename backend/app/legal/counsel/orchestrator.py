"""One chat turn, end to end (docs/echomind-legal/05 "Answer state machine").

    requested → authorized (scope) → intent → existing pipeline → verified components → validated envelope → stored

The turn never retrieves anything itself. It hands an `AuthorizedScope` (minted only by the policy engine for this
person and this matter) to the pipelines that already enforce isolation: research answers (quote-verified, abstaining)
and the conflict scan. Their results are turned into components from the trusted registry (schemas.py), validated as
a whole, stored, and only then sent as `ui.final`. Streamed text (`answer.delta`) is research's own paragraph text,
released only after each paragraph's quotes were checked.

Events, one JSON object per line: turn.started, turn.understood, task.progress, answer.delta, answer.held, citation.verified,
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
import re
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
from ..retrieval import search as searchmod
from . import (amounts, approvals, briefing, cases, chronology, citations, definitions_check, deliberation, discrepancies, drafting,
               casestatus, intent as intents, live,
               parties, schemas, store, tasks)

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


def _live_props(principal: dict, comp: dict) -> Optional[dict]:
    """A component drawn from records that change, rebuilt from them as the person may see them now (None: gone)."""
    stored, mid = comp["props"] or {}, comp["matter_id"]
    if comp["type"] == "checklist":
        return tasks.build(principal, _matter(mid), only=stored.get("obligation_ids") or [])[0]
    if comp["type"] == "approval_request":
        refs = stored.get("items") or []
        return approvals.build(principal, [_matter(m) for m in dict.fromkeys([mid] + [r["matter_id"] for r in refs])], stored=refs)[0]
    if comp["type"] == "document_preview":
        a = drafting.newest(principal, mid, stored.get("artifact_id") or "")
        return drafting.preview(principal, _matter(mid)["number"], a) if a else None
    if comp["type"] == "council_summary":
        return deliberation.summary(principal, mid, stored.get("review_id") or "")
    if comp["type"] == "live_transcript":
        return live.build(principal, mid, stored.get("meeting_id") or "")
    return None


def materialize(principal: dict, comp: dict, conv_matter: Optional[str] = None) -> dict:
    """A stored component as the person may see it now. An evidence comparison is resolved from its discrepancy (its
    state may have changed, or a document may no longer be visible); anything gone becomes `unavailable`. Anything else that
    drew on a document the person may no longer read (its access list narrowed after the answer) is withheld whole: its
    wording came from that document (commercial-readiness review F3, 2026-10-28). Components drawn from records that change
    (schemas.LIVE_TYPES) store only references and are rebuilt here; keys starting with "_" are the server's own."""
    from ..documents import repo as doc_repo
    if comp["type"] in schemas.LIVE_TYPES:
        props = _live_props(principal, comp)
        if props is None:
            out = {"id": comp["id"], "type": "unavailable", "props": {"text": "This item is no longer available."}}
        else:
            out = {"id": comp["id"], "type": comp["type"], "data_ref": comp["id"], "props": props}
        if conv_matter and comp["matter_id"] != conv_matter:
            out["matter_id"] = comp["matter_id"]
        return out
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
        out = {"id": comp["id"], "type": comp["type"], "props": {k: v for k, v in (comp["props"] or {}).items() if not k.startswith("_")}}
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


def _misquote_note(m: dict) -> str:
    """Another document quotes the named one with words it does not contain (research/routes._misquotes)."""
    return (f"{m.get('by') or 'Another document'} quotes {m.get('of') or 'the named document'} as “{m.get('quote', '')}”; "
            f"those words are not in {m.get('of') or 'it'}.")[:400]


def _twice_note(t: dict) -> str:
    """The document defines the term asked about more than once, with different meanings (research/routes._defined_twice)."""
    clauses = t.get("clauses") or []
    where = f" (clauses {' and '.join(clauses)})" if len(clauses) > 1 else ""
    return (f"{t.get('document') or 'The document'} defines “{t.get('term', '')}” more than once, with different meanings{where}, "
            "so it is ambiguous on this point. Found by rule.")[:400]


def _condition_note(q: dict) -> str:
    """A cited passage puts a condition on a sentence that the sentence leaves out (research/support.py, 2026-11-07)."""
    said = q.get("sentence") or ""
    said = said if len(said) <= 140 else said[:139].rsplit(" ", 1)[0] + "…"
    how = "Found by rule." if q.get("by") == "rule" else "Found on checking the answer against its sources."
    return f"The source adds a condition to “{said}”: “{q.get('words', '')}”. {how}"[:400]


def _choice_offers(choices: Optional[dict], question: str) -> tuple[list[str], list[dict]]:
    """What the question could mean, offered as asks narrowed to one document or one party (research/routes._choices,
    2026-11-07 review point 10). The answer is given either way; these let the person say which they meant."""
    notes, actions = [], []
    if not choices:
        return notes, actions
    if choices.get("documents"):
        n, kind = choices.get("count") or len(choices["documents"]), choices.get("kind") or "document"
        notes.append((f"This matter has {n} documents that are a{'n' if kind[:1] in 'aeiou' else ''} “{kind}”"
                      + (", and the answer may not cover all of them" if n > 4 else "") + ". To ask about one alone, choose it below.")[:400])
        for i, d in enumerate(choices["documents"][:4]):
            label = f"Only {d['title']}"
            actions.append({"id": f"only_document_{i}", "label": label if len(label) <= 48 else label[:47].rstrip() + "…",
                            "action_type": "ask", "prompt": question[:300], "document_ids": [d["id"]]})
    elif (parties := choices.get("parties")) and parties.get("names"):
        said, names = parties["said"], parties["names"]
        notes.append((f"“{said}” could be " + (", ".join(names[:-1]) + f" or {names[-1]}") + ". To ask about one of them, choose it below.")[:400])
        for i, name in enumerate(names[:4]):
            asked = re.sub(rf"\b{re.escape(said)}\b", name, question, count=1, flags=re.I)
            label = f"Only {name}"
            if len(asked) <= 300:
                actions.append({"id": f"only_party_{i}", "label": label if len(label) <= 48 else label[:47].rstrip() + "…",
                                "action_type": "ask", "prompt": asked})
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
               extra_scopes: tuple = (), as_written: bool = False):
    hist = [] if as_written else await run_in_threadpool(store.history, conv["id"], principal["sub"], turn)
    prefer = None
    if hist:
        yield event("task.progress", {"label": "Reading your follow-up with the conversation so far", "done": 0, "total": 0,
                                      "state": "running"})
        rewritten_q, rewritten = await followup.standalone(question, hist)
        if rewritten:
            question, built.understood = rewritten_q, rewritten_q
            yield event("turn.understood", {"question": rewritten_q})
            # the documents the last answer cited are searched first (2026-11-12, evaluation fix list item 4): "And did the
            # amendment change that?" is about the cap the last answer quoted, not about any cap in the matter
            prefer = (hist[-1].get("documents") or [])[:3] or None
    rturn = {"id": str(uuid.uuid4()), "org_id": principal["org_id"], "matter_id": conv["matter_id"], "owner_id": principal["sub"],
             "question": question, "auth": principal.get("auth"), "created_at": policy.now_iso()}
    built.research_turn_id = None if extra_scopes else rturn["id"]          # several matters: kept in the conversation only
    done = None
    # A paragraph whose quotes were not found in their sources is not streamed (2026-11-07 review point 6): only
    # `answer.held` says one is being kept back; the checked answer (ui.final) shows it, flagged.
    held = False
    async for line in research._answer(scope, principal, question, rturn, "signed", document_ids or None, None, extra_scopes,
                                       prefer=None if document_ids else prefer, progress=True):
        e = json.loads(line)
        kind = e.get("type")
        if kind == "progress":
            yield event("task.progress", {"label": (e.get("label") or "")[:120], "done": 0, "total": 0, "state": "running",
                                          **({"busy": True} if e.get("busy") else {})})
        elif kind == "paragraph":
            held = bool(e.get("failed"))
            if held:
                yield event("answer.held", {"index": e.get("index")})
        elif kind == "sources":
            docs = {c["document_id"] for c in e.get("citations") or []}
            n = len(e.get("citations") or [])
            if n:
                yield event("task.progress", {"label": f"Reading {n} passage{'s' if n != 1 else ''} from {len(docs)} "
                                                       f"document{'s' if len(docs) != 1 else ''}", "done": 0, "total": n, "state": "running"})
        elif kind == "chunk":
            if held:
                held = False
                continue
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
    choice_notes, choice_actions = _choice_offers(done.get("choices"), question)
    props = {"answer": done.get("answer") or "", "state": state,
             "citations": [{k: v for k, v in c.items() if k != "instructions"} for c in cites],
             # a decline supports nothing, so it lists no unsupported sentences; what it says is missing is shown instead (QA UX-001)
             "unsupported": [] if state == "abstained" else [s[:500] for s in (done.get("unsupported") or [])][:10],
             "notes": (([OTHER_DOCUMENTS] if done.get("abstention_reason") == "other_documents" else [])
                       + ([OTHER_PARTY] if done.get("abstention_reason") == "other_party" else [])
                       + ([NOT_IN_CITED] if done.get("abstention_reason") == "not_in_cited" else [])
                       + [_misquote_note(m) for m in (done.get("misquotes") or [])]
                       + ([] if state == "abstained" else [_twice_note(t) for t in (done.get("defined_twice") or [])])
                       + ([] if state == "abstained" else [_condition_note(q) for q in ((done.get("support") or {}).get("qualified") or [])])
                       + amend_notes + choice_notes + _notes(cites, done.get("conflicts") or []))[:3]}
    if built.understood:
        props["understood"] = built.understood[:400]
    if state == "abstained" and done.get("absence"):
        # its words, without the source marks it carried ("… not stated [2][3]" showed the marks as text, 2026-11-11)
        props["absence"] = (re.sub(r"\s*\[\d{1,2}(?::[^\]]*)?\]", "", done["absence"]).strip() or done["absence"])[:400]
    sd = done.get("stated_detail") or {}
    if state == "abstained" and sd.get("acted") and sd.get("verdict") == "not_given" and sd.get("asked"):
        props["checked"] = str(sd["asked"])[:300]
    # not when the cited passages were found to state what was asked (research/stated.py)
    if state != "abstained" and done.get("stated") is not True and research.prompt.opens_not_stated(done.get("answer") or ""):
        props["not_stated"] = True
    # one card for one calculation (2026-11-12, evaluation fix list item 11): the working by rule, when there is one, already
    # gives the dates the term calculation would show again in another style
    if state != "abstained" and done.get("calculation") and not (done.get("working") or {}).get("steps"):
        props["calculation"] = _calculation_props(done["calculation"], (await run_in_threadpool(_matter, conv["matter_id"]))["number"])
    if state != "abstained" and (done.get("working") or {}).get("steps"):
        props["working"] = _working_props(done["working"], (await run_in_threadpool(_matter, conv["matter_id"]))["number"])
    built.add("source_answer", props, [c["document_id"] for c in cites] + [a["by"]["document_id"] for a in changed])
    built.actions = (amend_actions + choice_actions)[:4]
    if state == "abstained":
        built.status = "abstained"
        built.summary = "No source found in these matters' documents." if extra_scopes else "No source found in this matter's documents."
    else:
        verified = sum(1 for c in cites if c["verified"])
        built.summary = f"Answered from {len(cites)} source{'s' if len(cites) != 1 else ''}, {verified} with a verified quote."


def _calculation_props(calc: dict, matter_number: str) -> dict:
    """The answer's calculated dates for the card, each fact opening its clause in the reader (QA RAG-003)."""
    number = calc.get("matter_number") or matter_number
    return {"document_title": calc["document_title"][:300],
            "facts": [{"label": f["label"][:60], "value": f["value"][:80], "quote": f["quote"][:300], "clause": (f.get("clause") or None),
                       "path": citations.reader_path(number, calc["document_id"], calc["version_id"], f["char_start"], f["char_end"])}
                      for f in calc["facts"][:6]],
            "derived": [{"label": d["label"][:80], "display": d["display"][:80]} for d in calc["derived"][:4]],
            "assumptions": [a[:400] for a in calc["assumptions"][:4]],
            "not_calculated": [n[:300] for n in calc["not_calculated"][:3]]}


def _working_props(w: dict, matter_number: str) -> dict:
    """Figures worked out by rule (research/working.py) for the card: each step, and each figure's words opening in the reader."""
    return {"steps": [{"label": s["label"][:120], "working": s["working"][:600], "result": s["result"][:80], "check": bool(s.get("check"))}
                      for s in w["steps"][:10]],
            "figures": [{"label": f["label"][:80], "value": f["value"][:80], "quote": f["quote"][:400],
                         "document_title": f["document_title"][:300], "clause": (f.get("clause") or "")[:40] or None,
                         "path": citations.reader_path(f.get("matter_number") or matter_number, f["document_id"], f["version_id"],
                                                       f.get("char_start"), f.get("char_end"))}
                        for f in w["figures"][:16]],
            "assumptions": [a[:400] for a in (w.get("assumptions") or [])[:3]]}


NOT_IN_CITED = ("The documents that were cited do not deal with what was asked; another document of this matter does, so the "
                "passages are not given as the answer. Ask about that document, or read the passages consulted.")
OTHER_PARTY = ("The passages found are another party's records, so they are not given as the answer: no document of the "
               "party asked about records this.")
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


INVENTORY_SHOWN = 12


def _inventory(principal: dict, matters: list[dict], built: Built) -> None:
    """The documents of the conversation's matters that this person may open, from their records: never from search, and
    never a document they may not open (2026-10-29 QA CHAT-001). A matter whose documents they may not read is said so."""
    from ..documents import repo as doc_repo
    parts, total = [], 0
    for m in matters:
        head = f"{m['number']}: " if len(matters) > 1 else f"{m['number']} "
        if not policy.authorize(principal, "documents.read", m["id"]).allowed:
            parts.append(f"{head}— your role here does not include reading its documents." if len(matters) == 1
                         else f"{head}your role does not include reading its documents.")
            continue
        docs = doc_repo.list_documents(m["id"], principal["sub"])
        total += len(docs)
        if not docs:
            parts.append(f"{head}has no documents that you can open." if len(matters) == 1 else f"{head}no documents that you can open.")
            continue
        def number(d):
            return (d.get("current_version") or {}).get("number")
        listed = "; ".join(f"{d['title'] or 'Untitled'} ({', '.join(x for x in (d.get('kind'), d.get('status')) if x)}"
                           f"{', version ' + str(number(d)) if number(d) else ''})" for d in docs[:INVENTORY_SHOWN])
        more = f"; and {len(docs) - INVENTORY_SHOWN} more" if len(docs) > INVENTORY_SHOWN else ""
        n = f"{len(docs)} document{'s' if len(docs) != 1 else ''}"
        parts.append(f"{head}has {n} that you can open: {listed}{more}." if len(matters) == 1 else f"{head}{n}: {listed}{more}.")
    text = " ".join(parts)
    if len(text) > 800:
        text = text[:797].rsplit(" ", 1)[0] + "…"
    actions = ([{"id": "open_documents", "label": "Open the documents", "action_type": "open_page",
                 "path": f"/m/{urlquote(matters[0]['number'], safe='')}/documents"}] if len(matters) == 1 and total else [])
    _notice(built, text, actions=actions,
            summary=(f"{total} document{'s' if total != 1 else ''} you can open." if total else "No documents that you can open."))


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


def _draft(principal: dict, matter: dict, built: Built, request: str = "", document_ids: Optional[list[str]] = None) -> None:
    base = f"/m/{urlquote(matter['number'], safe='')}"
    if not policy.authorize(principal, "ai.draft", matter["id"]).allowed:
        _notice(built, "Your role in this matter does not include drafting.", summary="Drafting is not part of your role here.")
        return
    asked = drafting.form(principal, matter, request, list(document_ids or []))
    if asked is None:
        _notice(built, "Your firm has no approved drafting template yet. A managing partner approves templates in Drafting; drafts "
                       "are then made from them, every fact cited to the matter's documents.",
                actions=[{"id": "open_drafting", "label": "Open Drafting", "action_type": "open_page", "path": f"{base}/drafting"}],
                summary="No approved template yet.")
        return
    props, private = asked
    built.add("guided_questions", props, list(document_ids or []), stored={**props, "_form": private})
    built.summary = "Three questions before the draft is written."


def _deliberate(principal: dict, matter: dict, built: Built, request: str = "", document_ids: Optional[list[str]] = None) -> None:
    base = f"/m/{urlquote(matter['number'], safe='')}"
    if settings.LEGAL_PANEL and not policy.authorize(principal, "ai.draft", matter["id"]).allowed:
        _notice(built, "Asking the Council is for the matter's lawyers and paralegals. You can read its sessions on the Council page.",
                actions=[{"id": "open_panel", "label": "Open the Council", "action_type": "open_page", "path": f"{base}/panel"}],
                summary="The Council is for the matter's lawyers.")
    elif settings.LEGAL_PANEL:
        props, private = deliberation.form(matter, request, list(document_ids or []))
        built.add("guided_questions", props, list(document_ids or []), stored={**props, "_form": private})
        built.summary = "Three questions before the Council starts."
    else:
        _notice(built, "Testing an argument from several perspectives (the Council) is not turned on for this firm. You can "
                       "still ask what the documents say for and against a point.",
                summary="The Council is not turned on.")


_SMALLTALK = ("Ask about this matter's documents, for example: what a clause says, which dates or amounts the documents "
              "disagree on, or “What matters today?”.")


def _social_reply(kind: Optional[str], matter: dict) -> str:
    """A short reply to small talk (spoken or typed): no search, no model, and nothing said about the matter's content."""
    about = f"Ask me about {matter['number']}: what a clause says, a date or amount in its documents, or what waits for your decision."
    return {
        "greeting": f"Hello. {about}",
        "check": f"Yes, I can hear you. {about}",
        "thanks": "You're welcome.",
        "ack": "All right.",
        "closing": "Goodbye. This conversation is saved in your chats.",
        "identity": (f"I'm EchoMind Legal. I answer questions about {matter['number']} only from its own documents, with the exact "
                     "words each answer relies on, and I can list deadlines, compare documents, build a chronology or start a "
                     "draft for a lawyer's review."),
        # asides (counsel/intent.aside, 2026-11-12): each got the decline sentence before
        "model": (f"I'm EchoMind Legal's assistant. I answer from {matter['number']}'s documents only, with a language model that "
                  "runs on this installation's own server, not a public AI service, and every answer is checked against the "
                  "documents' own words before it is shown."),
        "today": (f"Today is {_today()}. My answers come from {matter['number']}'s documents; to find a date in them, ask what a "
                  "document says, for example “When does the term end?”."),
        "offtopic": (f"That's outside what I do. I work only with {matter['number']}'s documents and records, so I don't write code, "
                     "tell jokes or look things up on the internet. I can say what a clause says, compare documents, build a "
                     "chronology or start a draft letter."),
    }.get(kind or "", _SMALLTALK)


def _today() -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    try:
        now = datetime.now(ZoneInfo(settings.LEGAL_TIMEZONE))
    except Exception:  # noqa: BLE001 — an unknown zone name: UTC
        now = datetime.now(ZoneInfo("UTC"))
    return f"{now:%A} {now.day} {now:%B %Y}"


def _recall(conv: dict, principal: dict, turn: dict, text: str) -> str:
    """"What did I ask you first?" answered from this conversation's own questions (2026-11-12), never from the documents."""
    asked = store.asked(conv["id"], principal["sub"], turn)
    if not asked:
        return "This is the first question in this conversation."
    if re.search(r"\b(?:last|previous|just|before\s+that)\b", text, re.I):
        return f"Your last question was: “{asked[-1][:300]}”."
    out = f"Your first question in this conversation was: “{asked[0][:300]}”."
    if len(asked) > 1:
        rest = "; ".join(f"“{q[:120]}”" for q in asked[1:6])
        out += f" Since then you asked: {rest}" + (f", and {len(asked) - 6} more." if len(asked) > 6 else ".")
    return out


_OFFTOPIC_ACTIONS = [{"id": "ask_cap", "label": "What is the liability cap?", "action_type": "ask", "prompt": "What is the liability cap?"},
                     {"id": "ask_dates", "label": "Build a chronology", "action_type": "ask", "prompt": "Build a chronology of the key dates"}]


# ── The turn ─────────────────────────────────────────────────────────────────────
async def _definitions(principal: dict, turn: dict, matter: dict, text: str, document_ids: list[str], built: Built) -> AsyncIterator[dict]:
    """The rules-only definitions check of the one document asked about, shown as a cited answer (counsel/definitions_check.py)."""
    document, docs = await run_in_threadpool(definitions_check.choose, principal, matter, text, document_ids)
    if document is None:
        _notice(built, definitions_check.ask_which(matter, docs))
        return
    yield event("task.progress", {"label": f"Checking the definitions, numbering and cross-references of “{document['title'][:80]}”",
                                  "done": 0, "total": 0, "state": "running"})
    answer, done, check = await run_in_threadpool(definitions_check.build, principal, matter, document)
    record("check.definitions", principal=principal, object_type="document", object_id=document["id"], matter_id=matter["id"],
           details={"check": check["id"], "from": "chat", **check["summary"]})
    cites = await run_in_threadpool(citations.from_research, turn, done)
    for c in cites:
        yield event("citation.verified", {"citation_id": c["id"], "n": c["n"], "document_id": c["document_id"], "page": c.get("page")})
    built.add("source_answer", {"answer": answer, "state": "answered",
                                "citations": [{k: v for k, v in c.items() if k != "instructions"} for c in cites],
                                "unsupported": [], "notes": []}, [document["id"]])
    page = f"/m/{urlquote(matter['number'], safe='')}/documents/{document['id']}/definitions"
    built.actions = [{"id": "open_definitions", "label": "Open the definitions check", "action_type": "open_page", "path": page}]
    n = len(check["findings"])
    built.summary = f"Definitions check of {document['title'][:200]}: {n} finding{'s' if n != 1 else ''}, found by rule."[:400]


def _live(principal: dict, matter: dict, built: Built) -> None:
    base = f"/m/{urlquote(matter['number'], safe='')}"
    meetings = [{"id": "open_meetings", "label": "Open Meetings", "action_type": "open_page", "path": f"{base}/meetings"}]
    if not policy.authorize(principal, "meetings.read", matter["id"]).allowed:
        _notice(built, "Your role in this matter does not include its meetings.", summary="Meetings are not part of your role here.")
        return
    mid = live.pick(matter["id"])
    props = live.build(principal, matter["id"], mid) if mid else None
    if not props:
        _notice(built, f"No meeting of {matter['number']} is being recorded, and none has a transcript yet. Recording starts on the "
                       "meeting page, only with every participant's consent.", actions=meetings, summary="No meeting transcript.")
        return
    built.add("live_transcript", props, [], data_ref=True, stored={"meeting_id": mid})
    built.summary = (f"{props['title']}: recording." if props["state"] == "recording" else f"{props['title']}: paused."
                     if props["state"] == "paused" else f"{props['title']}: not being recorded now.")


def form_values(props: dict, values: dict) -> dict:
    """The answers to a guided form, each checked against the questions the server asked (choices only from their options)."""
    out = {}
    for q in props["questions"]:
        v = values.get(q["id"])
        if v is None or not str(v).strip():
            v = q.get("default")
        if q["kind"] == "choice":
            if v not in {o["value"] for o in q["options"]}:
                raise TurnFailed(f"Choose one of the options for “{q['label']}”.")
        else:
            v = " ".join(str(v or "").split())[:q.get("max_length", 200)]
            if q.get("required", True) and not v:
                raise TurnFailed(f"“{q['label']}” is needed.")
        out[q["id"]] = v
    return out


async def _answer_form(principal: dict, conv: dict, matter: dict, form: dict, built: Built):
    """A guided form's answers: write the draft, or queue the Council (the review panel)."""
    comp, values = form["component"], form["values"]
    stored = comp["props"]
    private = stored.get("_form") or {}
    asked = form_values(stored, values)
    target = await run_in_threadpool(_matter, comp["matter_id"])
    if private.get("purpose") == "draft":
        yield event("task.progress", {"label": f"Writing “{asked['title'][:80]}”: each field is read from the documents and cited",
                                      "done": 0, "total": 0, "state": "running"})
        try:
            a = await drafting.run(principal, target, asked, private)
        except LookupError:
            raise TurnFailed("That template has no approved version any more. Choose another.")
        props = await run_in_threadpool(drafting.preview, principal, target["number"], a)
        built.add("document_preview", props, [], data_ref=True, stored={"artifact_id": a["id"]},
                  matter_id=target["id"] if target["id"] != conv["matter_id"] else None)
        built.summary = (f"Draft “{props['title']}” written" + (f", {props['gaps']} gap{'s' if props['gaps'] != 1 else ''} to fill" if props["gaps"] else "")
                         + ". A second lawyer approves it before anything leaves the matter.")[:400]
    elif private.get("purpose") == "panel":
        rid = await run_in_threadpool(deliberation.start, principal, target, asked, private)
        props = await run_in_threadpool(deliberation.summary, principal, target["id"], rid)
        built.add("council_summary", props, [], data_ref=True, stored={"review_id": rid},
                  matter_id=target["id"] if target["id"] != conv["matter_id"] else None)
        built.summary = "The Council is working on it."
    else:
        raise TurnFailed(FAILED)


async def run(principal: dict, conv: dict, text: str, client_request_id: Optional[str], document_ids: list[str],
              focus: Optional[str], parent=store._FOLLOW_NEWEST, form: Optional[dict] = None,
              as_written: bool = False) -> AsyncIterator[dict]:
    if form:
        kind = "draft_run" if (form["component"]["props"].get("_form") or {}).get("purpose") == "draft" else "panel_run"
    else:
        kind = "compare" if focus else intents.classify(text)
    turn = await run_in_threadpool(store.start_turn, conv, principal["sub"], text, kind, client_request_id, parent)
    built = Built()
    final_status = "failed"
    ids = conv.get("matter_ids") or [conv["matter_id"]]
    several = len(ids) > 1
    yield event("turn.started", {"turn_id": turn["id"], "conversation_id": conv["id"], "matter_id": conv["matter_id"], "intent": kind,
                                 "parent_turn_id": turn["parent_turn_id"], **({"matter_ids": ids} if several else {})})
    try:
        scopes = [await run_in_threadpool(scopemod.authorize, principal, m) for m in ids]
        scope = scopes[0]
        matters = [await run_in_threadpool(_matter, m) for m in ids]
        matter = matters[0]
        if kind == "ask_fact":
            async for e in _ask(principal, scope, conv, turn, text, document_ids, built, tuple(scopes[1:]), as_written):
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
                # a chronology of something: the passages a search for it finds narrow the matter's events to it
                focus = chronology.topic(text)
                if focus:
                    got = await searchmod.search(scope, focus["query"], k=focus["k"], versions="signed", wide=True)
                    focus["ranges"] = {}
                    for p in got["passages"]:
                        c = p["citation"]
                        if c.get("char_start") is not None:
                            focus["ranges"].setdefault(c["version_id"], []).append((c["char_start"], c["char_end"]))
                props, named = await run_in_threadpool(chronology.build, principal, matter, focus)
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
        elif kind == "documents":
            await run_in_threadpool(_inventory, principal, matters, built)
        elif kind == "draft":
            await run_in_threadpool(_draft, principal, matter, built, text, document_ids)
        elif kind == "deliberate":
            await run_in_threadpool(_deliberate, principal, matter, built, text, document_ids)
        elif kind == "case_status":
            for m in matters[:schemas.MAX_COMPONENTS]:
                props = await run_in_threadpool(casestatus.build, principal, m)
                built.add("case_status", props, [], matter_id=m["id"] if m["id"] != conv["matter_id"] else None)
            built.summary = props["summary"] if len(matters) == 1 else f"Where {len(matters[:3])} matters stand."
            built.actions = [{"id": "open_case", "label": "Post an update" if props["can_post"] else "Open the case",
                              "action_type": "open_page", "path": props["path"]}]
        elif kind == "tasks":
            plan = None
            if not several and await run_in_threadpool(casestatus.has_plan, matter["id"]):
                plan = await run_in_threadpool(casestatus.build, principal, matter)        # the plan a person keeps, first
                built.add("case_status", plan, [])
            for m in matters[:schemas.MAX_COMPONENTS - len(built.components)]:
                props, named, shown = await run_in_threadpool(tasks.build, principal, m)
                built.add("checklist", props, named, data_ref=True, stored={"obligation_ids": shown},
                          matter_id=m["id"] if m["id"] != conv["matter_id"] else None)
            built.summary = props["summary"] if len(matters) == 1 else f"Open obligations in {len(matters[:3])} matters."
            if plan:
                built.summary = (f"{plan['open_tasks']} open task{'s' if plan['open_tasks'] != 1 else ''} in the plan; " + built.summary)[:400]
        elif kind == "approvals":
            props, refs = await run_in_threadpool(approvals.build, principal, matters)
            built.add("approval_request", props, [], data_ref=True, stored={"items": refs})
            built.summary = props["summary"]
        elif kind == "live":
            await run_in_threadpool(_live, principal, matter, built)
        elif kind == "definitions":
            async for e in _definitions(principal, turn, matter, text, document_ids, built):
                yield e
        elif kind == "amounts":
            props, named = await run_in_threadpool(amounts.build, principal, matter, document_ids)
            built.add("numeric_chart", props, named)
            built.summary = props["summary"]
        elif kind == "parties":
            props, named = await run_in_threadpool(parties.build, principal, matter, document_ids)
            built.add("graph_excerpt", props, named)
            built.summary = props["summary"]
        elif kind in ("draft_run", "panel_run"):
            async for e in _answer_form(principal, conv, matter, form, built):
                yield e
        else:
            said = intents.social(text) or intents.aside(text)
            if said == "memory":
                _notice(built, await run_in_threadpool(_recall, conv, principal, turn, text))
            else:
                _notice(built, _social_reply(said, matter), actions=_OFFTOPIC_ACTIONS if said == "offtopic" else None)
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
