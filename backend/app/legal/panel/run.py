"""The four-perspective review panel (S4.5, OPT-05; decision D-057).

One question about a matter, answered four times:
- by a deal lawyer, by risk and compliance, by a litigator, and by a specialist in a chosen area;
- each in its own model call, with the same sources and without seeing the others;
- each citing every statement with a quote that is checked, as research answers are (S2.2).

A moderator then reads the four answers, and only them, into a disagreement matrix. The moderator is told to add
nothing, and a figure its summary gives that no answer gave is flagged.

If the answers depend on facts the documents do not give (the moderator says so, or three of the four perspectives
say facts are missing), the panel stops and asks the lawyer. It runs again with
those facts, marked as the lawyer's. A partner (the matter's lead or a managing partner) other than the person who
asked approves the result.

It is four perspectives drafted by the same model, not independent experts, and it says so.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from ...core.config import settings
from ...core.db import get_conn
from .. import artifacts, policy
from ..research import llm as llm_mod
from ..research import prompt as research_prompt
from ..retrieval import scope as scopemod
from ..retrieval import search as searchmod

VERSION = "panel-s4.5c"
NOTICE = ("Four perspectives drafted by the same AI model to test the question from different angles. They are not independent "
          "experts and not legal advice; a partner reviews them before anyone relies on them.")
SPECIALISMS = {"commercial": "commercial contracts", "corporate": "corporate law", "employment": "employment law", "ip": "intellectual property",
               "data": "data protection", "tax": "tax", "real_estate": "real estate", "competition": "competition law",
               "finance": "banking and finance", "regulatory": "financial regulation"}
ROLES = [("deal", "Deal lawyer", "the deal lawyer: you look at how the documents affect the transaction and the client's commercial aims"),
         ("risk", "Risk and compliance", "the risk and compliance lawyer: you look for exposure, obligations the client must meet, and "
                                         "regulatory or compliance points"),
         ("litigation", "Litigator", "the litigator: you look at disputes, remedies, notice requirements, governing law and how a court "
                                     "would read the words"),
         ("specialist", "Specialist ({area})", "a specialist in {area}: you look at the points particular to that area")]
MISSING = re.compile(r"^\s*MISSING FACTS:\s*(.*)\s*$", re.I | re.M)
STANCE_MAX = 80
MAX_FACTS = 8                 # the facts form takes this many; the first ones listed are kept
_NUMBER = re.compile(r"\d[\d,.]*\d|\d")


def role_system(role: str, area: str) -> str:
    who = next(d for r, _, d in ROLES if r == role).format(area=area)
    return (f"PERSPECTIVE: {role}\nYou are {who}. Answer the question from that angle only.\n\n" + research_prompt.SYSTEM +
            "\n7. After your answer, write one last line: MISSING FACTS: the facts your answer depends on that the sources do not give, "
            "separated by semicolons, or MISSING FACTS: none.")


def _facts_block(facts: list[dict]) -> str:
    if not facts:
        return ""
    return ("\n\nFacts given by the lawyer (not from the documents; use them, and say when you rely on them):\n" +
            "\n".join(f"- {research_prompt.withhold(f['question'])}: {research_prompt.withhold(f['answer'])}" for f in facts))


MODERATOR = (
    "You moderate a review panel. You are given a question and four answers to it, by a deal lawyer, a risk and compliance "
    "lawyer, a litigator and a specialist. Compare the answers only: never add facts, figures or views of your own, and never "
    "use outside knowledge. List the issues the answers address, and each answer's position on each issue in a few words "
    "(write \"not addressed\" when an answer does not deal with it). Mark an issue as a disagreement when the positions differ "
    "in substance. List the facts the answers say they lack, and set stop to true when the question cannot properly be answered "
    "without them. Answer only in JSON: {\"issues\": [{\"issue\": \"…\", \"positions\": {\"deal\": \"…\", \"risk\": \"…\", "
    "\"litigation\": \"…\", \"specialist\": \"…\"}, \"disagreement\": true, \"summary\": \"…\"}], \"missing_facts\": [\"…\"], "
    "\"stop\": false}, with at most 8 issues."
)


def _json(raw: str) -> dict:
    i = (raw or "").find("{")
    while i >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw[i:])
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
        i = raw.find("{", i + 1)
    return {}


_ECHO = re.compile(r"\b(?:the facts your answer depends on|separated by semicolons|missing facts\s*:)", re.I)


def unique_facts(facts) -> list[str]:
    """Missing facts once each (ignoring case), without an echo of the instruction (s4.5b)."""
    out, seen = [], set()
    for f in facts:
        f = " ".join(str(f).split()).rstrip(".")
        if not f or f.lower() == "none" or _ECHO.search(f) or f.casefold() in seen:
            continue
        seen.add(f.casefold())
        out.append(f[:200])
    return out


def _missing(raw: str) -> tuple[str, list[str]]:
    m = list(MISSING.finditer(raw or ""))
    if not m:
        return raw, []
    last = m[-1]
    return (raw[:last.start()] + raw[last.end():]).strip(), unique_facts(last.group(1).split(";"))[:8]


def _strip_citations(text: str) -> str:
    return re.sub(r"\s*\[\d{1,2}(?::\s*\"(?:\\.|[^\"\\])*\")?\]", "", text)


async def perspective(role: str, label: str, area: str, question: str, passages: list[dict], sources: list[dict], facts: list[dict]) -> dict:
    from ..research.routes import _check, _evidence
    messages = research_prompt.build(question + _facts_block(facts), passages)
    messages[0] = {"role": "system", "content": role_system(role, area)}
    raw = await llm_mod.complete(messages, temperature=0.1, max_tokens=settings.LEGAL_RESEARCH_MAX_TOKENS)
    from ..drafting.engine import _numbered            # s4.5b: "[9.1: …]" (a clause) is matched to the source that holds the quote
    body, missing = _missing(_numbered(research_prompt.clean(raw or ""), passages))
    shown, claims, unsupported, failed = [], [], [], []
    for para in [p for p in re.split(r"\n\s*\n", body) if p.strip()]:
        out, para_claims = _check(para, sources)
        shown.append(out["text"])
        claims.extend(para_claims)
        unsupported.extend(out["unsupported"])
        failed.extend(out["failed"])
    evidence = _evidence(claims, sources)
    text = re.sub(r"(\[\d{1,2}\])(?:\1)+", r"\1", "\n\n".join(shown))       # "[1][1]" reads as one citation
    abstained = research_prompt.is_abstention(text) and not evidence
    return {"role": role, "label": label.format(area=area), "text": text, "evidence": evidence, "unsupported": unsupported, "failed": failed,
            "missing_facts": missing, "abstained": abstained}


def _figures(text: str, said: str) -> list[str]:
    have = {n.rstrip(".,") for n in _NUMBER.findall(said)}
    return [n.rstrip(".,") for n in _NUMBER.findall(text or "") if n.rstrip(".,") not in have]


async def moderate(question: str, views: list[dict], facts: list[dict]) -> dict:
    answers = "\n\n".join(f"<answer role=\"{v['role']}\">\n{research_prompt.withhold(_strip_citations(v['text']))}\n"
                          f"Missing facts: {'; '.join(v['missing_facts']) or 'none'}\n</answer>" for v in views)
    raw = await llm_mod.complete([{"role": "system", "content": MODERATOR},
                                  {"role": "user", "content": f"Question: {question}{_facts_block(facts)}\n\n{answers}"}],
                                 temperature=0.0, max_tokens=1200)
    out = _json(raw)
    said = " ".join(v["text"] for v in views) + " " + question + " " + " ".join(f["answer"] for f in facts)
    issues = []
    for it in (out.get("issues") or [])[:8]:
        if not isinstance(it, dict) or not str(it.get("issue") or "").strip():
            continue
        pos = it.get("positions") if isinstance(it.get("positions"), dict) else {}
        positions = {r: (" ".join(str(pos.get(r) or "not addressed").split())[:STANCE_MAX] or "not addressed") for r, _, _ in ROLES}
        summary = " ".join(str(it.get("summary") or "").split())[:400]
        issues.append({"issue": " ".join(str(it["issue"]).split())[:200], "positions": positions, "disagreement": bool(it.get("disagreement")),
                       "summary": summary or None, "figures_unsupported": _figures(summary + " " + " ".join(positions.values()), said)})
    mod_missing = [" ".join(str(x).split()).rstrip(".")[:200] for x in (out.get("missing_facts") or []) if str(x).strip()][:8]
    return {"matrix": issues, "missing_facts": mod_missing, "stop": bool(out.get("stop"))}


async def run(job: dict) -> Optional[str]:
    rid = job["object_id"]
    with get_conn() as conn:
        r = conn.execute("SELECT matter_id, question, specialist, facts_json, created_by, documents_json FROM legal_panel_reviews WHERE id=?",
                         (rid,)).fetchone()
        if not r:
            return "no_review"
        conn.execute("UPDATE legal_panel_reviews SET status='running' WHERE id=?", (rid,))
        conn.commit()
    matter_id, question, specialist, facts_json, by, documents_json = r
    chosen = json.loads(documents_json or "[]") or None
    facts = json.loads(facts_json or "[]")
    area = SPECIALISMS.get(specialist, specialist)
    status, artifact_id = "failed", None
    try:
        principal = {"sub": by, "org_id": job["org_id"]}
        scope = scopemod.authorize(principal, matter_id, "research.query")
        result = await searchmod.search(scope, question, k=settings.LEGAL_RESEARCH_PASSAGES, documents=chosen)
        passages = result["passages"]
        from ..research.routes import _source
        sources = [_source(n, p) for n, p in enumerate(passages, start=1)]
        if not passages:
            views = [{"role": role, "label": label.format(area=area), "text": research_prompt.ABSTAIN, "evidence": [], "unsupported": [], "failed": [],
                      "missing_facts": [], "abstained": True} for role, label, _ in ROLES]
            mod = {"matrix": [], "missing_facts": [], "stop": False}
        else:
            views = [await perspective(role, label, area, question, passages, sources, facts) for role, label, _ in ROLES]
            scopemod.recheck(scope)
            mod = await moderate(question, views, facts)
        missing = unique_facts([m for v in views for m in v["missing_facts"]] + mod["missing_facts"])[:MAX_FACTS]   # what the lawyer can answer
        lacking = sum(1 for v in views if v["missing_facts"])
        stopped = bool(missing) and (mod["stop"] or lacking >= 3)       # s4.5c: most perspectives lacking facts also stops it
        content = {"title": f"Panel review — {question[:120]}", "question": question, "specialist": specialist, "area": area,
                   "perspectives": views, "matrix": mod["matrix"], "missing_facts": missing, "stopped": stopped, "facts_given": facts,
                   "notice": NOTICE, "sources": len(passages), "documents": chosen or []}
        artifact_id = artifacts.create(matter_id, "panel", "panel_review", rid, content,
                                       {"model": settings.LLM_MODEL, "prompt_version": VERSION, "passages": len(passages)}, by, "model")["id"]
        status = "needs_facts" if stopped else "done"
        return None
    except scopemod.ScopeDenied:
        status = "not_allowed"
        return "not_allowed"
    except scopemod.ScopeRevoked:
        status = "not_allowed"
        return "not_allowed"
    finally:
        with get_conn() as conn:
            conn.execute("UPDATE legal_panel_reviews SET status=?, finished_at=?, artifact_id=COALESCE(?, artifact_id) WHERE id=?",
                         (status, policy.now_iso(), artifact_id, rid))
            conn.commit()


def is_partner(p: dict, matter_id: str) -> bool:
    """The matter's lead, or a managing partner (S4.5: a panel review is approved by a partner)."""
    return policy.authorize(p, "matter.update", matter_id).allowed or "managing_partner" in policy.firm_roles_of(p["sub"])
