"""A due-diligence run (S3.7, DOC-06; decision D-051): a background job (`diligence.run`).

1. Each checklist item is matched to the matter documents that answer it, by the words that name them
   in the document's title or its opening lines. Items nothing answers are the missing-document list.
2. Each matched document is asked the item's questions, by rules (no model), on its current version:
   - **change of control**, **assignment** (consent), **termination** (convenience, break dates, other
     rights) and **renewal** (non-renewal deadlines and term ends) come from the S3.6 obligations reader;
   - **governing law** is compared with the law the checklist expects;
   - **liability**: a cap is reported; "unlimited" liability is high.
   Every issue quotes its sentence at its offsets.
3. Everything is **preliminary** until a reviewer verifies or dismisses it.

The documents read are those the person who started the run can see; what is shown afterwards is
filtered again for each viewer.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import date, datetime, timezone
from typing import Optional

from ...core.db import get_conn
from .. import policy
from ..compare.facts import _LAW
from ..documents import numbering
from ..documents import repo as documents
from ..obligations import extract as ox

VERSION = "diligence-s3.7"
_HEAD = 400


def _fmt(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.day} {d.strftime('%B')} {d.year}"


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[’'`]", "", re.sub(r"[_\-–—/.,()]+", " ", s or "")).casefold().split())


def match(item: dict, title: str, head: str) -> int:
    """How strongly a document answers an item: its title names it (3), or its opening lines do (1)."""
    t, h, score = _norm(title), _norm(head), 0
    for phrase in item["expect"]:
        p = _norm(phrase)
        if re.search(rf"(?<!\w){re.escape(p)}(?!\w)", t):
            score = max(score, 3)
        elif re.search(rf"(?<!\w){re.escape(p)}(?!\w)", h):
            score = max(score, 1)
    return score


def _first_line(text: str) -> str:
    """The document's own title line: what it says it is, not what its body mentions ("approved the supply agreement")."""
    for line in (text or "")[:_HEAD].splitlines():
        if line.strip(" #*>\t"):
            return line.strip(" #*>\t")
    return ""


def _version(matter_id: str, document_id: str) -> Optional[tuple[str, str, list[dict]]]:
    with get_conn() as conn:
        r = conn.execute("SELECT v.id, t.text FROM legal_documents d JOIN legal_document_versions v ON v.id = d.current_version_id "
                         "JOIN legal_version_text t ON t.version_id = v.id WHERE d.id=? AND d.matter_id=?", (document_id, matter_id)).fetchone()
        if not r:
            return None
        rows = [{"number": n, "heading": h, "level": lv, "parent": p, "start": s, "end": e} for n, h, lv, p, s, e in conn.execute(
            "SELECT number, heading, level, parent, char_start, char_end FROM legal_clauses WHERE version_id=? ORDER BY char_start, level", (r[0],))]
    return r[0], r[1], rows


def _sentence(text: str, rows: list[dict], at: int) -> dict:
    """The sentence around `at`, starting at its own clause (not a heading read into it)."""
    from ..research.prompt import document_sentences
    a, b = next(((x, y) for x, y in document_sentences(text) if x <= at < y), (at, min(len(text), at + 200)))
    modal = ox.MODAL.search(text, a, b)
    at = modal.start() if modal else at                   # the clause that binds, not a heading read into the sentence
    numbered = [c for c in numbering.complete(text, rows) if numbering.split(c["number"])]
    inside = [c for c in numbered if c["start"] <= at < max(c["end"], c["start"] + 1)]
    leaf = max(inside, key=lambda c: (c["level"], c["start"])) if inside else None
    if leaf and a < leaf["start"] <= at:
        a = leaf["start"]
    q = text[a:b].rstrip()
    return {"start": a, "end": a + len(q), "quote": q, "clause": leaf["number"] if leaf else None}


def ask(questions: list[str], text: str, rows: list[dict], expected_law: Optional[str], today: date) -> list[dict]:
    """The issues one document gives for the item's questions."""
    out, used = [], set()
    obs = ox.read(text, rows) if set(questions) & {"change_of_control", "assignment", "termination", "renewal"} else []

    def add(kind, severity, title, detail, ev):
        if (ev["start"], kind) in used:
            return
        used.add((ev["start"], kind))
        out.append({"kind": kind, "severity": severity, "title": title, "detail": detail,
                    "evidence": {"start": ev["start"], "end": ev["end"], "quote": ev["quote"]}, "clause": ev.get("clause")})
    coc_sentences = set()
    if "change_of_control" in questions:                  # every binding sentence about a change of control, period or not
        for m in re.finditer(r"change\s+of\s+control", text, re.I):
            ev = _sentence(text, rows, m.start())
            if not ev["clause"] or not ox.MODAL.search(ev["quote"]) or ev["start"] in coc_sentences:
                continue
            coc_sentences.add(ev["start"])
            q = ev["quote"]
            verb = ("lets the other party end the agreement" if re.search(r"\bterminat", q, re.I) else
                    "needs consent" if re.search(r"\bconsent", q, re.I) else "needs notice" if re.search(r"\bnotice|notify", q, re.I)
                    else "has consequences")
            add("change_of_control", "high", f"Change of control (clause {ev['clause']})", f"A change of control {verb}.", ev)
    if "assignment" in questions:
        for o in obs:
            if o["kind"] == "consent" and re.search(r"\bassign", o["evidence"]["quote"], re.I):
                who, act = o["party"] or "A party", o["action"]
                verb = ("shall" if act.startswith("not ") else "may" if re.match(r"(?i)neither\b", who) else "may not")   # never "may not not"
                add("assignment", "medium", f"Assignment needs consent (clause {o['clause']})",
                    f"{who} {verb} {act} {o['when'] or ''}".strip() + ".", {**o["evidence"], "clause": o["clause"]})
    if "termination" in questions:
        for o in obs:
            if o["kind"] != "termination" or o["evidence"]["start"] in coc_sentences:
                continue
            q = o["evidence"]["quote"]
            if re.search(r"convenience", q, re.I):
                title, sev = "Termination for convenience", "medium"
            elif re.search(r"anniversary", q, re.I):
                title, sev = "Break right", "medium"
            else:
                title, sev = "Termination right", "low"
            when = f" Notice is due by {_fmt(o['due_date'])}." if o.get("due_date") else ""
            add("termination", sev, f"{title} (clause {o['clause']})", f"{o['party'] or 'A party'} may {o['action']}: {o['when'] or ''}.{when}",
                {**o["evidence"], "clause": o["clause"]})
    if "renewal" in questions:
        for o in obs:
            if o["kind"] not in ("renewal", "expiry") or not o.get("due_date"):
                continue
            due = date.fromisoformat(o["due_date"])
            days = (due - today).days
            if o["kind"] == "renewal":
                if days < 0:
                    sev, lead = "high", f"The notice deadline ({_fmt(o['due_date'])}) has passed: the agreement renews unless the other party agrees otherwise."
                else:
                    sev, lead = ("medium" if days <= 183 else "low"), f"Notice of non-renewal is due by {_fmt(o['due_date'])}."
                add("renewal", sev, f"Automatic renewal (clause {o['clause']})", f"{lead} {o['due_note']}", {**o["evidence"], "clause": o["clause"]})
            elif days <= 366:
                add("expiry", "medium" if days >= 0 else "high", f"Term ends {_fmt(o['due_date'])} (clause {o['clause']})", o["due_note"],
                    {**o["evidence"], "clause": o["clause"]})
    if "governing_law" in questions:
        m = _LAW.search(text)
        if m:
            law = m.group("j").strip()
            law = f"the {law}" if re.match(r"(?:State|Commonwealth|Province|Republic|Kingdom|Federal Republic) of\b", law) else law
            if expected_law and _norm(expected_law) not in _norm(law) and _norm(law) not in _norm(expected_law):
                add("governing_law", "medium", f"Governed by the laws of {law}", f"The checklist expects {expected_law}.", _sentence(text, rows, m.start()))
    if "liability" in questions:
        for m in re.finditer(r"\bliabilit\w*[^.\n]{0,120}?\b(?:shall not exceed|is limited to|limited to|capped at|unlimited)\b", text, re.I):
            ev = _sentence(text, rows, m.start())
            if re.search(r"\bunlimited\b", m.group(0), re.I):
                add("liability", "high", f"Unlimited liability (clause {ev['clause']})", "Liability is not capped.", ev)
            else:
                add("liability", "low", f"Liability is capped (clause {ev['clause']})", "Check the cap against the deal value.", ev)
    return out


async def run(job: dict) -> Optional[str]:
    """The job: read every document the person who started the run can see, against the checklist."""
    run_id = job["object_id"]
    with get_conn() as conn:
        r = conn.execute("SELECT r.matter_id, r.created_by, c.content_json FROM legal_dd_runs r JOIN legal_dd_checklists c ON c.id = r.checklist_id "
                         "WHERE r.id=?", (run_id,)).fetchone()
        if not r:
            return "no_run"
        conn.execute("UPDATE legal_dd_runs SET status='running' WHERE id=?", (run_id,))
        conn.commit()
    matter_id, by, cj = r
    content = json.loads(cj)
    try:
        today = datetime.now(timezone.utc).date()
        docs = [d for d in documents.list_documents(matter_id, by) if d.get("current_version") and d["current_version"]["index_status"] == "indexed"]
        texts: dict[str, tuple] = {}
        for d in docs:
            v = _version(matter_id, d["id"])
            if v:
                texts[d["id"]] = v
        items, issues = [], []
        for it in content["items"]:
            found = []
            for d in docs:
                if d["id"] not in texts:
                    continue
                score = match(it, d["title"] or "", _first_line(texts[d["id"]][1]))
                if score:
                    found.append({"id": d["id"], "title": d["title"], "score": score})
            found.sort(key=lambda x: (-x["score"], x["title"]))
            items.append({"item_id": it["id"], "title": it["title"], "section": it["section"], "documents": found, "missing": not found})
            for d in found:
                vid, text, rows = texts[d["id"]]
                for i in ask(it.get("questions") or [], text, rows, content.get("expected_law"), today):
                    issues.append({**i, "item_id": it["id"], "document_id": d["id"], "version_id": vid})
        now = policy.now_iso()
        with get_conn() as conn:
            for i in issues:
                iid = str(uuid.uuid4())
                conn.execute("INSERT INTO legal_dd_issues (id, run_id, matter_id, item_id, document_id, version_id, kind, severity, title, detail, "
                             "evidence_json, status, reviewer_id, decided_at, version, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                             (iid, run_id, matter_id, i["item_id"], i["document_id"], i["version_id"], i["kind"], i["severity"], i["title"],
                              i["detail"], json.dumps({**i["evidence"], "clause": i.get("clause")}, ensure_ascii=False), "preliminary", None, None, 1, now))
                conn.execute("INSERT INTO legal_dd_issue_events (id, issue_id, action, reason, by, at) VALUES (?,?,?,?,?,?)",
                             (str(uuid.uuid4()), iid, "found", None, by, now))
            conn.execute("UPDATE legal_dd_runs SET status='done', results_json=?, finished_at=? WHERE id=?",
                         (json.dumps({"items": items, "engine": VERSION, "documents_read": len(texts)}, ensure_ascii=False), now, run_id))
            conn.commit()
        return None
    except Exception as e:                                   # noqa: BLE001 — recorded on the run; the job keeps the class
        with get_conn() as conn:
            conn.execute("UPDATE legal_dd_runs SET status='failed', error=?, finished_at=? WHERE id=?", (type(e).__name__, policy.now_iso(), run_id))
            conn.commit()
        raise
