"""Council Discovery (EchoMind Counsel Intelligence, iteration 2; owner's request 2026-11-05).

Questions worth putting to the Council, found by rules in records the matter already holds — never written by the model:
- `inconsistency`: a potential inconsistency in the register (counsel/discrepancies.py) that no one has resolved;
- `amended`: a sentence of a matter document that replaces or changes a clause of another agreement (research/amendments.py);
- `deadline`: a deadline read from a clause (obligations) that no lawyer has confirmed and that falls within the next
  LEGAL_DISCOVERY_DEADLINE_DAYS days.

Each question is a template filled with the documents' titles, clause numbers and values, and keeps its sources (document,
version, characters). Its fingerprint (the detector and those exact sources) keeps it from being suggested twice: a dismissed
question stays dismissed until its sources change (a new version is a new fingerprint), and a suggested one whose finding is
gone (resolved, confirmed, replaced) becomes stale. A question is shown only to someone who may read every document it quotes,
at most MAX_SHOWN per matter, highest score first; the weights are fixed below so a reviewer can see why one ranks first.

Discovery calls no model. It runs as a job after a document is read, after the register or the obligations change, and once a
day (deadlines come nearer); convening a question is a person's decision, and the Council run is the review panel's.
"""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta
from typing import Optional

from ...core.config import settings
from ...core.db import get_conn
from .. import policy
from ..counsel import discrepancies
from ..documents import repo as doc_repo
from ..documents import versions as doc_versions
from ..research import amendments

MAX_SHOWN = 3
SEVERITY = {"high": 3, "medium": 2}                 # score = 10 × severity + urgency (deadlines only: up to 10 as the date nears)
_HIGH_KINDS = ("date", "amount", "period")          # an inconsistency in these can move a deadline or a payment
_HIGH_DAYS = 14                                     # a deadline this close is high priority
_TITLE = 300


def enabled() -> bool:
    return bool(settings.LEGAL_PANEL and settings.LEGAL_COUNCIL_DISCOVERY)


def _q(detector: str, fingerprint: str, title: str, reason: str, priority: str, sources: list[dict], urgency: float = 0.0) -> dict:
    return {"detector": detector, "fingerprint": f"{detector}:{fingerprint}", "title": " ".join(title.split())[:_TITLE],
            "reason": " ".join(reason.split())[:400], "priority": priority, "score": 10 * SEVERITY[priority] + urgency, "sources": sources}


def _src(document_id, version_id, title, start, end) -> dict:
    return {"document_id": document_id, "version_id": version_id, "document_title": title,
            "start": start if isinstance(start, int) else None, "end": end if isinstance(end, int) else None}


def inconsistencies(conn, matter_id: str) -> list[dict]:
    out = []
    for rid, fp, kind, da, db, va, vb, detail in conn.execute(
            "SELECT id, fingerprint, kind, document_a_id, document_b_id, version_a_id, version_b_id, detail_json FROM legal_discrepancies "
            f"WHERE matter_id=? AND status IN ({','.join('?' * len(discrepancies.OPEN))}) ORDER BY created_at",
            (matter_id, *discrepancies.OPEN)).fetchall():
        d = json.loads(detail or "{}")
        a, b = d.get("a") or {}, d.get("b") or {}
        ta, tb = a.get("document_title") or "one document", b.get("document_title") or "another"
        what = discrepancies._KIND.get(kind, kind)
        out.append(_q("inconsistency", fp, f"Are “{a.get('value')}” in {ta} and “{b.get('value')}” in {tb} genuinely inconsistent?",
                      f"A potential inconsistency in {what} is in the register and no one has resolved it.",
                      "high" if kind in _HIGH_KINDS else "medium",
                      [_src(da, va, ta, a.get("value_start"), a.get("value_end")), _src(db, vb, tb, b.get("value_start"), b.get("value_end"))]))
    return out


def amended(conn, matter_id: str) -> list[dict]:
    pick = f"COALESCE({doc_versions._LATEST_SIGNED.format(doc='d.id')}, d.current_version_id)"
    like = " OR ".join("t.text LIKE ?" for _ in amendments._CANDIDATE_WORDS)
    out = []
    for did, title, vid, status, text in conn.execute(
            f"SELECT d.id, d.title, v.id, v.status, t.text FROM legal_documents d JOIN legal_document_versions v ON v.id = {pick} "
            f"JOIN legal_version_text t ON t.version_id = v.id WHERE d.matter_id=? AND d.deleted_at IS NULL AND ({like}) "
            "ORDER BY d.created_at LIMIT 50", (matter_id, *[f"%{w}%" for w in amendments._CANDIDATE_WORDS])).fetchall():
        for ch in amendments.changes(text or ""):
            signed = status == "executed"
            out.append(_q("amended", f"{vid}:{ch.start}", f"How does {title} change clause {ch.clause} of the {ch.title}?",
                          f"{title}{'' if signed else ' (not signed)'} says clause {ch.clause} of the {ch.title} is {ch.how}.",
                          "high" if signed else "medium", [_src(did, vid, title, ch.start, ch.end)]))
    return out


def deadlines(conn, matter_id: str, today: Optional[date] = None) -> list[dict]:
    today = today or datetime.now().date()
    window = int(settings.LEGAL_DISCOVERY_DEADLINE_DAYS)
    out = []
    for oid, did, vid, clause, kind, action, due, evidence, title in conn.execute(
            "SELECT o.id, o.document_id, o.version_id, o.clause, o.kind, o.action_text, o.due_date, o.evidence_json, d.title "
            "FROM legal_obligations o JOIN legal_documents d ON d.id = o.document_id WHERE o.matter_id=? AND o.status='proposed' "
            "AND o.due_date IS NOT NULL AND o.due_date >= ? AND o.due_date <= ? AND d.deleted_at IS NULL ORDER BY o.due_date",
            (matter_id, today.isoformat(), (today + timedelta(days=window)).isoformat())).fetchall():
        days = (date.fromisoformat(due[:10]) - today).days
        ev = json.loads(evidence or "{}")
        where = f"clause {clause} of {title}" if clause else title
        out.append(_q("deadline", f"{oid}:{due}", f"What must be ready before {due} under {where}?",
                      f"A {kind} deadline read from the document, {days} day{'s' if days != 1 else ''} away and not yet confirmed by a "
                      f"lawyer: {action[:200]}", "high" if days <= _HIGH_DAYS else "medium",
                      [_src(did, vid, title, ev.get("start"), ev.get("end"))], urgency=10 * (window - days) / max(window, 1)))
    return out


DETECTORS = (inconsistencies, amended, deadlines)


def run_for_matter(org_id: str, matter_id: str) -> dict:
    """Find the matter's questions now and bring the stored ones up to date. Returns counts."""
    with get_conn() as conn:
        found = {q["fingerprint"]: q for detect in DETECTORS for q in detect(conn, matter_id)}
        have = {fp: (qid, status) for qid, fp, status in conn.execute(
            "SELECT id, fingerprint, status FROM legal_council_questions WHERE matter_id=? AND origin='discovered'", (matter_id,))}
        now, added, stale = policy.now_iso(), 0, 0
        for fp, q in found.items():
            if fp in have:
                qid, status = have[fp]
                conn.execute("UPDATE legal_council_questions SET title=?, reason=?, priority=?, score=?, sources_json=?, updated_at=?, "
                             "status=CASE WHEN status='stale' THEN 'suggested' ELSE status END WHERE id=?",
                             (q["title"], q["reason"], q["priority"], q["score"], json.dumps(q["sources"], ensure_ascii=False), now, qid))
                continue
            conn.execute("INSERT INTO legal_council_questions (id, org_id, matter_id, origin, detector, fingerprint, title, reason, priority, score, "
                         "sources_json, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (str(uuid.uuid4()), org_id, matter_id, "discovered", q["detector"], fp, q["title"], q["reason"], q["priority"],
                          q["score"], json.dumps(q["sources"], ensure_ascii=False), "suggested", now, now))
            added += 1
        for fp, (qid, status) in have.items():
            if fp not in found and status == "suggested":
                conn.execute("UPDATE legal_council_questions SET status='stale', updated_at=? WHERE id=?", (now, qid))
                stale += 1
        conn.commit()
    return {"found": len(found), "added": added, "stale": stale}


async def run(job: dict) -> Optional[str]:
    """The `discovery.run` job: rules only."""
    if not enabled():
        return None
    run_for_matter(job["org_id"], job["matter_id"])
    return None


def enqueue(conn, org_id: str, matter_id: str, by: str) -> bool:
    """Queue discovery for a matter, once: a run already waiting covers this change too."""
    if not enabled() or conn.execute("SELECT 1 FROM legal_jobs WHERE kind='discovery.run' AND matter_id=? AND status='queued'",
                                     (matter_id,)).fetchone():
        return False
    conn.execute("INSERT INTO legal_jobs (id, kind, org_id, matter_id, object_id, requested_by, created_at) VALUES (?,?,?,?,?,?,?)",
                 (str(uuid.uuid4()), "discovery.run", org_id, matter_id, matter_id, by, policy.now_iso()))
    return True


def _row(r) -> dict:
    keys = ("id", "matter_id", "origin", "detector", "title", "reason", "priority", "score", "sources_json", "status", "review_id",
            "created_at", "updated_at")
    d = dict(zip(keys, r))
    d["sources"] = json.loads(d.pop("sources_json") or "[]")
    return d


_COLS = "id, matter_id, origin, detector, title, reason, priority, score, sources_json, status, review_id, created_at, updated_at"


def visible(row: dict, uid: str) -> bool:
    """Only to someone who may read every document the question quotes."""
    return all(doc_repo.visible(row["matter_id"], s["document_id"], uid) for s in row["sources"])


def listed(principal: dict, matter_id: str) -> tuple[list[dict], int]:
    """The suggested questions this person may see, best first: at most MAX_SHOWN, and how many more."""
    with get_conn() as conn:
        rows = [_row(r) for r in conn.execute(f"SELECT {_COLS} FROM legal_council_questions WHERE matter_id=? AND status='suggested' "
                                              "ORDER BY score DESC, created_at DESC", (matter_id,)).fetchall()]
    mine = [r for r in rows if visible(r, principal["sub"])]
    return mine[:MAX_SHOWN], max(0, len(mine) - MAX_SHOWN)


def get(principal: dict, matter_id: str, question_id: str) -> Optional[dict]:
    with get_conn() as conn:
        r = conn.execute(f"SELECT {_COLS} FROM legal_council_questions WHERE id=? AND matter_id=?", (question_id, matter_id)).fetchone()
    row = _row(r) if r else None
    return row if row and visible(row, principal["sub"]) else None


def decide(question_id: str, status: str, by: str, reason: Optional[str] = None, review_id: Optional[str] = None) -> bool:
    """A person convenes or dismisses a suggested question; False when it is no longer suggested."""
    with get_conn() as conn:
        n = conn.execute("UPDATE legal_council_questions SET status=?, decided_by=?, decided_reason=?, review_id=COALESCE(?, review_id), "
                         "updated_at=? WHERE id=? AND status='suggested'", (status, by, reason, review_id, policy.now_iso(), question_id)).rowcount
        conn.commit()
    return n == 1


def shown(row: dict, number: str) -> dict:
    """A question as the page shows it: its sources open at the quoted characters."""
    from urllib.parse import quote
    def href(s: dict) -> Optional[str]:
        base = f"/m/{quote(number, safe='')}/documents/{quote(s['document_id'], safe='')}/read?v={quote(s['version_id'] or '', safe='')}"
        return base + (f"&at={s['start']}-{s['end']}" if s.get("start") is not None and s.get("end") is not None else "")
    return {"id": row["id"], "title": row["title"], "reason": row["reason"], "priority": row["priority"], "detector": row["detector"],
            "status": row["status"], "created_at": row["created_at"],
            "sources": [{"document_title": s.get("document_title"), "href": href(s)} for s in row["sources"]]}
