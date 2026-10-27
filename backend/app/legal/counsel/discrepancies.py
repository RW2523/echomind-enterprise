"""Potential inconsistencies between a matter's documents, and the people who resolve them (FR06, AC05, Journey D).

The existing conflict scan (compare/scan.py) proposes them: facts of the same kind that two related documents state
differently, each checked in context by the model or reported by a rule. Here they are kept per matter, de-duplicated
by fingerprint (the two exact spans), so asking twice neither calls the model twice nor loses a resolution.

States: `potential` (only the machine has looked) → `in_review` → `explained` / `confirmed` / `dismissed`. Only a
person changes the state (`findings.review` on the matter), with a reason for every resolution; each change is written
once to legal_discrepancy_reviews and to the audit trail. Wording stays evidence-relative: "potential inconsistency",
never a verdict about anyone's honesty.

A discrepancy is shown only to someone who may read both documents it quotes (matter access and source permissions).
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Optional
from urllib.parse import quote as urlquote

from ...core.db import get_conn
from .. import policy
from ..documents import repo as doc_repo
from ..documents import versions as doc_versions
from . import store

OPEN = ("potential", "in_review")
RESOLVED = ("explained", "confirmed", "dismissed")
REVIEW_TO = ("in_review",) + RESOLVED
_STATUS_RANK = {"potential": 0, "in_review": 1, "explained": 2, "confirmed": 2, "dismissed": 3}

_KIND = {"date": "dates", "amount": "amounts", "period": "periods", "percent": "percentages", "law": "governing law",
         "entity": "party names", "reference": "a referenced agreement's date"}
_CAUTION = {
    "date": "The two dates may describe different events (for example dispatch and arrival), or one document may correct "
            "the other. Read both passages before resolving.",
    "amount": "The amounts may refer to different items, periods or currencies. Read both passages before resolving.",
    "period": "The periods may apply to different obligations. Read both passages before resolving.",
}
_DEFAULT_CAUTION = "The documents may describe different things. Read both passages before resolving."


class NotFound(Exception):
    pass


class Conflict(Exception):
    def __init__(self, message: str, current: Optional[dict] = None):
        super().__init__(message)
        self.current = current


class Invalid(Exception):
    pass


# ── From a scan ──────────────────────────────────────────────────────────────────
def fingerprint(c: dict) -> str:
    """The two exact spans, order-independent: the same pair found again is the same discrepancy."""
    sides = sorted([(c["a"]["version_id"], c["a"]["value_start"], c["a"]["value_end"]),
                    (c["b"]["version_id"], c["b"]["value_start"], c["b"]["value_end"])])
    raw = c["kind"] + "|" + "|".join(f"{v}:{s}-{e}" for v, s, e in sides)
    return hashlib.sha256(raw.encode()).hexdigest()


def versions_sha(docs: list[dict]) -> str:
    return hashlib.sha256("|".join(sorted(d["version_id"] for d in docs)).encode()).hexdigest()


def already_scanned(matter_id: str, sha: str) -> bool:
    with get_conn() as conn:
        return conn.execute("SELECT 1 FROM legal_discrepancy_scans WHERE matter_id=? AND versions_sha=?", (matter_id, sha)).fetchone() is not None


def record_scan(matter_id: str, sha: str, by: str, documents: int, candidates: int, found: int) -> None:
    with get_conn() as conn:
        conn.execute("INSERT OR REPLACE INTO legal_discrepancy_scans (matter_id, versions_sha, scanned_by, documents, candidates, found, "
                     "scanned_at) VALUES (?,?,?,?,?,?,?)", (matter_id, sha, by, documents, candidates, found, policy.now_iso()))
        conn.commit()


def _detail(c: dict) -> dict:
    keep = ("document_id", "document_title", "version_id", "version_number", "version_status", "clause", "page", "line",
            "value", "value_start", "value_end", "context", "context_start", "context_end")
    explanation = c.get("explanation") or ""
    if c.get("failed"):                       # a sentence whose quote was not found is never shown as support
        explanation = ""
    return {"a": {k: c["a"].get(k) for k in keep}, "b": {k: c["b"].get(k) for k in keep}, "subject": c.get("subject") or "",
            "explanation": explanation[:1500]}


def upsert(org_id: str, matter_id: str, checked: list[dict], by: str) -> list[str]:
    """Keep the scan's conflicts and unclear findings. A known one keeps its state; an unreviewed one is refreshed."""
    ids, now = [], policy.now_iso()
    with get_conn() as conn:
        for c in checked:
            if c.get("verdict") not in ("conflict", "unclear"):
                continue
            fp = fingerprint(c)
            row = conn.execute("SELECT id, status FROM legal_discrepancies WHERE matter_id=? AND fingerprint=?", (matter_id, fp)).fetchone()
            detail = json.dumps(_detail(c), ensure_ascii=False)
            method = "rule" if c.get("method") == "rule" else "model"
            if row:
                if row[1] == "potential":
                    conn.execute("UPDATE legal_discrepancies SET verdict=?, method=?, score=?, detail_json=?, updated_at=? WHERE id=?",
                                 (c["verdict"], method, float(c.get("score") or 0), detail, now, row[0]))
                ids.append(row[0])
                continue
            did = store.new_id("dsc")
            conn.execute("INSERT INTO legal_discrepancies (id, org_id, matter_id, fingerprint, kind, verdict, method, status, score, "
                         "document_a_id, document_b_id, version_a_id, version_b_id, detail_json, found_by, created_at, updated_at) "
                         "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (did, org_id, matter_id, fp, c["kind"], c["verdict"], method, "potential", float(c.get("score") or 0),
                          c["a"]["document_id"], c["b"]["document_id"], c["a"]["version_id"], c["b"]["version_id"], detail, by, now, now))
            ids.append(did)
        conn.commit()
    return ids


# ── Reading ──────────────────────────────────────────────────────────────────────
_COLS = ("id", "matter_id", "kind", "verdict", "method", "status", "score", "document_a_id", "document_b_id",
         "version_a_id", "version_b_id", "detail_json", "version", "created_at", "updated_at")


def _row(r) -> dict:
    d = dict(zip(_COLS, r))
    d["detail"] = json.loads(d.pop("detail_json"))
    return d


def _visible(row: dict, uid: str) -> bool:
    return doc_repo.visible(row["matter_id"], row["document_a_id"], uid) and doc_repo.visible(row["matter_id"], row["document_b_id"], uid)


def get(principal: dict, matter_id: str, discrepancy_id: str) -> Optional[dict]:
    with get_conn() as conn:
        r = conn.execute(f"SELECT {', '.join(_COLS)} FROM legal_discrepancies WHERE id=? AND matter_id=?",
                         (discrepancy_id, matter_id)).fetchone()
    row = _row(r) if r else None
    return row if row and _visible(row, principal["sub"]) else None


_WORD = re.compile(r"[a-z0-9]{3,}")


def listed(principal: dict, matter_id: str, statuses=OPEN, query: str = "") -> list[dict]:
    """Discrepancies the person may see, most important first: unresolved before reviewed, conflicts before unclear,
    then those whose passages share the most words with the question, then the scan's own score."""
    marks = ",".join("?" * len(statuses))
    with get_conn() as conn:
        rows = [_row(r) for r in conn.execute(f"SELECT {', '.join(_COLS)} FROM legal_discrepancies WHERE matter_id=? AND status IN ({marks})",
                                              (matter_id, *statuses)).fetchall()]
    rows = [r for r in rows if _visible(r, principal["sub"])]
    words = set(_WORD.findall(query.lower()))

    def overlap(r: dict) -> int:
        text = (r["detail"]["a"].get("context") or "") + " " + (r["detail"]["b"].get("context") or "")
        return len(words & set(_WORD.findall(text.lower())))
    rows.sort(key=lambda r: (_STATUS_RANK.get(r["status"], 9), r["verdict"] != "conflict", -overlap(r), -r["score"], r["created_at"]))
    return rows


def _username(conn, uid: str) -> str:
    r = conn.execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone()
    return r[0] if r else "someone"


def history(discrepancy_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT to_status, by_user, at, note FROM legal_discrepancy_reviews WHERE discrepancy_id=? ORDER BY at",
                            (discrepancy_id,)).fetchall()
        return [{"to_status": r[0], "by": _username(conn, r[1]), "at": r[2], **({"note": r[3]} if r[3] else {})} for r in rows]


def _stale(conn, document_id: str, version_id: str) -> bool:
    """A newer version than the one quoted is now what answers come from."""
    r = conn.execute(f"SELECT d.current_version_id, {doc_versions._LATEST_SIGNED.format(doc='d.id')} FROM legal_documents d WHERE d.id=?",
                     (document_id,)).fetchone()
    return bool(r) and version_id not in (r[0], r[1])


def title(row: dict) -> str:
    a, b = row["detail"]["a"], row["detail"]["b"]
    return f"Potential inconsistency in {_KIND.get(row['kind'], 'the facts')}: “{a['value']}” and “{b['value']}”"[:160]


def _side(conn, number: str, s: dict) -> dict:
    ctx = s.get("context") or ""
    vs = max(0, min(len(ctx), (s.get("value_start") or 0) - (s.get("context_start") or 0)))
    ve = max(vs, min(len(ctx), (s.get("value_end") or 0) - (s.get("context_start") or 0)))
    page = s.get("page")
    return {"document_id": s["document_id"], "document_title": s.get("document_title") or "", "version_id": s["version_id"],
            "version_number": s.get("version_number"), "version_status": s.get("version_status"), "clause": s.get("clause"),
            "page": page if isinstance(page, int) and page >= 1 else None, "value": (s.get("value") or "")[:300],
            "context": ctx[:1500], "value_start": min(vs, 1500), "value_end": min(ve, 1500),
            "reader_path": f"/m/{urlquote(number, safe='')}/documents/{s['document_id']}/read?v={s['version_id']}"
                           f"&at={s.get('value_start') or 0}-{max((s.get('value_end') or 0), (s.get('value_start') or 0) + 1)}",
            "stale": _stale(conn, s["document_id"], s["version_id"])}


def _explanation(text: str) -> str:
    return re.sub(r"\[([12])\]", lambda m: "[A]" if m.group(1) == "1" else "[B]", text or "")


def render(principal: dict, row: dict) -> dict:
    """The evidence_comparison props for one discrepancy, as the person may see it now."""
    with get_conn() as conn:
        number = conn.execute("SELECT number FROM matters WHERE id=?", (row["matter_id"],)).fetchone()[0]
        a, b = _side(conn, number, row["detail"]["a"]), _side(conn, number, row["detail"]["b"])
    return {"discrepancy_id": row["id"], "kind": row["kind"], "title": title(row), "status": row["status"], "a": a, "b": b,
            "explanation": _explanation(row["detail"].get("explanation") or "")[:1500],
            "caution": _CAUTION.get(row["kind"], _DEFAULT_CAUTION), "method": row["method"], "history": history(row["id"])[-20:],
            "can_review": policy.authorize(principal, "findings.review", row["matter_id"]).allowed}


# ── Resolving ────────────────────────────────────────────────────────────────────
def review(principal: dict, matter_id: str, discrepancy_id: str, to_status: str, note: Optional[str], expected_version: int) -> dict:
    """Change a discrepancy's state. Raises NotFound, Invalid or Conflict (someone changed it first)."""
    if to_status not in REVIEW_TO:
        raise Invalid("Choose in review, explained, confirmed or dismissed.")
    note = " ".join((note or "").split()) or None
    if to_status in RESOLVED and (not note or len(note) < 3):
        raise Invalid("Say briefly why, so the next reader knows how this was resolved.")
    if note and len(note) > 1000:
        raise Invalid("Keep the reason under 1,000 characters.")
    row = get(principal, matter_id, discrepancy_id)
    if not row:
        raise NotFound(discrepancy_id)
    if row["version"] != expected_version:
        raise Conflict("Someone changed this item while you were reading it. Here is its current state.", row)
    if row["status"] == to_status:
        raise Invalid("It is already in that state.")
    now = policy.now_iso()
    with get_conn() as conn:
        cur = conn.execute("UPDATE legal_discrepancies SET status=?, version=version+1, updated_at=? WHERE id=? AND version=?",
                           (to_status, now, discrepancy_id, expected_version))
        if cur.rowcount != 1:
            conn.rollback()
            raise Conflict("Someone changed this item while you were reading it. Here is its current state.",
                           get(principal, matter_id, discrepancy_id))
        conn.execute("INSERT INTO legal_discrepancy_reviews (id, discrepancy_id, matter_id, from_status, to_status, note, by_user, at) "
                     "VALUES (?,?,?,?,?,?,?,?)", (store.new_id("dsr"), discrepancy_id, matter_id, row["status"], to_status, note,
                                                  principal["sub"], now))
        conn.commit()
    return {**get(principal, matter_id, discrepancy_id), "from_status": row["status"]}


def purge_document(conn, matter_id: str, document_id: str) -> None:
    ids = [r[0] for r in conn.execute("SELECT id FROM legal_discrepancies WHERE matter_id=? AND (document_a_id=? OR document_b_id=?)",
                                      (matter_id, document_id, document_id))]
    if ids:
        marks = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM legal_discrepancy_reviews WHERE discrepancy_id IN ({marks})", ids)
        conn.execute(f"DELETE FROM legal_discrepancies WHERE id IN ({marks})", ids)
    conn.execute("DELETE FROM legal_discrepancy_scans WHERE matter_id=?", (matter_id,))
