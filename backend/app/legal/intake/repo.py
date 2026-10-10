"""Intake records (S4.1, D-054): enquiries, their screening, consent and recording, replies and conversion.

Who sees what:
- front-desk staff and lawyers see the answers and a screening that says only whether to refer;
- the conflicts team and managing partners also see the matches and the decision notes;
- an enquirer with a link sees the disclaimer, the questions and, once a lawyer approved it, the reply.

No model is used anywhere here.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from ...core.config import settings
from ...core.db import get_conn
from ..documents import blobs
from ..matters import repo as matters
from ..policy import now_iso
from . import replies, screening
from .script import DISCLAIMER, SCRIPT, SCRIPT_VERSION, names, validate

MESSAGES = {"not_run": "Not screened yet: the answers are not in.",
            "no_match": "No match found by the automatic check. The conflicts team confirms before we act.",
            "possible_conflict": "Possible conflict: refer to the conflicts team."}
_COLS = ("id", "org_id", "reference", "channel", "status", "script_version", "disclaimer_version", "disclaimer_accepted_at", "answers_json",
         "screening", "screened_at", "decision", "link_hash", "link_expires_at", "link_submitted_at", "matter_id", "created_by", "created_at",
         "updated_at", "version")


class Conflict(Exception):
    pass


class Invalid(ValueError):
    pass


def _id() -> str:
    return str(uuid.uuid4())


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _event(conn, intake_id: str, action: str, actor: Optional[str], note: Optional[str] = None) -> None:
    conn.execute("INSERT INTO legal_intake_events (id, intake_id, action, actor_id, at, note) VALUES (?,?,?,?,?,?)",
                 (_id(), intake_id, action, actor, now_iso(), note))


def _row(conn, intake_id: str, org_id: Optional[str] = None) -> Optional[dict]:
    q = f"SELECT {', '.join(_COLS)} FROM legal_intakes WHERE id=?" + (" AND org_id=?" if org_id else "")
    r = conn.execute(q, (intake_id, org_id) if org_id else (intake_id,)).fetchone()
    return dict(zip(_COLS, r)) if r else None


def _touch(conn, intake_id: str, **fields) -> None:
    sets = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE legal_intakes SET {sets}{', ' if sets else ''}updated_at=?, version=version+1 WHERE id=?",
                 (*fields.values(), now_iso(), intake_id))


def _reference(conn, org_id: str) -> str:
    year = datetime.now(timezone.utc).year
    n = conn.execute("SELECT COUNT(*) FROM legal_intakes WHERE org_id=? AND reference LIKE ?", (org_id, f"INT-{year}-%")).fetchone()[0]
    return f"INT-{year}-{n + 1:04d}"


# ── Creating ─────────────────────────────────────────────────────────────────────
def create(org_id: str, channel: str, by: str) -> str:
    iid = _id()
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_intakes (id, org_id, reference, channel, status, script_version, created_by, created_at) "
                     "VALUES (?,?,?,?,?,?,?,?)", (iid, org_id, _reference(conn, org_id), channel, "open", SCRIPT_VERSION, by, now_iso()))
        _event(conn, iid, "created", by)
        conn.commit()
    return iid


def create_link(org_id: str, by: str) -> tuple[str, str]:
    """A new intake the enquirer answers through a single-use link; the token is returned once and only its hash kept."""
    iid, token = _id(), secrets.token_urlsafe(32)
    expires = (datetime.now(timezone.utc) + timedelta(days=settings.LEGAL_INTAKE_LINK_DAYS)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_intakes (id, org_id, reference, channel, status, script_version, link_hash, link_expires_at, created_by, "
                     "created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (iid, org_id, _reference(conn, org_id), "link", "invited", SCRIPT_VERSION, token_hash(token), expires, by, now_iso()))
        _event(conn, iid, "invited", by)
        conn.commit()
    return iid, token


def by_token(token: str) -> Optional[dict]:
    """The intake a live link opens, or None (unknown, expired or purged)."""
    with get_conn() as conn:
        r = conn.execute(f"SELECT {', '.join(_COLS)} FROM legal_intakes WHERE link_hash=?", (token_hash(token),)).fetchone()
    if not r:
        return None
    d = dict(zip(_COLS, r))
    if d["status"] in ("purged", "closed") or not d["link_expires_at"] or d["link_expires_at"] < now_iso():
        return None
    return d


# ── The script ───────────────────────────────────────────────────────────────────
def accept_disclaimer(intake_id: str, version: str, by: Optional[str]) -> None:
    if version != DISCLAIMER["version"]:
        raise Conflict("The disclaimer has changed. Read the current one.")
    with get_conn() as conn:
        d = _row(conn, intake_id)
        if d["status"] not in ("open", "invited"):
            raise Conflict("This enquiry is already submitted.")
        _touch(conn, intake_id, disclaimer_version=version, disclaimer_accepted_at=now_iso())
        _event(conn, intake_id, "disclaimer_accepted", by)
        conn.commit()


def submit(intake_id: str, answers: dict, by: Optional[str]) -> None:
    """Save the answers and screen the names they give (the matches are kept for the conflicts team)."""
    try:
        clean = validate(answers)
    except ValueError as e:
        raise Invalid(str(e)) from None
    with get_conn() as conn:
        d = _row(conn, intake_id)
        if d["disclaimer_version"] != DISCLAIMER["version"]:
            raise Conflict("The disclaimer comes first: it must be accepted before the questions.")
        if d["channel"] == "link" and d["link_submitted_at"]:
            raise Conflict("These answers are already in.")
        if d["status"] not in ("open", "invited", "submitted"):
            raise Conflict("The conflicts team has decided on this enquiry. Start a new one for new details.")
    hits = screening.screen(d["org_id"], names(clean))
    now = now_iso()
    with get_conn() as conn:
        conn.execute("DELETE FROM legal_intake_matches WHERE intake_id=?", (intake_id,))
        for h in hits:
            conn.execute("INSERT INTO legal_intake_matches (id, intake_id, searched_name, matched_name, kind, client_id, matter_id, score, created_at) "
                         "VALUES (?,?,?,?,?,?,?,?,?)", (_id(), intake_id, h["searched"], h["matched"], h["kind"], h["client_id"], h["matter_id"],
                                                        h["score"], now))
        _touch(conn, intake_id, answers_json=json.dumps(clean, ensure_ascii=False), status="submitted",
               screening="possible_conflict" if hits else "no_match", screened_at=now,
               **({"link_submitted_at": now} if d["channel"] == "link" else {}))
        _event(conn, intake_id, "submitted", by)
        _event(conn, intake_id, "screened", None)
        conn.commit()


def decide(intake_id: str, decision: str, note: Optional[str], by: str) -> None:
    with get_conn() as conn:
        d = _row(conn, intake_id)
        if d["status"] not in ("submitted", "cleared", "conflict"):
            raise Conflict("The conflicts team decides once the answers are in, and before a matter is opened.")
        _touch(conn, intake_id, decision=decision, status=decision)
        _event(conn, intake_id, decision, by, (note or "").strip() or None)
        conn.commit()


def close(intake_id: str, by: str) -> None:
    with get_conn() as conn:
        d = _row(conn, intake_id)
        if d["status"] in ("converted", "closed", "purged"):
            raise Conflict(f"This enquiry is already {d['status']}.")
        _touch(conn, intake_id, status="closed")
        _event(conn, intake_id, "closed", by)
        conn.commit()


# ── Consent and recording ────────────────────────────────────────────────────────
def _live_consent(conn, intake_id: str):
    return conn.execute("SELECT id, method, script_version, recorded_by, recorded_at FROM legal_intake_consents WHERE intake_id=? "
                        "AND withdrawn_at IS NULL ORDER BY recorded_at DESC LIMIT 1", (intake_id,)).fetchone()


def _live_recording(conn, intake_id: str):
    return conn.execute("SELECT id, wrapped_key, mime, bytes, uploaded_by, uploaded_at FROM legal_intake_recordings WHERE intake_id=? "
                        "AND removed_at IS NULL ORDER BY uploaded_at DESC LIMIT 1", (intake_id,)).fetchone()


def _remove_recording(conn, intake_id: str, why: str) -> bool:
    r = _live_recording(conn, intake_id)
    if not r:
        return False
    blobs.remove(intake_id, r[0], folder="intake")
    conn.execute("UPDATE legal_intake_recordings SET removed_at=?, removed_why=?, wrapped_key=NULL WHERE id=?", (now_iso(), why, r[0]))
    return True


def record_consent(intake_id: str, method: str, by: str) -> None:
    with get_conn() as conn:
        if _live_consent(conn, intake_id):
            raise Conflict("The caller's consent is already recorded.")
        conn.execute("INSERT INTO legal_intake_consents (id, intake_id, method, script_version, recorded_by, recorded_at) VALUES (?,?,?,?,?,?)",
                     (_id(), intake_id, method, DISCLAIMER["version"], by, now_iso()))
        _event(conn, intake_id, "consent", by)
        conn.commit()


def withdraw_consent(intake_id: str, by: str) -> None:
    """Consent withdrawn: recording stops and the recording made under it is removed."""
    with get_conn() as conn:
        c = _live_consent(conn, intake_id)
        if not c:
            raise Conflict("No consent is recorded.")
        conn.execute("UPDATE legal_intake_consents SET withdrawn_at=?, withdrawn_by=? WHERE id=?", (now_iso(), by, c[0]))
        _event(conn, intake_id, "consent_withdrawn", by)
        if _remove_recording(conn, intake_id, "consent withdrawn"):
            _event(conn, intake_id, "recording_removed", by)
        conn.commit()


def add_recording(intake_id: str, data: bytes, mime: str, by: str) -> None:
    with get_conn() as conn:
        c = _live_consent(conn, intake_id)
        if not c:
            raise Conflict("Record the caller's consent first: a call is recorded only with consent.")
        if _remove_recording(conn, intake_id, "replaced"):
            _event(conn, intake_id, "recording_removed", by)
        rid = _id()
        wrapped = blobs.store(intake_id, rid, data, folder="intake")
        conn.execute("INSERT INTO legal_intake_recordings (id, intake_id, consent_id, wrapped_key, mime, bytes, sha256, uploaded_by, uploaded_at) "
                     "VALUES (?,?,?,?,?,?,?,?,?)", (rid, intake_id, c[0], wrapped, mime, len(data), hashlib.sha256(data).hexdigest(), by, now_iso()))
        _event(conn, intake_id, "recording_added", by)
        conn.commit()


def recording(intake_id: str) -> Optional[tuple[bytes, str]]:
    with get_conn() as conn:
        r = _live_recording(conn, intake_id)
    if not r:
        return None
    return blobs.load(intake_id, r[0], r[1], folder="intake"), r[2]


# ── Replies ──────────────────────────────────────────────────────────────────────
def _firm_names(conn, d: dict) -> list[str]:
    reg = [e["name"] for e in screening.register(d["org_id"])]
    matched = [r[0] for r in conn.execute("SELECT matched_name FROM legal_intake_matches WHERE intake_id=?", (d["id"],))]
    return list(dict.fromkeys(reg + matched))


def draft_reply(intake_id: str, template: str, request: Optional[str], by: str) -> str:
    with get_conn() as conn:
        d = _row(conn, intake_id)
        if d["status"] in ("open", "invited", "purged"):
            raise Conflict("A reply follows the enquirer's answers.")
        if template in replies.NEEDS_CLEARANCE and d["decision"] != "cleared":
            raise Conflict("The conflicts team must clear this enquiry before we offer to help.")
        answers = json.loads(d["answers_json"])
        try:
            body = replies.render(template, d["reference"], answers, request)
        except replies.Invalid as e:
            raise Invalid(str(e)) from None
        if replies.leaks(body, _firm_names(conn, d), names(answers)):
            raise Invalid("The reply names another client or party of the firm: remove it.")
        conn.execute("UPDATE legal_intake_replies SET status='withdrawn' WHERE intake_id=? AND status='draft'", (intake_id,))
        rid = _id()
        conn.execute("INSERT INTO legal_intake_replies (id, intake_id, template, body, body_sha256, status, drafted_by, drafted_at) "
                     "VALUES (?,?,?,?,?,?,?,?)", (rid, intake_id, template, body, replies.sha(body), "draft", by, now_iso()))
        _touch(conn, intake_id)
        _event(conn, intake_id, "reply_drafted", by)
        conn.commit()
    return rid


def approve_reply(intake_id: str, reply_id: str, body_sha256: str, by: str) -> None:
    with get_conn() as conn:
        r = conn.execute("SELECT status, drafted_by, body_sha256, template FROM legal_intake_replies WHERE id=? AND intake_id=?",
                         (reply_id, intake_id)).fetchone()
        if not r:
            raise LookupError(reply_id)
        status, author, digest, template = r
        if status != "draft":
            raise Conflict(f"This reply is {status}.")
        if author == by:
            raise Conflict("A reply is approved by a lawyer who did not write it.")
        if digest != body_sha256:
            raise Conflict("This is not the text you read. Reload it and check it again.")
        d = _row(conn, intake_id)
        if template in replies.NEEDS_CLEARANCE and d["decision"] != "cleared":
            raise Conflict("The conflicts team must clear this enquiry before we offer to help.")
        conn.execute("UPDATE legal_intake_replies SET status='withdrawn' WHERE intake_id=? AND status='released'", (intake_id,))
        conn.execute("UPDATE legal_intake_replies SET status='released', approved_by=?, approved_at=? WHERE id=?", (by, now_iso(), reply_id))
        _touch(conn, intake_id)
        _event(conn, intake_id, "reply_released", by)
        conn.commit()


def released_reply(intake_id: str) -> Optional[str]:
    with get_conn() as conn:
        r = conn.execute("SELECT body FROM legal_intake_replies WHERE intake_id=? AND status='released'", (intake_id,)).fetchone()
    return r[0] if r else None


# ── Conversion ───────────────────────────────────────────────────────────────────
def convert(intake_id: str, org_id: str, by: str, client_id: Optional[str], client_name: Optional[str], client_code: Optional[str],
            number: str, title: str) -> dict:
    with get_conn() as conn:
        d = _row(conn, intake_id, org_id)
        if d["status"] == "converted":
            raise Conflict("This enquiry already became a matter.")
        if d["decision"] != "cleared" or d["status"] != "cleared":
            raise Conflict("The conflicts team must clear this enquiry before a matter is opened.")
        if conn.execute("SELECT 1 FROM matters WHERE org_id=? AND number=?", (org_id, number.strip())).fetchone():
            raise Conflict("A matter with that number already exists.")
    answers = json.loads(d["answers_json"])
    if client_id:
        c = matters.get_client(client_id)
        if not c or c["org_id"] != org_id:
            raise LookupError(client_id)
    else:
        if not (client_name or "").strip():
            raise Invalid("Name the client, or choose an existing one.")
        try:
            c = matters.create_client(org_id, client_name.strip(), (client_code or "").strip() or None, by)
        except matters.Conflict as e:
            raise Conflict(str(e)) from None
    try:
        m = matters.create_matter(org_id, c["id"], number, title, by)
    except matters.Conflict as e:
        raise Conflict(str(e)) from None
    now = now_iso()
    with get_conn() as conn:
        for name in answers.get("other_parties") or []:
            conn.execute("INSERT INTO legal_matter_parties (id, matter_id, name, role, added_by, added_at) VALUES (?,?,?,?,?,?)",
                         (_id(), m["id"], name, "adverse", by, now))
        _touch(conn, intake_id, status="converted", matter_id=m["id"])
        _event(conn, intake_id, "converted", by)
        conn.commit()
    return m


# ── Records ──────────────────────────────────────────────────────────────────────
def purge(intake_id: str, by: str) -> None:
    """Remove the enquirer's personal data: answers, matches, replies, the recording and decision notes. The trail of
    what was done, by whom and when, stays."""
    with get_conn() as conn:
        d = _row(conn, intake_id)
        if d["status"] == "purged":
            raise Conflict("This enquiry is already purged.")
        _remove_recording(conn, intake_id, "purged")
        conn.execute("UPDATE legal_intake_recordings SET mime='', removed_why='purged' WHERE intake_id=?", (intake_id,))
        conn.execute("DELETE FROM legal_intake_matches WHERE intake_id=?", (intake_id,))
        conn.execute("DELETE FROM legal_intake_replies WHERE intake_id=?", (intake_id,))
        conn.execute("UPDATE legal_intake_events SET note=NULL WHERE intake_id=? AND note IS NOT NULL", (intake_id,))
        _touch(conn, intake_id, answers_json="{}", status="purged", link_hash=None)
        _event(conn, intake_id, "purged", by)
        conn.commit()


# ── Views ────────────────────────────────────────────────────────────────────────
def _username(conn, uid: Optional[str]) -> Optional[str]:
    r = conn.execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone() if uid else None
    return r[0] if r else None


def get_row(intake_id: str, org_id: str) -> Optional[dict]:
    with get_conn() as conn:
        return _row(conn, intake_id, org_id)


def view(intake_id: str, org_id: str, conflicts: bool) -> Optional[dict]:
    """The intake for firm staff; `conflicts` adds the matches and the decision notes."""
    with get_conn() as conn:
        d = _row(conn, intake_id, org_id)
        if not d:
            return None
        consent = _live_consent(conn, intake_id)
        rec = _live_recording(conn, intake_id)
        reps = conn.execute("SELECT id, template, body, body_sha256, status, drafted_by, drafted_at, approved_by, approved_at "
                            "FROM legal_intake_replies WHERE intake_id=? ORDER BY drafted_at, rowid", (intake_id,)).fetchall()
        events = conn.execute("SELECT action, actor_id, at, note FROM legal_intake_events WHERE intake_id=? ORDER BY at, rowid",
                              (intake_id,)).fetchall()
        number = conn.execute("SELECT number FROM matters WHERE id=?", (d["matter_id"],)).fetchone() if d["matter_id"] else None
        out = {
            "id": d["id"], "reference": d["reference"], "channel": d["channel"], "status": d["status"], "created_at": d["created_at"],
            "created_by": _username(conn, d["created_by"]), "updated_at": d["updated_at"], "version": d["version"],
            "disclaimer": {**DISCLAIMER, "accepted_at": d["disclaimer_accepted_at"] if d["disclaimer_version"] == DISCLAIMER["version"] else None},
            "script": SCRIPT, "answers": json.loads(d["answers_json"]),
            "screening": {"result": d["screening"], "message": MESSAGES[d["screening"]], "decision": d["decision"]},
            "consent": {"method": consent[1], "recorded_by": _username(conn, consent[3]), "recorded_at": consent[4]} if consent else None,
            "recording": {"mime": rec[2], "bytes": rec[3], "uploaded_by": _username(conn, rec[4]), "uploaded_at": rec[5]} if rec else None,
            "replies": [{"id": r[0], "template": r[1], "body": r[2], "body_sha256": r[3], "status": r[4], "drafted_by": _username(conn, r[5]),
                         "drafted_at": r[6], "approved_by": _username(conn, r[7]), "approved_at": r[8]} for r in reps],
            "events": [{"action": a, "actor": _username(conn, actor), "at": at, **({"note": note} if conflicts and note else {})}
                       for a, actor, at, note in events],
            "matter_number": number[0] if number else None,
            "link": {"expires_at": d["link_expires_at"], "submitted_at": d["link_submitted_at"]} if d["channel"] == "link" else None,
        }
        if not conflicts:
            return {"intake": out}
        rows = conn.execute("SELECT x.searched_name, x.matched_name, x.kind, x.score, c.name, m.number FROM legal_intake_matches x "
                            "LEFT JOIN clients c ON c.id = x.client_id LEFT JOIN matters m ON m.id = x.matter_id WHERE x.intake_id=? "
                            "ORDER BY x.score DESC, x.searched_name", (intake_id,)).fetchall()
        return {"intake": out, "matches": [{"searched": s, "matched": mt, "kind": k, "score": sc, "client": cn, "matter_number": num}
                                           for s, mt, k, sc, cn, num in rows]}


def listing(org_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT id, reference, channel, status, answers_json, screening, decision, created_at, updated_at FROM legal_intakes "
                            "WHERE org_id=? ORDER BY created_at DESC", (org_id,)).fetchall()
    return [{"id": i, "reference": ref, "channel": ch, "status": st, "name": json.loads(a).get("your_name") or None,
             "organisation": json.loads(a).get("organisation") or None, "screening": {"result": sc, "message": MESSAGES[sc], "decision": dec},
             "created_at": ca, "updated_at": ua} for i, ref, ch, st, a, sc, dec, ca, ua in rows]


def public_view(d: dict) -> dict:
    """What a link shows its enquirer: never the screening."""
    status = {"invited": "invited"}.get(d["status"], "submitted")
    reply = released_reply(d["id"])
    return {"status": "replied" if reply else status, "reference": d["reference"],
            "disclaimer": {**DISCLAIMER, "accepted": d["disclaimer_version"] == DISCLAIMER["version"]},
            "script": SCRIPT, "reply": {"body": reply} if reply else None}
