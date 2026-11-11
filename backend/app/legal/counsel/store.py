"""Conversations, turns, stored components and action idempotency (migrations 0036, 0039).

Every function takes the owner and the matter and answers None when either does not match, so a guessed id of
someone else's conversation, turn or component behaves exactly like one that does not exist. A conversation its owner
deleted behaves the same way (it is kept as work product, never listed, opened or continued).

Turns form a tree (0039): each names the turn it follows. Asking again, or with the question edited, adds a sibling
(another version of that step), so every answer that was shown stays on record; the browser shows one path through it.
"""
from __future__ import annotations

import json
import secrets
from typing import Optional

from ...core.db import get_conn
from .. import policy

TITLE_MAX = 80
NEW_TITLE = "New conversation"
STALE_RUNNING_SEC = 600          # a turn still "running" after this long was cut off by a restart


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(16)}"


def _title(question: str) -> str:
    t = " ".join(question.split())
    return t if len(t) <= TITLE_MAX else t[:TITLE_MAX - 1].rstrip() + "…"


# ── Conversations ────────────────────────────────────────────────────────────────
def create_conversation(org_id: str, matter_ids: list[str], owner_id: str) -> dict:
    """A conversation about one matter, or several (the first is its own matter). The caller has authorized each."""
    cid, now = new_id("conv"), policy.now_iso()
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_conversations (id, org_id, matter_id, owner_id, title, created_at, updated_at) "
                     "VALUES (?,?,?,?,?,?,?)", (cid, org_id, matter_ids[0], owner_id, NEW_TITLE, now, now))
        if len(matter_ids) > 1:
            conn.executemany("INSERT INTO legal_conversation_matters (conversation_id, matter_id, position) VALUES (?,?,?)",
                             [(cid, m, i) for i, m in enumerate(matter_ids)])
        conn.commit()
    return {"id": cid, "matter_id": matter_ids[0], "matter_ids": list(matter_ids), "title": NEW_TITLE, "created_at": now,
            "updated_at": now, "turns": 0}


def matter_ids(conn, conversation_id: str, primary: str) -> list[str]:
    rows = conn.execute("SELECT matter_id FROM legal_conversation_matters WHERE conversation_id=? ORDER BY position",
                        (conversation_id,)).fetchall()
    return [r[0] for r in rows] or [primary]


def conversation(conversation_id: str, owner_id: str) -> Optional[dict]:
    with get_conn() as conn:
        r = conn.execute("SELECT id, org_id, matter_id, owner_id, title, created_at, updated_at FROM legal_conversations "
                         "WHERE id=? AND owner_id=? AND deleted_at IS NULL", (conversation_id, owner_id)).fetchone()
        if not r:
            return None
        conv = dict(zip(("id", "org_id", "matter_id", "owner_id", "title", "created_at", "updated_at"), r))
        conv["matter_ids"] = matter_ids(conn, conv["id"], conv["matter_id"])
    return conv


def _like(q: str) -> str:
    return "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _snippet(question: str, q: str, width: int = 110) -> str:
    """The words around the first match of `q` in a question, for a search result."""
    at = question.lower().find(q.lower())
    if at < 0 or len(question) <= width:
        return question[:width]
    start = max(0, at - width // 3)
    end = min(len(question), start + width)
    return ("…" if start else "") + question[start:end].strip() + ("…" if end < len(question) else "")


def list_conversations(principal: dict, matter_id: Optional[str] = None, limit: int = 50, q: Optional[str] = None) -> list[dict]:
    """The owner's conversations whose matters they can all still open, newest first. With `q`: those whose title or one
    of whose questions contains it (case-insensitive), each with the words around the first matching question."""
    args = {"uid": principal["sub"], "org": principal["org_id"], "now": policy.now_iso(), "lim": int(limit)}
    extra = ""
    if matter_id:
        extra, args["mid"] = " AND c.matter_id = :mid", matter_id
    if q:
        extra += (" AND (c.title LIKE :q ESCAPE '\\' OR EXISTS (SELECT 1 FROM legal_conversation_turns t "
                  "WHERE t.conversation_id = c.id AND t.question LIKE :q ESCAPE '\\'))")
        args["q"] = _like(q)
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT c.id, c.title, c.created_at, c.updated_at, m.id, m.number, m.title, "
            "(SELECT COUNT(*) FROM legal_conversation_turns t WHERE t.conversation_id = c.id) "
            f"FROM legal_conversations c JOIN matters m ON m.id = c.matter_id WHERE c.owner_id = :uid AND c.deleted_at IS NULL{extra} "
            f"AND {policy.visible_matters_sql()} ORDER BY c.updated_at DESC LIMIT :lim", args).fetchall()
        out = []
        for r in rows:
            ids = matter_ids(conn, r[0], r[4])
            if len(ids) > 1 and not all(policy.authorize(principal, "research.query", m).allowed for m in ids):
                continue                                   # a conversation is listed only while every one of its matters is open
            others = [conn.execute("SELECT id, number, title FROM matters WHERE id=?", (m,)).fetchone() for m in ids[1:]]
            item = {"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3],
                    "matter": {"id": r[4], "number": r[5], "title": r[6]},
                    "matters": [{"id": r[4], "number": r[5], "title": r[6]}] + [{"id": o[0], "number": o[1], "title": o[2]}
                                                                               for o in others if o],
                    "turns": r[7]}
            if q:
                hit = conn.execute("SELECT question FROM legal_conversation_turns WHERE conversation_id=? AND question LIKE ? "
                                   "ESCAPE '\\' ORDER BY seq LIMIT 1", (r[0], args["q"])).fetchone()
                if hit:
                    item["match"] = _snippet(hit[0], q)
            out.append(item)
    return out


def rename(conversation_id: str, owner_id: str, title: str) -> bool:
    with get_conn() as conn:
        n = conn.execute("UPDATE legal_conversations SET title=? WHERE id=? AND owner_id=? AND deleted_at IS NULL",
                         (_title(title), conversation_id, owner_id)).rowcount
        conn.commit()
    return n == 1


def delete(conversation_id: str, owner_id: str) -> bool:
    """Off its owner's list for good. The rows stay (work product, purge and retention apply), unreachable through the chat."""
    with get_conn() as conn:
        n = conn.execute("UPDATE legal_conversations SET deleted_at=? WHERE id=? AND owner_id=? AND deleted_at IS NULL",
                         (policy.now_iso(), conversation_id, owner_id)).rowcount
        conn.commit()
    return n == 1


def touch(conversation_id: str, title_from: Optional[str] = None) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE legal_conversations SET updated_at=?, title=CASE WHEN title=? AND ? IS NOT NULL THEN ? ELSE title END "
                     "WHERE id=?", (policy.now_iso(), NEW_TITLE, title_from, _title(title_from or ""), conversation_id))
        conn.commit()


# ── Turns ────────────────────────────────────────────────────────────────────────
_TURN_COLS = ("id", "conversation_id", "matter_id", "owner_id", "seq", "client_request_id", "question", "understood", "intent",
              "status", "summary", "component_ids_json", "actions_json", "research_turn_id", "created_at", "finished_at",
              "parent_turn_id")


def _turn(r) -> dict:
    t = dict(zip(_TURN_COLS, r))
    t["component_ids"] = json.loads(t.pop("component_ids_json") or "[]")
    t["actions"] = json.loads(t.pop("actions_json") or "[]")
    return t


def turn_by_request(owner_id: str, client_request_id: str) -> Optional[dict]:
    with get_conn() as conn:
        r = conn.execute(f"SELECT {', '.join(_TURN_COLS)} FROM legal_conversation_turns WHERE owner_id=? AND client_request_id=?",
                         (owner_id, client_request_id)).fetchone()
    return _turn(r) if r else None


def turn(turn_id: str, owner_id: str) -> Optional[dict]:
    with get_conn() as conn:
        r = conn.execute(f"SELECT {', '.join(_TURN_COLS)} FROM legal_conversation_turns WHERE id=? AND owner_id=?",
                         (turn_id, owner_id)).fetchone()
    return _turn(r) if r else None


def newest_turn(conversation_id: str) -> Optional[str]:
    with get_conn() as conn:
        r = conn.execute("SELECT id FROM legal_conversation_turns WHERE conversation_id=? ORDER BY seq DESC LIMIT 1",
                         (conversation_id,)).fetchone()
    return r[0] if r else None


def turns(conversation_id: str, owner_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(f"SELECT {', '.join(_TURN_COLS)} FROM legal_conversation_turns WHERE conversation_id=? AND owner_id=? "
                            "ORDER BY seq", (conversation_id, owner_id)).fetchall()
    return [_turn(r) for r in rows]


def running_turn(conversation_id: str) -> Optional[str]:
    """A turn of this conversation still being answered (one at a time); one left by a restart does not count."""
    import time
    from datetime import datetime
    with get_conn() as conn:
        r = conn.execute("SELECT id, created_at FROM legal_conversation_turns WHERE conversation_id=? AND status='running' "
                         "ORDER BY seq DESC LIMIT 1", (conversation_id,)).fetchone()
    if not r:
        return None
    try:
        started = datetime.fromisoformat(r[1].replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None
    return r[0] if time.time() - started < STALE_RUNNING_SEC else None


_FOLLOW_NEWEST = object()


def start_turn(conversation: dict, owner_id: str, question: str, intent: str, client_request_id: Optional[str],
               parent_turn_id=_FOLLOW_NEWEST) -> dict:
    """A new turn following `parent_turn_id` (None: a first question, or another version of one). By default it follows
    the conversation's newest turn, as a conversation without versions always did."""
    tid, now = new_id("turn"), policy.now_iso()
    with get_conn() as conn:
        seq = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM legal_conversation_turns WHERE conversation_id=?",
                           (conversation["id"],)).fetchone()[0]
        if parent_turn_id is _FOLLOW_NEWEST:
            r = conn.execute("SELECT id FROM legal_conversation_turns WHERE conversation_id=? ORDER BY seq DESC LIMIT 1",
                             (conversation["id"],)).fetchone()
            parent_turn_id = r[0] if r else None
        conn.execute("INSERT INTO legal_conversation_turns (id, conversation_id, matter_id, owner_id, seq, client_request_id, question, "
                     "intent, status, created_at, parent_turn_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                     (tid, conversation["id"], conversation["matter_id"], owner_id, seq, client_request_id, question, intent,
                      "running", now, parent_turn_id))
        conn.commit()
    return {"id": tid, "conversation_id": conversation["id"], "matter_id": conversation["matter_id"], "owner_id": owner_id,
            "seq": seq, "question": question, "intent": intent, "status": "running", "created_at": now, "parent_turn_id": parent_turn_id}


def finish_turn(turn_id: str, status: str, summary: str, component_ids: list[str], actions: list[dict],
                understood: Optional[str] = None, research_turn_id: Optional[str] = None) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE legal_conversation_turns SET status=?, summary=?, component_ids_json=?, actions_json=?, understood=?, "
                     "research_turn_id=?, finished_at=? WHERE id=?",
                     (status, summary, json.dumps(component_ids), json.dumps(actions, ensure_ascii=False), understood,
                      research_turn_id, policy.now_iso(), turn_id))
        conn.commit()


def history(conversation_id: str, owner_id: str, turn: dict, limit: int = 4) -> list[dict]:
    """The answered questions this turn follows on its own path (never another version's), oldest first, as research's
    follow-up rewriting wants them. A comparison counts too, with its own title as the answer ("Potential inconsistency in
    dates: “May 12, 2026” and “May 15, 2026”"), so "Could these refer to different events?" is understood (demo script
    step 4); the follow-up is still answered only from the documents."""
    with get_conn() as conn:
        rows = {r[0]: r[1:] for r in conn.execute(
            "SELECT id, parent_turn_id, question, understood, component_ids_json, status, intent, summary FROM legal_conversation_turns "
            "WHERE conversation_id=? AND owner_id=?", (conversation_id, owner_id)).fetchall()}
        chain, seen, pid = [], set(), turn.get("parent_turn_id")
        while pid and pid in rows and pid not in seen and len(chain) < limit:
            seen.add(pid)
            parent, q, understood, cj, status, intent, summary = rows[pid]
            if status in ("completed", "abstained") and intent == "ask_fact":
                chain.append((q, understood, cj, None))
            elif status == "completed" and intent == "compare":
                chain.append((q, understood, "[]", _compared(conn, cj, owner_id) or summary or ""))
            pid = parent
        out = []
        for q, understood, cj, said in reversed(chain):
            answer, documents = said or "", []
            for cid in json.loads(cj or "[]"):
                r = conn.execute("SELECT type, props_json FROM legal_counsel_components WHERE id=?", (cid,)).fetchone()
                if r and r[0] == "source_answer":
                    props = json.loads(r[1])
                    answer = props.get("answer", "")
                    # the documents the answer cited (2026-11-12): a follow-up searches them first; never shown to the model
                    documents = list(dict.fromkeys(c["document_id"] for c in props.get("citations") or [] if c.get("document_id")))
                    break
            out.append({"question": understood or q, "answer": answer[:2000], "documents": documents})
    return out


def asked(conversation_id: str, owner_id: str, turn: dict, limit: int = 50) -> list[str]:
    """The questions asked before this turn on its own path, oldest first, whatever answered them ("What did I ask you
    first?", 2026-11-12)."""
    with get_conn() as conn:
        rows = {r[0]: r[1:] for r in conn.execute(
            "SELECT id, parent_turn_id, question FROM legal_conversation_turns WHERE conversation_id=? AND owner_id=?",
            (conversation_id, owner_id)).fetchall()}
    chain, seen, pid = [], set(), turn.get("parent_turn_id")
    while pid and pid in rows and pid not in seen and len(chain) < limit:
        seen.add(pid)
        parent, q = rows[pid]
        chain.append(q)
        pid = parent
    return list(reversed(chain))


def _compared(conn, component_ids_json: Optional[str], owner_id: str) -> str:
    """What a comparison turn showed, in words: each card's title and its two documents (those the person may still see)."""
    from . import discrepancies
    said = []
    for cid in json.loads(component_ids_json or "[]"):
        r = conn.execute("SELECT type, props_json, matter_id FROM legal_counsel_components WHERE id=?", (cid,)).fetchone()
        if not r or r[0] != "evidence_comparison":
            continue
        row = discrepancies.get({"sub": owner_id}, r[2], json.loads(r[1]).get("discrepancy_id") or "")
        if row:
            a, b = row["detail"]["a"], row["detail"]["b"]
            said.append(f"{discrepancies.title(row)}, in {a.get('document_title') or 'one document'} and {b.get('document_title') or 'another'}.")
    return " ".join(said[:3])


# ── Components ───────────────────────────────────────────────────────────────────
def save_component(component: dict, turn: dict, document_ids: list[str], matter_id: Optional[str] = None) -> None:
    """A shown component, under the matter it belongs to (in a conversation about several, not always the first)."""
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_counsel_components (id, conversation_id, turn_id, matter_id, owner_id, type, props_json, "
                     "document_ids_json, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                     (component["id"], turn["conversation_id"], turn["id"], matter_id or turn["matter_id"], turn["owner_id"], component["type"],
                      json.dumps(component["props"], ensure_ascii=False), json.dumps(sorted(set(document_ids))), policy.now_iso()))
        conn.commit()


def component(component_id: str, owner_id: str) -> Optional[dict]:
    with get_conn() as conn:
        r = conn.execute("SELECT id, conversation_id, turn_id, matter_id, owner_id, type, props_json, document_ids_json "
                         "FROM legal_counsel_components WHERE id=? AND owner_id=?", (component_id, owner_id)).fetchone()
    if not r:
        return None
    return {"id": r[0], "conversation_id": r[1], "turn_id": r[2], "matter_id": r[3], "owner_id": r[4], "type": r[5],
            "props": json.loads(r[6]), "document_ids": json.loads(r[7] or "[]")}


def components(ids: list[str], owner_id: str) -> list[dict]:
    return [c for c in (component(i, owner_id) for i in ids) if c]


# ── Action idempotency ───────────────────────────────────────────────────────────
def replay(owner_id: str, key: str) -> Optional[dict]:
    with get_conn() as conn:
        r = conn.execute("SELECT component_id, action_id, result_json FROM legal_counsel_actions WHERE owner_id=? AND idempotency_key=?",
                         (owner_id, key)).fetchone()
    return {"component_id": r[0], "action_id": r[1], "result": json.loads(r[2])} if r else None


def remember(owner_id: str, key: str, component_id: str, action_id: str, result: dict) -> None:
    with get_conn() as conn:
        conn.execute("INSERT OR IGNORE INTO legal_counsel_actions (owner_id, idempotency_key, component_id, action_id, result_json, "
                     "created_at) VALUES (?,?,?,?,?,?)",
                     (owner_id, key, component_id, action_id, json.dumps(result, ensure_ascii=False), policy.now_iso()))
        conn.commit()


# ── Feedback ─────────────────────────────────────────────────────────────────────
FEEDBACK_REASONS = ("wrong_source", "missed", "not_asked", "unclear", "other")


def set_feedback(turn: dict, org_id: str, rating: Optional[str], reason: Optional[str], note: Optional[str]) -> Optional[dict]:
    """The asker's rating of one answer (None removes it)."""
    now = policy.now_iso()
    with get_conn() as conn:
        if rating is None:
            conn.execute("DELETE FROM legal_counsel_feedback WHERE turn_id=? AND owner_id=?", (turn["id"], turn["owner_id"]))
            conn.commit()
            return None
        conn.execute("INSERT INTO legal_counsel_feedback (turn_id, owner_id, org_id, matter_id, rating, reason, note, created_at, updated_at) "
                     "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(turn_id, owner_id) DO UPDATE SET rating=excluded.rating, "
                     "reason=excluded.reason, note=excluded.note, updated_at=excluded.updated_at",
                     (turn["id"], turn["owner_id"], org_id, turn["matter_id"], rating, reason, note, now, now))
        conn.commit()
    return {"rating": rating, **({"reason": reason} if reason else {})}


def feedback(conversation_id: str, owner_id: str) -> dict[str, dict]:
    """turn id -> the asker's rating, for one conversation."""
    with get_conn() as conn:
        rows = conn.execute("SELECT f.turn_id, f.rating, f.reason FROM legal_counsel_feedback f JOIN legal_conversation_turns t "
                            "ON t.id = f.turn_id WHERE t.conversation_id=? AND f.owner_id=?", (conversation_id, owner_id)).fetchall()
    return {r[0]: {"rating": r[1], **({"reason": r[2]} if r[2] else {})} for r in rows}


# ── Briefing bookkeeping ─────────────────────────────────────────────────────────
def last_seen(user_id: str, matter_id: str) -> Optional[str]:
    with get_conn() as conn:
        r = conn.execute("SELECT seen_at FROM legal_counsel_seen WHERE user_id=? AND matter_id=?", (user_id, matter_id)).fetchone()
    return r[0] if r else None


def mark_seen(user_id: str, matter_id: str) -> None:
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_counsel_seen (user_id, matter_id, seen_at) VALUES (?,?,?) "
                     "ON CONFLICT(user_id, matter_id) DO UPDATE SET seen_at=excluded.seen_at", (user_id, matter_id, policy.now_iso()))
        conn.commit()
