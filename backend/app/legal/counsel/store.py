"""Conversations, turns, stored components and action idempotency (migration 0036).

Every function takes the owner and the matter and answers None when either does not match, so a guessed id of
someone else's conversation, turn or component behaves exactly like one that does not exist.
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
                         "WHERE id=? AND owner_id=?", (conversation_id, owner_id)).fetchone()
        if not r:
            return None
        conv = dict(zip(("id", "org_id", "matter_id", "owner_id", "title", "created_at", "updated_at"), r))
        conv["matter_ids"] = matter_ids(conn, conv["id"], conv["matter_id"])
    return conv


def list_conversations(principal: dict, matter_id: Optional[str] = None, limit: int = 50) -> list[dict]:
    """The owner's conversations whose matters they can all still open, newest first."""
    args = {"uid": principal["sub"], "org": principal["org_id"], "now": policy.now_iso(), "lim": int(limit)}
    extra = ""
    if matter_id:
        extra, args["mid"] = " AND c.matter_id = :mid", matter_id
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT c.id, c.title, c.created_at, c.updated_at, m.id, m.number, m.title, "
            "(SELECT COUNT(*) FROM legal_conversation_turns t WHERE t.conversation_id = c.id) "
            f"FROM legal_conversations c JOIN matters m ON m.id = c.matter_id WHERE c.owner_id = :uid{extra} "
            f"AND {policy.visible_matters_sql()} ORDER BY c.updated_at DESC LIMIT :lim", args).fetchall()
        out = []
        for r in rows:
            ids = matter_ids(conn, r[0], r[4])
            if len(ids) > 1 and not all(policy.authorize(principal, "research.query", m).allowed for m in ids):
                continue                                   # a conversation is listed only while every one of its matters is open
            others = [conn.execute("SELECT id, number, title FROM matters WHERE id=?", (m,)).fetchone() for m in ids[1:]]
            out.append({"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3],
                        "matter": {"id": r[4], "number": r[5], "title": r[6]},
                        "matters": [{"id": r[4], "number": r[5], "title": r[6]}] + [{"id": o[0], "number": o[1], "title": o[2]}
                                                                                   for o in others if o],
                        "turns": r[7]})
    return out


def touch(conversation_id: str, title_from: Optional[str] = None) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE legal_conversations SET updated_at=?, title=CASE WHEN title=? AND ? IS NOT NULL THEN ? ELSE title END "
                     "WHERE id=?", (policy.now_iso(), NEW_TITLE, title_from, _title(title_from or ""), conversation_id))
        conn.commit()


# ── Turns ────────────────────────────────────────────────────────────────────────
_TURN_COLS = ("id", "conversation_id", "matter_id", "owner_id", "seq", "client_request_id", "question", "understood", "intent",
              "status", "summary", "component_ids_json", "actions_json", "research_turn_id", "created_at", "finished_at")


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


def start_turn(conversation: dict, owner_id: str, question: str, intent: str, client_request_id: Optional[str]) -> dict:
    tid, now = new_id("turn"), policy.now_iso()
    with get_conn() as conn:
        seq = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM legal_conversation_turns WHERE conversation_id=?",
                           (conversation["id"],)).fetchone()[0]
        conn.execute("INSERT INTO legal_conversation_turns (id, conversation_id, matter_id, owner_id, seq, client_request_id, question, "
                     "intent, status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (tid, conversation["id"], conversation["matter_id"], owner_id, seq, client_request_id, question, intent,
                      "running", now))
        conn.commit()
    return {"id": tid, "conversation_id": conversation["id"], "matter_id": conversation["matter_id"], "owner_id": owner_id,
            "seq": seq, "question": question, "intent": intent, "status": "running", "created_at": now}


def finish_turn(turn_id: str, status: str, summary: str, component_ids: list[str], actions: list[dict],
                understood: Optional[str] = None, research_turn_id: Optional[str] = None) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE legal_conversation_turns SET status=?, summary=?, component_ids_json=?, actions_json=?, understood=?, "
                     "research_turn_id=?, finished_at=? WHERE id=?",
                     (status, summary, json.dumps(component_ids), json.dumps(actions, ensure_ascii=False), understood,
                      research_turn_id, policy.now_iso(), turn_id))
        conn.commit()


def history(conversation_id: str, owner_id: str, before_seq: int, limit: int = 4) -> list[dict]:
    """Earlier answered questions of the conversation, oldest first, as research's follow-up rewriting wants them."""
    with get_conn() as conn:
        rows = conn.execute("SELECT t.question, t.understood, t.component_ids_json FROM legal_conversation_turns t "
                            "WHERE t.conversation_id=? AND t.owner_id=? AND t.seq < ? AND t.status IN ('completed','abstained') "
                            "AND t.intent = 'ask_fact' "
                            "ORDER BY t.seq DESC LIMIT ?", (conversation_id, owner_id, before_seq, limit)).fetchall()
        out = []
        for q, understood, cj in reversed(rows):
            answer = ""
            for cid in json.loads(cj or "[]"):
                r = conn.execute("SELECT type, props_json FROM legal_counsel_components WHERE id=?", (cid,)).fetchone()
                if r and r[0] == "source_answer":
                    answer = json.loads(r[1]).get("answer", "")
                    break
            out.append({"question": understood or q, "answer": answer[:2000]})
    return out


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
