"""The reviews queue (S2.7, decision D-043): everything waiting for a person's decision, and what of
theirs waits for someone else.

Waiting for you:
- **AI drafts** (minutes, research notes): the latest draft of each, on a matter where you may
  approve (`artifacts.approve`), that you did not make, whose sources you can all see, and that no
  one has sent back for changes.
- **Exports** waiting for a second lawyer (`export.approve`), not asked for by you, of a document
  you can see.
- **Firm decisions** for those who hold them: deletions (`deletion.review`), holds and their release
  (`holds.approve`), walls (`walls.approve`) and emergency access (`breakglass.approve`), never your own.

Only matters you can open are read (membership, walls, break-glass rules), so the queue never
discloses another matter. Items carry a title, the matter, who asked and when, and where to decide.
"""
from __future__ import annotations

from ...core.db import get_conn
from .. import artifacts, policy

_KIND_PATH = {"minutes": "/m/{number}/meetings/{subject}?tab=minutes", "research_note": "/m/{number}/notes/{id}",
              "legal_draft": "/m/{number}/drafting/{id}", "redline": "/m/{number}/redlines/{id}", "summary": "/m/{number}/summaries/{id}"}


def _name(conn, uid):
    r = conn.execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone() if uid else None
    return r[0] if r else None


def _matters(p: dict) -> dict[str, dict]:
    with get_conn() as conn:
        rows = conn.execute(f"SELECT m.id, m.number, m.title FROM matters m WHERE {policy.visible_matters_sql()}",
                            {"uid": p["sub"], "org": p["org_id"], "now": policy.now_iso()}).fetchall()
    return {r[0]: {"id": r[0], "number": r[1], "title": r[2]} for r in rows}


def _drafts(p: dict, matters: dict, waiting: list, mine: list) -> None:
    if not matters:
        return
    approver = {mid for mid in matters if policy.authorize(p, "artifacts.approve", mid).allowed}
    ids = list(matters)
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT a.id, a.matter_id FROM legal_ai_artifacts a WHERE a.matter_id IN ({','.join('?' * len(ids))}) AND a.status='draft' "
            "AND a.version = (SELECT MAX(b.version) FROM legal_ai_artifacts b WHERE b.subject_type=a.subject_type "
            "AND b.subject_id=a.subject_id AND b.kind=a.kind) "
            "AND (a.subject_type <> 'meeting' OR EXISTS (SELECT 1 FROM legal_meetings mt WHERE mt.id=a.subject_id "
            "AND mt.transcript_deleted_at IS NULL)) ORDER BY a.created_at", ids).fetchall()
    for aid, mid in rows:
        a = artifacts.get(mid, aid)
        if not a or not artifacts.visible_to(a, p["sub"]):
            continue
        last = a["decisions"][-1] if a["decisions"] else None
        sent_back = bool(last and last["decision"] == "changes_requested")
        m = matters[mid]
        item = {"kind": a["kind"], "id": a["id"], "title": a["title"], "version": a["version"], "matter": m,
                "by": a["created_by_name"], "at": a["created_at"],
                "path": _KIND_PATH[a["kind"]].format(number=m["number"], subject=a["subject_id"], id=a["id"])}
        if a["created_by"] == p["sub"]:
            mine.append({**item, "state": "changes_requested" if sent_back else "waiting",
                         "reason": last["reason"] if sent_back else None, "decided_by": last["by"] if sent_back else None})
        elif mid in approver and not sent_back:
            waiting.append(item)


def _exports(p: dict, matters: dict, waiting: list, mine: list) -> None:
    if not matters:
        return
    from ..documents import repo as documents
    approver = {mid for mid in matters if policy.authorize(p, "export.approve", mid).allowed}
    ids = list(matters)
    with get_conn() as conn:
        rows = conn.execute(f"SELECT id, matter_id, object_type, object_id, title, requested_by, submitted_at, format FROM legal_exports "
                            f"WHERE status='pending_approval' AND matter_id IN ({','.join('?' * len(ids))}) ORDER BY submitted_at", ids).fetchall()
        names = {r[5]: _name(conn, r[5]) for r in rows}
    for eid, mid, otype, oid, title, by, at, fmt in rows:
        if otype == "document" and not documents.visible(mid, oid, p["sub"]):
            continue
        m = matters[mid]
        item = {"kind": "export", "id": eid, "title": f"{title} ({fmt.upper()})", "matter": m, "by": names[by], "at": at,
                "path": f"/m/{m['number']}/exports/{eid}"}
        if by == p["sub"]:
            mine.append({**item, "state": "waiting"})
        elif mid in approver:
            waiting.append(item)


def _firm(p: dict, waiting: list, mine: list) -> None:
    roles = policy.firm_roles_of(p["sub"])
    can = lambda action: bool(roles & policy.ORG_ACTIONS[action])          # noqa: E731
    with get_conn() as conn:
        for did, label, by, at, number, title in conn.execute(
                "SELECT d.id, d.object_label, d.requested_by, d.requested_at, m.number, m.title FROM legal_deletions d "
                "LEFT JOIN matters m ON m.id = d.matter_id WHERE d.org_id=? AND d.status='requested' ORDER BY d.requested_at",
                (p["org_id"],)).fetchall():
            item = {"kind": "deletion", "id": did, "title": f"Delete: {label}", "matter": {"number": number, "title": title},
                    "by": _name(conn, by), "at": at, "path": "/admin/records"}
            if by == p["sub"]:
                mine.append({**item, "state": "waiting"})
            elif can("deletion.review"):
                waiting.append(item)
        if can("holds.approve"):
            for hid, name, status, by, at, rel_by, rel_at in conn.execute(
                    "SELECT id, name, status, requested_by, requested_at, release_requested_by, release_requested_at FROM legal_holds "
                    "WHERE org_id=? AND status IN ('requested','release_requested') ORDER BY requested_at", (p["org_id"],)).fetchall():
                asker, when = (by, at) if status == "requested" else (rel_by, rel_at)
                if asker != p["sub"]:
                    waiting.append({"kind": "hold", "id": hid, "title": ("Place hold: " if status == "requested" else "Release hold: ") + name,
                                    "matter": None, "by": _name(conn, asker), "at": when, "path": "/admin/records"})
        if can("walls.approve"):
            for wid, name, status, by, at, rel_by in conn.execute(
                    "SELECT id, name, status, requested_by, requested_at, release_requested_by FROM ethical_walls "
                    "WHERE org_id=? AND status IN ('requested','release_requested') ORDER BY requested_at", (p["org_id"],)).fetchall():
                asker = by if status == "requested" else rel_by
                if asker != p["sub"]:
                    waiting.append({"kind": "wall", "id": wid, "title": ("Approve wall: " if status == "requested" else "Release wall: ") + name,
                                    "matter": None, "by": _name(conn, asker), "at": at, "path": "/admin/walls"})
        if can("breakglass.approve"):
            for gid, uid, number, at in conn.execute(
                    "SELECT id, user_id, requested_number, requested_at FROM break_glass_grants WHERE org_id=? AND status='requested' "
                    "ORDER BY requested_at", (p["org_id"],)).fetchall():
                if uid != p["sub"]:
                    waiting.append({"kind": "emergency_access", "id": gid, "title": f"Emergency access to {number}", "matter": None,
                                    "by": _name(conn, uid), "at": at, "path": "/admin/emergency-access"})


def for_principal(p: dict) -> dict:
    waiting, mine = [], []
    matters = _matters(p)
    _drafts(p, matters, waiting, mine)
    _exports(p, matters, waiting, mine)
    _firm(p, waiting, mine)
    waiting.sort(key=lambda i: i["at"] or "")
    mine.sort(key=lambda i: i["at"] or "", reverse=True)
    return {"waiting": waiting, "mine": mine, "count": len(waiting)}
