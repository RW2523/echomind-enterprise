"""S4.3 — time-entry drafts from audited activity (OPT-03; decision D-056).

Off unless the firm turns it on (LEGAL_TIME_ENTRIES). A person's own audited work on a matter is grouped, per day
in the firm's timezone, into sessions:
- events closer than 15 minutes run together;
- each session counts from its first event to its last, plus 6 minutes;
- totals are rounded up to 6-minute units.

The narrative names the kinds of work, never content. Entries are drafts until the person edits and approves them;
only approved entries are exported, as CSV. There is no billing integration, and nothing is submitted
anywhere. Matters the person can no longer open are left out.
"""
from __future__ import annotations

import csv
import io
import json

import pytest

from conftest import as_user, csrf_headers

DAY = "2026-10-05"


@pytest.fixture
def tm(world, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "LEGAL_TIME_ENTRIES", True)
    monkeypatch.setattr(settings, "LEGAL_TIMEZONE", "Europe/London")             # BST: UTC+1 in October
    world.update(base="/api/legal/time")
    return world


def act(w, who, action, matter, at, outcome="allow"):
    """One audited event by `who` at `at` (UTC, ISO), appended to the real hash chain with its own time (no clock is
    patched, so nothing running alongside can see a false time)."""
    import uuid
    from app.legal.audit import chain
    from app.legal.orgs import default_org_id
    u = w["users"][who]
    chain.append({"id": str(uuid.uuid4()), "ts": at, "org_id": default_org_id(), "actor_user_id": u["id"], "actor_username": u["username"],
                  "actor_roles": None, "auth": "session", "session_ref": None, "ip": None, "action": action, "object_type": "document",
                  "object_id": "doc-QXZ", "matter_id": w["matters"][matter]["id"], "outcome": outcome, "reason": None, "details_json": None})


def day_of_work(w):
    """alice on M-101: 09:00–09:20 reading and research (BST), a break, 11:00–11:05 drafting; on M-114 one search."""
    for at, action in (("08:00:00", "document.read"), ("08:05:00", "documents.search"), ("08:12:00", "research.query"),
                       ("08:20:00", "document.read"), ("10:00:00", "draft.create"), ("10:05:00", "draft.edit")):
        act(w, "alice", action, "M-101", f"{DAY}T{at}.000Z")
    act(w, "alice", "documents.search", "M-114", f"{DAY}T13:30:00.000Z")
    act(w, "alice", "matter.read", "M-114", f"{DAY}T13:40:00.000Z")          # opening a matter is not work
    act(w, "alice", "document.read", "M-101", f"{DAY}T15:00:00.000Z", outcome="deny")   # a refused request is not work
    act(w, "bob", "document.read", "M-202", f"{DAY}T08:00:00.000Z")          # someone else's work is not hers


def suggest(w, who="alice", frm=DAY, to=DAY):
    c = as_user(w, who)
    return c.post(f"{w['base']}/suggest", headers=csrf_headers(c), json={"from": frm, "to": to})


def test_time_entries_are_off_unless_the_firm_turns_them_on(world):
    alice = as_user(world, "alice")
    assert alice.post("/api/legal/time/suggest", headers=csrf_headers(alice), json={"from": DAY, "to": DAY}).status_code == 404
    assert alice.get("/api/legal/time").status_code == 404
    assert alice.get("/api/auth/me").json()["features"]["time_entries"] is False


def test_a_day_of_audited_work_becomes_one_draft_per_matter(tm, monkeypatch):
    day_of_work(tm)
    r = suggest(tm)
    assert r.status_code == 200, r.text
    got = {e["matter"]["number"]: e for e in r.json()["entries"]}
    assert set(got) == {"M-101", "M-114"}
    m101 = got["M-101"]
    assert m101["work_date"] == DAY and m101["status"] == "draft" and m101["source"] == "suggested"
    # 08:00–08:20 (20 + 6 = 26 min) and 10:00–10:05 (5 + 6 = 11 min): 37 minutes, rounded up to 42 (0.7 h)
    assert m101["minutes_suggested"] == 42 and m101["minutes"] == 42
    assert m101["narrative"] == "Document review (3); legal research (1); drafting (2)"
    assert m101["sessions"] == [{"from": "09:00", "to": "09:20"}, {"from": "11:00", "to": "11:05"}]          # in the firm's timezone
    assert got["M-114"]["minutes"] == 6 and got["M-114"]["narrative"] == "Document review (1)"
    assert all("QXZ" not in json.dumps(e) for e in r.json()["entries"])        # the planted object id (never hex) is in no entry


def test_suggesting_again_updates_untouched_drafts_and_keeps_edited_ones(tm, monkeypatch):
    day_of_work(tm)
    alice = as_user(tm, "alice")
    first = {e["matter"]["number"]: e for e in suggest(tm).json()["entries"]}
    e = first["M-114"]
    r = alice.patch(f"{tm['base']}/{e['id']}", headers=csrf_headers(alice), json={"minutes": 30, "narrative": "Searched the data room.", "version": e["version"]})
    assert r.status_code == 200 and r.json()["entry"]["minutes"] == 30 and r.json()["entry"]["edited"] is True
    act(tm, "alice", "research.query", "M-101", f"{DAY}T16:00:00.000Z")
    again = {e["matter"]["number"]: e for e in suggest(tm).json()["entries"]}
    assert again["M-101"]["minutes"] == 48 and again["M-101"]["narrative"] == "Document review (3); legal research (2); drafting (2)"   # a third session
    assert again["M-114"]["minutes"] == 30 and again["M-114"]["narrative"] == "Searched the data room."     # her edit stands
    assert again["M-114"]["minutes_suggested"] == 6
    assert len(alice.get(f"{tm['base']}?from={DAY}&to={DAY}").json()["entries"]) == 2                          # no duplicates


def test_edits_are_checked_and_only_the_owner_sees_or_changes_an_entry(tm, monkeypatch):
    day_of_work(tm)
    alice, bob = as_user(tm, "alice"), as_user(tm, "bob")
    e = suggest(tm).json()["entries"][0]
    url = f"{tm['base']}/{e['id']}"
    for bad in ({"minutes": 7}, {"minutes": 0}, {"minutes": 1500}, {"narrative": ""}, {"narrative": "x" * 1001}):
        assert alice.patch(url, headers=csrf_headers(alice), json={**bad, "version": e["version"]}).status_code == 422, bad
    assert alice.patch(url, headers=csrf_headers(alice), json={"minutes": 12, "version": e["version"] + 5}).status_code == 409
    assert bob.patch(url, headers=csrf_headers(bob), json={"minutes": 12, "version": e["version"]}).status_code == 404
    assert bob.get(f"{tm['base']}?from={DAY}&to={DAY}").json()["entries"] == []


def test_approved_entries_are_locked_and_exported_as_csv_with_no_formula_cells(tm, monkeypatch):
    day_of_work(tm)
    alice = as_user(tm, "alice")
    entries = {e["matter"]["number"]: e for e in suggest(tm).json()["entries"]}
    e = entries["M-101"]
    e = alice.patch(f"{tm['base']}/{e['id']}", headers=csrf_headers(alice),
                    json={"narrative": "=HYPERLINK(\"http://x\") Reviewed the MSA; research on caps", "version": e["version"]}).json()["entry"]
    r = alice.post(f"{tm['base']}/{e['id']}/approve", headers=csrf_headers(alice), json={"version": e["version"]})
    assert r.status_code == 200 and r.json()["entry"]["status"] == "approved"
    assert alice.patch(f"{tm['base']}/{e['id']}", headers=csrf_headers(alice), json={"minutes": 12, "version": r.json()["entry"]["version"]}).status_code == 409
    d = entries["M-114"]
    assert alice.post(f"{tm['base']}/{d['id']}/discard", headers=csrf_headers(alice), json={"version": d["version"]}).json()["entry"]["status"] == "discarded"
    r = alice.get(f"{tm['base']}/export.csv?from={DAY}&to={DAY}")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"] and r.headers["cache-control"] == "no-store"
    rows = list(csv.reader(io.StringIO(r.text)))
    assert rows[0] == ["Date", "Matter", "Client", "Hours", "Minutes", "Narrative", "Approved"]
    assert len(rows) == 2 and rows[1][:5] == [DAY, "M-101", "Alpha Holdings", "0.7", "42"]
    assert rows[1][5].startswith("'=HYPERLINK")                                                         # a formula is written as text
    now = next(x for x in alice.get(f"{tm['base']}?from={DAY}&to={DAY}").json()["entries"] if x["id"] == e["id"])
    reopened = alice.post(f"{tm['base']}/{e['id']}/reopen", headers=csrf_headers(alice), json={"version": now["version"]})
    assert reopened.status_code == 200 and reopened.json()["entry"]["status"] == "draft"


def test_a_matter_she_can_no_longer_open_is_left_out(tm, monkeypatch):
    day_of_work(tm)
    act(tm, "alice", "document.read", "M-202", f"{DAY}T12:00:00.000Z")       # she cannot open M-202
    numbers = {e["matter"]["number"] for e in suggest(tm).json()["entries"]}
    assert numbers == {"M-101", "M-114"}


def test_a_range_is_bounded_and_the_audit_keeps_counts_only(tm, monkeypatch):
    day_of_work(tm)
    assert suggest(tm, frm="2026-01-01", to="2026-12-31").status_code == 422          # at most 31 days at a time
    assert suggest(tm, frm="2026-10-06", to="2026-10-05").status_code == 422
    suggest(tm)
    from app.core.db import get_conn
    with get_conn() as conn:
        rows = json.dumps(conn.execute("SELECT details_json FROM audit_events WHERE action LIKE 'time.%'").fetchall())
    assert "Document review" not in rows and "entries" in rows


def test_approved_entries_are_kept_through_an_attempted_rollback(tm, fresh_db):
    import sqlite3
    from app.legal.migrations import runner
    day_of_work(tm)
    alice = as_user(tm, "alice")
    e = suggest(tm).json()["entries"][0]
    alice.post(f"{tm['base']}/{e['id']}/approve", headers=csrf_headers(alice), json={"version": e["version"]})
    with sqlite3.connect(fresh_db) as c:
        with pytest.raises(runner.MigrationError, match="time entr"):
            runner.migrate_down(c, 31)


def test_a_sitting_of_seconds_is_one_unit_not_two(tm):
    """Journey 32 showed three actions within 45 seconds suggested as 0.2 h: the seconds were rounded up a whole extra unit."""
    for s in ("05", "20", "50"):
        act(tm, "alice", "document.read", "M-101", f"{DAY}T09:11:{s}.000Z")
    assert suggest(tm).json()["entries"][0]["minutes"] == 6
