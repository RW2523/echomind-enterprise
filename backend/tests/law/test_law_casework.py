"""Case management (2026-11-07): a new matter with its team, parties, track and stage; the updates log; the plan; stage
moves; the pipeline; importing matters from CSV; importing a zip of documents. Every read and write is authorised per
matter, the updates log is append-only, and audit events never carry what was written."""
from __future__ import annotations

import io
import json
import sqlite3
import zipfile

import pytest

from conftest import as_user, csrf_headers, numbers


def _mid(world, number):
    return world["matters"][number]["id"]


def _post(c, path, **kw):
    return c.post(path, headers=csrf_headers(c), **kw)


def _audit(db, action):
    with sqlite3.connect(db) as c:
        return [json.loads(r[0] or "{}") for r in c.execute("SELECT details_json FROM audit_events WHERE action=? ORDER BY seq", (action,))]


# ── A new matter ─────────────────────────────────────────────────────────────────
def test_a_new_matter_takes_its_team_parties_track_and_stage(world):
    alice = as_user(world, "alice")
    r = _post(alice, "/api/legal/matters", json={
        "client_id": world["clients"]["Alpha"]["id"], "number": "M-150", "title": "Supply dispute", "practice_area": "Commercial litigation",
        "description": "Late deliveries under the supply agreement.", "stage": "pleadings",
        "members": [{"username": "carol", "role": "paralegal"}], "parties": [{"name": "Example Supplier plc", "role": "adverse"}]})
    assert r.status_code == 200, r.text
    m = r.json()["matter"]
    assert (m["track"], m["stage"], m["description"], m["my_role"]) == ("disputes", "pleadings", "Late deliveries under the supply agreement.", "lead")
    assert "M-150" in numbers(as_user(world, "carol").get("/api/legal/matters"))
    parties = alice.get(f"/api/legal/matters/{m['id']}/parties").json()["parties"]
    assert [(p["name"], p["role"]) for p in parties] == [("Example Supplier plc", "adverse")]
    roles = {x["username"]: x["role"] for x in alice.get(f"/api/legal/matters/{m['id']}/memberships").json()["memberships"]}
    assert roles == {"alice": "lead", "carol": "paralegal"}


def test_a_new_matter_refuses_a_stage_outside_its_track_and_an_unknown_colleague(world):
    alice = as_user(world, "alice")
    base = {"client_id": world["clients"]["Alpha"]["id"], "number": "M-151", "title": "Advice"}
    assert _post(alice, "/api/legal/matters", json={**base, "track": "advisory", "stage": "pleadings"}).status_code == 422
    assert _post(alice, "/api/legal/matters", json={**base, "members": [{"username": "nobody", "role": "attorney"}]}).status_code == 400
    m = _post(alice, "/api/legal/matters", json=base).json()["matter"]
    assert (m["track"], m["stage"]) == ("advisory", "instruction")             # no practice area: advisory, first stage


def test_a_colleague_walled_from_the_client_cannot_be_put_on_a_new_matter(world):
    from test_law_walls import _wall
    _wall(world, screened=("carol",), clients=("Alpha",))
    alice = as_user(world, "alice")
    r = _post(alice, "/api/legal/matters", json={"client_id": world["clients"]["Alpha"]["id"], "number": "M-152", "title": "x",
                                                 "members": [{"username": "carol", "role": "paralegal"}]})
    assert r.status_code == 409 and "ethical wall" in r.json()["detail"]


# ── Updates ──────────────────────────────────────────────────────────────────────
def test_updates_are_posted_by_the_team_read_by_members_and_hidden_from_everyone_else(world, fresh_db):
    alice, carol, bob = as_user(world, "alice"), as_user(world, "carol"), as_user(world, "bob")
    base = f"/api/legal/matters/{_mid(world, 'M-101')}/updates"
    r = _post(alice, base, json={"kind": "call", "body": "Client confirmed the renewal terms on the phone.", "occurred_on": "2026-11-02"})
    assert r.status_code == 200, r.text
    u = r.json()["update"]
    assert (u["kind"], u["author"], u["occurred_on"]) == ("call", "alice", "2026-11-02")
    assert [x["id"] for x in carol.get(base).json()["updates"]] == [u["id"]]               # read-only member reads
    assert _post(carol, base, json={"body": "x"}).status_code == 403                         # … but does not post
    assert bob.get(base).status_code == 404 and _post(bob, base, json={"body": "x"}).status_code == 404
    assert all("renewal" not in json.dumps(d) for d in _audit(fresh_db, "matter.update_posted"))   # never the words


def test_an_update_is_never_changed_a_correction_names_it(world):
    alice = as_user(world, "alice")
    base = f"/api/legal/matters/{_mid(world, 'M-101')}/updates"
    first = _post(alice, base, json={"kind": "filing", "body": "Filed the defence on 3 November."}).json()["update"]
    fix = _post(alice, f"{base}/{first['id']}/correct", json={"body": "Filed the defence on 4 November."})
    assert fix.status_code == 200, fix.text
    assert fix.json()["update"]["corrects_id"] == first["id"] and fix.json()["update"]["kind"] == "filing"
    listed = {x["id"]: x for x in alice.get(base).json()["updates"]}
    assert listed[first["id"]]["corrected_by"] == fix.json()["update"]["id"]
    assert _post(alice, f"{base}/{first['id']}/correct", json={"body": "again"}).status_code == 409
    from app.core.db import get_conn
    with get_conn() as conn:
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("UPDATE legal_matter_updates SET body='changed' WHERE id=?", (first["id"],))
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("DELETE FROM legal_matter_updates WHERE id=?", (first["id"],))


def test_an_update_records_the_past_and_links_only_a_document_of_its_matter(docs):
    from conftest import upload
    alice = as_user(docs, "alice")
    base = f"/api/legal/matters/{_mid(docs, 'M-101')}/updates"
    assert _post(alice, base, json={"body": "Hearing", "occurred_on": "2099-01-01"}).status_code == 422
    other = upload(alice, _mid(docs, "M-114"), "other.txt", "Another matter's letter.").json()["document"]
    assert _post(alice, base, json={"body": "See the letter", "document_id": other["id"]}).status_code == 422
    mine = upload(alice, _mid(docs, "M-101"), "letter.txt", "The client's letter.").json()["document"]
    u = _post(alice, base, json={"body": "See the letter", "document_id": mine["id"]}).json()["update"]
    assert u["document_title"] == mine["title"]


# ── Plan and stage ───────────────────────────────────────────────────────────────
def test_tasks_are_kept_by_the_team_with_their_history(world):
    alice, carol = as_user(world, "alice"), as_user(world, "carol")
    mid = _mid(world, "M-101")
    base = f"/api/legal/matters/{mid}/tasks"
    bob_id = world["users"]["bob"]["id"]
    assert _post(alice, base, json={"title": "File the reply", "assignee_id": bob_id}).status_code == 422   # not on the matter
    t = _post(alice, base, json={"title": "File the reply", "kind": "filing", "due_on": "2020-01-10",
                                 "assignee_id": world["users"]["carol"]["id"]}).json()["task"]
    assert (t["kind"], t["assignee"], t["overdue"], t["status"]) == ("filing", "carol", True, "open")
    assert _post(carol, base, json={"title": "x"}).status_code == 403                   # read-only
    done = alice.patch(f"{base}/{t['id']}", headers=csrf_headers(alice), json={"status": "done", "version": t["version"]})
    assert done.status_code == 200 and done.json()["task"]["status"] == "done" and not done.json()["task"]["overdue"]
    stale = alice.patch(f"{base}/{t['id']}", headers=csrf_headers(alice), json={"title": "Changed", "version": t["version"]})
    assert stale.status_code == 409
    from app.core.db import get_conn
    with get_conn() as conn:
        assert [r[0] for r in conn.execute("SELECT action FROM legal_matter_task_events WHERE task_id=? ORDER BY at", (t["id"],))] == ["created", "done"]


def test_a_stages_standard_tasks_are_added_once(world):
    alice = as_user(world, "alice")
    mid = _mid(world, "M-101")
    m = alice.get(f"/api/legal/matters/{mid}").json()["matter"]
    first = _post(alice, f"/api/legal/matters/{mid}/plan/standard-tasks", json={}).json()
    assert first["added"] >= 2 and all(t["source"] == "standard" and t["stage"] == m["stage"] for t in first["tasks"])
    assert _post(alice, f"/api/legal/matters/{mid}/plan/standard-tasks", json={}).json()["added"] == 0


def test_a_stage_move_is_made_by_the_lawyers_and_shows_in_the_updates(world):
    alice, carol = as_user(world, "alice"), as_user(world, "carol")
    mid = _mid(world, "M-101")
    m = alice.get(f"/api/legal/matters/{mid}").json()["matter"]
    path = f"/api/legal/matters/{mid}/stage"
    assert _post(carol, path, json={"stage": "research", "version": m["version"]}).status_code == 403
    assert _post(alice, path, json={"stage": "pleadings", "version": m["version"]}).status_code == 422      # not this track
    assert _post(alice, path, json={"stage": m["stage"], "version": m["version"]}).status_code == 409
    moved = _post(alice, path, json={"stage": "research", "version": m["version"]})
    assert moved.status_code == 200 and moved.json()["matter"]["stage"] == "research"
    entry = alice.get(f"/api/legal/matters/{mid}/updates").json()["updates"][0]
    assert (entry["kind"], entry["stage_from"], entry["stage_to"]) == ("stage", m["stage"], "research")
    assert entry["body"] == "Moved from Instruction to Research & analysis."
    assert _post(alice, f"/api/legal/matters/{mid}/updates/{entry['id']}/correct", json={"body": "x"}).status_code == 409
    switched = _post(alice, path, json={"track": "disputes", "stage": "assessment", "version": moved.json()["matter"]["version"],
                                        "effective_on": "2026-10-20"})
    assert switched.json()["matter"]["track"] == "disputes" and switched.json()["matter"]["stage_since"].startswith("2026-10-20")
    assert alice.get(f"/api/legal/matters/{mid}/updates").json()["updates"][-1]["occurred_on"] == "2026-10-20"   # oldest, by when


def test_a_closed_matter_takes_no_updates_tasks_or_stage_moves(world):
    alice = as_user(world, "alice")
    mid = _mid(world, "M-101")
    m = alice.get(f"/api/legal/matters/{mid}").json()["matter"]
    alice.patch(f"/api/legal/matters/{mid}", headers=csrf_headers(alice), json={"status": "closed", "version": m["version"]})
    assert _post(alice, f"/api/legal/matters/{mid}/updates", json={"body": "x"}).status_code == 409
    assert _post(alice, f"/api/legal/matters/{mid}/tasks", json={"title": "x"}).status_code == 409
    assert alice.get(f"/api/legal/matters/{mid}/updates").status_code == 200                # still readable


# ── Pipeline ─────────────────────────────────────────────────────────────────────
def test_the_pipeline_shows_only_the_matters_a_person_may_see_with_their_latest(world):
    alice, bob = as_user(world, "alice"), as_user(world, "bob")
    mid = _mid(world, "M-101")
    _post(alice, f"/api/legal/matters/{mid}/updates", json={"kind": "meeting", "body": "Met the client about the renewal."})
    _post(alice, f"/api/legal/matters/{mid}/tasks", json={"title": "Send the draft", "due_on": "2030-01-01"})
    rows = {r["number"]: r for r in alice.get("/api/legal/pipeline").json()["matters"]}
    assert set(rows) == {"M-101", "M-114"}
    assert rows["M-101"]["last_update"]["excerpt"] == "Met the client about the renewal."
    assert rows["M-101"]["next_task"]["title"] == "Send the draft" and rows["M-101"]["leads"] == ["alice"]
    assert {r["number"] for r in bob.get("/api/legal/pipeline").json()["matters"]} == {"M-202", "M-215"}


# ── Import from CSV ──────────────────────────────────────────────────────────────
CSV = ("Matter number,Title,Client,Practice area,Sensitivity,Track,Stage,Other parties\n"
       "M-301,Lease renewal,Alpha Holdings,Real estate,standard,,Due diligence,Landlord Ltd\n"
       "M-302,Wrongful dismissal claim,New Client Ltd,Employment,,,Pleadings,Former Employer plc; Its Parent plc\n"
       "M-101,Duplicate,Alpha Holdings,,,,,\n"
       "M-303,,Alpha Holdings,,,,,\n"
       "M-304,Bad stage,Alpha Holdings,,,advisory,Hearing / trial,\n")


def test_a_csv_is_previewed_row_by_row_and_only_good_rows_are_opened(world, fresh_db):
    alice = as_user(world, "alice")
    files = {"file": ("matters.csv", CSV.encode(), "text/csv")}
    pre = _post(alice, "/api/legal/matters-import", files=files)
    assert pre.status_code == 200, pre.text
    rows = {r["number"]: r for r in pre.json()["rows"]}
    assert pre.json()["ready"] == 2 and pre.json()["created"] == []
    assert (rows["M-301"]["track"], rows["M-301"]["stage"], rows["M-301"]["new_client"]) == ("transactions", "diligence", False)
    assert (rows["M-302"]["track"], rows["M-302"]["stage"], rows["M-302"]["new_client"]) == ("disputes", "pleadings", True)
    assert rows["M-101"]["errors"] == ["A matter with this number already exists."]
    assert rows["M-303"]["errors"] == ["The title is missing."]
    assert "is not a stage of the advisory track" in rows["M-304"]["errors"][0]
    assert "M-301" not in numbers(alice.get("/api/legal/matters"))                         # a preview opens nothing
    done = _post(alice, "/api/legal/matters-import?preview=false", files={"file": ("matters.csv", CSV.encode(), "text/csv")})
    assert [c["number"] for c in done.json()["created"]] == ["M-301", "M-302"]
    assert {"M-301", "M-302"} <= numbers(alice.get("/api/legal/matters"))
    m302 = next(m for m in alice.get("/api/legal/matters").json()["matters"] if m["number"] == "M-302")
    assert m302["my_role"] == "lead" and m302["client"]["name"] == "New Client Ltd"
    parties = alice.get(f"/api/legal/matters/{m302['id']}/parties").json()["parties"]
    assert sorted(p["name"] for p in parties) == ["Former Employer plc", "Its Parent plc"]
    assert len([d for d in _audit(fresh_db, "matter.create") if d.get("method") == "import"]) == 2


def test_only_lawyers_who_open_matters_may_import_them_and_the_file_must_be_a_table(world):
    carol, alice = as_user(world, "carol"), as_user(world, "alice")
    assert _post(carol, "/api/legal/matters-import", files={"file": ("m.csv", CSV.encode(), "text/csv")}).status_code == 403
    assert _post(alice, "/api/legal/matters-import", files={"file": ("m.csv", b"just,some\nwords,here\n", "text/csv")}).status_code == 422
    assert _post(alice, "/api/legal/matters-import", files={"file": ("m.xlsx", b"PK\x03\x04\x00\x00", "text/csv")}).status_code == 422


# ── A zip of documents ───────────────────────────────────────────────────────────
def _zip(entries: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_a_zip_imports_each_supported_file_and_says_why_others_were_skipped(docs, fresh_db):
    alice = as_user(docs, "alice")
    mid = _mid(docs, "M-101")
    blob = _zip({"letters/letter.txt": "Dear client, the renewal is agreed.", "notes.md": "# Notes\nCall on Monday.",
                 "tool.exe": b"MZ\x90\x00", "inner.zip": _zip({"a.txt": "x"}), "__MACOSX/._letter.txt": b"\x00\x05",
                 "../../escape.txt": "Outside the folder?", "big.txt": "a" * (3 * 1024 * 1024)})
    r = _post(alice, f"/api/legal/matters/{mid}/documents/import-zip", files={"file": ("bundle.zip", blob, "application/zip")})
    assert r.status_code == 202, r.text
    got = sorted(x["name"] for x in r.json()["imported"])
    assert got == ["escape.txt", "letter.txt", "notes.md"]                                  # only the base name is ever used
    reasons = {x["name"]: x["reason"] for x in r.json()["skipped"]}
    assert set(reasons) == {"tool.exe", "inner.zip", "big.txt"}
    assert "zip inside" in reasons["inner.zip"] and "expands" in reasons["big.txt"]
    imports = [d for d in _audit(fresh_db, "document.import") if d.get("method") == "zip"]
    assert len(imports) == 3
    titles = {d["title"] for d in alice.get(f"/api/legal/matters/{mid}/documents").json()["documents"]}
    assert {"letter", "notes", "escape"} <= titles


def test_a_zip_is_refused_to_readers_and_when_it_is_not_a_zip_or_too_full(docs, monkeypatch):
    from app.legal.documents import routes as droutes
    alice, carol = as_user(docs, "alice"), as_user(docs, "carol")
    path = f"/api/legal/matters/{_mid(docs, 'M-101')}/documents/import-zip"
    blob = _zip({"a.txt": "one", "b.txt": "two", "c.txt": "three"})
    assert _post(carol, path, files={"file": ("x.zip", blob, "application/zip")}).status_code == 403
    assert _post(alice, path, files={"file": ("x.zip", b"not a zip", "application/zip")}).status_code == 422
    monkeypatch.setattr(droutes, "ZIP_MAX_FILES", 2)
    assert _post(alice, path, files={"file": ("x.zip", blob, "application/zip")}).status_code == 413


# ── In the chat ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text, intent", [
    ("What's the latest on this case?", "case_status"), ("Where are we with M-101?", "case_status"), ("Status update", "case_status"),
    ("What stage is the matter at?", "case_status"), ("Show me the recent updates", "case_status"),
    ("Draft a client status update", "draft"), ("What is the latest version of the agreement?", "ask_fact"),
    ("What's the latest date to give notice?", "ask_fact"), ("Brief me", "inspect_case")])
def test_questions_about_where_the_case_stands_get_the_case_status(text, intent):
    from app.legal.counsel import intent as intents
    assert intents.classify(text) == intent


def test_the_chat_shows_where_the_case_stands_from_its_own_records(world):
    from app.legal.counsel import schemas
    from test_law_counsel import final, new_conversation, turn
    alice, carol = as_user(world, "alice"), as_user(world, "carol")
    mid = _mid(world, "M-101")
    _post(alice, f"/api/legal/matters/{mid}/updates", json={"kind": "instruction", "body": "Client wants the renewal signed by Friday."})
    _post(alice, f"/api/legal/matters/{mid}/tasks", json={"title": "Send the signing pack", "due_on": "2020-01-01"})
    env = final(turn(alice, new_conversation(alice, mid), "What's the latest on this case?"))
    card = env["components"][0]
    schemas.validate_component(card)
    p = card["props"]
    assert card["type"] == "case_status" and p["number"] == "M-101" and p["can_post"]
    assert p["updates"][0]["text"] == "Client wants the renewal signed by Friday." and p["updates"][0]["kind_label"] == "Client instruction"
    assert p["tasks"][0]["title"] == "Send the signing pack" and p["tasks"][0]["overdue"] and p["overdue"] == 1
    assert "1 open task, 1 overdue" in env["summary"] and env["suggested_actions"][0]["label"] == "Post an update"
    seen = final(turn(carol, new_conversation(carol, mid), "Status update"))["components"][0]["props"]
    assert not seen["can_post"] and seen["updates"]                                       # read-only: reads, cannot post
    plan = final(turn(alice, new_conversation(alice, mid), "What are the open tasks?"))
    assert [c["type"] for c in plan["components"]][:1] == ["case_status"]                 # the plan first, then obligations
