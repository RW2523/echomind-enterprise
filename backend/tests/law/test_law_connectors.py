"""S1.11 / SEC-08 — the connector framework with the local-folder connector (spec test T15).

Fail closed throughout: no permissions file → not imported; permissions name a person → only if
they are also on the matter; a connector that has not synced successfully recently → its
documents are hidden; paths never leave the configured root. No real DMS provider is offered
(P-010): the framework is tested with the local-folder connector only.
"""
from __future__ import annotations

import json
import os
import sqlite3

import pytest

from conftest import as_user, csrf_headers, index_all, search_in

CANARY_MSA, CANARY_SIDE = "KESTREL-5150", "PLOVER-6161"


def write(folder, name, text, users=None, raw_acl=None):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(text)
    if users is not None:
        (folder / f"{name}.acl.json").write_text(json.dumps({"users": users}))
    elif raw_acl is not None:
        (folder / f"{name}.acl.json").write_text(raw_acl)


@pytest.fixture
def dms(docs, tmp_path, monkeypatch):
    from app.core.config import settings
    root = tmp_path / "dms"
    folder = root / "Harrowgate" / "MSA"
    write(folder, "msa.txt", f"Master services agreement. The {CANARY_MSA} clause caps liability.", users=["alice", "dave"])
    write(folder, "side-letter.txt", f"Side letter. The {CANARY_SIDE} rebate applies.", users=["alice", "carol"])
    write(folder, "no-acl.txt", "No permissions file for this one.")
    write(folder, "bad-acl.txt", "Broken permissions file.", raw_acl="{not json")
    monkeypatch.setattr(settings, "LEGAL_CONNECTOR_LOCAL_ROOTS", str(root))
    admin, alice = as_user(docs, "admin"), as_user(docs, "alice")
    c = admin.post("/api/legal/connectors", headers=csrf_headers(admin),
                   json={"kind": "local_folder", "name": "Firm DMS (test folder)", "root": str(root)})
    assert c.status_code == 200, c.text
    cid = c.json()["connector"]["id"]
    mid = docs["matters"]["M-101"]["id"]
    link = alice.post(f"/api/legal/matters/{mid}/connector-links", headers=csrf_headers(alice),
                      json={"connector_id": cid, "remote_path": "Harrowgate/MSA"})
    assert link.status_code == 200, link.text
    docs.update(root=root, folder=folder, cid=cid, mid=mid)
    return docs


def sync(w):
    admin = as_user(w, "admin")
    r = admin.post(f"/api/legal/connectors/{w['cid']}/sync", headers=csrf_headers(admin))
    assert r.status_code == 200, r.text
    index_all()
    return r.json()["run"]


def titles(c, w):
    r = c.get(f"/api/legal/matters/{w['mid']}/documents")
    assert r.status_code == 200, r.text
    return {d["title"]: d for d in r.json()["documents"]}


# ── Import and permissions ───────────────────────────────────────────────────────
def test_only_files_with_valid_permissions_are_imported(dms):
    run = sync(dms)
    assert run["status"] == "partial" and run["report"]["imported"] == 2
    assert run["report"]["acl_missing"] == 2
    seen = titles(as_user(dms, "alice"), dms)
    assert set(seen) == {"msa", "side-letter"}
    assert seen["msa"]["source"] == {"kind": "connector", "name": "Firm DMS (test folder)", "state": "current", "path": "msa.txt"}


def test_source_permissions_are_intersected_with_matter_membership(dms):
    sync(dms)
    alice, carol = as_user(dms, "alice"), as_user(dms, "carol")
    assert set(titles(carol, dms)) == {"side-letter"}                       # not named on msa
    msa = titles(alice, dms)["msa"]
    base = f"/api/legal/matters/{dms['mid']}/documents/{msa['id']}"
    assert carol.get(base).status_code == 404
    assert carol.get(f"{base}/versions/{msa['current_version']['id']}/file").status_code == 404
    assert search_in(carol, dms["mid"], CANARY_MSA).json()["passages"] == []
    assert CANARY_SIDE in json.dumps(search_in(carol, dms["mid"], CANARY_SIDE).json())
    assert CANARY_MSA in json.dumps(search_in(alice, dms["mid"], CANARY_MSA).json())
    # dave is named on msa but is not on the matter: still nothing
    assert as_user(dms, "dave").get(f"/api/legal/matters/{dms['mid']}/documents").status_code == 404


def test_a_permission_change_applies_at_the_next_sync(dms):
    sync(dms)
    write(dms["folder"], "msa.txt", f"Master services agreement. The {CANARY_MSA} clause caps liability.", users=["carol"])
    run = sync(dms)
    assert run["report"]["acl_changed"] == 1
    alice, carol = as_user(dms, "alice"), as_user(dms, "carol")
    assert "msa" not in titles(alice, dms) and "msa" in titles(carol, dms)
    assert search_in(alice, dms["mid"], CANARY_MSA).json()["passages"] == []


def test_changed_files_become_new_versions_and_unchanged_ones_do_not(dms):
    sync(dms)
    again = sync(dms)
    assert again["report"]["unchanged"] == 2 and again["report"]["updated"] == 0
    write(dms["folder"], "side-letter.txt", f"Side letter, amended. The {CANARY_SIDE} rebate applies twice.", users=["alice", "carol"])
    run = sync(dms)
    assert run["report"]["updated"] == 1
    doc = titles(as_user(dms, "alice"), dms)["side-letter"]
    assert doc["current_version"]["number"] == 2
    assert "applies twice" in json.dumps(search_in(as_user(dms, "alice"), dms["mid"], f"{CANARY_SIDE} rebate twice").json())


def test_a_file_removed_at_the_source_is_hidden_at_once_and_queued_for_purge(dms, fresh_db):
    sync(dms)
    os.remove(dms["folder"] / "msa.txt")
    run = sync(dms)
    assert run["report"]["removed"] == 1
    alice = as_user(dms, "alice")
    assert "msa" not in titles(alice, dms)
    assert search_in(alice, dms["mid"], CANARY_MSA).json()["passages"] == []
    with sqlite3.connect(fresh_db) as c:
        rows = c.execute("SELECT origin, status FROM legal_deletions").fetchall()
    assert rows == [("connector", "requested")]


def test_a_legal_hold_keeps_material_removed_at_the_source(dms):
    from conftest import make_user
    for name in ("rhea", "rex"):
        dms["users"][name] = make_user(name, mfa=True, firm_roles=["records_admin"])
    sync(dms)
    rhea, rex = as_user(dms, "rhea"), as_user(dms, "rex")
    h = rhea.post("/api/legal/holds", headers=csrf_headers(rhea), json={"name": "h", "reason": "r", "matter_ids": [dms["mid"]]}).json()["hold"]
    rex.post(f"/api/legal/holds/{h['id']}/approve", headers=csrf_headers(rex))
    os.remove(dms["folder"] / "msa.txt")
    run = sync(dms)
    assert run["report"]["removed"] == 0 and run["report"]["held"] == 1
    msa = titles(as_user(dms, "alice"), dms)["msa"]
    assert msa["source"]["state"] == "removed_in_source"


def test_documents_are_hidden_while_their_connector_is_out_of_date(dms, fresh_db):
    from conftest import upload
    sync(dms)
    alice = as_user(dms, "alice")
    upload(alice, dms["mid"], "own.txt", "An uploaded note.")
    index_all()
    with sqlite3.connect(fresh_db) as c:
        c.execute("UPDATE legal_connectors SET last_ok_at='2000-01-01T00:00:00.000Z'")
    assert set(titles(alice, dms)) == {"own.txt"}                            # uploads unaffected
    assert search_in(alice, dms["mid"], CANARY_SIDE).json()["passages"] == []
    sync(dms)
    assert {"msa", "side-letter", "own.txt"} <= set(titles(alice, dms))


def test_a_failed_sync_is_reported_and_changes_nothing(dms, fresh_db):
    import shutil
    sync(dms)
    with sqlite3.connect(fresh_db) as c:
        before = c.execute("SELECT last_ok_at FROM legal_connectors").fetchone()[0]
    shutil.rmtree(dms["root"])
    run = sync(dms)
    assert run["status"] == "failed"
    with sqlite3.connect(fresh_db) as c:
        after = c.execute("SELECT last_ok_at, last_status FROM legal_connectors").fetchone()
    assert after == (before, "failed")
    assert {"msa", "side-letter"} <= set(titles(as_user(dms, "alice"), dms))   # still fresh enough


def test_a_disabled_connector_hides_its_documents(dms):
    sync(dms)
    admin = as_user(dms, "admin")
    admin.post(f"/api/legal/connectors/{dms['cid']}/disable", headers=csrf_headers(admin))
    assert titles(as_user(dms, "alice"), dms) == {}


# ── Boundaries ───────────────────────────────────────────────────────────────────
def test_paths_never_leave_the_root(dms, tmp_path):
    alice, admin = as_user(dms, "alice"), as_user(dms, "admin")
    for bad in ("../outside", "/etc", "Harrowgate/../../outside"):
        r = alice.post(f"/api/legal/matters/{dms['mid']}/connector-links", headers=csrf_headers(alice),
                       json={"connector_id": dms["cid"], "remote_path": bad})
        assert r.status_code == 422, bad
    outside = tmp_path / "outside"
    write(outside, "secret.txt", "Outside the root.", users=["alice"])
    os.symlink(outside / "secret.txt", dms["folder"] / "linked.txt")
    os.symlink(outside / "secret.txt.acl.json", dms["folder"] / "linked.txt.acl.json")
    run = sync(dms)
    assert run["report"]["skipped_unsafe"] >= 1 and "linked" not in titles(alice, dms)
    r = admin.post("/api/legal/connectors", headers=csrf_headers(admin), json={"kind": "local_folder", "name": "x", "root": str(outside)})
    assert r.status_code == 422                                              # not an allowed root


def test_connector_settings_are_sealed_at_rest(dms, fresh_db):
    with sqlite3.connect(fresh_db) as c:
        sealed = c.execute("SELECT config_sealed FROM legal_connectors").fetchone()[0]
    assert str(dms["root"]) not in sealed and "dms" not in sealed


def test_who_may_manage_connectors_and_links(dms, fresh_db):
    alice, carol, bob, admin = (as_user(dms, n) for n in ("alice", "carol", "bob", "admin"))
    assert alice.post("/api/legal/connectors", headers=csrf_headers(alice), json={"kind": "local_folder", "name": "x", "root": str(dms["root"])}).status_code == 403
    assert alice.get("/api/legal/connectors").status_code == 403
    link = {"connector_id": dms["cid"], "remote_path": "Harrowgate"}
    assert carol.post(f"/api/legal/matters/{dms['mid']}/connector-links", headers=csrf_headers(carol), json=link).status_code == 403
    assert bob.post(f"/api/legal/matters/{dms['mid']}/connector-links", headers=csrf_headers(bob), json=link).status_code == 404
    links = alice.get(f"/api/legal/matters/{dms['mid']}/connector-links").json()
    assert links["links"][0]["remote_path"] == "Harrowgate/MSA" and links["connectors"][0]["name"] == "Firm DMS (test folder)"
    listed = admin.get("/api/legal/connectors").json()["connectors"][0]
    assert listed["links"] == 1 and "root" not in json.dumps(listed)                 # no matter names or paths for the admin
    sync(dms)
    with sqlite3.connect(fresh_db) as c:
        acts = {r[0] for r in c.execute("SELECT action FROM audit_events")}
    assert {"connector.create", "connector.link", "connector.sync"} <= acts


def test_synced_permissions_also_bind_exports(dms):
    """Only someone the source names may export the document, and only such a person may approve."""
    from app.legal.matters import repo
    sync(dms)
    mid, alice, carol = dms["mid"], as_user(dms, "alice"), as_user(dms, "carol")
    msa = titles(alice, dms)["msa"]
    mm = [m for m in repo.list_live_memberships(mid) if m["username"] == "carol"][0]
    repo.change_membership(mid, mm["id"], by=dms["users"]["alice"]["id"], role="attorney", can_export=True)
    body = {"object_type": "document", "object_id": msa["id"], "format": "pdf", "purpose": "x"}
    assert carol.post(f"/api/legal/matters/{mid}/exports", headers=csrf_headers(carol), json=body).status_code == 404
    e = alice.post(f"/api/legal/matters/{mid}/exports", headers=csrf_headers(alice), json=body).json()["export"]
    alice.post(f"/api/legal/matters/{mid}/exports/{e['id']}/submit", headers=csrf_headers(alice))
    r = carol.post(f"/api/legal/matters/{mid}/exports/{e['id']}/approve", headers=csrf_headers(carol), json={})
    assert r.status_code == 404                                              # carol is not named on msa
    assert carol.get(f"/api/legal/matters/{mid}/exports").json()["exports"] == []   # nor sees that it exists
    assert carol.get(f"/api/legal/matters/{mid}/exports/{e['id']}").status_code == 404
