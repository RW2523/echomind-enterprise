"""Access checked again late, and untrusted files read safely (commercial-readiness review, 2026-10-28: F12, F13).

- A queued summary or due-diligence run is refused when the person who asked has since lost the matter.
- An answer in progress stops when the person's account is disabled, not only when their membership ends.
- Stored Word files are read without resolving entities, loading a DTD or reaching the network.
"""
from __future__ import annotations

import asyncio
import sqlite3

import pytest

from conftest import as_user


ORG = None


@pytest.fixture(autouse=True)
def _org(world):
    global ORG
    from app.legal.orgs import default_org_id
    ORG = default_org_id()


def _end_membership(db, matter_id, user_id):
    with sqlite3.connect(db) as c:
        c.execute("UPDATE matter_memberships SET valid_to='2000-01-01T00:00:00.000Z' WHERE matter_id=? AND user_id=?", (matter_id, user_id))


def test_a_queued_summary_is_refused_once_the_requester_left_the_matter(world, fresh_db):
    from app.legal.summaries import run
    m, alice = world["matters"]["M-101"], world["users"]["alice"]
    job = {"matter_id": m["id"], "object_id": "any-document", "requested_by": alice["id"], "org_id": ORG}
    assert asyncio.run(run.run(job)) == "not_visible"                   # still a member: it goes on to the document
    _end_membership(fresh_db, m["id"], alice["id"])
    assert asyncio.run(run.run(job)) == "not_allowed"


def test_a_queued_diligence_run_is_refused_once_the_requester_left_the_matter(world, fresh_db):
    from app.legal.diligence import run
    m, alice = world["matters"]["M-101"], world["users"]["alice"]
    with sqlite3.connect(fresh_db) as c:
        c.execute("INSERT INTO legal_dd_checklists (id, org_id, family_id, version, name, content_json, content_sha256, status, created_by, "
                  "created_at) VALUES ('cl1', ?, 'f1', 1, 'c', '{\"items\": []}', 'x', 'approved', ?, '2026-10-28T00:00:00.000Z')",
                  (ORG, alice["id"]))
        c.execute("INSERT INTO legal_dd_runs (id, matter_id, checklist_id, status, created_by, created_at) "
                  "VALUES ('r1', ?, 'cl1', 'queued', ?, '2026-10-28T00:00:00.000Z')", (m["id"], alice["id"]))
    _end_membership(fresh_db, m["id"], alice["id"])
    assert asyncio.run(run.run({"object_id": "r1", "org_id": ORG})) == "not_allowed"
    with sqlite3.connect(fresh_db) as c:
        assert c.execute("SELECT status, error FROM legal_dd_runs WHERE id='r1'").fetchone() == ("failed", "not_allowed")


def test_an_answer_in_progress_stops_when_the_account_is_disabled(world):
    from app.legal.identity import accounts
    from app.legal.retrieval import scope
    alice, m = world["users"]["alice"], world["matters"]["M-101"]
    s = scope.authorize({"sub": alice["id"], "org_id": ORG}, m["id"])
    scope.recheck(s)
    accounts.set_status("alice", "disabled")
    with pytest.raises(scope.ScopeRevoked):
        scope.recheck(s)


def test_stored_word_xml_is_read_without_resolving_entities(tmp_path):
    from lxml import etree
    from app.legal.redlines import word
    secret = tmp_path / "secret.txt"
    secret.write_text("XXE-CANARY-4471")
    data = (f'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY x SYSTEM "file://{secret}">]><r>&x;</r>').encode()
    el = word._xml(data)
    assert b"XXE-CANARY-4471" not in etree.tostring(el) and "XXE-CANARY-4471" not in (el.text or "")
