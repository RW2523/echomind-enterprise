"""S1.6 / SEC-03 — partitioned retrieval: canary leakage suite v1, T02, T06, scope capability.

Each matter's documents live in their own dense and sparse index (decision D-005). Searching is
only possible with an AuthorizedScope minted by the policy engine, and is re-authorized on use.
"""
from __future__ import annotations

import json
import os
import sqlite3

import pytest

from conftest import as_user, csrf_headers, index_all, search_in, upload

CANARY = {"M-101": "ZEBRA-1017", "M-114": "OTTER-1147", "M-202": "LYNX-2027", "M-215": "MOOSE-2157"}
# What each person may read in the T01 world (memberships live; dave expired; carol revoked on M-215).
READS = {"alice": {"M-101", "M-114"}, "bob": {"M-202", "M-215"}, "carol": {"M-101"}, "dave": set(),
         "mp": set(), "mp2": set(), "admin": set()}
MODES = [["dense"], ["sparse"], ["keyword"], ["dense", "sparse", "keyword"]]


@pytest.fixture
def seeded(docs):
    """One document per matter carrying that matter's canary, imported by each matter's lead."""
    lead = {"M-101": "alice", "M-114": "alice", "M-202": "bob", "M-215": "bob"}
    for num, token in CANARY.items():
        c = as_user(docs, lead[num])
        r = upload(c, docs["matters"][num]["id"], f"{num}.txt",
                   f"Privileged memo for {num}. The reference code {token} identifies the settlement position.")
        assert r.status_code == 202, r.text
    assert index_all() == 4
    return docs


def test_canary_suite_v1_no_matter_ever_returns_another_matters_text(seeded):
    for person, readable in READS.items():
        c = as_user(seeded, person)
        for num, m in seeded["matters"].items():
            for token in CANARY.values():
                for modes in MODES:
                    r = search_in(c, m["id"], f"reference code {token} settlement", modes=modes)
                    if num not in readable:
                        assert r.status_code == 404 and r.json() == {"detail": "Not Found"}, (person, num)
                        continue
                    assert r.status_code == 200, (person, num, r.text)
                    body = json.dumps(r.json())
                    for other_num, other in CANARY.items():
                        if other_num != num:
                            assert other not in body, (person, num, token, modes)
                    for p in r.json()["passages"]:
                        assert p["citation"]["matter_id"] == m["id"]


def test_each_matter_finds_its_own_canary(seeded):
    alice = as_user(seeded, "alice")
    for num in ("M-101", "M-114"):
        for modes in MODES:
            hits = search_in(alice, seeded["matters"][num]["id"], CANARY[num], modes=modes).json()["passages"]
            assert hits and CANARY[num] in hits[0]["text"], (num, modes)
            assert hits[0]["citation"]["document_title"] == f"{num}.txt"


def test_t02_revoking_membership_ends_access_on_the_next_query(seeded):
    carol, alice = as_user(seeded, "carol"), as_user(seeded, "alice")
    mid = seeded["matters"]["M-101"]["id"]
    assert search_in(carol, mid, CANARY["M-101"]).status_code == 200
    mm = [m for m in alice.get(f"/api/legal/matters/{mid}/memberships").json()["memberships"] if m["username"] == "carol"][0]
    alice.delete(f"/api/legal/matters/{mid}/memberships/{mm['id']}", headers=csrf_headers(alice))
    assert search_in(carol, mid, CANARY["M-101"]).status_code == 404


def test_t06_the_answer_sits_in_a_forbidden_matter_so_the_permitted_search_abstains(seeded):
    bob, alice = as_user(seeded, "bob"), as_user(seeded, "alice")
    upload(bob, seeded["matters"]["M-202"]["id"], "terms.txt", "The board approved a settlement amount of 4.2 million euros.")
    index_all()
    r = search_in(alice, seeded["matters"]["M-101"]["id"], "board approved settlement amount euros")
    body = r.json()
    assert r.status_code == 200 and "4.2 million" not in json.dumps(body)
    assert body["abstained"] is True and body["passages"] == []


def test_a_wall_or_break_glass_changes_search_like_everything_else(seeded):
    mp, mp2, admin = (as_user(seeded, n) for n in ("mp", "mp2", "admin"))
    m114 = seeded["matters"]["M-114"]["id"]
    w = mp.post("/api/legal/walls", headers=csrf_headers(mp), json={"name": "screen", "rationale": "r", "screened": ["alice"],
                "protected_matters": [m114], "protected_clients": []}).json()["wall"]
    mp2.post(f"/api/legal/walls/{w['id']}/approve", headers=csrf_headers(mp2))
    assert search_in(as_user(seeded, "alice"), m114, CANARY["M-114"]).status_code == 404
    g = admin.post("/api/legal/break-glass", headers=csrf_headers(admin), json={"matter_number": "M-215", "reason": "x", "minutes": 30}).json()["request"]
    mp.post(f"/api/legal/break-glass/{g['id']}/approve", headers=csrf_headers(mp))
    r = search_in(admin, seeded["matters"]["M-215"]["id"], CANARY["M-215"])
    assert r.status_code == 200 and CANARY["M-215"] in r.json()["passages"][0]["text"]
    assert upload(admin, seeded["matters"]["M-215"]["id"], "x.txt", "no imports").status_code == 403     # read only


# ── The capability itself ───────────────────────────────────────────────────────
def test_a_scope_cannot_be_made_outside_the_policy_engine(seeded):
    import asyncio
    from app.legal.retrieval import scope as scopemod
    from app.legal.retrieval import search
    with pytest.raises(TypeError):
        scopemod.AuthorizedScope(principal_id="x", org_id="o", matter_id=seeded["matters"]["M-202"]["id"],
                                 partition="m:x", role="lead", break_glass=False)
    with pytest.raises(TypeError):
        asyncio.run(search.search({"partition": "m:" + seeded["matters"]["M-202"]["id"]}, "LYNX-2027"))


def test_a_scope_is_rechecked_when_used(seeded):
    """A scope minted before a revocation is refused at search time, not trusted."""
    import asyncio
    from app.core.db import get_conn
    from app.legal.matters import repo
    from app.legal.orgs import default_org_id
    from app.legal.retrieval import scope as scopemod
    from app.legal.retrieval import search
    mid = seeded["matters"]["M-101"]["id"]
    principal = {"sub": seeded["users"]["alice"]["id"], "org_id": default_org_id()}
    s = scopemod.authorize(principal, mid)
    assert asyncio.run(search.search(s, CANARY["M-101"]))["passages"]
    lead = [m for m in repo.list_live_memberships(mid) if m["username"] == "alice"][0]
    with get_conn() as conn:                     # out-of-band revocation of alice's own membership
        conn.execute("UPDATE matter_memberships SET revoked_at='2000-01-01T00:00:00.000Z' WHERE id=?", (lead["id"],))
        conn.commit()
    with pytest.raises(scopemod.ScopeRevoked):
        asyncio.run(search.search(s, CANARY["M-101"]))


# ── Ingestion and the partitions on disk ─────────────────────────────────────────
def test_ingestion_rolls_back_if_the_uploader_loses_rights_mid_job(docs, fresh_db, monkeypatch):
    from conftest import HashEmbedder
    from app.legal.matters import repo
    from app.legal.retrieval import embedder
    carol_uid = docs["users"]["carol"]["id"]
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    mm = [m for m in repo.list_live_memberships(mid) if m["username"] == "carol"][0]
    repo.change_membership(mid, mm["id"], by=docs["users"]["alice"]["id"], role="paralegal")
    carol = as_user(docs, "carol")
    assert upload(carol, mid, "carol.txt", "Paralegal note with WALRUS-77 inside.").status_code == 202

    class Revoking(HashEmbedder):
        async def embed(self, texts, kind="query"):
            live = [m for m in repo.list_live_memberships(mid) if m["user_id"] == carol_uid]
            for m in live:
                repo.revoke_membership(mid, m["id"], by=docs["users"]["alice"]["id"], reason="left")
            return await super().embed(texts, kind)
    monkeypatch.setattr(embedder, "_override", Revoking())
    index_all()
    with sqlite3.connect(fresh_db) as c:
        assert c.execute("SELECT COUNT(*) FROM legal_chunks").fetchone()[0] == 0
        status = c.execute("SELECT index_status, index_error FROM legal_document_versions").fetchone()
        audit = c.execute("SELECT outcome, reason FROM audit_events WHERE action='document.index'").fetchall()
    assert status == ("failed", "rights_revoked")
    assert audit == [("error", "rights_revoked")]
    monkeypatch.setattr(embedder, "_override", HashEmbedder())
    assert search_in(alice, mid, "WALRUS-77").json()["passages"] == []


def test_each_matter_has_its_own_partition_on_disk(seeded, fresh_db):
    root = fresh_db.parent / "legal" / "partitions"
    names = sorted(os.listdir(root))
    assert len(names) == 4
    for n in names:
        meta = json.loads((root / n / "meta.json").read_text())
        assert meta["partition"].startswith("m:") and len(meta["chunk_ids"]) >= 1
        others = [t for num, t in CANARY.items() if seeded["matters"][num]["id"] not in meta["partition"]]
        raw = (root / n / "meta.json").read_text()
        assert not any(t.lower() in raw.lower() for t in others)


def test_a_partition_rebuilds_from_the_database_and_never_trusts_a_misplaced_file(seeded, fresh_db):
    import shutil
    from app.legal.retrieval import registry
    root = fresh_db.parent / "legal" / "partitions"
    m101, m202 = seeded["matters"]["M-101"]["id"], seeded["matters"]["M-202"]["id"]
    d101, d202 = root / registry.dirname("m:" + m101), root / registry.dirname("m:" + m202)
    shutil.rmtree(d101)
    shutil.copytree(d202, d101)                     # M-202's index files placed under M-101
    registry.clear()
    alice = as_user(seeded, "alice")
    r = search_in(alice, m101, "reference code settlement")
    assert r.status_code == 200 and CANARY["M-202"] not in json.dumps(r.json())
    assert CANARY["M-101"] in json.dumps(search_in(alice, m101, CANARY["M-101"]).json())


# ── Long passages: the clause that matched is found and quoted (E2E regression) ──
def test_a_clause_beyond_the_rerankers_reading_window_is_found_and_quoted(docs, monkeypatch):
    """The real cross-encoder reads ~1,000 characters. A passage whose matching clause sits after
    that point was scored as irrelevant and the search abstained; candidates are now scored in
    sentence windows and the best window is what the result quotes."""
    from conftest import overlap_rerank
    from app.legal.retrieval import search

    async def short_sighted(query, hits, top_k):           # reads one window, far less than the 2,000-character passage
        return await overlap_rerank(query, [{**h, "text": h["text"][:search.WINDOW_CHARS]} for h in hits], top_k)
    monkeypatch.setattr(search, "rerank", short_sighted)
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    filler = " ".join(f"Clause {i}.1 The parties record routine operational matter number {i}." for i in range(30))
    upload(alice, mid, "long.txt", filler + " Clause 31.1 The QUAIL-88 cap limits aggregate liability to the annual fees.")
    index_all()
    hits = search_in(alice, mid, "QUAIL-88 cap aggregate liability").json()["passages"]
    assert hits and "QUAIL-88" in hits[0]["text"] and len(hits[0]["text"]) <= 700
    assert hits[0]["text"].startswith("Clause") and "31.1" in hits[0]["text"]       # "31.1" is not split


def test_windows_follow_sentences_and_never_split_clause_numbers():
    from app.legal.retrieval.search import windows
    text = "Limitation of Liability 8.1 Each party's liability shall not exceed USD 250,000.00 in total. " * 12
    spans = windows(text, 200)
    assert spans and all(b - a <= 200 for a, b in spans)
    assert all(text[a:b].startswith("Limitation") for a, b in spans)
    assert windows("") == [] and windows("short") == [(0, 5)]
    md = "8.3 Clauses 8.1 and 8.2 do not limit liability for fraud. ## 9. Indemnification 9.1 Provider shall indemnify Client."
    assert [md[a:b] for a, b in windows(md)][0] == "8.3 Clauses 8.1 and 8.2 do not limit liability for fraud."
