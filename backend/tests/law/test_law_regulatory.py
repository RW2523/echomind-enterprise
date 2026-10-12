"""S4.4 — regulatory updates (OPT-04; decision D-068).

New public legislation and rules are fetched daily on the host from official sources and imported as data; the
Law containers stay offline. Every field is checked again on import (the bundle is untrusted text). Each update
is matched to open matters by jurisdiction and practice-area terms, with the terms kept as the reason. A lawyer
sees an update with only the matters they may see; on a matter, a lawyer marks a match reviewed or not relevant
(with a reason). Off unless the firm turns it on.
"""
from __future__ import annotations

import importlib.util
import io
import json
import pathlib
import sqlite3

import pytest

from conftest import as_user, csrf_headers

ROOT = pathlib.Path(__file__).resolve().parents[3]


def item(source="legislation.gov.uk", ext="uksi/2026/901", title="The Employment Rights Act 2025 (Commencement No. 3) Regulations 2026",
         url=None, **kw):
    host = "www." + source
    return {"source": source, "external_id": ext, "title": title, "summary": kw.pop("summary", ""),
            "url": url or f"https://{host}/{ext}/made", "published": kw.pop("published", "2026-10-09"),
            "kind": kw.pop("kind", "UK Statutory Instrument"), "subjects": kw.pop("subjects", []), **kw}


BUNDLE = {"format": "echomind-law-regulatory/1", "fetched_at": "2026-10-12T06:15:00Z", "items": [
    item(),
    item("federalregister.gov", "2026-20111", "Residential Lease Disclosures", summary="Landlords must disclose fees to a tenant.",
         kind="Final rule", subjects=["Housing and Urban Development Department"], licence="all rights reserved"),
    item(ext="nisr/2026/176", title="The Parking Places (Cookstown) (Amendment) Order (Northern Ireland) 2026"),
    item(ext="uksi/2026/950", title="The Late Payment‮ of Commercial Debts\x00 (Amendment) Regulations 2026"),
    item(ext="uksi/2026/951", url="https://evil.example/uksi/2026/951"),           # a link off the source's own site
    item(source="example.org", ext="x/1"),                                          # an unknown source
]}


@pytest.fixture
def reg(world, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "LEGAL_REGULATORY", True)
    from app.core.db import get_conn
    with get_conn() as conn:
        for number, area, where in (("M-101", "Employment", "England & Wales"), ("M-202", "Employment", "England & Wales"),
                                    ("M-215", "Real estate", "New York"), ("M-114", "Corporate", "Delaware")):
            conn.execute("UPDATE matters SET practice_area=?, jurisdiction=? WHERE id=?", (area, where, world["matters"][number]["id"]))
        conn.commit()
    return world


def load(bundle=BUNDLE):
    from app.legal.orgs import default_org_id
    from app.legal.regulatory import store
    return store.import_bundle(json.dumps(bundle).encode(), default_org_id())


def q(db, sql, *args):
    with sqlite3.connect(db) as c:
        return c.execute(sql, args).fetchall()


def test_off_unless_the_firm_turns_it_on(world, monkeypatch):
    from app.legal.regulatory import __main__ as cli
    alice = as_user(world, "alice")
    assert alice.get("/api/legal/regulatory").status_code == 404
    assert alice.get(f"/api/legal/matters/{world['matters']['M-101']['id']}/regulatory").status_code == 404
    assert alice.get("/api/auth/me").json()["features"]["regulatory"] is False
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(json.dumps(BUNDLE).encode())))
    assert cli.main(["import"]) == 2


def test_import_checks_every_field_and_matches_by_jurisdiction_and_practice_area(reg, fresh_db):
    assert load() == {"items": 6, "skipped": 2, "new": 4, "known": 0, "matches": 3}
    rows = dict(q(fresh_db, "SELECT external_id, title FROM legal_regulatory_updates"))
    assert rows["uksi/2026/950"] == "The Late Payment of Commercial Debts (Amendment) Regulations 2026"    # control characters gone
    assert "uksi/2026/951" not in rows and "x/1" not in rows
    lic = dict(q(fresh_db, "SELECT source, licence FROM legal_regulatory_updates"))
    assert lic == {"legislation.gov.uk": "OGL-3.0", "federalregister.gov": "public-domain"}   # from our table, never the bundle
    matched = q(fresh_db, "SELECT u.external_id, m.number, r.terms_json FROM legal_regulatory_matches r "
                          "JOIN legal_regulatory_updates u ON u.id = r.update_id JOIN matters m ON m.id = r.matter_id ORDER BY 1, 2")
    assert [(e, n, json.loads(t)) for e, n, t in matched] == [
        ("2026-20111", "M-215", ["lease", "landlord", "tenant", "housing"]),   # US real estate (plurals count); not the UK matters
        ("uksi/2026/901", "M-101", ["employment"]), ("uksi/2026/901", "M-202", ["employment"])]
    assert load() == {"items": 6, "skipped": 2, "new": 0, "known": 4, "matches": 0}               # a day's overlap is not repeated


def test_generic_words_do_not_match_and_specific_ones_do():
    """From the first real import: "Commercial" in a fishery quota, "enforce" in a Coast Guard safety zone and "freight" in
    a rail notice matched commercial and investigations matters. The terms are now specific legal phrases."""
    from app.legal.regulatory import topics
    us = lambda title, summary="": {"jurisdiction": "US", "title": title, "summary": summary, "subjects": []}
    commercial, inv = {"practice_area": "Commercial contracts", "jurisdiction": "Delaware"}, {"practice_area": "Investigations", "jurisdiction": "New York"}
    assert topics.match(us("Fisheries of the Northeastern United States; Summer Flounder Fishery; 2026 Commercial Quota"), commercial) == []
    assert topics.match(us("Advance Traveler Information for Rail", "Freight and passenger rail carriers"), commercial) == []
    assert topics.match(us("Safety Zone; Chicago River", "The Coast Guard will enforce a safety zone."), inv) == []
    assert topics.match(us("Arbitration Agreements in Consumer Contracts"), commercial) == ["arbitration", "consumer contracts"]
    assert topics.match(us("Publication of Venezuela Sanctions Regulations Web General License 60"), inv) == ["sanctions"]


def test_rematch_follows_new_terms_and_keeps_decisions(reg, fresh_db, monkeypatch):
    from app.legal.orgs import default_org_id
    from app.legal.regulatory import store, topics
    load()
    m101 = reg["matters"]["M-101"]["id"]
    alice = as_user(reg, "alice")
    uid = alice.get(f"/api/legal/matters/{m101}/regulatory").json()["matches"][0]["update"]["id"]
    alice.post(f"/api/legal/matters/{m101}/regulatory/{uid}", headers=csrf_headers(alice), json={"status": "reviewed", "version": 1})
    monkeypatch.setitem(topics.AREAS, ("employment", "labour", "labor"), ("late payment",))     # the terms change
    assert store.rematch(default_org_id()) == {"updates": 4, "added": 2, "removed": 1, "kept_decided": 1}
    rows = q(fresh_db, "SELECT m.number, u.external_id, r.status FROM legal_regulatory_matches r JOIN matters m ON m.id = r.matter_id "
                       "JOIN legal_regulatory_updates u ON u.id = r.update_id WHERE m.number IN ('M-101', 'M-202') ORDER BY 1, 2")
    assert rows == [("M-101", "uksi/2026/901", "reviewed"),          # decided: kept as decided
                    ("M-101", "uksi/2026/950", "new"), ("M-202", "uksi/2026/950", "new")]   # M-202's undecided 901 no longer matches


@pytest.mark.parametrize("bad", [b"not json", json.dumps({"format": "other", "items": []}).encode(),
                                 json.dumps({"format": "echomind-law-regulatory/1", "items": {}}).encode()])
def test_a_bundle_that_is_not_one_is_refused_whole(reg, bad):
    from app.legal.orgs import default_org_id
    from app.legal.regulatory import store
    with pytest.raises(store.Invalid):
        store.import_bundle(bad, default_org_id())


def test_lawyers_see_each_update_with_only_the_matters_they_may_see(reg):
    load()
    seen = {}
    for name in ("alice", "bob", "carol", "mp"):
        r = as_user(reg, name).get("/api/legal/regulatory")
        assert r.status_code == 200, r.text
        seen[name] = {u["external_id"]: [m["number"] for m in u["matters"]] for u in r.json()["updates"]}
    assert seen["alice"]["uksi/2026/901"] == ["M-101"]          # never M-202: highly restricted, and not hers
    assert seen["bob"]["uksi/2026/901"] == ["M-202"]
    assert seen["carol"]["uksi/2026/901"] == ["M-101"]          # read-only members see it too
    assert seen["mp"]["uksi/2026/901"] == []                     # the update is public law; the matters are not
    assert set(seen["alice"]) == {"uksi/2026/901", "2026-20111", "nisr/2026/176", "uksi/2026/950"}
    mine = as_user(reg, "alice").get("/api/legal/regulatory?mine=1").json()["updates"]
    assert [u["external_id"] for u in mine] == ["uksi/2026/901"]
    first = next(u for u in as_user(reg, "alice").get("/api/legal/regulatory?jurisdiction=US").json()["updates"])
    assert first["external_id"] == "2026-20111" and first["licence"] == "public-domain" and first["url"].startswith("https://www.federalregister.gov/")
    assert as_user(reg, "admin").get("/api/legal/regulatory").status_code == 403          # for the firm's lawyers


def test_on_a_matter_a_lawyer_decides_each_match_and_not_relevant_needs_a_reason(reg, fresh_db):
    load()
    m101, m202 = reg["matters"]["M-101"]["id"], reg["matters"]["M-202"]["id"]
    alice, carol, dave = (as_user(reg, n) for n in ("alice", "carol", "dave"))
    got = alice.get(f"/api/legal/matters/{m101}/regulatory").json()["matches"]
    assert [(g["update"]["external_id"], g["terms"], g["status"]) for g in got] == [("uksi/2026/901", ["employment"], "new")]
    base = f"/api/legal/matters/{m101}/regulatory/{got[0]['update']['id']}"
    r = alice.post(base, headers=csrf_headers(alice), json={"status": "not_relevant", "version": 1})
    assert r.status_code == 422 and "reason" in r.json()["detail"]
    r = alice.post(base, headers=csrf_headers(alice), json={"status": "not_relevant", "note": "Commencement only; no change for the claim.", "version": 1})
    assert r.status_code == 200, r.text
    assert r.json()["match"]["status"] == "not_relevant" and r.json()["match"]["decided_by"] == "alice"
    assert alice.post(base, headers=csrf_headers(alice), json={"status": "reviewed", "version": 1}).status_code == 409   # changed since
    assert carol.post(base, headers=csrf_headers(carol), json={"status": "reviewed", "version": 2}).status_code == 403  # read-only
    assert dave.get(f"/api/legal/matters/{m202}/regulatory").status_code == 404        # expired membership: not visible
    details = [d for (d,) in q(fresh_db, "SELECT details_json FROM audit_events WHERE action='regulatory.decide'") if d]
    assert {"status": "not_relevant", "reason_given": True} in [json.loads(d) for d in details]
    assert not any("Commencement" in d for d in details)                                   # the reason's words are not in the trail


def test_the_import_command_records_counts_only(reg, fresh_db, monkeypatch, capsys):
    from app.legal.regulatory import __main__ as cli
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(json.dumps(BUNDLE).encode())))
    assert cli.main(["import"]) == 0
    assert json.loads(capsys.readouterr().out)["new"] == 4
    (details,), = q(fresh_db, "SELECT details_json FROM audit_events WHERE action='regulatory.import'")
    assert json.loads(details) == {"items": 6, "skipped": 2, "new": 4, "known": 0, "matches": 3}


# ── The host-side fetcher (scripts/law/fetch_regulatory.py), offline ────────────
def fetcher():
    spec = importlib.util.spec_from_file_location("fetch_regulatory", ROOT / "scripts" / "law" / "fetch_regulatory.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


UK_FEED = b"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE feed [<!ENTITY secret SYSTEM "file:///etc/passwd">]>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:ukm="http://www.legislation.gov.uk/namespaces/metadata">
<entry><id>http://www.legislation.gov.uk/id/uksi/2026/901</id><title>The Employment Rights Regulations 2026 &secret;</title>
<link rel="self" href="http://www.legislation.gov.uk/id/uksi/2026/901"/><link href="http://www.legislation.gov.uk/uksi/2026/901/made"/>
<published>2026-10-12T07:35:11+01:00</published><ukm:DocumentMainType Value="UnitedKingdomStatutoryInstrument"/><summary/></entry>
<entry><id>http://www.legislation.gov.uk/uksi/2026/551/pdfs/uksics_20260551_en_001.pdf</id><title>Correction Slip</title>
<link href="http://www.legislation.gov.uk/uksi/2026/551/pdfs/uksics_20260551_en_001.pdf"/><published>2026-10-12T07:00:00+01:00</published></entry>
<entry><id>http://www.legislation.gov.uk/id/uksi/2026/800</id><title>Older</title><link href="http://www.legislation.gov.uk/uksi/2026/800/made"/>
<published>2026-10-01T07:35:11+01:00</published></entry>
</feed>"""


def test_the_fetcher_reads_the_official_feeds_and_never_resolves_entities():
    import datetime as dt
    f = fetcher()
    got, older = f.uk_entries(UK_FEED, dt.date(2026, 10, 10))
    assert older is True and [g["external_id"] for g in got] == ["uksi/2026/901", "uksi/2026/551/pdfs/uksics_20260551_en_001.pdf"]
    assert got[0]["url"] == "https://www.legislation.gov.uk/uksi/2026/901/made"
    from app.legal.regulatory import store
    assert store.clean(got[1])["external_id"].startswith("uksi/2026/551/")              # a correction slip is kept, not skipped
    assert got[0]["kind"] == "UK Statutory Instrument" and "root:" not in got[0]["title"]
    page = json.dumps({"results": [{"document_number": "2026-1", "title": "T", "abstract": "A", "type": "Proposed Rule",
                                    "html_url": "https://www.federalregister.gov/documents/2026/10/12/2026-1/t", "publication_date": "2026-10-12",
                                    "agencies": [{"name": "Labor Department"}], "topics": ["Wages"]}],
                       "next_page_url": "https://www.federalregister.gov/api/v1/documents.json?page=2"}).encode()
    items, nxt = f.us_page(page)
    assert items[0]["kind"] == "Proposed rule" and items[0]["subjects"] == ["Labor Department", "Wages"] and nxt.endswith("page=2")
    with pytest.raises(ValueError):
        f.us_page(json.dumps({"results": [{"document_number": "2026-2", "html_url": "https://evil.example/x"}]}).encode())
