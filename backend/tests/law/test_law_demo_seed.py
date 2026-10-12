"""S1.3 — the fictional demo world used by end-to-end tests and design reviews."""
from __future__ import annotations

import json
import os
import stat


def test_seed_demo_refuses_without_explicit_permission(fresh_db, law_edition, monkeypatch, capsys):
    from app.legal.cli import main
    monkeypatch.delenv("LEGAL_ALLOW_DEMO_SEED", raising=False)
    assert main(["seed-demo"]) == 2
    assert "refusing" in capsys.readouterr().err


def test_seed_demo_builds_the_t01_world_once(fresh_db, law_edition, monkeypatch):
    from app.core.config import settings
    from app.legal.cli import main
    from app.legal.matters import repo
    from app.core import auth
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "AUTH_SECRET", "s" * 48)
    monkeypatch.setenv("LEGAL_ALLOW_DEMO_SEED", "1")
    assert main(["seed-demo"]) == 0
    path = os.path.join(settings.DATA_DIR, "legal", "demo_credentials.json")
    creds = json.load(open(path))
    assert set(creds) == {"alice", "bob", "carol", "morgan", "nakamura", "fiona", "rhea", "iris", "conway"} and stat.S_IMODE(os.stat(path).st_mode) == 0o600
    from app.legal.intake import screening                  # S4.1: the other side on the demo matters is screened against
    from app.legal.orgs import default_org_id
    assert {(h["matched"], h["matter_number"]) for h in screening.screen(default_org_id(), ["Orrin Marine Limited"])} == {("Orrin Marine Ltd", "M-202")}
    def visible(name):
        u = auth.get_user_by_username(name)
        from app.legal.orgs import default_org_id
        return {m["number"] for m in repo.list_visible_matters({"sub": u["id"], "org_id": default_org_id()})}
    assert visible("alice") == {"M-101", "M-114"}          # her M-202 membership expired
    assert visible("bob") == {"M-202", "M-215"}
    assert visible("carol") == {"M-101"}                   # her M-215 membership was revoked
    assert main(["seed-demo"]) == 0                         # idempotent
    assert json.load(open(path)) == creds


def test_seed_demo_installs_the_firm_checklists_approved(fresh_db, law_edition, monkeypatch):
    """S3.7/S3.8: both presets, approved by the demo managing partner; the governance review asks its checks
    (journey 29 found the seed dropping them)."""
    from app.core.config import settings
    from app.legal.cli import main
    from app.legal.diligence import checklists
    from app.legal.orgs import default_org_id
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "AUTH_SECRET", "s" * 48)
    monkeypatch.setenv("LEGAL_ALLOW_DEMO_SEED", "1")
    assert main(["seed-demo"]) == 0
    got = {c["name"]: c for c in checklists.listing(default_org_id(), drafts=True)}
    assert all(c["status"] == "approved" for c in got.values())
    assert got["Corporate governance review"]["governance"] == ["share_capital", "authority", "constitution", "ip_chain"]
    assert got["Share acquisition — core documents"]["governance"] == []


# D-064: the practice group and the fictional cases, added to the demo world from a plan given on stdin.
PRACTICE = {
    "team": {"harper": {"roles": ["attorney"], "about": "senior associate"}, "omar": {"roles": ["paralegal"], "about": "paralegal"}},
    "matters": [{"number": "M-301", "title": "Northwind v Calder Freight", "client": {"name": "Northwind Logistics Ltd", "code": "NWL"},
                 "practice_area": "Commercial litigation", "jurisdiction": "England & Wales", "lead": "harper",
                 "members": {"omar": "paralegal", "alice": "attorney"}, "parties": [["Calder Freight Ltd", "adverse"]]},
                {"number": "R-001", "title": "Reference library: public law", "client": {"name": "Firm knowledge (internal)", "code": "KNOW"},
                 "practice_area": "Knowledge", "jurisdiction": "England & Wales; United States", "lead": "morgan",
                 "members": {"alice": "read_only", "harper": "read_only"}, "parties": []}],
}


def _practice(monkeypatch, plan=PRACTICE):
    import io
    import sys
    from app.legal.cli import main
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(plan)))
    return main(["seed-demo-practice"])


def test_seed_demo_practice_refuses_without_permission_or_the_demo_world(fresh_db, law_edition, monkeypatch, capsys):
    monkeypatch.delenv("LEGAL_ALLOW_DEMO_SEED", raising=False)
    assert _practice(monkeypatch) == 2 and "refusing" in capsys.readouterr().err
    monkeypatch.setenv("LEGAL_ALLOW_DEMO_SEED", "1")
    assert _practice(monkeypatch) == 2 and "seed-demo first" in capsys.readouterr().err


def test_seed_demo_practice_adds_the_group_and_the_cases_once(fresh_db, law_edition, monkeypatch, capsys):
    from app.core import auth
    from app.core.config import settings
    from app.legal.cli import main
    from app.legal.intake import screening
    from app.legal.matters import repo
    from app.legal.orgs import default_org_id
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "AUTH_SECRET", "s" * 48)
    monkeypatch.setenv("LEGAL_ALLOW_DEMO_SEED", "1")
    assert main(["seed-demo"]) == 0
    path = os.path.join(settings.DATA_DIR, "legal", "demo_credentials.json")
    before = json.load(open(path))
    capsys.readouterr()
    assert _practice(monkeypatch) == 0
    out = capsys.readouterr().out
    creds = json.load(open(path))
    assert set(creds) == set(before) | {"harper", "omar"} and all(creds[u] == before[u] for u in before)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert "demo " not in out and creds["harper"]["password"] not in out                # no password is printed

    def visible(name):
        u = auth.get_user_by_username(name)
        return {m["number"] for m in repo.list_visible_matters({"sub": u["id"], "org_id": default_org_id()})}
    assert visible("harper") == {"M-301", "R-001"} and visible("omar") == {"M-301"}
    assert visible("alice") == {"M-101", "M-114", "M-301", "R-001"}
    assert ("Calder Freight Ltd", "M-301") in {(h["matched"], h["matter_number"]) for h in screening.screen(default_org_id(), ["Calder Freight Limited"])}
    assert _practice(monkeypatch) == 0                                                  # idempotent
    assert json.load(open(path)) == creds and visible("alice") == {"M-101", "M-114", "M-301", "R-001"}
