"""S2.8 — evaluation runs recorded with the commit, models and dataset they measured (OPS-02; D-044).

Every evaluation report (`eval/law/reports/<time>-<suite>-<set>.json`) can be imported into the
Law database. A run keeps its suite, set, commit, models, the hash of the dataset it read, its
summary and where its full report is; never its items, which stay in the report file. Firm audit
readers can list the runs; nobody else can.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from conftest import as_user

ROOT = Path(__file__).resolve().parents[3]


def report(tmp_path, name, meta, **body):
    p = tmp_path / name
    p.write_text(json.dumps({"meta": meta, **body}))
    return p


def test_a_research_report_is_imported_with_its_commit_models_and_dataset(world, tmp_path, fresh_db):
    from app.legal import evaluations
    p = report(tmp_path, "20261010-120000-heldout.json",
               {"set": "heldout", "commit": "abc1234", "model": "nvidia/Qwen3-30B-A3B-FP4", "embeddings": "nomic-embed-text",
                "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2", "time": "20261010-120000"},
               summary={"items": 32, "answer_accuracy": "28/29", "citation_precision": 0.93}, items=[{"question": "secret item text"}])
    run = evaluations.import_report(p)
    assert run["suite"] == "research" and run["set"] == "heldout" and run["commit"] == "abc1234"
    assert run["models"] == {"model": "nvidia/Qwen3-30B-A3B-FP4", "embeddings": "nomic-embed-text",
                             "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
    assert run["dataset_sha256"] == hashlib.sha256((ROOT / "eval/law/heldout.jsonl").read_bytes()).hexdigest()
    assert run["summary"]["citation_precision"] == 0.93 and run["report"] == "20261010-120000-heldout.json"
    assert run["report_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    with sqlite3.connect(fresh_db) as c:
        stored = json.dumps(c.execute("SELECT * FROM legal_evaluation_runs").fetchall())
    assert "secret item text" not in stored                                   # items stay in the report file
    assert evaluations.import_report(p)["id"] == run["id"]                    # the same report is imported once


@pytest.mark.parametrize("name, suite, set_, dataset", [
    ("20261010-120000-compare-golden.json", "compare", "golden", "eval/law/compare/golden.json"),
    ("20261010-120000-assist-heldout.json", "assist", "heldout", "eval/law/assist/heldout.json"),
    ("20261010-120000-minutes-golden.json", "minutes", "golden", "eval/law/minutes/golden.json"),
    ("20261010-120000-transcription-script.json", "transcription", "script", "eval/law/transcription/scenarios.json"),
])
def test_each_suite_is_named_from_its_report_and_hashes_its_dataset(world, tmp_path, name, suite, set_, dataset):
    from app.legal import evaluations
    p = report(tmp_path, name, {"commit": "abc1234", "time": "20261010-120000"}, alignment={"pairs": 10}, explanations={"verified": 1.0})
    run = evaluations.import_report(p)
    assert (run["suite"], run["set"]) == (suite, set_)
    assert run["dataset_sha256"] == hashlib.sha256((ROOT / dataset).read_bytes()).hexdigest()
    assert run["summary"] == {"alignment": {"pairs": 10}, "explanations": {"verified": 1.0}}   # suites without a summary keep their sections


def test_a_file_that_is_not_an_evaluation_report_is_refused(world, tmp_path):
    from app.legal import evaluations
    bad = tmp_path / "notes.json"
    bad.write_text(json.dumps({"hello": 1}))
    with pytest.raises(ValueError):
        evaluations.import_report(bad)


def test_only_firm_audit_readers_see_the_runs(world, tmp_path):
    from app.legal import evaluations
    evaluations.import_report(report(tmp_path, "20261010-120000-golden.json", {"set": "golden", "commit": "abc1234", "time": "20261010-120000"},
                                     summary={"items": 70}))
    r = as_user(world, "mp").get("/api/legal/evaluations")
    assert r.status_code == 200 and [x["suite"] for x in r.json()["runs"]] == ["research"]
    assert as_user(world, "alice").get("/api/legal/evaluations").status_code == 403


def test_the_command_imports_reports_into_the_configured_database(world, tmp_path, capsys):
    from app.legal import evaluations
    a = report(tmp_path, "20261010-120000-golden.json", {"set": "golden", "commit": "abc1234", "time": "20261010-120000"}, summary={})
    b = report(tmp_path, "20261010-120100-minutes-heldout.json", {"set": "heldout", "commit": "abc1234", "time": "20261010-120100"}, summary={})
    assert evaluations.main(["import", str(a), str(b)]) == 0
    assert "2 run(s) recorded" in capsys.readouterr().out
    assert len(evaluations.listing()) == 2
