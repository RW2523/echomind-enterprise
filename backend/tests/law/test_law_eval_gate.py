"""The research evaluation is a gate, not a report (commercial-readiness review R8, 2026-10-28).

eval/law/run_law_eval.py --gate applies eval/law/thresholds.json and exits 2 when a bar is missed; it refuses (exit 3) a
tree with uncommitted changes or a held-out set that differs from its frozen checksum. A measure that could not be
computed misses its bar.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]


def _runner():
    spec = importlib.util.spec_from_file_location("run_law_eval", ROOT / "eval" / "law" / "run_law_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GOOD = {"t06_leaks": 0, "answer_accuracy": "29/31", "abstention_accuracy": "2/3", "citation_precision": 0.971,
        "quote_verification_rate": 0.955, "latency_total_p95_s": 4.0}


def test_a_run_that_meets_every_bar_passes_and_each_miss_is_named():
    m = _runner()
    assert m.check_bars("heldout", GOOD) == []
    misses = m.check_bars("heldout", {**GOOD, "t06_leaks": 1, "abstention_accuracy": "1/3", "citation_precision": None})
    assert any(x.startswith("t06_leaks 1") for x in misses)
    assert any(x.startswith("abstention_accuracy 0.333") for x in misses)
    assert any("citation_precision not measured" in x for x in misses)


def test_the_held_out_sets_are_frozen_by_their_checksums():
    import hashlib
    t = json.loads((ROOT / "eval" / "law" / "thresholds.json").read_text())
    for name in ("heldout", "heldout2", "heldout3", "heldout4", "heldout5", "heldout6", "heldout7"):
        assert hashlib.sha256((ROOT / "eval" / "law" / f"{name}.jsonl").read_bytes()).hexdigest() == t[f"{name}_sha256"], name


def test_the_later_held_out_sets_measure_abstention_on_enough_items_and_repeat_nothing():
    sets = ["golden", "heldout", "heldout2", "heldout3", "heldout4", "heldout5", "heldout6", "heldout7"]
    load = lambda n: [json.loads(line) for line in (ROOT / "eval" / "law" / f"{n}.jsonl").read_text().splitlines() if line.strip()]
    for k, name in enumerate(sets[2:], start=2):
        items = load(name)
        assert sum(i["expect"]["abstain"] for i in items) >= 12 and sum(not i["expect"]["abstain"] for i in items) >= 12, name
        earlier = {i["question"].lower() for n in sets[:k] for i in load(n)}
        assert not any(i["question"].lower() in earlier for i in items), name


def test_the_thresholds_say_they_await_sign_off():
    t = json.loads((ROOT / "eval" / "law" / "thresholds.json").read_text())
    assert "proposed" in t["status"] and set(t["bars"]) == {"golden", "heldout", "heldout2", "heldout3", "heldout4", "heldout5", "heldout6", "heldout7"}
    assert all(b["t06_leaks_max"] == 0 for b in t["bars"].values())


def _result(item: str, run: int, correct: bool, abstain: bool = False) -> dict:
    return {"id": item, "run": run, "expect_abstain": abstain, "status": "abstained" if abstain == correct else "answered",
            "correct": correct, "leaked": [], "citation_precision": 1.0 if correct else 0.5, "verified_quotes": 1, "quotes": 1,
            "conflict_expected": False, "conflict_found": False, "multi_source": None, "total_s": 2.0, "first_paragraph_s": 1.0}


def test_repeated_runs_are_pooled_and_name_the_items_that_changed():
    """EV14: one run of a small set moves by several items on sampling; N runs are scored together, with each run's scores,
    their range and the items right in some runs and wrong in others."""
    m = _runner()
    rs = [_result("a", 1, True), _result("a", 2, True), _result("a", 3, True),
          _result("b", 1, True), _result("b", 2, False), _result("b", 3, True),
          _result("c", 1, False, abstain=True), _result("c", 2, False, abstain=True), _result("c", 3, False, abstain=True)]
    pooled = m.summarise(rs) | m.repeated(rs, 3)
    assert pooled["answer_accuracy"] == "5/6" and pooled["abstention_accuracy"] == "0/3"
    assert pooled["runs"] == 3 and [p["answer_accuracy"] for p in pooled["per_run"]] == ["2/2", "1/2", "2/2"]
    assert pooled["spread"]["answer_accuracy"] == [0.5, 1.0]
    assert pooled["unstable_items"] == {"b": "2/3"} and pooled["never_correct"] == ["c"]
    assert m.check_bars("heldout", {**GOOD, "answer_accuracy": pooled["answer_accuracy"]}) == ["answer_accuracy 0.833 (below 0.85)"]


def test_a_gate_is_refused_with_fewer_runs_than_the_thresholds_ask_for(monkeypatch, tmp_path):
    m = _runner()
    t = json.loads(m.THRESHOLDS.read_text())
    frozen = tmp_path / "thresholds.json"
    frozen.write_text(json.dumps({**t, "gate_runs": 3}))
    monkeypatch.setattr(m, "THRESHOLDS", frozen)
    args = type("A", (), {"repeat": 1, "allow_dirty": True})()
    assert any("at least 3 times" in p for p in m.gate_preconditions(args))
    args.repeat = 3
    assert not any("at least" in p for p in m.gate_preconditions(args))
