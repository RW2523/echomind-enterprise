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


def test_the_held_out_set_is_frozen_by_its_checksum():
    import hashlib
    frozen = json.loads((ROOT / "eval" / "law" / "thresholds.json").read_text())["heldout_sha256"]
    assert hashlib.sha256((ROOT / "eval" / "law" / "heldout.jsonl").read_bytes()).hexdigest() == frozen


def test_the_thresholds_say_they_await_sign_off():
    t = json.loads((ROOT / "eval" / "law" / "thresholds.json").read_text())
    assert "proposed" in t["status"] and set(t["bars"]) == {"golden", "heldout"}
    assert all(b["t06_leaks_max"] == 0 for b in t["bars"].values())
