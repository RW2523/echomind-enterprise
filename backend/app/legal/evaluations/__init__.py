"""Evaluation runs (S2.8, OPS-02; decision D-044).

Each evaluation script writes `eval/law/reports/<YYYYMMDD-HHMMSS>-<suite>-<set>.json` (the research
suite has no suite name: `<time>-<set>.json`). Importing one records what was measured and on what:
the suite and set, the commit, the models, the SHA-256 of the dataset file it read, the summary, and
the report's name and hash. The items themselves (questions, answers, quotes) stay in the report.

    cd backend && python -m app.legal.evaluations import ../eval/law/reports/<report>.json [...]

A report is imported once (by its hash). Runs are never changed.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Optional

from ...core.db import get_conn
from ..policy import now_iso

ROOT = Path(__file__).resolve().parents[4]
_NAME = re.compile(r"^(\d{8}-\d{6})-(.+)\.json$")
_SUITES = ("research", "compare", "assist", "minutes", "transcription", "gate", "latency", "load", "injection", "lawyer", "drafting", "playbook", "redline", "citations")
_DATASETS = {"research": "eval/law/{set}.jsonl", "compare": "eval/law/compare/{set}.json", "assist": "eval/law/assist/{set}.json",
             "minutes": "eval/law/minutes/{set}.json", "transcription": "eval/law/transcription/scenarios.json",
             "drafting": "eval/law/drafting/{set}.json", "playbook": "eval/law/playbook/{set}.json",
             "citations": "eval/law/citations/{set}.json"}
_MODEL_KEYS = ("model", "embeddings", "reranker", "asr", "voice")
_COLS = ("id", "suite", "set_name", "commit_ref", "models_json", "dataset_sha256", "summary_json", "report", "report_sha256", "run_at", "imported_at")


def _named(path: Path, meta: dict) -> tuple[str, str]:
    m = _NAME.match(path.name)
    if not m:
        raise ValueError(f"{path.name} is not named like an evaluation report (<time>-<suite>-<set>.json)")
    rest = m.group(2)
    if rest in ("golden", "heldout"):
        return "research", rest
    suite, _, set_ = rest.partition("-")
    if suite not in _SUITES:
        raise ValueError(f"{path.name}: unknown suite {suite!r}")
    return suite, set_ or str(meta.get("set") or "")


def _dataset(suite: str, set_: str) -> Optional[str]:
    rel = _DATASETS.get(suite)
    p = ROOT / rel.format(set=set_) if rel else None
    return hashlib.sha256(p.read_bytes()).hexdigest() if p and p.is_file() else None


def _out(r) -> dict:
    d = dict(zip(_COLS, r))
    d["commit"], d["set"] = d.pop("commit_ref"), d.pop("set_name")
    d["models"], d["summary"] = json.loads(d.pop("models_json")), json.loads(d.pop("summary_json"))
    return d


def import_report(path) -> dict:
    """Record one report. Raises ValueError for a file that is not an evaluation report."""
    path = Path(path)
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
    except ValueError:
        raise ValueError(f"{path.name} is not JSON")
    if not isinstance(data, dict) or not isinstance(data.get("meta"), dict):
        raise ValueError(f"{path.name} has no evaluation metadata")
    meta = data["meta"]
    suite, set_ = _named(path, meta)
    summary = data["summary"] if isinstance(data.get("summary"), dict) else \
        {k: v for k, v in data.items() if k not in ("meta", "items", "scenarios", "planted", "invented", "found")}
    sha = hashlib.sha256(raw).hexdigest()
    with get_conn() as conn:
        found = conn.execute(f"SELECT {', '.join(_COLS)} FROM legal_evaluation_runs WHERE report_sha256=?", (sha,)).fetchone()
        if found:
            return _out(found)
        rid = str(uuid.uuid4())
        conn.execute("INSERT INTO legal_evaluation_runs (id, suite, set_name, commit_ref, models_json, dataset_sha256, summary_json, report, "
                     "report_sha256, run_at, imported_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                     (rid, suite, set_, meta.get("commit"), json.dumps({k: meta[k] for k in _MODEL_KEYS if meta.get(k)}),
                      _dataset(suite, set_), json.dumps(summary, ensure_ascii=False), path.name, sha, meta.get("time"), now_iso()))
        conn.commit()
        return _out(conn.execute(f"SELECT {', '.join(_COLS)} FROM legal_evaluation_runs WHERE id=?", (rid,)).fetchone())


def listing(suite: Optional[str] = None) -> list[dict]:
    q, args = f"SELECT {', '.join(_COLS)} FROM legal_evaluation_runs", []
    if suite:
        q += " WHERE suite=?"
        args.append(suite)
    with get_conn() as conn:
        return [_out(r) for r in conn.execute(q + " ORDER BY run_at DESC, imported_at DESC LIMIT 500", args).fetchall()]


def main(argv: Optional[list[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2 or argv[0] != "import":
        print("usage: python -m app.legal.evaluations import <report.json> [...]", file=sys.stderr)
        return 2
    n = 0
    for name in argv[1:]:
        try:
            run = import_report(name)
        except (OSError, ValueError) as e:
            print(f"skipped {name}: {e}", file=sys.stderr)
            continue
        n += 1
        print(f"{run['suite']} {run['set']} {run['commit'] or '-'} {run['report']}")
    print(f"{n} run(s) recorded")
    return 0 if n == len(argv) - 1 else 1
