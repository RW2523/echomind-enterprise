#!/usr/bin/env python3
"""EchoMind Law gates: every evaluation, once, on the code under test (S2.8, D-044; S3.9, D-053).

Runs each suite as its own script (each builds and deletes its own throwaway data folder) with the
real models on this host, then writes one gate summary next to the suite reports:

- research answers, golden and held-out (RAG-01; T05, T06);
- compare and conflicts, golden and held-out (RAG-02);
- Silent Assist, golden and held-out (LIVE-02);
- minutes, golden and held-out (MEET-01);
- transcription, scripted (LIVE-01; the word error rate on real audio needs the speech stack, 🔒);
- the prompt-injection probe (T08), with the defences on.

Held-out sets are run once on this version and never used to change it. Phase 3 (S3.9) runs drafting,
playbooks, redlines (T10), citations (T07), definitions, obligations, due diligence and governance; a
held-out report already made on the current version and held-out file is cited rather than run again.
Phase 3 then runs the security suite and the browser journeys, and keeps their counts as reports.

    cd backend && python3 ../eval/law/run_gate.py                     # Phase 2: everything (about 8 minutes on the DGX Spark)
    cd backend && python3 ../eval/law/run_gate.py --phase 3           # Phase 3: drafting to governance, then the security and browser suites
    cd backend && python3 ../eval/law/run_gate.py --phase 4           # Phase 4: summaries and the review panel, then the same suites
    cd backend && python3 ../eval/law/run_gate.py --only minutes,assist
    cd backend && python3 ../eval/law/run_gate.py --import-into ../.law-data   # also record the runs there

The summary is eval/law/reports/<time>-gate-phase<N>.json and .md. Nothing leaves this host.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BACKEND = ROOT / "backend"
REPORTS = HERE / "reports"
PY = sys.executable

STEPS = [
    ("research", "golden", [str(HERE / "run_law_eval.py"), "--set", "golden", "--gate"]),      # bars: eval/law/thresholds.json
    ("research", "heldout", [str(HERE / "run_law_eval.py"), "--set", "heldout", "--gate"]),
    ("research", "heldout2", [str(HERE / "run_law_eval.py"), "--set", "heldout2", "--gate"]),
    ("compare", "golden", [str(HERE / "run_compare_eval.py"), "--set", "golden"]),
    ("compare", "heldout", [str(HERE / "run_compare_eval.py"), "--set", "heldout"]),
    ("assist", "golden", [str(HERE / "assist" / "run_assist_eval.py"), "--set", "golden"]),
    ("assist", "heldout", [str(HERE / "assist" / "run_assist_eval.py"), "--set", "heldout"]),
    ("minutes", "golden", [str(HERE / "minutes" / "run_minutes_eval.py"), "--set", "golden"]),
    ("minutes", "heldout", [str(HERE / "minutes" / "run_minutes_eval.py"), "--set", "heldout"]),
    ("transcription", "script", [str(HERE / "transcription" / "run_transcription_eval.py"), "--asr", "script"]),
    ("injection", "probe", None),
]
# Phase 3 (S3.9): the document tools. Rules-only suites need no model; drafting, playbooks and the obligations
# model pass use the host's model.
PHASE3 = [
    ("drafting", "golden", [str(HERE / "drafting" / "run_drafting_eval.py"), "--set", "golden"]),
    ("drafting", "heldout", [str(HERE / "drafting" / "run_drafting_eval.py"), "--set", "heldout"]),
    ("playbook", "golden", [str(HERE / "playbook" / "run_playbook_eval.py"), "--set", "golden"]),
    ("playbook", "heldout", [str(HERE / "playbook" / "run_playbook_eval.py"), "--set", "heldout"]),
    ("redline", "t10", [str(HERE / "redline" / "run_redline_eval.py")]),
    ("citations", "golden", [str(HERE / "citations" / "run_citations_eval.py"), "--set", "golden"]),
    ("citations", "heldout", [str(HERE / "citations" / "run_citations_eval.py"), "--set", "heldout"]),
    ("definitions", "golden", [str(HERE / "definitions" / "run_definitions_eval.py"), "--set", "golden"]),
    ("definitions", "heldout", [str(HERE / "definitions" / "run_definitions_eval.py"), "--set", "heldout"]),
    ("obligations", "golden", [str(HERE / "obligations" / "run_obligations_eval.py"), "--set", "golden"]),
    ("obligations", "heldout", [str(HERE / "obligations" / "run_obligations_eval.py"), "--set", "heldout"]),
    ("diligence", "golden", [str(HERE / "diligence" / "run_diligence_eval.py"), "--set", "golden"]),
    ("diligence", "heldout", [str(HERE / "diligence" / "run_diligence_eval.py"), "--set", "heldout"]),
    ("governance", "golden", [str(HERE / "governance" / "run_governance_eval.py"), "--set", "golden"]),
    ("governance", "heldout", [str(HERE / "governance" / "run_governance_eval.py"), "--set", "heldout"]),
    ("security", "suite", None),         # scripts/law/security_suite.sh, every spec-test group
    ("browser", "e2e", None),            # the Law browser journeys (Playwright starts its own servers)
]
# Phase 4 (the Phase 4 gate): the optional features with a model. Intake (S4.1) and time entries (S4.3) are rules
# without a model, tested by the backend and security suites and the browser journeys.
PHASE4 = [
    ("summaries", "golden", [str(HERE / "summaries" / "run_summaries_eval.py"), "--set", "golden"]),
    ("summaries", "heldout", [str(HERE / "summaries" / "run_summaries_eval.py"), "--set", "heldout"]),
    ("panel", "golden", [str(HERE / "panel" / "run_panel_eval.py"), "--set", "golden"]),
    ("panel", "heldout", [str(HERE / "panel" / "run_panel_eval.py"), "--set", "heldout"]),
    ("security", "suite", None),
    ("browser", "e2e", None),
]
# Held-out sets run once per version (D-044, D-053): a held-out report made on the current rules or prompt version
# and the same held-out file is cited by the gate, not run again. A changed version or a new blind set runs.
VERSIONS = {"drafting": "drafting/engine.py", "playbook": "playbooks/check.py", "citations": "citations/check.py",
            "definitions": "definitions/check.py", "obligations": "obligations/extract.py", "diligence": "diligence/run.py",
            "governance": "diligence/run.py", "summaries": "summaries/summarize.py", "panel": "panel/run.py"}
# Latency figures each suite reports, gathered for the gate (seconds).
LATENCY = {"research": ["latency_first_paragraph_p50_s", "latency_first_paragraph_p95_s", "latency_total_p50_s", "latency_total_p95_s"],
           "assist": ["latency_p50_s", "latency_max_s"], "minutes": ["seconds"],
           "summaries": ["seconds_per_document"], "panel": ["seconds_per_panel"]}


def _pattern(suite: str, set_: str) -> str:
    return f"-{set_}.json" if suite == "research" else f"-{suite}-{set_}.json"


def _newest(suite: str, set_: str, since: set[str]) -> Path | None:
    end = _pattern(suite, set_)
    fresh = [p for p in REPORTS.glob("*.json") if p.name not in since and p.name.endswith(end)
             and (suite != "research" or p.name.count("-") == 2)]
    return max(fresh, key=lambda p: p.name) if fresh else None


def _version(suite: str) -> str | None:
    m = re.search(r'^VERSION = "([^"]+)"', (BACKEND / "app" / "legal" / VERSIONS[suite]).read_text(), re.M)
    return m.group(1) if m else None


def _already_run(suite: str, set_: str) -> Path | None:
    """The newest held-out report on this version, if it ran the held-out file as it is now."""
    if set_ != "heldout" or suite not in VERSIONS:
        return None
    want = _version(suite)
    for p in sorted(REPORTS.glob(f"*-{suite}-heldout.json"), reverse=True):
        meta = json.loads(p.read_text()).get("meta", {})
        if meta.get("prompt_version") != want:
            continue
        commit = (meta.get("commit") or "").split("+")[0]
        same = bool(commit) and subprocess.run(["git", "diff", "--quiet", commit, "--", f"eval/law/{suite}/heldout.json"], cwd=ROOT).returncode == 0
        return p if same else None
    return None


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _injection(args) -> tuple[int, Path]:
    """The T08 probe writes per-answer rows; the gate keeps a report like the other suites."""
    rows_file = Path(tempfile.mkstemp(suffix=".json")[1])
    env = {**os.environ, "LLM_BASE_URL": args.llm, "LLM_MODEL": args.model}
    p = subprocess.run([PY, str(ROOT / "scripts" / "law" / "injection_probe.py"), "--trials", str(args.trials), "--json", str(rows_file)],
                       cwd=BACKEND, env=env, capture_output=True, text=True)
    rows = json.loads(rows_file.read_text() or "[]") if p.returncode == 0 else []
    rows_file.unlink(missing_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    obeyed = sum(r["obeyed"] for r in rows)
    out = REPORTS / f"{stamp}-injection-probe.json"
    out.write_text(json.dumps({"meta": {"set": "probe", "commit": _commit(), "model": args.model, "time": stamp, "trials": args.trials},
                               "summary": {"answers": len(rows), "obeyed": obeyed, "obeyed_not_flagged": sum(r["obeyed"] and not r["flagged"] for r in rows)},
                               "items": rows}, indent=1, ensure_ascii=False))
    return p.returncode, out


def _command(suite: str, args) -> tuple[int, Path]:
    """The security suite and the browser journeys: run as they are, kept as a report with their counts."""
    if suite == "security":
        p = subprocess.run(["bash", str(ROOT / "scripts" / "law" / "security_suite.sh")], cwd=ROOT, capture_output=True, text=True)
        rows = [ln for ln in p.stdout.splitlines()[1:] if ln.strip()]
        summary = {"groups": len(rows), "failed_groups": sum(("failed" in ln or "error" in ln or "passed" not in ln) for ln in rows),
                   "tests_passed": sum(int(m) for ln in rows for m in re.findall(r"(\d+) passed", ln))}
        items = rows
    else:
        p = subprocess.run(["npx", "playwright", "test"], cwd=ROOT / "frontend", capture_output=True, text=True)
        tail = p.stdout.splitlines()[-40:]
        n = lambda word: sum(int(m) for ln in tail for m in re.findall(rf"^\s*(\d+) {word}\b", ln))
        summary = {"passed": n("passed"), "failed": n("failed"), "flaky": n("flaky"), "skipped": n("skipped")}
        items = [ln for ln in p.stdout.splitlines() if re.search(r"(✓|✘|ok|x)\s+\d+ ", ln)][-60:]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = REPORTS / f"{stamp}-{suite}-{'suite' if suite == 'security' else 'e2e'}.json"
    out.write_text(json.dumps({"meta": {"set": "suite" if suite == "security" else "e2e", "commit": _commit(), "time": stamp},
                               "summary": summary, "items": items}, indent=1, ensure_ascii=False))
    return p.returncode, out


def _commit() -> str:
    dirty = _git("status", "--porcelain", "--", ".", ":!eval/law/reports", ":!docs/echomind-law/design/screens")   # reports and journey screenshots are not code
    return _git("rev-parse", "--short", "HEAD") + ("+changes" if dirty else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", type=int, choices=[2, 3, 4], default=2,
                    help="2: research to minutes (S2.8); 3: the document tools (S3.9); 4: summaries and the review panel (Phase 4 gate)")
    ap.add_argument("--only", default="", help="comma-separated suites (phase 2: research, compare, assist, minutes, transcription, injection; "
                                                 "phase 3: drafting, playbook, redline, citations, definitions, obligations, diligence, governance, security, browser)")
    ap.add_argument("--llm", default="http://localhost:8355/v1")
    ap.add_argument("--model", default="nvidia/Qwen3-30B-A3B-FP4")
    ap.add_argument("--trials", type=int, default=3, help="injection probe trials per question")
    ap.add_argument("--import-into", help="a Law data folder to record the runs in (python -m app.legal.evaluations import)")
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    REPORTS.mkdir(exist_ok=True)
    started, results = time.strftime("%Y%m%d-%H%M%S"), []
    for suite, set_, cmd in {3: PHASE3, 4: PHASE4}.get(args.phase, STEPS):
        if only and suite not in only:
            continue
        before = {p.name for p in REPORTS.glob("*.json")}
        t0 = time.monotonic()
        print(f"… {suite} {set_}", flush=True)
        reused = _already_run(suite, set_)
        if reused:
            code, report = 0, reused
        elif cmd is None and suite in ("security", "browser"):
            code, report = _command(suite, args)
        elif cmd is None:
            code, report = _injection(args)
        else:
            p = subprocess.run([PY, *cmd], cwd=BACKEND, capture_output=True, text=True)
            code, report = p.returncode, _newest(suite, set_, before)
            if code != 0:
                sys.stderr.write(p.stderr[-2000:])
        seconds = round(time.monotonic() - t0, 1)
        summary = {}
        if report:
            data = json.loads(report.read_text())
            summary = data.get("summary") or {k: v for k, v in data.items() if k in ("alignment", "explanations", "scan")}
        results.append({"suite": suite, "set": set_, "ok": code == 0 and report is not None, "seconds": seconds,
                        "report": report.name if report else None, "summary": summary, "reused": bool(reused)})
        print(f"  {'ok' if results[-1]['ok'] else 'FAILED'} in {seconds}s → {results[-1]['report']}"
              + (" (held-out already run on this version; cited, not run again)" if reused else ""), flush=True)
    latency = {f"{r['suite']}_{r['set']}": {k: r["summary"][k] for k in LATENCY.get(r["suite"], []) if k in r["summary"]}
               for r in results if LATENCY.get(r["suite"])}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    meta = {"set": f"phase{args.phase}", "commit": _commit(), "model": args.model, "time": stamp, "started": started}
    gate = {"meta": meta, "summary": {"suites": len(results), "failed": sum(not r["ok"] for r in results), "latency_s": latency},
            "runs": results}
    out = REPORTS / f"{stamp}-gate-phase{args.phase}.json"
    out.write_text(json.dumps(gate, indent=1, ensure_ascii=False))
    md = [f"# EchoMind Law — Phase {args.phase} gate run", "", f"- Commit: `{meta['commit']}`", f"- Model: `{args.model}`", f"- Run: {started} → {stamp}", "",
          "| Suite | Set | Result | Time (s) | Report |", "|---|---|---|---|---|"]
    md += [f"| {r['suite']} | {r['set']} | {'ok' if r['ok'] else '**failed**'}{' (cited: already run on this version)' if r['reused'] else ''} "
           f"| {r['seconds']} | {r['report'] or '—'} |" for r in results]
    md += ["", "## Latency (seconds, from the suite reports)", ""] + [f"- {k}: {v}" for k, v in latency.items() if v]
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"gate: eval/law/reports/{out.name}")
    if args.import_into:
        names = [str(REPORTS / r["report"]) for r in results if r["report"]] + [str(out)]
        env = {**os.environ, "ECHOMIND_DATA_DIR": str(Path(args.import_into).resolve()), "ECHOMIND_EDITION": "law"}
        subprocess.run([PY, "-m", "app.legal.evaluations", "import", *names], cwd=BACKEND, env=env)
    return 0 if gate["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
