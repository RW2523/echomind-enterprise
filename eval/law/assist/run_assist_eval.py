#!/usr/bin/env python3
"""EchoMind Law Silent Assist evaluation (S2.5, LIVE-02; decision D-041).

Imports the fictional corpus (eval/law/corpus: the signed MSA, the board minutes and the rest) into
a throwaway matter. Each scripted call in the set (golden.json, the tuning set; heldout.json, never used to adjust the
packs) is then played through the real capture
socket: speakers are marked turn by turn and each turn's text comes from a scripted speech model,
so no audio is involved. The packs run as in a real meeting, with the real answer model on this
host checking subjects.

Reported, overall and per scenario:
- **recall:** planted issues raised (critical ones separately);
- **precision:** alerts that are planted or listed as acceptable;
- **false alerts per hour:** speech at 150 words a minute;
- **latency:** from the words being stored to the alert, p50 and max.

    cd backend && python3 ../eval/law/assist/run_assist_eval.py --set golden
    cd backend && python3 ../eval/law/assist/run_assist_eval.py --set heldout

The report goes to eval/law/reports/<time>-assist-<set>.json and .md. Nothing leaves this host.
"""
from __future__ import annotations

import json
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAW = HERE.parent
ROOT = LAW.parent.parent
sys.path.insert(0, str(LAW))


class Scripted:
    def __init__(self, lines):
        self.lines = list(lines)

    async def feed(self, pcm16, sample_rate):
        return [(self.lines.pop(0) + " ", 100)] if self.lines else []

    async def flush(self):
        return []

    def reset(self):
        pass


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default="http://localhost:8355/v1")
    ap.add_argument("--model", default="nvidia/Qwen3-30B-A3B-FP4")
    ap.add_argument("--embed", default="http://localhost:11435/api/embeddings")
    ap.add_argument("--set", choices=["golden", "heldout"], default="golden")
    args = ap.parse_args()
    from run_law_eval import setup_env
    data = setup_env(args)
    try:
        return run(args)
    finally:
        shutil.rmtree(data, ignore_errors=True)
        (data.parent / f"{data.name}.key").unlink(missing_ok=True)


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def run(args) -> int:
    import logging
    logging.disable(logging.INFO)
    from fastapi.testclient import TestClient
    from app.core.db import init_db
    init_db()
    import app.main as main_mod
    from app.legal.jobs import worker
    from app.legal.matters import repo
    from app.legal.meetings import capture
    from app.legal.orgs import default_org_id
    from conftest import csrf_headers, make_user, sign_in

    org = default_org_id()
    alice = make_user("alice", mfa=True, firm_roles=["attorney"])
    client = repo.create_client(org, "Voss-Lindqvist AB", "VOSS", alice["id"])
    matter = repo.create_matter(org, client["id"], "EVAL-4", "Harrowgate renewal (assist evaluation)", alice["id"])
    mid = matter["id"]
    c = TestClient(main_mod.app, base_url="https://testserver")
    sign_in(c, alice)
    manifest = json.loads((LAW / "corpus" / "manifest.json").read_text())
    ids = {}
    for d in manifest["documents"]:
        if d["matter"] == "forbidden":
            continue
        form = {"kind": d["kind"], "status": d["status"], "title": d["title"]}
        if d.get("version_of"):
            form["version_of"] = ids[d["version_of"]]
        r = c.post(f"/api/legal/matters/{mid}/documents/import", headers=csrf_headers(c),
                   files={"file": (Path(d["file"]).name, (LAW / d["file"]).read_bytes(), "text/markdown")}, data=form)
        ids.setdefault(d["key"], r.json()["document"]["id"])
        worker.run_pending()

    scenarios = json.loads((HERE / f"{args.set}.json").read_text())["scenarios"]
    rows, all_latency = [], []
    for sc in scenarios:
        base = f"/api/legal/matters/{mid}/meetings"
        m = c.post(base, headers=csrf_headers(c), json={"title": sc["id"], "kind": sc["kind"]}).json()["meeting"]
        base = f"{base}/{m['id']}"
        c.patch(f"{base}/assist", headers=csrf_headers(c), json={"pack": sc["pack"]})
        pids = {}
        for s in sc["speakers"]:
            p = c.post(f"{base}/participants", headers=csrf_headers(c), json=s).json()["participant"]
            c.post(f"{base}/participants/{p['id']}/consent", headers=csrf_headers(c), json={"method": "written", "statement_read": True})
            pids[s["name"]] = p["id"]
        capture.transcriber_factory = (lambda lines: (lambda: Scripted(lines)))([t["text"] for t in sc["turns"]])
        started = time.monotonic()
        with c.websocket_connect(f"wss://testserver{base}/capture", headers={"origin": "https://testserver"}) as ws:
            ws.receive_json()
            for t in sc["turns"]:
                ws.send_json({"type": "speaker", "participant_id": pids[t["speaker"]]})
                while ws.receive_json()["type"] != "speaker":
                    pass
                ws.send_bytes(b"\x01\x00" * 1600)
            ws.send_json({"type": "stop"})
            while ws.receive_json()["type"] != "stopped":
                pass
        seconds = time.monotonic() - started
        segs = c.get(f"{base}/transcript").json()["segments"]
        turn_of = {s["id"]: k for k, s in enumerate(segs)}
        alerts = c.get(f"{base}/alerts").json()["alerts"]
        # Consecutive turns by one speaker can form one segment, so an alert is placed on the turn that
        # holds the words it quotes (the first such turn at or after its segment's position).
        texts = [t["text"] for t in sc["turns"]]
        for a in alerts:
            seg_turn = turn_of.get(a["said"]["segment_id"]) or 0
            holding = [k for k, t in enumerate(texts) if a["said"]["quote"] and a["said"]["quote"] in t]
            a["turn"] = next((k for k in holding if k >= seg_turn), holding[0] if holding else seg_turn)
            all_latency.append((_parse(a["raised_at"]) - _parse(a["heard_at"])).total_seconds())

        def matches(a, p):
            return a["turn"] == p["turn"] and a["rule"] == p["rule"] and (p.get("verdict") is None or a["verdict"] == p["verdict"])
        found = [p for p in sc["planted"] if any(matches(a, p) for a in alerts)]
        good = [a for a in alerts if any(matches(a, p) for p in sc["planted"])]
        okay = [a for a in alerts if a not in good and any(matches(a, p) for p in sc.get("acceptable", []))]
        false = [a for a in alerts if a not in good and a not in okay]
        words = sum(len(t["text"].split()) for t in sc["turns"])
        rows.append({"id": sc["id"], "pack": sc["pack"], "planted": len(sc["planted"]), "found": len(found),
                     "critical_planted": sum(1 for p in sc["planted"] if p["severity"] == "critical"),
                     "critical_found": sum(1 for p in found if p["severity"] == "critical"),
                     "alerts": len(alerts), "acceptable": len(okay), "false": len(false), "minutes_spoken": round(words / 150, 2),
                     "missed": [f"turn {p['turn']} {p['rule']}" for p in sc["planted"] if p not in found],
                     "false_alerts": [f"turn {a['turn']} {a['rule']} {a['verdict']}: {a['said']['quote']}" for a in false],
                     "all": [f"turn {a['turn']} {a['rule']} {a['verdict']} ({a['method']}, {a['confidence']}): {a['title']} — said "
                             f"“{a['said']['quote']}”" + (f" — source “{(a['source'] or {}).get('quote', '')[:90]}”" if a.get("source") else "")
                             for a in alerts],
                     "seconds": round(seconds, 2)})
        print(f"{sc['id']}: found {len(found)}/{len(sc['planted'])}, false {len(false)}, alerts {len(alerts)}", flush=True)
    planted = sum(r["planted"] for r in rows)
    found = sum(r["found"] for r in rows)
    alerts = sum(r["alerts"] for r in rows)
    false = sum(r["false"] for r in rows)
    minutes = sum(r["minutes_spoken"] for r in rows)
    summary = {"scenarios": len(rows), "recall": round(found / planted, 3) if planted else None,
               "critical_recall": f"{sum(r['critical_found'] for r in rows)}/{sum(r['critical_planted'] for r in rows)}",
               "precision": round((alerts - false) / alerts, 3) if alerts else None, "alerts": alerts, "false_alerts": false,
               "false_alerts_per_hour": round(false / (minutes / 60), 1) if minutes else None,
               "latency_p50_s": round(statistics.median(all_latency), 2) if all_latency else None,
               "latency_max_s": round(max(all_latency), 2) if all_latency else None}
    out = LAW / "reports"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    meta = {"set": args.set, "commit": commit + ("+changes" if dirty else ""), "model": args.model, "time": stamp}
    (out / f"{stamp}-assist-{args.set}.json").write_text(json.dumps({"meta": meta, "summary": summary, "scenarios": rows}, indent=1, ensure_ascii=False))
    lines = [f"# EchoMind Law Silent Assist evaluation — {args.set}", "", f"- Commit: `{meta['commit']}`", f"- Model: `{args.model}`", f"- Run: {stamp}", "",
             "| Measure | Result |", "|---|---|"] + [f"| {k.replace('_', ' ')} | {v} |" for k, v in summary.items()]
    for r in rows:
        lines += ["", f"## {r['id']} ({r['pack']})", "", f"Found {r['found']}/{r['planted']} · false {r['false']} · acceptable {r['acceptable']}",
                  "", "Alerts:"] + [f"- {x}" for x in r["all"]] + (["", "Missed: " + "; ".join(r["missed"])] if r["missed"] else []) + \
            (["", "False: " + "; ".join(r["false_alerts"])] if r["false_alerts"] else [])
    (out / f"{stamp}-assist-{args.set}.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary))
    print(f"report: eval/law/reports/{stamp}-assist-{args.set}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
