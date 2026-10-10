#!/usr/bin/env python3
"""EchoMind Law minutes evaluation (S2.6, MEET-01; decision D-042).

A fictional meeting (golden.json, the tuning set; heldout.json, never used to adjust the minutes) is
held through the real capture socket: agenda items and speakers are marked turn by turn, and each
turn's text comes from a scripted speech model, so no audio is involved. The meeting is ended and its
minutes are drafted by the real answer model on this host, through the real endpoint.

Reported:
- **recall:** planted decisions, actions and unresolved points found (kind and key words);
- **owner and deadline** of each planted action;
- **verified:** items whose cited quotes are found in their lines;
- **invented specifics:** numbers and names in the minutes that were never said. This must be 0;
- **coverage:** batches and lines sent (the whole transcript), and the time taken.

    cd backend && python3 ../eval/law/minutes/run_minutes_eval.py --set golden

The report goes to eval/law/reports/<time>-minutes-<set>.json and .md. Nothing leaves this host.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAW = HERE.parent
ROOT = LAW.parent.parent
sys.path.insert(0, str(LAW))
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
_STOPNAMES = {"The", "A", "An", "I", "We", "It", "This", "That", "They", "Then", "Thank", "All", "Item", "Finally", "Given", "Good",
              "Agreed", "Could", "Let", "Operations", "Finance", "Shareholders", "Board", "Committee", "Minutes", "Noted", "Directors"}


class Scripted:
    def __init__(self, lines):
        self.lines = list(lines)

    async def feed(self, pcm16, sample_rate):
        return [(self.lines.pop(0) + " ", 100)] if self.lines else []

    async def flush(self):
        return []

    def reset(self):
        pass


def numbers(text: str) -> set[float]:
    """Figures in text, digits or words ("two point four million" → 2400000, "USD 400,000" → 400000)."""
    from app.legal.meetings import accuracy
    toks = accuracy.normalised(text.replace("%", " percent"))
    out, i = set(), 0
    while i < len(toks):
        t = toks[i]
        if re.fullmatch(r"\d+(?:\.\d+)?", t):
            value = float(t)
            if i + 2 < len(toks) and toks[i + 1] == "point" and toks[i + 2].isdigit():
                value = float(f"{t}.{toks[i + 2]}")
                i += 2
            scale = toks[i + 1] if i + 1 < len(toks) else ""
            if scale in _SCALE:
                value *= _SCALE[scale]
            elif i + 1 < len(toks) and re.fullmatch(r"\d+", toks[i + 1]) and float(toks[i + 1]) >= 1000 and value < 10:
                value = value * float(toks[i + 1])            # "2.4 million" read as 2.4 then 1000000
            out.add(round(value, 2))
        i += 1
    return out


def names(text: str) -> set[str]:
    found = set()
    for m in re.finditer(r"\b[A-Z][a-zäöåé]+(?:\s+[A-Z][a-zäöåé]+)*", text):
        words = [w for w in m.group(0).split() if w not in _STOPNAMES]
        if words:
            found.add(" ".join(words))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["golden", "heldout"], default="golden")
    ap.add_argument("--llm", default="http://localhost:8355/v1")
    ap.add_argument("--model", default="nvidia/Qwen3-30B-A3B-FP4")
    ap.add_argument("--embed", default="http://localhost:11435/api/embeddings")
    args = ap.parse_args()
    from run_law_eval import setup_env
    data = setup_env(args)
    try:
        return run(args)
    finally:
        shutil.rmtree(data, ignore_errors=True)
        (data.parent / f"{data.name}.key").unlink(missing_ok=True)


def run(args) -> int:
    import logging
    logging.disable(logging.INFO)
    from fastapi.testclient import TestClient
    from app.core.db import init_db
    init_db()
    import app.main as main_mod
    from app.legal.matters import repo
    from app.legal.meetings import capture
    from app.legal.orgs import default_org_id
    from conftest import csrf_headers, make_user, sign_in

    sc = json.loads((HERE / f"{args.set}.json").read_text())
    org = default_org_id()
    alice = make_user("alice", mfa=True, firm_roles=["attorney"])
    client = repo.create_client(org, "Voss-Lindqvist AB", "VOSS", alice["id"])
    matter = repo.create_matter(org, client["id"], "EVAL-5", "Minutes evaluation", alice["id"])
    c = TestClient(main_mod.app, base_url="https://testserver")
    sign_in(c, alice)
    base = f"/api/legal/matters/{matter['id']}/meetings"
    m = c.post(base, headers=csrf_headers(c), json=sc["meeting"]).json()["meeting"]
    base = f"{base}/{m['id']}"
    pids = {}
    for s in sc["speakers"]:
        p = c.post(f"{base}/participants", headers=csrf_headers(c), json=s).json()["participant"]
        c.post(f"{base}/participants/{p['id']}/consent", headers=csrf_headers(c), json={"method": "written", "statement_read": True})
        pids[s["name"]] = p["id"]
    items = [c.post(f"{base}/agenda", headers=csrf_headers(c), json={"title": t}).json()["item"]["id"] for t in sc["agenda"]]
    capture.transcriber_factory = (lambda lines: (lambda: Scripted(lines)))([t["text"] for t in sc["turns"]])
    with c.websocket_connect(f"wss://testserver{base}/capture", headers={"origin": "https://testserver"}) as ws:
        ws.receive_json()
        current = None
        for t in sc["turns"]:
            if t["item"] != current:
                ws.send_json({"type": "agenda", "item_id": items[t["item"] - 1]})
                while ws.receive_json()["type"] != "agenda":
                    pass
                current = t["item"]
            ws.send_json({"type": "speaker", "participant_id": pids[t["speaker"]]})
            while ws.receive_json()["type"] != "speaker":
                pass
            ws.send_bytes(b"\x01\x00" * 1600)
        ws.send_json({"type": "stop"})
        while ws.receive_json()["type"] != "stopped":
            pass
    c.post(f"{base}/end", headers=csrf_headers(c))
    started = time.monotonic()
    r = c.post(f"{base}/minutes/draft", headers=csrf_headers(c))
    seconds = time.monotonic() - started
    assert r.status_code == 200, r.text
    minutes = r.json()["minutes"]
    all_items = [dict(i, section=s["title"]) for s in minutes["content"]["sections"] for i in s["items"]]
    spoken = " ".join(t["text"] for t in sc["turns"])
    said_numbers, said_names = numbers(spoken), names(spoken) | {s["name"] for s in sc["speakers"]} | set(sc["agenda"]) | {sc["meeting"]["title"]}

    def has(item, words):
        text = " ".join(filter(None, [item["text"], item.get("owner") or "", item.get("deadline") or ""]))
        return all(re.search(w, text, re.I) for w in words)
    found, owners, deadlines = [], 0, 0
    for p in sc["planted"]:
        hit = next((i for i in all_items if i["kind"] == p["kind"] and has(i, p["must"])), None)
        found.append({"planted": p, "item": hit["text"] if hit else None})
        if hit and p["kind"] == "action":
            owners += bool(hit.get("owner")) and p["owner"].lower() in (hit.get("owner") or "").lower()
            deadlines += "deadline" not in p or bool(hit.get("deadline")) and p["deadline"].lower() in (hit.get("deadline") or "").lower()
    invented = []
    for i in all_items:
        text = " ".join(filter(None, [i["text"], i.get("owner") or "", i.get("deadline") or ""]))
        bad_numbers = [n for n in numbers(text) if n not in said_numbers]
        bad_names = [n for n in names(text) if not any(n.lower() in s.lower() or s.lower() in n.lower() for s in said_names)]
        if bad_numbers or bad_names:
            invented.append({"item": i["text"], "numbers": bad_numbers, "names": bad_names})
    actions = [p for p in sc["planted"] if p["kind"] == "action"]
    summary = {"items": len(all_items), "planted": len(sc["planted"]), "recall": round(sum(1 for f in found if f["item"]) / len(found), 3),
               "action_owners": f"{owners}/{len(actions)}", "action_deadlines": f"{deadlines}/{len(actions)}",
               "verified": round(sum(1 for i in all_items if i["status"] == "verified") / len(all_items), 3) if all_items else None,
               "invented_specifics": len(invented), "batches": minutes["inputs"]["batches"], "failed_batches": minutes["inputs"]["failed_batches"],
               "lines": minutes["inputs"]["segments"], "turns": len(sc["turns"]), "seconds": round(seconds, 1)}
    out = LAW / "reports"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    meta = {"set": args.set, "commit": commit + ("+changes" if dirty else ""), "model": args.model, "time": stamp}
    (out / f"{stamp}-minutes-{args.set}.json").write_text(json.dumps({"meta": meta, "summary": summary, "planted": found, "invented": invented,
                                                                     "items": all_items}, indent=1, ensure_ascii=False))
    lines = [f"# EchoMind Law minutes evaluation — {args.set}", "", f"- Commit: `{meta['commit']}`", f"- Model: `{args.model}`", f"- Run: {stamp}", "",
             "| Measure | Result |", "|---|---|"] + [f"| {k.replace('_', ' ')} | {v} |" for k, v in summary.items()]
    lines += ["", "| Planted | Found as |", "|---|---|"] + [f"| {f['planted']['kind']}: {' + '.join(f['planted']['must'])} | {f['item'] or '**missed**'} |" for f in found]
    lines += ["", "## Minutes", ""] + [f"- [{i['section']}] {i['kind']} ({i['status']}): {i['text']}" +
                                       (f" — owner {i.get('owner')}, by {i.get('deadline')}" if i["kind"] == "action" else "") for i in all_items]
    if invented:
        lines += ["", "## Invented specifics", ""] + [f"- {x['item']} — numbers {x['numbers']}, names {x['names']}" for x in invented]
    (out / f"{stamp}-minutes-{args.set}.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary))
    print(f"report: eval/law/reports/{stamp}-minutes-{args.set}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
