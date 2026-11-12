#!/usr/bin/env python3
"""The demo's end-to-end chat set (2026-11-05, 165 questions): every answer-key question asked through the chat, as alice.

    python3 eval/law/run_chat_keys.py --base http://127.0.0.1:8131/api --label baseline        # a private copy of the demo
    python3 eval/law/run_chat_keys.py --rescore eval/law/reports/<time>-chat-<label>.json       # grade stored answers again
    python3 eval/law/run_chat_keys.py --keys 'eval/fixtures/synthetic/answer_key_*.jsonl' ...    # another set (2026-11-12)

Each item of eval/fixtures/demo/answer_key_*.jsonl is asked in a new conversation of its matter (a follow-up item asks its
turns in one conversation). An item passes when every `expect` group has one alternative in the answer and no `must_not`
value appears; an item with no `expect` (the documents do not say) passes when the answer declines or says it is not stated.
The answer read is the source_answer card's answer (or its absence); another card is read whole (a chronology, a comparison).

Numbers are compared as whole numbers (2026-11-07): "36,000" is no longer found inside "360,000", and "USD 36000" still
matches "USD 36,000". Run it only against a private copy (fictional data): it opens one conversation per item.

The report goes to eval/law/reports/<time>-chat-<label>.json and .md: per item, per kind and per matter.
"""
from __future__ import annotations

import argparse
import http.cookiejar
import json
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
KEYS = sorted((ROOT / "eval" / "fixtures" / "demo").glob("answer_key_*.jsonl"))
sys.path.insert(0, str(ROOT / "scripts" / "law"))


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-").replace("**", "")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s.lower())        # thousands separators: 36,000 -> 36000
    return " ".join(s.replace(",", " ").split())


def has(alt: str, text: str) -> bool:
    a, t = norm(alt), norm(text)
    if not a:
        return False
    pat = re.escape(a)
    if a[0].isdigit():
        pat = r"(?<![\d.])" + pat                           # 36000 is not inside 360000 or 1.36000
    if a[-1].isdigit():
        pat += r"(?![\d])"
    return re.search(pat, t) is not None


DECLINED = re.compile(r"\b(do(es)? not (state|specify|say|provide|mention)|not stated|no (information|provision|mention))")


def grade(step: dict, res: dict) -> bool:
    exp, bad = step.get("expect") or [], step.get("must_not") or []
    text = res["text"]
    if not exp:
        ok = res["state"] == "abstained" or bool(DECLINED.search(norm(text)))
        res["missing"] = []
    else:
        res["missing"] = [g for g in exp if not any(has(a, text) for a in g)]
        ok = not res["missing"]
    res["leaked"] = [b for b in bad if has(b, text)]
    return ok and not res["leaked"]


def items_of(only: list[str] | None) -> list[dict]:
    items = [json.loads(line) for k in KEYS for line in k.read_text().splitlines() if line.strip()]
    return [i for i in items if not only or i["matter"] in only or i["id"] in only]


def steps_of(it: dict) -> list[dict]:
    steps = it.get("turns") or [{"question": it["question"], "expect": it.get("expect")}]
    return [{**st, "must_not": st.get("must_not", it.get("must_not"))} for st in steps]


def ask_all(base: str, items: list[dict]) -> list[dict]:
    from law_client import _LoopbackSecurePolicy
    jar = http.cookiejar.CookieJar(policy=_LoopbackSecurePolicy())
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

    def call(method: str, path: str, body=None, timeout: int = 300) -> str:
        csrf = next((c.value for c in jar if c.name.endswith("csrf")), "")
        req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None, method=method,
                                     headers={"Content-Type": "application/json", "X-CSRF-Token": csrf})
        return op.open(req, timeout=timeout).read().decode()

    def ask(conv: str, question: str) -> dict:
        t0 = time.time()
        ev = [json.loads(x) for x in call("POST", f"/legal/counsel/conversations/{conv}/turns", {"text": question}).splitlines() if x.strip()]
        fin = next((e["data"] for e in ev if e["event"] == "ui.final"), {"components": []})
        sa = next((c["props"] for c in fin.get("components", []) if c["type"] == "source_answer"), None)
        text = (sa or {}).get("answer") or (sa or {}).get("absence") or fin.get("summary") or ""
        if not sa:                                          # a card answer: read all its words
            text = json.dumps(fin.get("components", []), ensure_ascii=False)
        return {"state": (sa or {}).get("state"), "text": text, "secs": round(time.time() - t0, 1),
                "cards": [c["type"] for c in fin.get("components", []) if c["type"] != "source_answer"],
                "cited": sorted({c["document_title"] for c in (sa or {}).get("citations", [])}), "notes": (sa or {}).get("notes", [])}

    call("POST", "/auth/demo/alice")
    matters = {m["number"]: m["id"] for m in json.loads(call("GET", "/legal/matters"))["matters"]}
    results = []
    for it in items:
        mid = matters.get(it["matter"])
        if not mid:
            results.append({"id": it["id"], "matter": it["matter"], "kind": it.get("kind"), "ok": False, "turns": [],
                            "error": "matter not visible"})
            continue
        conv = json.loads(call("POST", "/legal/counsel/conversations", {"matter_id": mid}))["conversation"]["id"]
        out, ok = [], True
        for st in steps_of(it):
            try:
                res = ask(conv, st["question"])
            except Exception as e:                          # noqa: BLE001 — recorded as a failed item
                res = {"state": "error", "text": str(e)[:200], "cited": [], "notes": [], "cards": [], "secs": 0}
            res["ok"] = grade(st, res)
            ok = ok and res["ok"]
            out.append({"question": st["question"], **res})
        results.append({"id": it["id"], "matter": it["matter"], "kind": it.get("kind"), "source": it.get("source"), "ok": ok,
                        "turns": out, "conv": conv})
        print(f'{"PASS" if ok else "FAIL"} {it["id"]:16} {it.get("kind", ""):11} {out[-1]["secs"]:5}s {out[-1]["state"]}', flush=True)
    return results


def rescore(results: list[dict]) -> list[dict]:
    by_id = {i["id"]: i for i in items_of(None)}
    for r in results:
        it = by_id.get(r["id"])
        if not it or not r.get("turns"):
            continue
        ok = True
        for st, res in zip(steps_of(it), r["turns"]):
            res["ok"] = grade(st, res)
            ok = ok and res["ok"]
        r["ok"] = ok
    return results


def summary(results: list[dict]) -> dict:
    def count(key):
        out: dict[str, list[int]] = {}
        for r in results:
            c = out.setdefault(r.get(key) or "?", [0, 0])
            c[0] += bool(r["ok"])
            c[1] += 1
        return {k: f"{v[0]}/{v[1]}" for k, v in sorted(out.items())}
    secs = sorted(t["secs"] for r in results for t in r.get("turns", []) if t.get("secs"))
    pct = (lambda q: secs[min(len(secs) - 1, int(q * len(secs)))] if secs else None)
    return {"passed": sum(bool(r["ok"]) for r in results), "items": len(results), "by_kind": count("kind"),
            "by_matter": count("matter"), "p50_s": pct(0.5), "p95_s": pct(0.95)}


def write(results: list[dict], label: str, extra: dict) -> Path:
    import subprocess
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = HERE / "reports" / f"{stamp}-chat-{label}"
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    s = summary(results)
    out.with_suffix(".json").write_text(json.dumps({"suite": "chat", "label": label, "commit": commit, **extra, "summary": s,
                                                    "items": results}, indent=1, ensure_ascii=False))
    lines = [f"# Chat set ({label})", "", f"Commit {commit}. Passed **{s['passed']}/{s['items']}**; answer time p50 {s['p50_s']} s, "
             f"p95 {s['p95_s']} s.", "", "| Kind | Passed |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in s["by_kind"].items()]
    lines += ["", "| Matter | Passed |", "|---|---|"] + [f"| {k} | {v} |" for k, v in s["by_matter"].items()]
    lines += ["", "Failed: " + (", ".join(r["id"] for r in results if not r["ok"]) or "none")]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8131/api", help="a private copy's API (loopback only)")
    ap.add_argument("--label", default="run")
    ap.add_argument("--only", default="", help="comma-separated matter numbers or item ids")
    ap.add_argument("--rescore", default="", help="grade the answers stored in an earlier report (or run_keys.py output) again")
    ap.add_argument("--keys", default="", help="other answer keys: comma-separated globs from the repository root "
                                                "(e.g. eval/fixtures/synthetic/answer_key_*.jsonl); default the demo's")
    args = ap.parse_args()
    if args.keys:
        KEYS[:] = sorted({p for g in args.keys.split(",") if g for p in ROOT.glob(g)})
        if not KEYS:
            print(f"no answer keys match {args.keys}", file=sys.stderr)
            return 2
    if args.rescore:
        raw = json.loads(Path(args.rescore).read_text())
        results = rescore(raw["items"] if isinstance(raw, dict) else raw)
        out = write(results, args.label, {"rescored_from": Path(args.rescore).name})
    else:
        if not re.match(r"^https?://(127\.0\.0\.1|localhost)(:\d+)?/", args.base):
            print("refusing: run the chat set against a local private copy only", file=sys.stderr)
            return 2
        results = ask_all(args.base, items_of([x for x in args.only.split(",") if x] or None))
        out = write(results, args.label, {"base": args.base, "keys": [str(k.relative_to(ROOT)) for k in KEYS]})
    s = summary(results)
    print(f"SUMMARY {s['passed']}/{s['items']} by kind {json.dumps(s['by_kind'])} -> {out.with_suffix('.md').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
