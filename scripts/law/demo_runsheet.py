"""The demo run-sheet's chat steps (the product guide's 20-minute walk-through), each asked in a new conversation as alice;
prints PASS/FAIL with what was looked for and deletes the conversations afterwards. Fictional demo data only.
Usage: python3 -I scripts/law/demo_runsheet.py <base> [origin]   (base: https://… or http://127.0.0.1:<private copy>)
Part of scripts/law/release_gate.sh (2026-11-12)."""
import http.cookiejar, json, sys, urllib.request

BASE = sys.argv[1].rstrip("/")
ORIGIN = sys.argv[2] if len(sys.argv) > 2 else BASE
API = BASE if BASE.endswith("/api") else BASE + "/api"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from law_client import _LoopbackSecurePolicy  # noqa: E402
jar = http.cookiejar.CookieJar(policy=_LoopbackSecurePolicy()) if "127.0.0.1" in BASE else http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def call(m, p, body=None, timeout=300):
    csrf = next((c.value for c in jar if c.name.endswith("csrf")), "")
    r = urllib.request.Request(API + p, data=json.dumps(body).encode() if body is not None else None, method=m,
                               headers={"Content-Type": "application/json", "X-CSRF-Token": csrf, "User-Agent": UA, "Origin": ORIGIN})
    return op.open(r, timeout=timeout).read().decode()


call("POST", "/auth/demo/alice")
ms = {m["number"]: m["id"] for m in json.loads(call("GET", "/legal/matters"))["matters"]}
STEPS = [
    ("MAT-1042", "What's the latest on this case?", lambda c, t: "case_status" in c),
    ("MAT-1042", "Give me a chronology of the delivery, from dispatch to acceptance", lambda c, t: "timeline" in c),
    ("MAT-1042", "What is the overall cap on each party's liability?", lambda c, t: "620,000" in t),
    ("MAT-1042", "Find conflicting dates or amounts", lambda c, t: "evidence_comparison" in c or "potential inconsistenc" in t.lower()),
    ("MAT-1042", "Draft a letter before claim to Apex", lambda c, t: "guided_questions" in c),
    ("M-302", "At the policy rates, what would 83 nights and 12 weekend days come to? Does that match the GBP 18,400 claimed?",
     lambda c, t: "19,340" in t and "does not hold" in t),
]
made, bad = [], 0
for number, q, ok in STEPS:
    conv = json.loads(call("POST", "/legal/counsel/conversations", {"matter_id": ms[number]}))["conversation"]["id"]
    made.append(conv)
    ev = [json.loads(x) for x in call("POST", f"/legal/counsel/conversations/{conv}/turns", {"text": q}).splitlines() if x.strip()]
    fin = next((e["data"] for e in ev if e["event"] == "ui.final"), {"components": []})
    kinds = [c["type"] for c in fin.get("components", [])]
    text = json.dumps(fin, ensure_ascii=False)
    good = ok(kinds, text)
    bad += not good
    print(f"{'PASS' if good else 'FAIL'}  {number:9} {q[:60]:60} cards={kinds}", flush=True)
for conv in made:
    try:
        call("DELETE", f"/legal/counsel/conversations/{conv}")
    except Exception as e:  # noqa: BLE001
        print("could not delete", conv, e)
print("run-sheet:", "all PASS" if not bad else f"{bad} FAIL", f"({len(made)} conversations deleted)")
sys.exit(1 if bad else 0)
