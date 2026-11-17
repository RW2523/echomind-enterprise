"""The demo run-sheet's chat steps, each asked in a new conversation as alice; prints PASS/FAIL with what was looked for,
the cards, the answer's state and the time, and deletes the conversations afterwards. Fictional demo data only.

Usage: python3 -I scripts/law/demo_runsheet.py <base> [origin] [--steps FILE] [--repeat N] [--only ID,ID]
  base: https://… or http://127.0.0.1:<private copy>
  --steps: a run-sheet file (eval/fixtures/webinar/runsheet.json: {"steps": [{"id", "matter", "question", "cards_any",
           "text_all", "text_any", "text_none", "max_secs"}]}). Without it, the webinar run-sheet is used when the instance
           has its matters, otherwise the original product-guide steps (MAT-1042, M-302).
  --repeat: ask every step N times (the webinar bar: 5 of 5), each in a new conversation.
Part of scripts/law/release_gate.sh (2026-11-12); steps from a file since the webinar (2026-11-16)."""
import http.cookiejar, json, os, statistics, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
args = [a for a in sys.argv[1:]]


def opt(name, default=None):
    if name in args:
        i = args.index(name)
        val = args[i + 1]
        del args[i:i + 2]
        return val
    return default


STEPS_FILE = opt("--steps")
REPEAT = int(opt("--repeat", "1"))
ONLY = {s for s in (opt("--only", "") or "").split(",") if s}
BASE = args[0].rstrip("/")
ORIGIN = args[1] if len(args) > 1 else BASE
API = BASE if BASE.endswith("/api") else BASE + "/api"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
sys.path.insert(0, HERE)
from law_client import _LoopbackSecurePolicy  # noqa: E402
jar = http.cookiejar.CookieJar(policy=_LoopbackSecurePolicy()) if "127.0.0.1" in BASE else http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

LEGACY = [
    {"id": "status", "matter": "MAT-1042", "question": "What's the latest on this case?", "cards_any": ["case_status"]},
    {"id": "chronology", "matter": "MAT-1042", "question": "Give me a chronology of the delivery, from dispatch to acceptance", "cards_any": ["timeline"]},
    {"id": "cap", "matter": "MAT-1042", "question": "What is the overall cap on each party's liability?", "text_all": ["620,000"]},
    {"id": "conflicts", "matter": "MAT-1042", "question": "Find conflicting dates or amounts",
     "cards_any": ["evidence_comparison"], "text_any_alt": ["potential inconsistenc"]},
    {"id": "draft", "matter": "MAT-1042", "question": "Draft a letter before claim to Apex", "cards_any": ["guided_questions"]},
    {"id": "policy-rates", "matter": "M-302",
     "question": "At the policy rates, what would 83 nights and 12 weekend days come to? Does that match the GBP 18,400 claimed?",
     "text_all": ["19,340", "does not hold"]},
]


def call(m, p, body=None, timeout=300):
    csrf = next((c.value for c in jar if c.name.endswith("csrf")), "")
    r = urllib.request.Request(API + p, data=json.dumps(body).encode() if body is not None else None, method=m,
                               headers={"Content-Type": "application/json", "X-CSRF-Token": csrf, "User-Agent": UA, "Origin": ORIGIN})
    return op.open(r, timeout=timeout).read().decode()


def judge(step, kinds, text):
    low = text.lower()
    why = []
    cards = step.get("cards_any")
    card_ok = not cards or any(k in kinds for k in cards) or any(a.lower() in low for a in step.get("text_any_alt", []))
    if not card_ok:
        why.append(f"no {'/'.join(cards)} card")
    for t in step.get("text_all", []):
        if t.lower() not in low:
            why.append(f"missing {t!r}")
    if step.get("text_any") and not any(t.lower() in low for t in step["text_any"]):
        why.append(f"none of {step['text_any']}")
    for t in step.get("text_none", []):
        if t.lower() in low:
            why.append(f"says {t!r}")
    return not why, why


call("POST", "/auth/demo/alice")
ms = {m["number"]: m["id"] for m in json.loads(call("GET", "/legal/matters"))["matters"]}
if STEPS_FILE:
    steps = json.load(open(STEPS_FILE))["steps"]
else:
    default = os.path.join(ROOT, "eval", "fixtures", "webinar", "runsheet.json")
    webinar = json.load(open(default))["steps"] if os.path.exists(default) else []
    steps = webinar if webinar and all(s["matter"] in ms for s in webinar) else LEGACY
steps = [s for s in steps if not ONLY or s["id"] in ONLY]

made, bad, report = [], 0, []
for step in steps:
    if step["matter"] not in ms:
        print(f"FAIL  {step['id']:<14} {step['matter']} is not on this instance", flush=True)
        bad += 1
        continue
    passes, secs = 0, []
    for n in range(REPEAT):
        conv = json.loads(call("POST", "/legal/counsel/conversations", {"matter_id": ms[step["matter"]]}))["conversation"]["id"]
        made.append(conv)
        t = time.time()
        ev = [json.loads(x) for x in call("POST", f"/legal/counsel/conversations/{conv}/turns", {"text": step["question"]}).splitlines() if x.strip()]
        secs.append(time.time() - t)
        fin = next((e["data"] for e in ev if e["event"] == "ui.final"), {"components": []})
        kinds = [c["type"] for c in fin.get("components", [])]
        state = next((c.get("props", {}).get("state") for c in fin.get("components", []) if c["type"] == "source_answer"), None)
        good, why = judge(step, kinds, json.dumps(fin, ensure_ascii=False))
        slow = step.get("max_secs") and secs[-1] > step["max_secs"]
        passes += good
        print(f"{'PASS' if good else 'FAIL'}  {step['id']:<14} {step['matter']:9} {secs[-1]:5.1f}s {('SLOW ' if slow else '')}"
              f"cards={kinds} state={state} {'; '.join(why)}", flush=True)
    bad += passes < REPEAT
    report.append((step["id"], passes, statistics.median(secs)))
for conv in made:
    try:
        call("DELETE", f"/legal/counsel/conversations/{conv}")
    except Exception as e:  # noqa: BLE001
        print("could not delete", conv, e)
if REPEAT > 1:
    for sid, passes, med in report:
        print(f"  {sid:<14} {passes}/{REPEAT}  median {med:.1f} s")
print("run-sheet:", "all PASS" if not bad else f"{bad} FAIL", f"({len(made)} conversations deleted)")
sys.exit(1 if bad else 0)
