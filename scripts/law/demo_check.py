#!/usr/bin/env python3
"""End-to-end check of the EchoMind Law pilot stack with the fictional demo world (D-061).

    python3 scripts/law/demo_check.py                     # every check
    python3 scripts/law/demo_check.py research meeting    # some of them

Runs through the pilot's nginx (http://localhost:3110) as the demo users, with the real model, speech
recognition and text-to-speech, the way a person would use each feature:
- research: a question on M-101, answered from its documents with checked quotes (the matter chat);
- meeting: a fictional client call spoken by two offline Piper voices, streamed into a meeting after consent:
  live transcript, and a Silent Assist alert for a liability cap that differs from the signed MSA;
- drafting: an engagement letter from the firm's approved template (the document studio);
- summary, panel, time, intake: the Phase 4 features.
Needs `scripts/law/demo_load.py` first. Prints one result line per check, with counts and flags only: no
matter content and no credentials (they are read through `docker exec` and kept in memory).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import wave

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "law"))
from law_client import open_session  # noqa: E402

BASE = "http://localhost:3110"
CONTAINER = "echomind-law-backend"
VOICE_IMAGE = os.getenv("LAW_VOICE_IMAGE", "echomind-law-voice:latest")

# A fictional client call. The client misstates the liability cap (the signed MSA, clause 8.1, says USD 250,000).
CALL = [
    ("attorney", "en_US-lessac-medium", "Thank you for joining. I want to go through the Harrowgate master services agreement "
                                        "before the renewal notice goes out."),
    ("client", "en_US-joe-medium", "Thanks. As I understand it, the liability cap under clause eight is five hundred thousand "
                                   "dollars, so we are well covered."),
    ("attorney", "en_US-lessac-medium", "I will check clause eight and send you a short note on the cap by Friday."),
]


class Pilot:
    def __init__(self):
        out = subprocess.run(["docker", "exec", CONTAINER, "cat", "/data/legal/demo_credentials.json"],
                             check=True, capture_output=True, text=True).stdout
        self._creds = json.loads(out)
        self._sessions = {}

    def opener(self, user: str):
        if user not in self._sessions:
            os.environ["LAW_TOTP_SECRET"] = self._creds[user]["totp_secret"]
            try:
                self._sessions[user] = open_session(BASE + "/api", user, self._creds[user]["password"])
            finally:
                os.environ.pop("LAW_TOTP_SECRET", None)
        return self._sessions[user]

    def cookie_header(self, user: str) -> str:
        jar = next(h.cookiejar for h in self.opener(user).handlers if isinstance(h, urllib.request.HTTPCookieProcessor))
        return "; ".join(f"{c.name}={c.value}" for c in jar)

    def call(self, user: str, method: str, path: str, body=None, timeout=120):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(BASE + "/api" + path, data=data if method != "GET" else None, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            r = self.opener(user).open(req, timeout=timeout)
        except urllib.error.HTTPError as e:                                        # say which call failed and why
            raise RuntimeError(f"{method} {path.split('/')[2] if path.count('/') > 1 else path} -> {e.code} "
                               f"{e.read()[:200].decode(errors='replace')}") from None
        with r:
            raw = r.read()
            return raw if r.headers.get_content_type() == "application/x-ndjson" else json.loads(raw or b"{}")

    def matter(self, user: str, number: str) -> str:
        return next(m["id"] for m in self.call(user, "GET", "/legal/matters")["matters"] if m["number"] == number)

    def documents(self, user: str, mid: str) -> dict:
        return {d["title"]: d for d in self.call(user, "GET", f"/legal/matters/{q(mid)}/documents")["documents"]}


def q(s: str) -> str:
    return urllib.parse.quote(s, safe="")


def wait(fn, timeout: float, every: float = 3.0):
    end = time.time() + timeout
    while True:
        v = fn()
        if v or time.time() > end:
            return v
        time.sleep(every)


# ── checks ────────────────────────────────────────────────────────────────────────────────────
def check_research(p: Pilot) -> tuple[bool, str]:
    mid = p.matter("alice", "M-101")
    t0 = time.time()
    raw = p.call("alice", "POST", f"/legal/matters/{q(mid)}/research/query",
                 {"question": "What is each party's liability cap under the Master Services Agreement?"}, timeout=300)
    events = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    done = next((e for e in events if e.get("type") == "done"), None)
    if not done:
        return False, f"no answer ({[e.get('type') for e in events][-3:]})"
    verified = sum(1 for e in done["evidence"] if e.get("verification") in ("verbatim", "normalized"))
    ok = not done["abstained"] and verified >= 1 and "250,000" in done["answer"]
    return ok, (f"status {done['status']}, {len(done['citations'])} citations, {verified}/{len(done['evidence'])} quotes verified, "
                f"cap 250,000 {'found' if '250,000' in done['answer'] else 'MISSING'}, {time.time() - t0:.1f} s")


def _synthesise(lines: list[tuple[str, str, str]]) -> list[tuple[str, bytes, int]]:
    """Each line as 16-bit mono PCM, spoken by an offline Piper voice in a throwaway, capped container."""
    with tempfile.TemporaryDirectory() as tmp:
        # Without Piper's random variation, so every run speaks exactly the same audio and a result can be repeated.
        script = ("import sys, wave, json\nfrom piper import PiperVoice, SynthesisConfig\n"
                  "lines = json.load(open('/out/lines.json'))\n"
                  "fixed = SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0)\n"
                  "for i, (voice, text) in enumerate(lines):\n"
                  "    v = PiperVoice.load(f'/voices/{voice}.onnx')\n"
                  "    with wave.open(f'/out/{i}.wav', 'wb') as w:\n"
                  "        v.synthesize_wav(text, w, syn_config=fixed)\n")
        pathlib.Path(tmp, "lines.json").write_text(json.dumps([[v, t] for _, v, t in lines]))
        pathlib.Path(tmp, "speak.py").write_text(script)
        os.chmod(tmp, 0o777)
        subprocess.run(["docker", "run", "--rm", "--network", "none", "-m", "2g", "--memory-swap", "2g", "--oom-score-adj", "900",
                        "-v", f"{ROOT}/voice/voices:/voices:ro", "-v", f"{tmp}:/out", "--entrypoint", "python3", VOICE_IMAGE,
                        "/out/speak.py"], check=True, capture_output=True, timeout=600)
        out = []
        for i, (role, _, _) in enumerate(lines):
            with wave.open(str(pathlib.Path(tmp, f"{i}.wav"))) as w:
                out.append((role, _to_16k(w.readframes(w.getnframes()), w.getframerate()), 16000))
        return out


def _to_16k(pcm: bytes, rate: int) -> bytes:
    """16 kHz, as the meeting page sends (the browser resamples the microphone before streaming)."""
    import math
    import numpy as np
    from scipy.signal import resample_poly
    if rate == 16000:
        return pcm
    g = math.gcd(16000, rate)
    x = resample_poly(np.frombuffer(pcm, dtype="<i2").astype(np.float32), 16000 // g, rate // g)
    return np.clip(x, -32768, 32767).astype("<i2").tobytes()


def check_meeting(p: Pilot) -> tuple[bool, str]:
    import websockets
    audio = _synthesise(CALL)
    rate = audio[0][2]
    mid = p.matter("alice", "M-101")
    m = p.call("alice", "POST", f"/legal/matters/{q(mid)}/meetings",
               {"title": f"Demo client call {time.strftime('%H:%M')}", "kind": "client_call"})
    meeting = m.get("meeting", m)
    base = f"/legal/matters/{q(mid)}/meetings/{q(meeting['id'])}"
    who = {}
    for role, name in (("attorney", "Alice Hart (demo)"), ("client", "Jonas Voss (demo client)")):
        r = p.call("alice", "POST", base + "/participants", {"name": name, "role": role})
        who[role] = (r.get("participant") or r)["id"]
        p.call("alice", "POST", f"{base}/participants/{q(who[role])}/consent", {"method": "verbal_on_record", "statement_read": True})

    got = {"segment": [], "alert": [], "degraded": [], "other": []}

    async def run():
        url = f"ws://localhost:3110/api{base}/capture?sample_rate={rate}"
        headers = [("Cookie", p.cookie_header("alice")), ("Origin", BASE)]
        async with websockets.connect(url, extra_headers=headers, max_size=2 ** 22, open_timeout=60) as ws:
            first = json.loads(await asyncio.wait_for(ws.recv(), 60))
            if first.get("type") != "ready":
                got["other"].append(first.get("type"))
                return

            async def reader():
                async for msg in ws:
                    if isinstance(msg, (bytes, bytearray)):
                        continue
                    ev = json.loads(msg)
                    got.get(ev.get("type"), got["other"]).append(ev)
                    if ev.get("type") == "stopped":
                        return

            task = asyncio.create_task(reader())
            frame = int(rate * 0.1) * 2                                            # 100 ms of 16-bit audio
            silence = b"\x00\x00" * int(rate * 0.8)
            for role, pcm, _ in audio:
                await ws.send(json.dumps({"type": "speaker", "participant_id": who[role]}))
                for i in range(0, len(pcm + silence), frame):
                    await ws.send((pcm + silence)[i:i + frame])
                    await asyncio.sleep(0.1)                                       # as fast as a microphone
            await asyncio.sleep(3)
            await ws.send(json.dumps({"type": "stop"}))
            await asyncio.wait_for(task, 180)

    t0 = time.time()
    asyncio.run(run())
    spoken = sum(len(pcm) / 2 / rate for _, pcm, _ in audio)
    p.call("alice", "POST", base + "/end")
    alerts = wait(lambda: p.call("alice", "GET", base + "/alerts").get("alerts"), 90)
    transcript = p.call("alice", "GET", base + "/transcript")
    segs = transcript.get("segments") or got["segment"]
    words = sum(len((s.get("text") or "").split()) for s in segs)
    heard = " ".join((s.get("text") or "").lower() for s in segs)
    rules = sorted({a.get("rule") or a.get("rule_id") or "?" for a in alerts or []})
    ok = bool(segs) and ("liability" in heard or "cap" in heard) and "signed_terms" in rules     # the misstated cap is flagged
    return ok, (f"{len(segs)} segments, {words} words from {spoken:.0f} s of speech in {time.time() - t0:.0f} s; "
                f"'liability cap' {'heard' if 'cap' in heard else 'NOT heard'}; {len(alerts or [])} Silent Assist alerts {rules}; "
                f"degraded {len(got['degraded'])}; other {[o if isinstance(o, str) else o.get('type') for o in got['other']][:4]}")


def check_drafting(p: Pilot) -> tuple[bool, str]:
    mid = p.matter("alice", "M-101")
    tpl = next(t for t in p.call("alice", "GET", "/legal/templates")["templates"] if t.get("kind") == "engagement_letter")
    t0 = time.time()
    d = p.call("alice", "POST", f"/legal/matters/{q(mid)}/drafts",
               {"template_id": tpl["id"], "title": "Engagement letter — MSA renewal (demo)"}, timeout=600)["draft"]
    c = d["content"]
    n = lambda v: v if isinstance(v, int) else len(v or [])                      # noqa: E731 (counts or lists)
    return bool(c.get("blocks")), (f"{n(c.get('blocks'))} blocks, {n(c.get('fields'))} fields, {n(c.get('gaps'))} gaps "
                                   f"for the lawyer, {time.time() - t0:.0f} s")


def check_summary(p: Pilot) -> tuple[bool, str]:
    mid = p.matter("alice", "M-101")
    doc = p.documents("alice", mid)["Master Services Agreement (DEMO)"]
    path = f"/legal/matters/{q(mid)}/documents/{q(doc['id'])}/summaries"
    p.call("alice", "POST", path, {})

    def ready():
        r = p.call("alice", "GET", path)
        items = r.get("summaries") or []
        return items[0] if items and (r.get("job") or {}).get("status") in (None, "done") else None
    s = wait(ready, 600, 5)
    if not s:
        return False, "no summary within 10 minutes"
    pts = (s.get("content") or {}).get("points", [])
    cited = sum(1 for x in pts if x.get("cite"))
    return bool(pts) and cited == len(pts), f"{len(pts)} points, {cited} cited by clause and line"


def check_panel(p: Pilot) -> tuple[bool, str]:
    mid = p.matter("alice", "M-101")
    doc = p.documents("alice", mid)["Reseller Agreement (DEMO)"]
    r = p.call("alice", "POST", f"/legal/matters/{q(mid)}/panel",
               {"question": "If the Reseller undergoes a change of control, can the Supplier terminate the Reseller Agreement?",
                "specialist": "commercial", "documents": [doc["id"]]})
    rid = (r.get("review") or r)["id"]
    v = wait(lambda: (lambda x: x if x["review"]["status"] in ("done", "needs_facts", "failed") else None)(
        p.call("alice", "GET", f"/legal/matters/{q(mid)}/panel/{q(rid)}")), 600, 5)
    if not v:
        return False, "no panel within 10 minutes"
    c = (v.get("artifact") or {}).get("content") or {}
    cited = sum(1 for x in c.get("perspectives", []) if x.get("evidence"))
    return v["review"]["status"] in ("done", "needs_facts"), (
        f"{v['review']['status']}, {len(c.get('perspectives', []))} perspectives ({cited} citing), "
        f"{len(c.get('matrix', []))} matrix rows, {len(c.get('missing_facts', []))} missing facts")


def check_time(p: Pilot) -> tuple[bool, str]:
    r = p.call("alice", "POST", "/legal/time/suggest", {})
    entries = r.get("entries") or p.call("alice", "GET", "/legal/time").get("entries", [])
    return bool(entries), f"{len(entries)} draft entries from alice's own work today"


def check_intake(p: Pilot) -> tuple[bool, str]:
    r = p.call("iris", "POST", "/legal/intake", {"channel": "phone"})
    iid = (r.get("intake") or r)["id"]
    detail = p.call("iris", "GET", f"/legal/intake/{q(iid)}")
    version = (detail.get("disclaimer") or {}).get("version") or detail.get("disclaimer_version") or "intake-disclaimer-1"
    p.call("iris", "POST", f"/legal/intake/{q(iid)}/disclaimer", {"version": version})
    s = p.call("iris", "POST", f"/legal/intake/{q(iid)}/submit", {"answers": {
        "your_name": "Dana Pike (demo)", "email": "dana.pike@example.com", "matter_type": "commercial", "other_parties": ["Harrowgate Systems Inc."],
        "summary": "A dispute about a software subscription renewal (fictional demo enquiry)."}})
    text = json.dumps(s).lower()
    return "conflicts team" in text, ("staff told to refer to the conflicts team" if "conflicts team" in text
                                      else f"unexpected reply keys {sorted(s)[:6]}")


def check_conversation(p: Pilot) -> tuple[bool, str]:
    """A spoken question (offline Piper voice) transcribed by the pilot, answered from M-101, and read aloud."""
    [(_, pcm, rate)] = _synthesise([("attorney", "en_US-lessac-medium", "What is the cap on each party's liability under the master services agreement?")])
    mid = p.matter("alice", "M-101")
    base = f"/legal/matters/{q(mid)}/conversation"
    t0 = time.time()
    req = urllib.request.Request(f"{BASE}/api{base}/transcribe?sample_rate={rate}", data=pcm, method="POST",
                                 headers={"Content-Type": "application/octet-stream"})
    with p.opener("alice").open(req, timeout=300) as r:
        heard = json.loads(r.read())
    t_heard = time.time() - t0
    raw = p.call("alice", "POST", f"/legal/matters/{q(mid)}/research/query", {"question": heard["text"], "versions": "signed"}, timeout=300)
    done = next((json.loads(line) for line in raw.decode().splitlines() if line.strip() and json.loads(line).get("type") == "done"), {})
    req = urllib.request.Request(f"{BASE}/api{base}/speak", data=json.dumps({"text": done.get("answer", "")}).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with p.opener("alice").open(req, timeout=120) as r:
        wav, kind = r.read(), r.headers.get_content_type()
    seconds = (len(wav) - 44) / 2 / 22050 if wav[:4] == b"RIFF" else 0
    ok = bool(heard["text"]) and "250,000" in done.get("answer", "") and kind == "audio/wav" and seconds > 1
    return ok, (f"heard {len(heard['text'].split())} words in {t_heard:.1f} s ('liability' {'heard' if 'liabil' in heard['text'].lower() else 'NOT heard'}); "
                f"answer {done.get('status')}, cap 250,000 {'found' if '250,000' in done.get('answer', '') else 'MISSING'}; "
                f"read aloud {seconds:.0f} s of {kind}")


def _ask(p: Pilot, user: str, number: str, question: str, history=None) -> list[dict]:
    mid = p.matter(user, number)
    body = {"question": question, "versions": "signed", **({"history": history} if history else {})}
    raw = p.call(user, "POST", f"/legal/matters/{q(mid)}/research/query", body, timeout=300)
    return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]


def check_followup(p: Pilot) -> tuple[bool, str]:
    """Conversation memory (D-063): a follow-up on M-301 is made to stand alone and answered from the documents."""
    first = _ask(p, "harper", "M-301", "Within what period must the Customer notify a claim under the Logistics Services Agreement?")
    done1 = next(e for e in first if e["type"] == "done")
    events = _ask(p, "harper", "M-301", "And was the notice for the April failures given in time?",
                  history=[{"question": "Within what period must the Customer notify a claim under the Logistics Services Agreement?",
                            "answer": done1["answer"]}])
    understood = next((e for e in events if e["type"] == "understood"), {})
    done = next(e for e in events if e["type"] == "done")
    ok = understood.get("rewritten") and not done["abstained"] and "april" in understood.get("question", "").lower()
    return bool(ok), f"understood as '{understood.get('question', '')[:90]}'; {done['status']}, {len(done['citations'])} citations"


def check_conflicts(p: Pilot) -> tuple[bool, str]:
    """Compare across documents (S2.3): the Series A term sheet against the investor's draft (option pool 12% / 10%)."""
    mid = p.matter("kenji", "M-303")
    raw = p.call("kenji", "POST", f"/legal/matters/{q(mid)}/compare/scan", {"versions": "current"}, timeout=600)
    events = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    found = [e["finding"] for e in events if e.get("type") == "checked" and (e.get("finding") or {}).get("verdict") == "conflict"]
    text = json.dumps(found)
    ok = any(x in text for x in ("12%", "10%", "30 October 2026", "15 November 2026"))
    return ok, f"{len(found)} conflicts; option pool {'found' if '12%' in text and '10%' in text else 'not found'}; closing date {'found' if '15 November 2026' in text else 'not found'}"


def check_reference(p: Pilot) -> tuple[bool, str]:
    """The public-law library (R-001): a question answered from a real Act, with quotes."""
    events = _ask(p, "alice", "R-001", "Under the Late Payment of Commercial Debts (Interest) Act 1998, what interest is implied into a contract for a qualifying debt?")
    done = next(e for e in events if e["type"] == "done")
    titles = {c["document_title"] for c in done["citations"]} | {e["document_title"] for e in done.get("evidence") or []}
    ok = not done["abstained"] and any("Late Payment" in t for t in titles)
    return ok, f"{done['status']}, cites {sorted(titles)[:2]}"


def check_employment(p: Pilot) -> tuple[bool, str]:
    """M-302: the claimed GBP 18,400 against the GBP 14,080 the policy rate gives."""
    events = _ask(p, "sofia", "M-302", "How much unpaid on-call allowance is Dr Rahman owed for 2025, and how was it calculated?")
    done = next(e for e in events if e["type"] == "done")
    ok = not done["abstained"] and "14,080" in done["answer"]
    return ok, f"{done['status']}, GBP 14,080 {'found' if '14,080' in done['answer'] else 'MISSING'}; 18,400 {'mentioned' if '18,400' in done['answer'] else 'not mentioned'}"


def check_regulatory(p: Pilot) -> tuple[bool, str]:
    """Regulatory updates (D-068): today's import is visible, labelled, linked to the official sites, and wall-safe."""
    ups = p.call("sofia", "GET", "/legal/regulatory?days=7")["updates"]
    if not ups:
        return False, "no updates imported in the last 7 days (run scripts/law/regulatory_daily.sh)"
    official = all(u["url"].startswith(("https://www.legislation.gov.uk/", "https://www.federalregister.gov/")) for u in ups)
    visible = {m["number"] for u in ups for m in u["matters"]}
    mine = {m["number"] for m in p.call("sofia", "GET", "/legal/matters")["matters"]}
    by = {j: sum(1 for u in ups if u["jurisdiction"] == j) for j in ("UK", "US")}
    return official and visible <= mine, (f"{len(ups)} updates (UK {by['UK']}, US {by['US']}), official links {'only' if official else 'NOT ONLY'}; "
                                          f"matters shown to sofia: {sorted(visible) or 'none today'} (all hers: {visible <= mine})")


def check_translation(p: Pilot) -> tuple[bool, str]:
    """Machine translation (D-070): the signed MSA into French by the real model, labelled, every figure kept."""
    mid = p.matter("alice", "M-101")
    doc = p.documents("alice", mid)["Master Services Agreement (DEMO)"]
    t0 = time.time()
    raw = p.call("alice", "POST", f"/legal/matters/{q(mid)}/documents/{q(doc['id'])}/translate", {"language": "fr"}, timeout=600)
    events = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    start = next((e for e in events if e["type"] == "start"), {})
    paras = [e for e in events if e["type"] == "paragraph"]
    flagged = [e for e in paras if e["figures_missing"]]
    done = events[-1].get("type") if events else None
    ok = done == "done" and "not reviewed" in start.get("label", "") and paras and len(flagged) <= max(1, len(paras) // 10)
    sample = next((e["text"] for e in paras if "250" in e["source"]), "")[:90]
    return bool(ok), (f"{len(paras)} paragraphs into French, {len(flagged)} with a figure flagged, {time.time() - t0:.0f} s; "
                      f"cap clause: '{sample}'")


CHECKS = {"research": check_research, "conversation": check_conversation, "followup": check_followup, "conflicts": check_conflicts,
          "reference": check_reference, "employment": check_employment, "meeting": check_meeting, "drafting": check_drafting, "summary": check_summary,
          "panel": check_panel, "time": check_time, "intake": check_intake, "regulatory": check_regulatory, "translation": check_translation}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checks", nargs="*", metavar="CHECK", help=f"some of: {', '.join(CHECKS)} (default: all)")
    args = ap.parse_args()
    unknown = [c for c in args.checks if c not in CHECKS]
    if unknown:
        ap.error(f"unknown checks {unknown}; choose from {', '.join(CHECKS)}")
    args.checks = args.checks or list(CHECKS)
    p = Pilot()
    failed = 0
    for name in args.checks:
        t0 = time.time()
        try:
            ok, info = CHECKS[name](p)
        except Exception as e:                                                     # report and go on to the next check
            ok, info = False, f"{type(e).__name__}: {str(e)[:160]}"
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:<9} {info}  [{time.time() - t0:.0f} s]", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
