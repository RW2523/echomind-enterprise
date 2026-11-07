#!/usr/bin/env python3
"""T12 load test (S2.8; decision D-044): concurrent research, search and live capture against a Law
DEVELOPMENT backend, reporting p50/p95 latency per operation, failures, database-busy errors and the
backend's memory.

The backend must be a loopback development backend seeded with the fictional demo world (the browser
tests' `scripts/law/e2e_backend.sh` does this on 127.0.0.1:8110). Demo sign-ins are read from its
`legal/demo_credentials.json`. Each virtual user is a different demo person (an authenticator code is
single-use).

What runs at once, for --seconds:
- **research:** alice and nakamura each ask M-101 questions in a loop (the real answer model);
- **search:** carol searches M-101 in a loop (the real embedding model);
- **capture:** alice records a consented meeting, streaming 100 ms audio frames in real time through
  the capture socket (consent gate, custody hashing, storage). Speech recognition is not on the
  development host, so the word error rate under load needs the speech stack (🔒).

The answer model and embeddings on this host are shared with production. A measured run therefore
needs an approved load-test window (🔒): pass --window-approved "<who, when>". Without it only
--smoke runs (one user per kind, at most 30 s), which is no heavier than the browser tests.

    python3 scripts/law/load_test.py --smoke
    python3 scripts/law/load_test.py --seconds 600 --research-users 2 --window-approved "owner, 2026-10-12 22:00"

The pilot stack (D-066): `--base http://localhost:3110 --origin http://localhost:3110 --container echomind-law-backend`
reads the demo sign-ins through `docker exec`, `--users harper:M-301,sofia:M-302,...` gives each researcher their own
matter, and `--stop-file PATH` ends every virtual user as soon as PATH exists (a watch on production writes it).

The demo, or a private copy of it (2026-11-07 review point 11): `--demo-signin` uses one-click demo sign-in (alice researches
and chats, carol searches), `--chat-users N` adds people asking the chat (time to the first text and to the checked answer),
`--ingest` adds a document upload every few seconds (time until it is searchable), `--council` runs Council reviews back to
back (time until each finishes or asks for facts), and the host is sampled every 2 s: GPU use (nvidia-smi), host CPU,
available memory, and the CPU and memory of each container in --sample-containers.

    python3 scripts/law/load_test.py --base http://127.0.0.1:8131 --origin http://127.0.0.1:8132 --demo-signin \
        --container echomind-legal-next-api --research-users 1 --chat-users 2 --ingest --council --seconds 180 \
        --sample-containers echomind-legal-next-api,echomind-trtllm --window-approved "<who, when>"

The report goes to eval/law/reports/<time>-load-<smoke|t12>.json and .md.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import hmac
import json
import re
import struct
import subprocess
import time
import urllib.parse
from pathlib import Path

import httpx
import websockets

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "eval" / "law" / "reports"
FIXTURE = ROOT / "eval" / "fixtures" / "law" / "Master Services Agreement (DEMO).md"
QUESTIONS = ["What is the cap on each party’s aggregate liability?", "When are invoices payable?",
             "How can either party terminate the agreement?", "Which law governs the agreement?"]
SEARCHES = ["cap on aggregate liability", "termination for convenience", "governing law", "payment terms invoice"]
CHAT = [("MAT-1042", "What is the overall cap on each party's liability?"), ("M-101", "When are invoices payable?"),
        ("MAT-1042", "What's the latest on this case?"), ("MAT-1042", "Give me a chronology of the delivery, from dispatch to acceptance"),
        ("M-101", "What does the agreement say about termination for convenience?"), ("MAT-1042", "Find conflicting dates or amounts")]
NOTE = ("FILE NOTE (fictional, load test {k})\n\nAttendance note of a call with the client about the renewal timetable. The client "
        "asked for the draft by Friday and confirmed the notice address. Reference LT-{k}-{tag}.\n")
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def totp(secret: str) -> str:
    key = base64.b32decode(secret.upper() + "=" * (-len(secret) % 8))
    d = hmac.new(key, struct.pack(">Q", int(time.time()) // 30), hashlib.sha1).digest()
    o = d[-1] & 0x0F
    return str((struct.unpack(">I", d[o:o + 4])[0] & 0x7FFFFFFF) % 1_000_000).zfill(6)


class Session:
    """A signed-in demo person. Cookies are kept by hand: the session cookies are Secure, and a
    loopback development backend is a secure context, as it is to browsers."""
    def __init__(self, base: str, origin: str):
        self.base, self.origin, self.cookies = base, origin, {}
        self.http = httpx.AsyncClient(base_url=base, timeout=240)

    def _headers(self, write: bool) -> dict:
        h = {"Cookie": "; ".join(f"{k}={v}" for k, v in self.cookies.items()), "Origin": self.origin}
        if write and "__Host-echomind_csrf" in self.cookies:
            h["X-CSRF-Token"] = self.cookies["__Host-echomind_csrf"]
        return h

    def _keep(self, r: httpx.Response) -> None:
        for raw in r.headers.get_list("set-cookie"):
            name, _, rest = raw.partition("=")
            self.cookies[name.strip()] = rest.split(";", 1)[0]

    async def request(self, method: str, path: str, **kw) -> httpx.Response:
        r = await self.http.request(method, path, headers=self._headers(method != "GET"), **kw)
        self._keep(r)
        return r

    async def demo(self, name: str) -> None:
        r = await self.request("POST", f"/api/auth/demo/{name}")
        if r.status_code != 200:
            raise SystemExit(f"one-click sign-in as {name} refused: {r.status_code}")

    async def sign_in(self, name: str, creds: dict) -> None:
        r = await self.request("POST", "/api/auth/login", json={"username": name, "password": creds["password"]})
        r.raise_for_status()
        for _ in range(2):
            r = await self.request("POST", "/api/auth/mfa/verify", json={"code": totp(creds["totp_secret"])})
            if r.status_code == 200:
                return
            await asyncio.sleep(31 - time.time() % 30)          # a code is single-use: wait for the next one
        raise SystemExit(f"could not sign {name} in: {r.status_code}")


class Stats:
    def __init__(self):
        self.lat: dict[str, list[float]] = {}
        self.fail: dict[str, int] = {}
        self.busy = 0
        self.rss_mb: list[float] = []
        self.last_error: dict[str, str] = {}
        self.samples: dict[str, list[float]] = {}

    def sample(self, name: str, value: float) -> None:
        self.samples.setdefault(name, []).append(value)

    def ok(self, op: str, seconds: float) -> None:
        self.lat.setdefault(op, []).append(seconds)

    def bad(self, op: str, detail: str = "") -> None:
        self.fail[op] = self.fail.get(op, 0) + 1
        self.last_error[op] = detail[-300:]
        if re.search(r"database is locked|SQLITE_BUSY", detail, re.I):
            self.busy += 1


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, int(round(q * (len(v) - 1))))], 2)


STOP: list = []                                                # a path whose existence ends the run (--stop-file)


def running(until: float) -> bool:
    return time.monotonic() < until and not (STOP and Path(STOP[0]).exists())


async def research(s: Session, mid: str, stats: Stats, until: float, n: int) -> None:
    i = n
    while running(until):
        q = QUESTIONS[i % len(QUESTIONS)]
        i += 1
        t0 = time.monotonic()
        try:
            r = await s.request("POST", f"/api/legal/matters/{mid}/research/query", json={"question": q})
            lines = [json.loads(x) for x in r.text.splitlines() if x.strip()]
            if r.status_code == 200 and lines and lines[-1].get("type") == "done":
                stats.ok("research_answer", time.monotonic() - t0)
            else:
                stats.bad("research_answer", r.text[-300:])
        except Exception as e:                                  # noqa: BLE001 — every failure counts
            stats.bad("research_answer", repr(e))


async def search(s: Session, mid: str, stats: Stats, until: float) -> None:
    i = 0
    while running(until):
        t0 = time.monotonic()
        try:
            r = await s.request("POST", f"/api/legal/matters/{mid}/documents/search", json={"query": SEARCHES[i % len(SEARCHES)]})
            if r.status_code == 200:
                stats.ok("search", time.monotonic() - t0)
            else:
                stats.bad("search", r.text)
        except Exception as e:                                  # noqa: BLE001
            stats.bad("search", repr(e))
        i += 1


async def capture(s: Session, mid: str, meeting: str, stats: Stats, until: float) -> None:
    url = s.base.replace("http", "ws", 1) + f"/api/legal/matters/{mid}/meetings/{meeting}/capture"
    frame = b"\x00\x00" * 1600                                 # 100 ms of silence at 16 kHz
    t0 = time.monotonic()
    try:
        async with websockets.connect(url, extra_headers={"Cookie": s._headers(False)["Cookie"]}, origin=s.origin, max_size=2**22) as ws:
            ready = json.loads(await asyncio.wait_for(ws.recv(), 300))   # a cold speech model loads first; a browser waits too
            if ready.get("type") != "ready":
                stats.bad("capture_open", json.dumps(ready))
                return
            stats.ok("capture_open", time.monotonic() - t0)
            sent, start = 0, time.monotonic()
            while running(until):
                await ws.send(frame)
                sent += 1
                lag = start + sent * 0.1 - time.monotonic()        # real time: one frame every 100 ms
                if lag > 0:
                    await asyncio.sleep(lag)
                else:
                    stats.ok("capture_frame_lag", -lag)
            t1 = time.monotonic()
            await ws.send(json.dumps({"type": "stop"}))
            while True:
                m = await asyncio.wait_for(ws.recv(), 320)          # audio still waiting is transcribed (LEGAL_CAPTURE_FINISH_SEC)
                if isinstance(m, str) and json.loads(m).get("type") == "stopped":
                    break
            stats.ok("capture_stop", time.monotonic() - t1)
            stats.ok("capture_frames", float(sent))
    except Exception as e:                                      # noqa: BLE001
        stats.bad("capture", repr(e))


async def chat(s: Session, ids: dict, stats: Stats, until: float, n: int) -> None:
    """One question per new conversation (no follow-up rewriting): time to the first streamed text and to the end."""
    i = n
    while running(until):
        number, q = CHAT[i % len(CHAT)]
        i += 1
        t0, first = time.monotonic(), None
        try:
            conv = (await s.request("POST", "/api/legal/counsel/conversations", json={"matter_id": ids[number]})).json()["conversation"]["id"]
            t0 = time.monotonic()
            end = None
            async with s.http.stream("POST", f"/api/legal/counsel/conversations/{conv}/turns", json={"text": q},
                                     headers=s._headers(True)) as r:
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    ev = json.loads(line).get("event")
                    if first is None and ev in ("answer.delta", "ui.partial", "ui.final"):
                        first = time.monotonic() - t0
                    if ev in ("turn.completed", "turn.abstained", "turn.failed", "scope_revoked"):
                        end = ev
            if end in ("turn.completed", "turn.abstained"):
                stats.ok("chat_answer", time.monotonic() - t0)
                if first is not None:
                    stats.ok("chat_first_text", first)
            else:
                stats.bad("chat_answer", str(end))
        except Exception as e:                                  # noqa: BLE001
            stats.bad("chat_answer", repr(e))


async def ingest(s: Session, mid: str, stats: Stats, until: float) -> None:
    """A fictional file note uploaded, then waited for until it is read and searchable."""
    k = 0
    while running(until):
        k += 1
        t0 = time.monotonic()
        try:
            body = NOTE.format(k=k, tag=hashlib.sha256(f"{time.time()}{k}".encode()).hexdigest()[:10]).encode()
            r = await s.request("POST", f"/api/legal/matters/{mid}/documents/import", data={"kind": "note", "status": "final",
                                "title": f"Load test file note {k} (fictional)"}, files={"file": (f"load-note-{k}.txt", body, "text/plain")})
            r.raise_for_status()
            doc = r.json()["document"]["id"]
            for _ in range(600):
                d = (await s.request("GET", f"/api/legal/matters/{mid}/documents/{doc}")).json()["document"]
                if d["current_version"]["index_status"] in ("indexed", "failed"):
                    break
                await asyncio.sleep(0.5)
            if d["current_version"]["index_status"] == "indexed":
                stats.ok("document_ready", time.monotonic() - t0)
            else:
                stats.bad("document_ready", d["current_version"]["index_status"])
        except Exception as e:                                  # noqa: BLE001
            stats.bad("document_ready", repr(e))
        await asyncio.sleep(5)


async def council(s: Session, mid: str, stats: Stats, until: float) -> None:
    """Council reviews one after another: time until each is done or asks for facts."""
    while running(until):
        t0 = time.monotonic()
        try:
            r = await s.request("POST", f"/api/legal/matters/{mid}/panel", json={
                "question": "Is the Supplier exposed to liability beyond the cap for the late delivery?", "specialist": "commercial"})
            r.raise_for_status()
            rid = r.json()["review"]["id"]
            status = "queued"
            while status in ("queued", "running") and time.monotonic() - t0 < 900:
                await asyncio.sleep(3)
                status = (await s.request("GET", f"/api/legal/matters/{mid}/panel/{rid}")).json()["review"]["status"]
            if status in ("done", "needs_facts"):
                stats.ok("council_review", time.monotonic() - t0)
            else:
                stats.bad("council_review", status)
        except Exception as e:                                  # noqa: BLE001
            stats.bad("council_review", repr(e))


def _cpu() -> tuple[int, int]:
    f = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    return sum(f), f[3] + f[4]                                 # total, idle + iowait


async def sampler(stats: Stats, until: float, containers: list[str]) -> None:
    """The host every 2 s: GPU use, CPU, available memory; each named container's CPU and memory every third sample."""
    last, k = _cpu(), 0
    while running(until):
        await asyncio.sleep(2)
        k += 1
        total, idle = _cpu()
        if total > last[0]:
            stats.sample("host_cpu_pct", round(100 * (1 - (idle - last[1]) / (total - last[0])), 1))
        last = (total, idle)
        mem = {x.split(":")[0]: int(x.split()[1]) for x in Path("/proc/meminfo").read_text().splitlines() if ":" in x}
        stats.sample("host_mem_available_gb", round(mem.get("MemAvailable", 0) / 2**20, 1))
        out = await asyncio.to_thread(subprocess.run, ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True)
        if out.stdout.strip().split("\n")[0].strip().isdigit():
            stats.sample("gpu_util_pct", float(out.stdout.strip().split("\n")[0]))
        if containers and k % 3 == 0:
            out = await asyncio.to_thread(subprocess.run, ["docker", "stats", "--no-stream", "--format", "{{.Name}} {{.CPUPerc}} {{.MemUsage}}",
                                                           *containers], capture_output=True, text=True)
            for line in out.stdout.splitlines():
                m = re.match(r"(\S+) ([\d.]+)% ([\d.]+)(MiB|GiB)", line)
                if m:
                    stats.sample(f"{m.group(1)}_cpu_pct", float(m.group(2)))
                    stats.sample(f"{m.group(1)}_mem_gb", round(float(m.group(3)) / (1024 if m.group(4) == "MiB" else 1), 2))


def backend_pid(port: int) -> int | None:
    out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True).stdout
    m = re.search(rf":{port}\s.*?pid=(\d+)", out)
    return int(m.group(1)) if m else None


async def container_memory(name: str, stats: Stats, until: float) -> None:
    while running(until):
        out = await asyncio.to_thread(subprocess.run, ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", name],
                                      capture_output=True, text=True)
        m = re.match(r"([\d.]+)(MiB|GiB)", out.stdout.strip())
        if m:
            stats.rss_mb.append(round(float(m.group(1)) * (1024 if m.group(2) == "GiB" else 1), 1))
        await asyncio.sleep(5)


async def memory(pid: int | None, stats: Stats, until: float) -> None:
    while pid and running(until):
        try:
            rss = next(int(x.split()[1]) for x in Path(f"/proc/{pid}/status").read_text().splitlines() if x.startswith("VmRSS"))
            stats.rss_mb.append(round(rss / 1024, 1))
        except (OSError, StopIteration):
            return
        await asyncio.sleep(1)


async def setup(alice: Session, number: str) -> tuple[str, str]:
    matters = (await alice.request("GET", "/api/legal/matters")).json()["matters"]
    mid = next(m["id"] for m in matters if m["number"] == number)
    docs = (await alice.request("GET", f"/api/legal/matters/{mid}/documents")).json()["documents"]
    if not any(d["title"].startswith("Master Services Agreement (DEMO)") for d in docs):
        r = await alice.request("POST", f"/api/legal/matters/{mid}/documents/import", data={"kind": "contract", "status": "executed"},
                                files={"file": (FIXTURE.name, FIXTURE.read_bytes(), "text/markdown")})
        r.raise_for_status()
        doc = r.json()["document"]["id"]
        for _ in range(180):
            d = (await alice.request("GET", f"/api/legal/matters/{mid}/documents/{doc}")).json()["document"]
            if d["current_version"]["index_status"] == "indexed":
                break
            await asyncio.sleep(1)
    m = (await alice.request("POST", f"/api/legal/matters/{mid}/meetings", json={"title": "Load test call (fictional)", "kind": "client_call"})).json()["meeting"]
    p = (await alice.request("POST", f"/api/legal/matters/{mid}/meetings/{m['id']}/participants", json={"name": "Helena Marsh", "role": "attorney"})).json()["participant"]
    await alice.request("POST", f"/api/legal/matters/{mid}/meetings/{m['id']}/participants/{p['id']}/consent",
                        json={"method": "written", "statement_read": True})
    return mid, m["id"]


async def run(args) -> int:
    host = urllib.parse.urlparse(args.base).hostname
    if host not in _LOOPBACK:
        raise SystemExit("the load test runs against a loopback backend only")
    if args.stop_file:
        STOP.append(args.stop_file)
    if not args.smoke and not args.window_approved:
        raise SystemExit("a measured run shares the host's models with production: pass --window-approved \"<who, when>\" (🔒), or --smoke")
    seconds = min(args.seconds, 30) if args.smoke else args.seconds
    research_users = 1 if args.smoke else args.research_users
    if args.demo_signin:
        creds = {}
    elif args.container:                                    # the pilot: sign-ins read through docker exec, never written out
        creds = json.loads(subprocess.run(["docker", "exec", args.container, "cat", "/data/legal/demo_credentials.json"],
                                          check=True, capture_output=True, text=True).stdout)
    else:
        creds = json.loads((Path(args.data) / "legal" / "demo_credentials.json").read_text())
    if args.demo_signin:                                    # one-click people: alice researches and chats, carol searches
        pairs = [(f"alice#{i}", args.matter) for i in range(research_users)]
    else:
        pairs = [tuple(x.split(":", 1)) for x in args.users.split(",") if x] if args.users else \
            [(n, args.matter) for n in ["alice", "nakamura"][:research_users]]   # the M-101 members who research (carol searches)
    research_users = len(pairs)
    sessions = {}
    chatters = [f"alice~{i}" for i in range(args.chat_users)]
    for name in dict.fromkeys([n for n, _ in pairs] + ["carol", "alice"] + chatters + (["alice!council"] if args.council else [])):
        sessions[name] = Session(args.base, args.origin)
        person = re.split(r"[#~!]", name)[0]
        await (sessions[name].demo(person) if args.demo_signin else sessions[name].sign_in(person, creds[person]))
    mid, meeting = await setup(sessions["alice"], args.matter)
    ids = {m["number"]: m["id"] for n in dict.fromkeys(n for n, _ in pairs)
           for m in (await sessions[n].request("GET", "/api/legal/matters")).json()["matters"]}
    stats = Stats()
    until = time.monotonic() + seconds
    watch = container_memory(args.container, stats, until) if args.container else \
        memory(backend_pid(urllib.parse.urlparse(args.base).port or 80), stats, until)
    tasks = [research(sessions[n], ids[num], stats, until, i) for i, (n, num) in enumerate(pairs)]
    tasks += [search(sessions["carol"], mid, stats, until), capture(sessions["alice"], mid, meeting, stats, until), watch]
    if chatters:
        every = {m["number"]: m["id"] for m in (await sessions[chatters[0]].request("GET", "/api/legal/matters")).json()["matters"]}
        tasks += [chat(sessions[c], every, stats, until, i) for i, c in enumerate(chatters)]
    if args.ingest:
        tasks.append(ingest(sessions["alice"], mid, stats, until))
    if args.council:
        tasks.append(council(sessions["alice!council"], mid, stats, until))
    tasks.append(sampler(stats, until, [c for c in args.sample_containers.split(",") if c]))
    started = time.strftime("%Y%m%d-%H%M%S")
    await asyncio.gather(*tasks)
    for s in sessions.values():
        await s.http.aclose()
    ops = {op: {"n": len(v), "p50_s": pct(v, .5), "p95_s": pct(v, .95), "max_s": round(max(v), 2)}
           for op, v in stats.lat.items() if op not in ("capture_frames",)}
    summary = {"seconds": seconds, "research_users": research_users, "users": [f"{n}:{m}" for n, m in pairs],
               "stopped_early": bool(STOP and Path(STOP[0]).exists()), "operations": ops, "failures": stats.fail, "last_error": stats.last_error,
               "database_busy": stats.busy, "capture_frames": int(sum(stats.lat.get("capture_frames", []))),
               "backend_rss_mb_max": max(stats.rss_mb) if stats.rss_mb else None,
               "backend_rss_mb_start": stats.rss_mb[0] if stats.rss_mb else None,
               "chat_users": args.chat_users, "ingest": args.ingest, "council": args.council,
               "system": {k: {"n": len(v), "p50": pct(v, .5), "p95": pct(v, .95), "max": max(v)} for k, v in stats.samples.items()}}
    kind = "smoke" if args.smoke else "t12"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    meta = {"set": kind, "commit": commit, "time": stamp, "started": started, "base": args.base, "window": args.window_approved}
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"{stamp}-load-{kind}.json"
    out.write_text(json.dumps({"meta": meta, "summary": summary}, indent=1))
    md = [f"# EchoMind Law load test — {kind}", "", f"- Commit: `{commit}`", f"- Backend: {args.base}", f"- Window: {args.window_approved or 'smoke (no window)'}",
          f"- Duration: {seconds} s; research users: {research_users}; search: 1; capture: 1", "",
          "| Operation | n | p50 (s) | p95 (s) | max (s) |", "|---|---|---|---|---|"]
    md += [f"| {op} | {v['n']} | {v['p50_s']} | {v['p95_s']} | {v['max_s']} |" for op, v in ops.items()]
    md += ["", f"- Failures: {stats.fail or 'none'}", f"- Database busy: {stats.busy}",
           f"- Capture frames streamed: {summary['capture_frames']}",
           f"- Backend memory (RSS): {summary['backend_rss_mb_start']} → max {summary['backend_rss_mb_max']} MB",
           f"- Chat users: {args.chat_users}; document uploads: {'yes' if args.ingest else 'no'}; Council: {'yes' if args.council else 'no'}"]
    if summary["system"]:
        md += ["", "| Host and containers (sampled) | n | p50 | p95 | max |", "|---|---|---|---|---|"]
        md += [f"| {k} | {v['n']} | {v['p50']} | {v['p95']} | {v['max']} |" for k, v in summary["system"].items()]
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(json.dumps(summary))
    print(f"report: eval/law/reports/{out.name}")
    return 0 if not stats.fail else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8110")
    ap.add_argument("--origin", default="http://localhost:5174", help="an origin the backend allows (its LAW_CORS_ORIGINS)")
    ap.add_argument("--data", default=str(ROOT / ".law-data-e2e"), help="the backend's data folder (for the demo sign-ins)")
    ap.add_argument("--matter", default="M-101")
    ap.add_argument("--seconds", type=int, default=300)
    ap.add_argument("--research-users", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--window-approved", default="", help="who approved the load-test window, and when (🔒)")
    ap.add_argument("--container", default="", help="read the demo sign-ins from this backend container (the pilot)")
    ap.add_argument("--users", default="", help="researchers and their matters: name:M-101,name:M-301,...")
    ap.add_argument("--stop-file", default="", help="end the run as soon as this file exists")
    ap.add_argument("--demo-signin", action="store_true", help="one-click demo sign-in (the demo or a private copy of it)")
    ap.add_argument("--chat-users", type=int, default=0, help="people asking the chat at once")
    ap.add_argument("--ingest", action="store_true", help="upload a fictional note every few seconds, timed until searchable")
    ap.add_argument("--council", action="store_true", help="run Council reviews back to back")
    ap.add_argument("--sample-containers", default="", help="containers whose CPU and memory are sampled (comma-separated)")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
