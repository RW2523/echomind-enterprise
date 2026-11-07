#!/usr/bin/env python3
"""Is the public demo ready? Run it right before a webinar or a client call (owner's request 2026-11-07).

    python3 scripts/law/demo_health.py                     # the public URL is read from the demo API container
    python3 scripts/law/demo_health.py https://<tunnel>    # or given

Checks, each one line PASS/FAIL: the demo containers are up and healthy; the model, embedding and voice services answer;
the public page and API answer through the tunnel; the one-click sign-in works; the chat answers (a case status, built
from the records without the model, and one question to the model); disk space. Fictional data only; the chat it
opens is deleted afterwards. Prints no credentials and no matter content.
"""
from __future__ import annotations

import http.cookiejar
import json
import shutil
import subprocess
import sys
import time
import urllib.request

PREFIX = "echomind-legal-demo"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
results: list[tuple[bool, str, str]] = []


def check(name: str):
    def wrap(fn):
        t = time.time()
        try:
            ok, info = fn()
        except Exception as e:                                             # noqa: BLE001 — report and go on
            ok, info = False, f"{type(e).__name__}: {str(e)[:160]}"
        results.append((ok, name, f"{info} [{time.time() - t:.1f} s]"))
        print(f"{'PASS' if ok else 'FAIL'}  {name:<14} {info}", flush=True)
        return fn
    return wrap


def docker(*args: str) -> str:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=30).stdout.strip()


def public_url() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1].rstrip("/")
    env = docker("inspect", f"{PREFIX}-api", "--format", "{{range .Config.Env}}{{println .}}{{end}}")
    return next((line.split("=", 1)[1] for line in env.splitlines() if line.startswith("PUBLIC_URL=")), "").rstrip("/")


def get(url: str, timeout: int = 15) -> int:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout) as r:
        return r.status


def main() -> int:
    base = public_url()

    @check("containers")
    def _():
        rows = docker("ps", "--filter", f"name={PREFIX}-", "--format", "{{.Names}}|{{.Status}}").splitlines()
        state = dict(r.split("|", 1) for r in rows if "|" in r)
        need = [f"{PREFIX}-{n}" for n in ("api", "web", "voice", "tunnel")]
        bad = [n for n in need if n not in state or "unhealthy" in state[n] or not state[n].startswith("Up")]
        return not bad, "all four up" + (", API healthy" if "healthy" in state.get(f"{PREFIX}-api", "") else "") if not bad else f"not ready: {bad}"

    @check("model")
    def _():
        return get("http://127.0.0.1:8355/v1/models") == 200, "answer model on :8355 answers"

    @check("voice")
    def _():
        return get("http://127.0.0.1:8112/health") == 200, "demo voice service answers"

    @check("public page")
    def _():
        return bool(base) and get(base + "/") == 200, base or "no public URL found"

    jar = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

    def call(method: str, path: str, body=None, timeout: int = 180):
        csrf = next((c.value for c in jar if c.name.endswith("csrf")), "")
        req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None, method=method,
                                     headers={"Content-Type": "application/json", "X-CSRF-Token": csrf, "User-Agent": UA, "Origin": base})
        with op.open(req, timeout=timeout) as r:
            raw = r.read().decode()
        return r.status, raw

    @check("sign-in")
    def _():
        status, _ = call("POST", "/api/auth/demo/alice")
        return status == 200, "one-click sign-in as alice"

    conv = {}

    @check("case status")
    def _():
        ms = {m["number"]: m["id"] for m in json.loads(call("GET", "/api/legal/matters")[1])["matters"]}
        conv["id"] = json.loads(call("POST", "/api/legal/counsel/conversations", {"matter_id": ms["MAT-1042"]})[1])["conversation"]["id"]
        ev = [json.loads(x) for x in call("POST", f"/api/legal/counsel/conversations/{conv['id']}/turns",
                                          {"text": "What's the latest on this case?"})[1].splitlines() if x.strip()]
        fin = next(e["data"] for e in ev if e["event"] == "ui.final")
        kinds = [c["type"] for c in fin["components"]]
        return kinds == ["case_status"], f"MAT-1042 card: {kinds}"

    @check("chat answer")
    def _():
        t = time.time()
        ev = [json.loads(x) for x in call("POST", f"/api/legal/counsel/conversations/{conv['id']}/turns",
                                          {"text": "What is the overall cap on each party's liability?"}, 300)[1].splitlines() if x.strip()]
        fin = next(e["data"] for e in ev if e["event"] == "ui.final")
        p = (fin["components"] or [{"props": {}}])[0]["props"]
        ok = "620,000" in (p.get("answer") or "")
        return ok, f"{p.get('state')}, cap {'found' if ok else 'NOT found'}, {len(p.get('citations', []))} citations, {time.time() - t:.0f} s"

    if conv.get("id"):
        try:
            call("DELETE", f"/api/legal/counsel/conversations/{conv['id']}")
        except Exception:                                                  # noqa: BLE001
            print("note: the health-check chat could not be deleted; delete it from the chat list", flush=True)

    @check("disk")
    def _():
        free = shutil.disk_usage("/").free / 2 ** 30
        return free > 20, f"{free:.0f} GB free"

    failed = [n for ok, n, _ in results if not ok]
    print("READY" if not failed else f"NOT READY: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
