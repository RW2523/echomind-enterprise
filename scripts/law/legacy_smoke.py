#!/usr/bin/env python3
"""T14 legacy smoke test, API level (S1.12): the STANDARD edition still works end to end.

Starts scripts/law/legacy_backend.sh (standard edition, fresh data folder, host model servers) and
exercises what EchoMind Enterprise customers use: sign-in, documents, knowledge chat (plain and
streamed), the voice service's answer path, transcripts and their analysis, Boardroom sessions,
Document Studio generation and export, and the live-transcription socket. Everything is fictional.

    python3 scripts/law/legacy_smoke.py            # exit 0 only if every step passes

Not covered here, and said so in the output: Boardroom transcription and analysis, and live speech
recognition, need the speech service, which is not installed on this development host.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import zipfile

import httpx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE = "http://127.0.0.1:8130"
DOC = ("Northwind Logistics fleet policy (fictional). Section 3. Every delivery van is serviced every 12,000 kilometres "
       "or every six months, whichever comes first. Section 4. Drivers report tyre damage to the depot manager the same day.")
TRANSCRIPT = ("Depot manager: The van servicing interval stays at 12,000 kilometres. Driver lead: Agreed, and we will log "
              "tyre damage the same day. Depot manager: Action for the driver lead: send the tyre log every Friday.")
results: list[tuple[str, bool, str]] = []


def step(name: str):
    def wrap(fn):
        try:
            detail = fn() or ""
            results.append((name, True, detail))
        except Exception as e:                               # every step is reported; the run fails at the end
            results.append((name, False, f"{type(e).__name__}: {e}"[:300]))
        print(("PASS " if results[-1][1] else "FAIL ") + name + (f" — {results[-1][2]}" if results[-1][2] else ""), flush=True)
        return fn
    return wrap


def main() -> int:
    import socket
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 8130)) == 0:
            print("port 8130 is already in use by another program; nothing was started")
            return 2
    log = open(os.path.join(ROOT, ".law-data-legacy.log"), "w")
    server = subprocess.Popen([os.path.join(ROOT, "scripts/law/legacy_backend.sh")], stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(240):
            try:
                if httpx.get(f"{BASE}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        else:
            print("the standard-edition backend did not start; see .law-data-legacy.log")
            return 2
        run()
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)} of {len(results)} steps passed")
    print("Not covered on this host: Boardroom transcription and analysis, live speech recognition (speech service not installed).")
    return 1 if failed else 0


def run() -> None:
    c = httpx.Client(base_url=BASE, timeout=300)
    state: dict = {}

    @step("the standard edition answers and is not the Law edition")
    def _():
        assert c.get("/health").status_code == 200
        cfg = c.get("/api/auth/config").json()
        assert cfg["edition"] == "standard" and cfg["auth_enabled"] is True, cfg
        assert c.get("/api/legal/matters").status_code == 401           # sign-in required, as before
        return f"edition={cfg['edition']}"

    @step("sign in")
    def _():
        password = open(os.path.join(ROOT, ".law-data-legacy", "dev_admin_password")).read().strip()
        r = c.post("/api/auth/login", json={"username": "admin", "password": password})
        assert r.status_code == 200, r.text
        state["token"] = r.json()["token"]
        c.headers["Authorization"] = f"Bearer {state['token']}"
        assert c.get("/api/auth/me").status_code == 200
        assert c.get("/api/legal/matters").status_code == 404           # the Law edition's routes stay closed

    @step("documents: upload and list")
    def _():
        r = c.post("/api/docs/upload", files={"file": ("northwind-fleet-policy.txt", DOC.encode(), "text/plain")})
        assert r.status_code == 200, r.text
        for _ in range(60):
            docs = c.get("/api/docs/list").json()
            items = docs if isinstance(docs, list) else docs.get("documents") or docs.get("docs") or []
            if any("northwind" in json.dumps(d).lower() for d in items):
                return f"{len(items)} document(s)"
            time.sleep(1)
        raise AssertionError("the uploaded document never appeared")

    @step("knowledge chat: an answer from the document, with a citation")
    def _():
        chat = c.post("/api/chat/create", json={"title": "Smoke"}).json()
        state["chat"] = chat["chat_id"]
        r = c.post("/api/chat/ask", json={"chat_id": state["chat"], "message": "How often is a Northwind delivery van serviced?"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert "12,000" in body["answer"] or "12000" in body["answer"], body["answer"][:200]
        assert body.get("citations"), "no citations"
        return body["answer"][:90].replace("\n", " ")

    @step("knowledge chat, streamed")
    def _():
        text = ""
        with c.stream("POST", "/api/chat/ask-stream", json={"chat_id": state["chat"], "message": "When is tyre damage reported?"}) as r:
            assert r.status_code == 200
            for line in r.iter_lines():
                if line.strip():
                    ev = json.loads(line)
                    text += ev.get("text") or ev.get("delta") or (ev.get("answer") if ev.get("type") == "done" else "") or ""
        assert "same day" in text.lower(), text[:200]
        return text[:90].replace("\n", " ")

    @step("voice service answer path (ask-voice-stream)")
    def _():
        text = ""
        with c.stream("POST", "/api/chat/ask-voice-stream", json={"message": "How often is a delivery van serviced?"}) as r:
            assert r.status_code == 200, r.read()[:200]
            for line in r.iter_lines():
                if line.strip():
                    ev = json.loads(line)
                    text += ev.get("text") or ev.get("delta") or ""
        assert text.strip(), "empty voice answer"
        return text[:90].replace("\n", " ")

    @step("transcripts: store, read and analyse")
    def _():
        r = c.post("/api/transcribe/store", json={"raw_text": TRANSCRIPT, "name": "Depot sync (fictional)"})
        assert r.status_code == 200, r.text
        tid = r.json()["transcript_id"]
        assert c.get("/api/transcribe/list").status_code == 200
        assert TRANSCRIPT[:20] in json.dumps(c.get(f"/api/transcribe/transcripts/{tid}").json())
        a = c.get(f"/api/transcribe/transcripts/{tid}/analysis")
        assert a.status_code == 200 and "cards" in a.json(), a.text[:200]
        assert c.get("/api/transcribe/scenarios").status_code == 200
        return f"{len(a.json()['cards'])} analysis card(s)"

    @step("Boardroom: create, list, read and delete a session")
    def _():
        sid = c.post("/api/boardroom/sessions", json={"transcript_id": None}).json()["session_id"]
        assert c.get("/api/boardroom/sessions?limit=5").status_code == 200
        assert c.get(f"/api/boardroom/sessions/{sid}").status_code == 200
        assert c.delete(f"/api/boardroom/sessions/{sid}").status_code == 200

    @step("Document Studio: generate from a template and export PDF and PowerPoint")
    def _():
        templates = c.get("/api/docgen/templates").json()["templates"]
        assert templates
        tpl = templates[0]["id"]
        r = c.post("/api/docgen/generate", json={"template_id": tpl, "mode": "brief", "title": "Fleet servicing note",
                                                 "brief": "A one-page internal note on the fictional Northwind van servicing policy."})
        assert r.status_code == 200, r.text
        job = r.json()["job_id"]
        for _ in range(300):
            j = c.get(f"/api/docgen/jobs/{job}").json()
            if j.get("status") in ("done", "error"):
                break
            time.sleep(2)
        assert j.get("status") == "done", j.get("error") or j.get("status")
        pdf = c.get(f"/api/docgen/jobs/{job}/export?format=pdf")
        assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
        deck = c.get(f"/api/docgen/jobs/{job}/export?format=pptx")
        assert deck.status_code == 200 and zipfile.is_zipfile(io.BytesIO(deck.content))
        return f"template {tpl}: {len(pdf.content)} byte PDF, {len(deck.content)} byte PPTX"

    @step("live-transcription socket accepts a signed-in client (closed only in Law)")
    def _():
        from websockets.exceptions import ConnectionClosed, InvalidStatus
        from websockets.sync.client import connect
        try:
            with connect("ws://127.0.0.1:8130/api/transcribe/ws", additional_headers={"Cookie": f"echomind_token={state['token']}"},
                         open_timeout=10) as ws:
                try:
                    ws.recv(timeout=2)
                except TimeoutError:
                    pass
                except ConnectionClosed as e:
                    assert e.rcvd is None or e.rcvd.code != 4404, "closed as in the Law edition"
                    return f"accepted, then closed by the server ({e.rcvd.code if e.rcvd else 'no code'}: speech service absent)"
            return "accepted"
        except InvalidStatus as e:
            raise AssertionError(f"handshake refused: {e.response.status_code}")


if __name__ == "__main__":
    sys.exit(main())
