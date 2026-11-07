#!/usr/bin/env python3
"""Live end-to-end check of EchoMind Legal's human-approval chains against a running backend.

    LAW_CHECK_BASE=http://127.0.0.1:8151 LAW_CHECK_CONTAINER=echomind-legal-wf-api python3 scripts/law/workflow_check.py

The unit tests and the browser journeys cover each step; this runs the whole chain on a live stack, as the seeded demo
people (backend/app/legal/cli.py DEMO_USERS), the way the firm would:
- enquiry: iris (front desk) takes an enquiry, it is screened for conflicts, iris may not decide (refused), conway
  (conflicts team) clears it, and alice converts it into a new client and matter;
- draft: alice drafts from the firm's approved engagement-letter template, fills its gaps, her own approval is refused
  (409), bob (a second lawyer on the matter) approves it, and it is filed to the matter as a Word document;
- export: alice asks to export the filed draft, she may not approve it herself, bob approves, and the released file
  carries the export's watermark;
- wall: morgan requests a wall screening bob from the matter, morgan may not approve it, nakamura does; bob then gets 404
  on the matter, its search and a citation he opened before; the wall is released (morgan asks, nakamura approves) and
  bob's access returns;
- emergency: fiona (firm admin) asks for break-glass access, may not approve it herself, morgan approves; it is read-only
  and it ends by itself, after which the managing partners' list shows it as ended (the shortest grant the firm allows is
  5 minutes, so this check waits for it; the other chains run meanwhile);
- audit: the chain verifies and every step above is in the trail with the right person and outcome, each refusal as
  "deny" whatever status code answered it ("error" is for faults).

LAW_CHECK_BASE / LAW_CHECK_CONTAINER choose the stack (no default: this check writes). Credentials are read through
`docker exec` and kept in memory. Each check prints one PASS/FAIL line with counts and status codes only: no matter
content and no credentials.

THIS CHANGES THE DATA IT RUNS ON. Each run adds an enquiry, a new client and matter (number WF-<time>) with bob and
alice's export right on it, one fictional document, a counsel conversation, a draft filed as a document, an export, a
released wall and an expired break-glass grant, and their audit events (the trail is append-only). Nothing existing is
changed and every grant it makes is over when it ends, so the data stays usable; run it on a private copy of the demo
(scripts/law/demo_up.sh with DEMO_PREFIX/DEMO_DATA_DIR), never on the public demo, which it refuses without --allow-demo.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from law_client import open_session  # noqa: E402

BASE = os.getenv("LAW_CHECK_BASE", "")
CONTAINER = os.getenv("LAW_CHECK_CONTAINER", "")
PUBLIC_DEMO_PREFIX = "echomind-legal-demo-"
RUN = datetime.now(timezone.utc).strftime("%m%d%H%M%S")

# Fictional only (the repository is public). The other party is a party on the demo's M-101, so the screen has a match
# for the conflicts team to look at.
CLIENT = f"Wrenfield Optics Ltd WF{RUN}"
ANSWERS = {"your_name": "Mara Lindqvist (fictional)", "organisation": CLIENT, "email": "mara.lindqvist@wrenfield.example",
           "phone": "", "matter_type": "commercial", "other_parties": ["Harrowgate Systems Inc."],
           "summary": "Renewal of a supply agreement (fictional workflow check).", "deadline": "", "heard_from": "referral"}
REQUEST = (f"ENGAGEMENT REQUEST (fictional, workflow check)\n\n{CLIENT} asks the firm to advise on the renewal of its supply "
           "agreement with Harrowgate Systems Inc.\n\nThe client contact is Mara Lindqvist, Head of Legal.\n\n"
           "The agreed fee is a fixed fee of USD 14,500 for the renewal negotiation.\n\n"
           "The work covers reviewing the supplier's redline and negotiating the service levels.\n")
QUESTION = "What fee was agreed for the renewal negotiation?"
CIT = re.compile(r"\bcit_[0-9a-f]{32}\b")
_ID = re.compile(r"(?<=/)(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|cit_[0-9a-f]{32}|[0-9a-f]{32})(?=/|$|\?)")


def q(s: str) -> str:
    return urllib.parse.quote(s, safe="")


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class Failed(Exception):
    pass


class Stack:
    """Signed-in sessions per demo person, plain calls, and the audit events each step should leave."""

    def __init__(self):
        out = subprocess.run(["docker", "exec", CONTAINER, "cat", "/data/legal/demo_credentials.json"],
                             check=True, capture_output=True, text=True).stdout
        self._creds = json.loads(out)
        self._sessions = {}
        self.expected: list[tuple[str, str, bool]] = []        # (action, actor, allowed)

    def opener(self, user: str):
        if user not in self._sessions:
            os.environ["LAW_TOTP_SECRET"] = self._creds[user]["totp_secret"]
            try:
                self._sessions[user] = open_session(BASE + "/api", user, self._creds[user]["password"])
            finally:
                os.environ.pop("LAW_TOTP_SECRET", None)
        return self._sessions[user]

    def send(self, user: str, method: str, path: str, body=None, *, data: bytes | None = None, ctype: str = "application/json",
             timeout: float = 120) -> tuple[int, bytes, str]:
        if body is not None:
            data = json.dumps(body).encode()
        if data is None and method != "GET":
            data = b""
        req = urllib.request.Request(BASE + "/api" + path, data=data, method=method, headers={"Content-Type": ctype})
        try:
            with self.opener(user).open(req, timeout=timeout) as r:
                return r.status, r.read(), r.headers.get_content_type()
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers.get_content_type() if e.headers else ""

    def call(self, user: str, method: str, path: str, body=None, *, ok=(200, 201, 202), audit: str | None = None, **kw) -> dict:
        code, raw, ctype = self.send(user, method, path, body, **kw)
        if code not in ok:
            try:
                detail = str(json.loads(raw).get("detail"))[:120]
            except Exception:
                detail = ""
            raise Failed(f"{user} {method} {_ID.sub('{id}', path)} -> {code} {detail}".strip())
        if audit:
            self.expect(audit, user, True)
        if ctype == "application/x-ndjson":
            return {"events": [json.loads(line) for line in raw.decode().splitlines() if line.strip()]}
        return json.loads(raw or b"{}") if ctype == "application/json" else {"bytes": raw}

    def refused(self, user: str, method: str, path: str, body=None, *, want: int, audit: str | None = None, **kw) -> int:
        """A step that must be refused with `want`; returns the code (and the audit trail should say who tried)."""
        code, _, _ = self.send(user, method, path, body, **kw)
        if code != want:
            raise Failed(f"{user} {method} {_ID.sub('{id}', path)} -> {code}, expected {want}")
        if audit:
            self.expect(audit, user, False)
        return code

    def expect(self, action: str, actor: str, allowed: bool) -> None:
        self.expected.append((action, actor, allowed))


def multipart(fields: dict, filename: str, content: bytes, mime: str = "text/plain") -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()]
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                 f"Content-Type: {mime}\r\n\r\n".encode() + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def wait(fn, timeout: float, every: float = 3.0):
    end = time.time() + timeout
    while True:
        v = fn()
        if v or time.time() > end:
            return v
        time.sleep(every)


# ── 1. Enquiry ───────────────────────────────────────────────────────────────────
def check_enquiry(s: Stack, w: dict) -> str:
    i = s.call("iris", "POST", "/legal/intake", {"channel": "phone"}, audit="intake.create")["intake"]
    base = f"/legal/intake/{q(i['id'])}"
    s.call("iris", "POST", f"{base}/disclaimer", {"version": i["disclaimer"]["version"]}, audit="intake.disclaimer")
    staff = s.call("iris", "POST", f"{base}/submit", {"answers": ANSWERS}, audit="intake.submit")
    screen = staff["intake"]["screening"]["result"]
    if screen != "possible_conflict" or "matches" in staff:
        raise Failed(f"screening {screen}, matches shown to the front desk: {'matches' in staff}")
    matches = len(s.call("conway", "GET", base)["matches"])
    if not matches:
        raise Failed("the conflicts team sees no match")
    decision = {"decision": "cleared", "note": "Different entity; checked the register (fictional)."}
    iris_code = s.refused("iris", "POST", f"{base}/decision", decision, want=403, audit="intake.decision")
    cleared = s.call("conway", "POST", f"{base}/decision", decision, audit="intake.decision")["intake"]["status"]
    if cleared != "cleared":
        raise Failed(f"after the decision the enquiry is {cleared}")
    number = f"WF-{RUN}"
    r = s.call("alice", "POST", f"{base}/convert", {"client_name": CLIENT, "client_code": f"WF{RUN}", "matter_number": number,
                                                    "matter_title": "Supply agreement renewal (workflow check)"}, audit="intake.convert")
    m = s.call("alice", "GET", f"/legal/matters/{q(r['matter']['id'])}", audit="matter.read")["matter"]
    if r["intake"]["status"] != "converted" or m["number"] != number or m["my_role"] != "lead":
        raise Failed(f"converted {r['intake']['status']}, role {m['my_role']}")
    parties = len(s.call("alice", "GET", f"/legal/matters/{q(m['id'])}/parties")["parties"])
    w.update(mid=m["id"], number=number)
    return (f"screen {screen} ({matches} matches for conway, none for iris), iris decision {iris_code}, conway cleared, "
            f"converted to a new client and matter ({parties} party)")


def setup_matter(s: Stack, w: dict) -> None:
    """bob joins as the second lawyer, alice may export, and one fictional document is indexed."""
    mid = q(w["mid"])
    s.call("alice", "POST", f"/legal/matters/{mid}/memberships", {"username": "bob", "role": "attorney"}, audit="membership.add")
    lead = next(m for m in s.call("alice", "GET", f"/legal/matters/{mid}/memberships")["memberships"] if m["username"] == "alice")
    if not lead.get("can_export"):
        s.call("alice", "PATCH", f"/legal/matters/{mid}/memberships/{q(lead['id'])}", {"can_export": True}, audit="membership.change")
    body, ctype = multipart({"title": "Engagement request (workflow check)", "kind": "correspondence", "status": "final"},
                            "engagement-request.txt", REQUEST.encode())
    doc = s.call("alice", "POST", f"/legal/matters/{mid}/documents/import", data=body, ctype=ctype, audit="document.import")["document"]

    def indexed():
        d = next((d for d in s.call("alice", "GET", f"/legal/matters/{mid}/documents")["documents"] if d["id"] == doc["id"]), {})
        st = (d.get("current_version") or {}).get("index_status")
        if st == "failed":
            raise Failed("the document failed to index")
        return st == "indexed"
    if not wait(indexed, 300):
        raise Failed("the document was not indexed within 300 s")
    w["doc"] = doc["id"]


# ── 2. Draft ─────────────────────────────────────────────────────────────────────
def _fill(field: dict, today: str) -> str:
    sug = (field.get("suggestion") or {}).get("value")
    if sug:
        return sug
    return {"party": CLIENT, "money": "USD 14,500", "date": today}.get(field.get("kind"), "Workflow check (fictional)")


def check_draft(s: Stack, w: dict) -> str:
    mid = q(w["mid"])
    tpl = next((t for t in s.call("alice", "GET", "/legal/templates")["templates"]
                if t.get("kind") == "engagement_letter" and any(v.get("status") == "approved" for v in t.get("versions", []))), None)
    if not tpl:
        raise Failed("no approved engagement-letter template")
    d = s.call("alice", "POST", f"/legal/matters/{mid}/drafts", {"template_id": tpl["id"], "title": "Engagement letter (workflow check)"},
               timeout=600, audit="draft.create")["draft"]
    first_gaps, edits = d["content"]["gaps"], 0
    today = datetime.now().strftime("%-d %B %Y")
    while d["content"]["gaps"] and edits < 3:
        fields = [{"key": f["key"], "value": _fill(f, today)} for f in d["content"]["fields"] if f.get("status") == "needs_input"]
        blocks = [{"index": n, "text": "The firm will review the supplier's redline and negotiate the service levels (fictional)."}
                  for n, b in enumerate(d["content"]["blocks"])
                  if b.get("origin") in ("written", "edited") and "[[NEEDS INPUT:" in b.get("text", "")]
        d = s.call("alice", "POST", f"/legal/matters/{mid}/drafts/{q(d['id'])}/edit",
                   {"content_sha256": d["content_sha256"], "fields": fields, "blocks": blocks, "reason": "Workflow check"},
                   audit="draft.edit")["draft"]
        edits += 1
    if d["content"]["gaps"]:
        raise Failed(f"{d['content']['gaps']} gaps left after {edits} edits")
    review = f"/legal/matters/{mid}/artifacts/{q(d['id'])}/review"
    approve = {"decision": "approved", "content_sha256": d["content_sha256"]}
    own = s.refused("alice", "POST", review, approve, want=409, audit="artifact.review")
    status = s.call("bob", "POST", review, approve, audit="artifact.review")["artifact"]["status"]
    if status != "approved":
        raise Failed(f"after bob's approval the draft is {status}")
    filed = s.call("alice", "POST", f"/legal/matters/{mid}/artifacts/{q(d['id'])}/file", audit="artifact.file")
    meta = s.call("alice", "GET", f"/legal/matters/{mid}/documents/{q(filed['document_id'])}")["document"]
    if not meta["current_version"]["filename"].endswith(".docx"):
        raise Failed("the filed draft is not a Word document")
    w["filed"] = filed["document_id"]
    return (f"template v{d['content']['template'].get('version', '?')}, {first_gaps} gaps filled in {edits} edit(s), "
            f"author's approval {own}, bob approved, filed as a Word document")


# ── 3. Export ────────────────────────────────────────────────────────────────────
def check_export(s: Stack, w: dict) -> str:
    base = f"/legal/matters/{q(w['mid'])}/exports"
    e = s.call("alice", "POST", base, {"object_type": "document", "object_id": w["filed"], "format": "docx",
                                       "purpose": "Workflow check (fictional)"}, audit="export.request")["export"]
    s.call("alice", "POST", f"{base}/{q(e['id'])}/submit", audit="export.submit")
    own = s.refused("alice", "POST", f"{base}/{q(e['id'])}/approve", {}, want=403, audit="export.approve")
    code, raw, _ = s.send("bob", "POST", f"{base}/{q(e['id'])}/approve", {})
    ack = False
    if code == 409 and "privileged" in raw.decode(errors="replace"):              # a filed draft may be labelled privileged
        ack = True
        code, raw, _ = s.send("bob", "POST", f"{base}/{q(e['id'])}/approve", {"acknowledge_privilege": True})
    if code != 200:
        raise Failed(f"bob's approval -> {code}")
    s.expect("export.approve", "bob", True)
    done = json.loads(raw)["export"]
    if done["status"] != "ready" or not done.get("watermark"):
        raise Failed(f"export {done['status']}, watermark {'set' if done.get('watermark') else 'missing'}")
    data = s.call("alice", "GET", f"{base}/{q(e['id'])}/file", audit="export.download")["bytes"]
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        text = "".join(z.read(n).decode("utf-8", "replace") for n in z.namelist() if n.endswith(".xml"))
    if done["watermark"] not in text:
        raise Failed("the released file does not carry the watermark")
    checks = (done.get("verification") or {}).get("checks") or {}
    return (f"requester's approval {own}, bob approved{' (privilege acknowledged)' if ack else ''}, released "
            f"{len(data)} bytes with the watermark, {sum(map(bool, checks.values()))}/{len(checks)} verification checks")


# ── 4. Wall ──────────────────────────────────────────────────────────────────────
def _bob_access(s: Stack, w: dict) -> tuple[int, int, int]:
    mid = q(w["mid"])
    return (s.send("bob", "GET", f"/legal/matters/{mid}")[0],
            s.send("bob", "POST", f"/legal/matters/{mid}/documents/search", {"query": "agreed fee"})[0],
            s.send("bob", "GET", f"/legal/counsel/citations/{q(w['cit'])}")[0])


def check_wall(s: Stack, w: dict) -> str:
    conv = s.call("bob", "POST", "/legal/counsel/conversations", {"matter_id": w["mid"]}, ok=(201,))["conversation"]["id"]
    events = s.call("bob", "POST", f"/legal/counsel/conversations/{q(conv)}/turns", {"text": QUESTION}, timeout=600,
                    audit="counsel.turn")["events"]
    final = next((e["data"] for e in events if e.get("event") == "ui.final"), None)
    cits = sorted(set(CIT.findall(json.dumps(final)))) if final else []
    if not cits:
        raise Failed(f"bob's question gave no citation ({[e.get('event') for e in events][-3:]})")
    w["cit"] = cits[0]
    before = _bob_access(s, w)
    if before != (200, 200, 200):
        raise Failed(f"before the wall bob gets {before}")
    for action in ("matter.read", "documents.search", "counsel.citation.open"):
        s.expect(action, "bob", True)
    wall = s.call("morgan", "POST", "/legal/walls", {
        "name": f"Workflow check WF{RUN}", "rationale": "Fictional conflict screen for the workflow check.",
        "screened": ["bob"], "protected_matters": [w["mid"]], "protected_clients": []}, audit="wall.request")["wall"]
    wid = q(wall["id"])
    try:
        own = s.refused("morgan", "POST", f"/legal/walls/{wid}/approve", want=403, audit="wall.approve")
        status = s.call("nakamura", "POST", f"/legal/walls/{wid}/approve", audit="wall.approve")["wall"]["status"]
        walled = _bob_access(s, w)
        for action in ("matter.read", "documents.search", "counsel.citation.open"):
            s.expect(action, "bob", False)
    finally:                                                                       # never leave a wall up
        cur = next((x for x in s.call("morgan", "GET", "/legal/walls")["walls"] if x["id"] == wall["id"]), {})
        if cur.get("status") == "requested":
            s.call("nakamura", "POST", f"/legal/walls/{wid}/reject")
        if cur.get("status") == "active":
            s.call("morgan", "POST", f"/legal/walls/{wid}/release", audit="wall.release.request")
            cur["status"] = "release_requested"
        if cur.get("status") == "release_requested":
            s.call("nakamura", "POST", f"/legal/walls/{wid}/release/approve", audit="wall.release.approve")
    after = _bob_access(s, w)
    for action in ("matter.read", "documents.search", "counsel.citation.open"):
        s.expect(action, "bob", True)
    if status != "active" or walled != (404, 404, 404) or after != (200, 200, 200):
        raise Failed(f"wall {status}; bob walled {walled}, after release {after}")
    return (f"{len(cits)} citation(s) for bob; requester's approval {own}, nakamura approved; bob walled: matter/search/citation "
            f"{'/'.join(map(str, walled))}; released (morgan asked, nakamura approved): {'/'.join(map(str, after))}")


# ── 5. Emergency access ──────────────────────────────────────────────────────────
def start_emergency(s: Stack, w: dict) -> None:
    g = s.call("fiona", "POST", "/legal/break-glass", {"matter_number": w["number"], "minutes": 5,
                                                       "reason": "Workflow check: lead unreachable (fictional)."},
               audit="breakglass.request")["request"]
    gid = q(g["id"])
    w["bg_own"] = s.refused("fiona", "POST", f"/legal/break-glass/{gid}/approve", want=403, audit="breakglass.approve")
    g = s.call("morgan", "POST", f"/legal/break-glass/{gid}/approve", audit="breakglass.approve")["request"]
    if g["status"] != "active":
        raise Failed(f"after morgan's approval the request is {g['status']}")
    mid = q(w["mid"])
    m = s.call("fiona", "GET", f"/legal/matters/{mid}", audit="matter.read")["matter"]
    if m["my_role"] != "break_glass":
        raise Failed(f"fiona's role on the matter is {m['my_role']}")
    w["bg_write"] = (s.refused("fiona", "PATCH", f"/legal/matters/{mid}", {"title": "changed", "version": m["version"]}, want=403,
                               audit="matter.update"),
                     s.refused("fiona", "POST", f"/legal/matters/{mid}/memberships", {"username": "carol", "role": "paralegal"},
                               want=403, audit="membership.add"))
    w["bg"] = g


def finish_emergency(s: Stack, w: dict) -> str:
    g = w["bg"]
    ends = datetime.strptime(g["expires_at"].replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
    left = ends - time.time()
    if left > 0:
        time.sleep(left + 2)
    code = s.send("fiona", "GET", f"/legal/matters/{q(w['mid'])}")[0]
    s.expect("matter.read", "fiona", code == 200)
    status = next((r["status"] for r in s.call("morgan", "GET", "/legal/break-glass")["requests"] if r["id"] == g["id"]), "?")
    if code != 404 or status != "ended":
        raise Failed(f"after the grant's end fiona gets {code} and the managing partners' list says {status}")
    return (f"self-approval {w['bg_own']}, morgan approved, read-only (edit {w['bg_write'][0]}, add member {w['bg_write'][1]}), "
            f"{g['minutes']} min grant ended: fiona then gets {code}, listed as {status}")


# ── 6. Audit ─────────────────────────────────────────────────────────────────────
def check_audit(s: Stack, w: dict, since: str) -> str:
    v = s.call("morgan", "GET", "/legal/audit/verify")
    if not v.get("ok"):
        raise Failed(f"the audit chain does not verify ({v.get('events')} events)")
    seen, outcomes, before, pages = collections.Counter(), collections.defaultdict(collections.Counter), None, 0
    while True:
        page = s.call("morgan", "GET", "/legal/audit?" + urllib.parse.urlencode(
            {"since": since, "limit": 200, **({"before": before} if before else {})}))
        for e in page["events"]:
            key = (e["action"], (e.get("actor") or {}).get("username"), e["outcome"] == "allow")
            seen[key] += 1
            outcomes[key][e["outcome"]] += 1
        pages += 1
        before = page.get("next_before")
        if not before or pages > 50:
            break
    want = collections.Counter(s.expected)
    missing = sorted(f"{a} by {u} ({'allow' if ok else 'refused'})" for (a, u, ok), n in want.items() if seen[(a, u, ok)] < n)
    if missing:
        raise Failed(f"{len(missing)} of {len(want)} expected steps missing: {', '.join(missing[:6])}")
    # A refusal is recorded as "deny" (not "error", which is for faults), whatever status code it was answered with.
    not_denied = sorted(f"{a} by {u} ({', '.join(f'{o} {k}' for o, k in sorted(outcomes[(a, u, ok)].items()))})"
                        for (a, u, ok), n in want.items() if not ok and outcomes[(a, u, ok)]["deny"] < n)
    if not_denied:
        raise Failed(f"{len(not_denied)} refused steps not recorded as deny: {', '.join(not_denied[:6])}")
    refusals = sum(n for (_, _, ok), n in want.items() if not ok)
    return (f"chain intact ({v.get('events')} events); all {sum(want.values())} expected steps found with their actor and "
            f"outcome ({refusals} refusals, each recorded as deny)")


# ── Running ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--allow-demo", action="store_true", help="run on the public demo's containers too (it adds records)")
    args = ap.parse_args()
    if not BASE or not CONTAINER:
        ap.error("set LAW_CHECK_BASE (e.g. http://127.0.0.1:8151) and LAW_CHECK_CONTAINER (the backend container)")
    if CONTAINER.startswith(PUBLIC_DEMO_PREFIX) and not args.allow_demo:
        ap.error(f"{CONTAINER} is the public demo; this check adds records, so run it on a private copy (or pass --allow-demo)")
    since = iso_now()
    s, w = Stack(), {}
    print(f"run WF-{RUN} on {BASE}: adds an enquiry, client and matter WF-{RUN} and the records of each chain", flush=True)
    failed = 0

    def run(name: str, fn, *extra, needs: tuple = ()) -> bool:
        nonlocal failed
        t0 = time.time()
        missing = [k for k in needs if k not in w]
        try:
            if missing:
                raise Failed(f"skipped: needs {', '.join(missing)} from an earlier chain")
            ok, info = True, fn(s, w, *extra)
        except Failed as e:
            ok, info = False, str(e)
        except Exception as e:                                                     # report and go on to the next check
            ok, info = False, f"{type(e).__name__}: {str(e)[:160]}"
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:<9} {info}  [{time.time() - t0:.0f} s]", flush=True)
        return ok

    if run("enquiry", check_enquiry):
        try:
            setup_matter(s, w)
        except Exception as e:
            print(f"FAIL  setup     {type(e).__name__}: {str(e)[:160]}", flush=True)
            failed += 1
        try:
            start_emergency(s, w)                                                  # its 5 minutes run during the next chains
        except Exception as e:
            w["bg_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    run("draft", check_draft, needs=("doc",))
    run("export", check_export, needs=("filed",))
    run("wall", check_wall, needs=("doc",))
    if "bg_error" in w:
        print(f"FAIL  emergency {w['bg_error']}", flush=True)
        failed += 1
    else:
        run("emergency", finish_emergency, needs=("bg",))
    run("audit", check_audit, since)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
