"""Execute the 2026 EchoMind test cases against the running reference instance and record
exactly what happened. Credentials come from the environment and are never written out."""
import base64, datetime as dt, http.cookiejar, json, os, re, subprocess, time, urllib.error, urllib.request, uuid

BASE = "http://localhost:3000"
ROOT = "/home/echomind/Documents/echomind/echomind-enterprise"
OUT = os.environ.get("QA_OUT") or (os.path.dirname(os.path.abspath(__file__)) + "/results")
os.makedirs(OUT, exist_ok=True)
ADMIN_U, ADMIN_P = os.environ["AUTH_ADMIN_USER"], os.environ["AUTH_ADMIN_PASSWORD"]
UA = {"User-Agent": "Mozilla/5.0 echomind-qa"}
RESULTS, LOG = {}, []


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(line):
    LOG.append(f"[{now()}] {line}")
    print(LOG[-1], flush=True)


class Client:
    def __init__(self):
        self.cj = http.cookiejar.CookieJar()
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cj))
        self.bearer = None

    def req(self, method, path, body=None, form=None, raw=False):
        headers = dict(UA)
        data = None
        if self.bearer:
            headers["Authorization"] = "Bearer " + self.bearer
        if form is not None:
            b = uuid.uuid4().hex
            parts = []
            for k, v in form.items():
                if isinstance(v, tuple):
                    fn, content = v
                    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"; filename="{fn}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode() + content + b"\r\n")
                else:
                    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
            data = b"".join(parts) + f"--{b}--\r\n".encode()
            headers["Content-Type"] = f"multipart/form-data; boundary={b}"
        elif body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        r = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
        try:
            with self.op.open(r, timeout=180) as resp:
                txt = resp.read().decode("utf-8", "replace")
                return resp.status, (txt if raw else (json.loads(txt) if txt.strip().startswith(("{", "[")) else txt))
        except urllib.error.HTTPError as e:
            txt = e.read().decode("utf-8", "replace")
            try:
                return e.code, json.loads(txt)
            except Exception:
                return e.code, txt[:300]

    def token(self):
        return next((c.value for c in self.cj if c.name == "echomind_token"), None)


def tc(tid, title):
    RESULTS[tid] = {"id": tid, "title": title, "executed": now(), "steps": [], "notes": []}
    log(f"=== {tid} {title}")
    return RESULTS[tid]


def step(t, sid, action, expected, ok, actual, evidence=""):
    t["steps"].append({"id": sid, "action": action, "expected": expected, "actual": actual,
                       "status": "Pass" if ok is True else ("Fail" if ok is False else ok), "evidence": evidence})
    log(f"  {sid} {'PASS' if ok is True else ('FAIL' if ok is False else ok)} — {actual}")


def finish(t):
    st = [s["status"] for s in t["steps"]]
    t["result"] = "Fail" if "Fail" in st else ("Blocked" if "Blocked" in st else ("Not executed" if st and all(s == "Not executed" for s in st) else "Pass"))
    log(f"  => {t['id']} {t['result']}")


def sh(cmd, timeout=900):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)


admin = Client()

# ── TC-EM-AUTH-101 ────────────────────────────────────────────────────────────────
t = tc("TC-EM-AUTH-101", "Login, session and access control")
anon = Client()
c, b = anon.req("GET", "/api/auth/config")
step(t, "AC-101.1", "GET /api/auth/config", "auth_enabled true", c == 200 and b.get("auth_enabled") is True, f"HTTP {c} {b}")
c, b = admin.req("POST", "/api/auth/login", {"username": ADMIN_U, "password": ADMIN_P})
step(t, "AC-101.2", "Log in with valid admin credentials", "HTTP 200, session cookie issued, role admin",
     c == 200 and admin.token() is not None and b.get("user", {}).get("role") == "admin", f"HTTP {c}, cookie {'set' if admin.token() else 'missing'}, role {b.get('user', {}).get('role') if isinstance(b, dict) else '?'}")
c, b = Client().req("POST", "/api/auth/login", {"username": ADMIN_U, "password": "wrong-" + uuid.uuid4().hex[:6]})
step(t, "AC-101.3", "Log in with an invalid password", "HTTP 401, no session", c == 401, f"HTTP {c} {b if isinstance(b, dict) else ''}")
codes = {p: anon.req("GET", p)[0] for p in ("/api/docs/list", "/api/chat/list", "/api/auth/users")}
codes["POST /api/docs/delete-all"] = anon.req("POST", "/api/docs/delete-all")[0]
step(t, "AC-101.4", "Call protected APIs without a session", "HTTP 401 for every call", all(v == 401 for v in codes.values()), ", ".join(f"{k} → {v}" for k, v in codes.items()))
qa_user, qa_pass = "qa_tc101_" + uuid.uuid4().hex[:5], "Qa!" + uuid.uuid4().hex[:12]
c, b = admin.req("POST", "/api/auth/users", {"username": qa_user, "password": qa_pass, "role": "user"})
created = c == 200
user = Client()
c2, _ = user.req("POST", "/api/auth/login", {"username": qa_user, "password": qa_pass})
c3, _ = user.req("GET", "/api/auth/users")
c4, _ = user.req("GET", "/api/docs/list")
step(t, "AC-101.5", "Standard (non-admin) user calls an admin-only API", "Admin API refused (403); normal API allowed (200)",
     created and c2 == 200 and c3 == 403 and c4 == 200, f"create user HTTP {c}; login {c2}; GET /api/auth/users → {c3}; GET /api/docs/list → {c4}")
tok = admin.token() or ""
try:
    p = json.loads(base64.urlsafe_b64decode(tok.split(".")[1] + "=="))
    ttl = (p["exp"] - p["iat"]) / 60
    step(t, "AC-101.6", "Inspect session token lifetime", "Token expires (finite lifetime configured)", ttl > 0, f"exp − iat = {ttl:.0f} minutes ({ttl/60:.0f} h); configured by AUTH_TOKEN_TTL_MIN")
except Exception as e:
    step(t, "AC-101.6", "Inspect session token lifetime", "Token expires", False, f"could not decode token: {e}")
old = user.token()
c5, _ = user.req("POST", "/api/auth/logout")
replay = Client(); replay.bearer = old
c6, _ = replay.req("GET", "/api/docs/list")
step(t, "AC-101.7", "Log out, then replay the old session token", "Old token rejected (401) after logout", c6 == 401,
     f"logout HTTP {c5}; replayed token → HTTP {c6}" + (" — token still accepted after logout (stateless JWT, no revocation)" if c6 == 200 else ""))
c, b = admin.req("GET", "/api/auth/audit")
evs = b.get("events", []) if isinstance(b, dict) else []
hit = [e for e in evs if qa_user in json.dumps(e)]
step(t, "AC-101.8", "Check the activity log for the test user's activity", "Entries with timestamp, user and action", len(hit) > 0,
     f"{len(evs)} recent events; {len(hit)} for {qa_user}" + (f"; e.g. {json.dumps(hit[0])[:160]}" if hit else ""))
ck = sh(f"curl -s -o /dev/null -w '%{{http_code}}' -m 10 -H 'Connection: Upgrade' -H 'Upgrade: websocket' -H 'Sec-WebSocket-Version: 13' -H 'Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==' http://localhost:8002/ws")[1]
ck2 = sh(f"curl -s -o /dev/null -w '%{{http_code}}' -m 10 --cookie 'echomind_token={tok}' -H 'Connection: Upgrade' -H 'Upgrade: websocket' -H 'Sec-WebSocket-Version: 13' -H 'Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==' http://localhost:8002/ws")[1]
step(t, "AC-101.9", "Open the voice WebSocket without and with a session", "403 without, 101 with", ck == "403" and ck2 == "101", f"without session → {ck}; with session → {ck2}")
c, _ = admin.req("DELETE", f"/api/auth/users/{qa_user}")
t["notes"].append(f"Temporary user {qa_user} created for the test and deleted afterwards (HTTP {c}).")
finish(t)

# ── TC-EM-DOC-102 ─────────────────────────────────────────────────────────────────
t = tc("TC-EM-DOC-102", "Document upload, validation and activity log")
marker = "QA-TC102-" + uuid.uuid4().hex[:8]
content = (f"EchoMind QA upload test document {marker}.\nThe calibration interval for the QA reference widget is "
           f"fourteen days. This file exists only to verify upload, indexing and deletion.\n").encode()
c, b = admin.req("POST", "/api/docs/upload", form={"file": (f"{marker}.txt", content), "namespace": "qa-test"})
doc_id = b.get("doc_id") if isinstance(b, dict) else None
step(t, "DOC-102.1", "Upload a small text document to namespace qa-test", "HTTP 200 with a document id", c == 200 and bool(doc_id), f"HTTP {c}; doc_id {doc_id}")
c, b = admin.req("GET", "/api/docs/list")
listed = any(d.get("id") == doc_id for d in (b.get("documents", []) if isinstance(b, dict) else []))
step(t, "DOC-102.2", "List documents", "Uploaded document appears", listed, f"present in list: {listed}")
c, b = admin.req("POST", "/api/docs/upload", form={"file": (f"{marker}-copy.txt", content), "namespace": "qa-test"})
step(t, "DOC-102.3", "Upload identical content again (duplicate)", "Recognised as duplicate, no second copy indexed",
     c == 200 and isinstance(b, dict) and b.get("duplicate_of") == doc_id, f"HTTP {c}; duplicate_of {b.get('duplicate_of') if isinstance(b, dict) else b}")
c, b = admin.req("POST", "/api/docs/upload", form={"file": ("empty.txt", b""), "namespace": "qa-test"})
step(t, "DOC-102.4", "Upload an empty file", "Rejected with an explanatory error, nothing stored", 400 <= c < 500, f"HTTP {c}; {str(b)[:140]}")
binary = bytes((i * 37 + 11) % 256 for i in range(4096))
c, b = admin.req("POST", "/api/docs/upload", form={"file": ("tool.exe", b"MZ" + binary), "namespace": "qa-test"})
exe_id = b.get("doc_id") if isinstance(b, dict) else None
step(t, "DOC-102.5", "Upload a binary executable (.exe)", "Rejected — unsupported type, nothing stored", 400 <= c < 500,
     f"HTTP {c}; " + (f"ACCEPTED and indexed as {exe_id}" if exe_id else str(b)[:140]))
c, b = admin.req("GET", "/api/auth/audit")
up = [e for e in (b.get("events", []) if isinstance(b, dict) else []) if "upload" in json.dumps(e)]
step(t, "DOC-102.6", "Check the activity log for the upload", "Upload recorded with user and time", len(up) > 0, f"{len(up)} upload events; latest {json.dumps(up[0])[:160] if up else '-'}")
c1, _ = admin.req("DELETE", f"/api/docs/{doc_id}") if doc_id else (None, None)
if exe_id:
    admin.req("DELETE", f"/api/docs/{exe_id}")
c, b = admin.req("GET", "/api/docs/list")
gone = not any(d.get("id") in (doc_id, exe_id) for d in (b.get("documents", []) if isinstance(b, dict) else []))
step(t, "DOC-102.7", "Delete the test document(s)", "Removed from the knowledge base", c1 == 200 and gone, f"DELETE HTTP {c1}; absent from list: {gone}")
finish(t)

# ── TC-EM-RAG-103 ─────────────────────────────────────────────────────────────────
t = tc("TC-EM-RAG-103", "Grounded answers with citations; no fabricated sources")
c, b = admin.req("POST", "/api/chat/create", {"title": "QA test TC-EM-RAG-103 (2026-10-06)"})
chat_id = b.get("chat_id") or b.get("id") if isinstance(b, dict) else None


def ask(q, ns=None):
    body = {"chat_id": chat_id, "message": q}
    if ns:
        body["namespace"] = ns
    t0 = time.time(); c, b = admin.req("POST", "/api/chat/ask", body); el = time.time() - t0
    cits = (b.get("citations") or b.get("sources") or []) if isinstance(b, dict) else []
    files = sorted({(x.get("filename") or x.get("doc") or x.get("title") or "?") for x in cits if isinstance(x, dict)})
    return c, (b.get("answer", "") if isinstance(b, dict) else str(b)), files, el


c, ans, files, el = ask("What is the acknowledgement time for a Severity 1 incident in the Meridian Bank service level schedule?")
step(t, "RAG-103.1", "Ask a question answered by a knowledge-base document", "Specific answer citing the Meridian service-level document",
     c == 200 and any("Service_Level" in f for f in files), f"HTTP {c} in {el:.1f}s; cites {files}; answer: {ans[:170]}")
c, ans2, files2, el = ask("What does the Meridian Bank contract say about colonising Mars?")
step(t, "RAG-103.2", "Ask about something not in any document", "Says it is not in the documents; no fabricated citation",
     c == 200 and not any("Mars" in f for f in files2), f"HTTP {c} in {el:.1f}s; cites {files2 or 'none'}; answer: {ans2[:170]}")
c, ans3, files3, el = ask("Hello! How are you today?")
step(t, "RAG-103.3", "Small talk", "Natural reply with no citations", c == 200 and not files3, f"HTTP {c} in {el:.1f}s; cites {files3 or 'none'}; answer: {ans3[:120]}")
t["notes"].append(f"One chat titled 'QA test TC-EM-RAG-103 (2026-10-06)' was created (id {chat_id}); the API has no chat delete, so it remains as test evidence.")
finish(t)

# ── TC-EM-ISO-104 ─────────────────────────────────────────────────────────────────
t = tc("TC-EM-ISO-104", "Knowledge-base namespace isolation between vertical packs")
q = "What is the acknowledgement time for a Severity 1 incident in the Meridian Bank service level schedule?"
res = {ns: ask(q, ns) for ns in ("bank", "health", "law")}
for ns, (c, a, f, el) in res.items():
    log(f"    namespace {ns}: HTTP {c}; cites {f}")
mer = {ns: any("Meridian" in x for x in v[2]) for ns, v in res.items()}
step(t, "ISO-104.1", "Ask a bank-document question in the health and law namespaces", "No Meridian Bank document cited outside its own namespace",
     not mer["health"] and not mer["law"], "; ".join(f"{ns}: cites Meridian = {v}" for ns, v in mer.items()))
step(t, "ISO-104.2", "Record which namespace holds the Meridian documents", "Informational", "Info",
     "Meridian cited in: " + (", ".join(ns for ns, v in mer.items() if v) or "none of bank/health/law — documents are in the default namespace"))
finish(t)

# ── TC-EM-OPS-105 ─────────────────────────────────────────────────────────────────
t = tc("TC-EM-OPS-105", "Service stack health, controlled restart and configuration rejection")
rc, out = sh("docker compose --profile public ps --format '{{.Service}}={{.Status}}'")
svc = dict(l.split("=", 1) for l in out.strip().splitlines() if "=" in l)
healthy = all(("healthy" in v) or (k in ("frontend", "cloudflared") and v.startswith("Up")) for k, v in svc.items())
step(t, "OPS-105.1", "List the six services", "All up; healthchecked services healthy", len(svc) == 6 and healthy, "; ".join(f"{k}: {v}" for k, v in svc.items()))
before = sh("docker inspect -f '{{.Name}} {{.State.StartedAt}}' echomind-backend echomind-voice echomind-trtllm")[1].strip()
rc, out = sh("printf 'services:\\n  backend:\\n    mem_limit: notanumber\\n' > /tmp/claude-1000/qa_bad_override.yml && docker compose -f docker-compose.yml -f /tmp/claude-1000/qa_bad_override.yml config -q; echo rc=$?")
after = sh("docker inspect -f '{{.Name}} {{.State.StartedAt}}' echomind-backend echomind-voice echomind-trtllm")[1].strip()
step(t, "OPS-105.2", "Validate a deliberately invalid compose override", "Rejected; running services untouched", "rc=0" not in out and before == after,
     f"{out.strip().splitlines()[-2][:150] if len(out.strip().splitlines()) > 1 else out.strip()}; running services unchanged: {before == after}")
t0 = time.time(); sh("docker restart echomind-frontend")
ok, code = False, "000"
for _ in range(30):
    code = sh("curl -s -o /dev/null -w '%{http_code}' -m 5 http://localhost:3000/")[1]
    if code == "200":
        ok = True; break
    time.sleep(1)
step(t, "OPS-105.3", "Restart the frontend service", "Returns HTTP 200 within 30 s", ok, f"HTTP {code} after {time.time()-t0:.1f}s")
c, b = admin.req("GET", "/api/version")
step(t, "OPS-105.4", "Smoke test after restart (API via frontend proxy)", "API reachable", c == 200, f"/api/version HTTP {c}")
finish(t)

# ── TC-EM-REL-106 ─────────────────────────────────────────────────────────────────
t = tc("TC-EM-REL-106", "Release identification and traceability")
c, b = admin.req("GET", "/api/version")
ver = b.get("version", "") if isinstance(b, dict) else ""
tag_commit = sh(f"git rev-list -n1 v{ver} 2>/dev/null")[1].strip()
step(t, "REL-106.1", "GET /api/version", "Reports a released version, commit and build date", c == 200 and bool(re.fullmatch(r"\d+\.\d+\.\d+", ver)), f"HTTP {c} {b}")
step(t, "REL-106.2", "Compare the reported commit with the tag of that version", "Identical", bool(tag_commit) and isinstance(b, dict) and b.get("commit") == tag_commit, f"reported {b.get('commit', '?')[:12]}; tag v{ver} → {tag_commit[:12] or 'missing'}")
cl = open(f"{ROOT}/CHANGELOG.md").read()
step(t, "REL-106.3", f"CHANGELOG has a section for {ver}", "Present", f"## [{ver}]" in cl, f"## [{ver}] present" if f"## [{ver}]" in cl else "missing")
vh = sh("curl -s -m 5 http://localhost:8002/health")[1]
step(t, "REL-106.4", "Voice /health reports the same build", f"Version {ver}", f'"{ver}"' in vh, vh[:150])
finish(t)

# ── TC-EM-EXP-108 ─────────────────────────────────────────────────────────────────
t = tc("TC-EM-EXP-108", "Secure export gateway — PII and secret detection with redaction")
sample = ("Contact Jane Q. Example at jane.example@example.com or +1 (555) 010-4477. "
          "Her SSN is 078-05-1120. Deploy key: sk-test-4f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c.")
c, b = admin.req("POST", "/api/export/evaluate", {"text": sample})
red = json.dumps(b.get("redacted") or b.get("redacted_text") or "") if isinstance(b, dict) else ""
fnd = b.get("findings", []) if isinstance(b, dict) else []
kinds = sorted({(f.get("type") or f.get("kind") or f.get("category") or "?") for f in fnd if isinstance(f, dict)})
step(t, "EXP-108.1", "Evaluate text containing an email, phone, SSN and API key (synthetic values)", "Each flagged", c == 200 and len(fnd) >= 3, f"HTTP {c}; {len(fnd)} findings: {kinds}")
leaked = [v for v in ("jane.example@example.com", "078-05-1120", "sk-test-4f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c") if v in red]
step(t, "EXP-108.2", "Inspect the redacted copy", "No original sensitive value remains", bool(red) and not leaked, f"redacted copy present: {bool(red)}; values still present: {leaked or 'none'}")
finish(t)

# ── TC-EM-UNIT-109 ────────────────────────────────────────────────────────────────
t = tc("TC-EM-UNIT-109", "Automated regression suites (backend, voice, frontend type-check)")
SCR = "docker exec {c} sh -c 'rm -rf /tmp/hc && mkdir -p /tmp/hc' && docker cp {src}/app {c}:/tmp/hc/app && docker cp {src}/tests {c}:/tmp/hc/tests && docker exec -w /tmp/hc {env} -e CUDA_VISIBLE_DEVICES= {c} timeout 600 python -m pytest tests -q -p no:cacheprovider --no-header 2>&1 | tail -3; docker exec {c} rm -rf /tmp/hc"
rc, out = sh(SCR.format(c="echomind-backend", src="backend", env="-e ECHOMIND_DATA_DIR=/tmp/hc/data"))
m = re.search(r"(\d+) passed", out); f = re.search(r"(\d+) failed", out)
step(t, "UNIT-109.1", "Backend unit suite (repo HEAD, in the backend container)", "All pass", bool(m) and not f, out.strip().splitlines()[-1])
open(f"{OUT}/backend_pytest.txt", "w").write(out)
rc, out = sh(SCR.format(c="echomind-voice", src="voice", env=""))
m = re.search(r"(\d+) passed", out); f = re.search(r"(\d+) failed", out)
step(t, "UNIT-109.2", "Voice unit suite", "All pass", bool(m) and not f, out.strip().splitlines()[-1])
open(f"{OUT}/voice_pytest.txt", "w").write(out)
rc, out = sh("cd frontend && timeout 300 npx --no-install tsc --noEmit -p . 2>&1; echo exit=$?")
step(t, "UNIT-109.3", "Frontend type-check (tsc --noEmit)", "0 errors", "exit=0" in out, f"{out.count('error TS')} errors; {out.strip().splitlines()[-1]}")
finish(t)

# ── TC-EM-DGEN-110 / TC-EM-BRD-111 smoke ──────────────────────────────────────────
t = tc("TC-EM-DGEN-110", "Document Studio availability (smoke)")
c, b = admin.req("GET", "/api/docgen/templates")
n = len(b.get("templates", b) if isinstance(b, dict) else b) if c == 200 else 0
step(t, "DGEN-110.1", "List document templates", "Templates returned", c == 200 and n > 0, f"HTTP {c}; {n} templates")
c, b = admin.req("GET", "/api/docgen/image-status")
step(t, "DGEN-110.2", "Query on-device image generator status", "Status returned", c == 200, f"HTTP {c}; {str(b)[:150]}")
t["notes"].append("Generation itself not exercised: it loads SDXL-Turbo (~18 GB GPU) and the shared GPU had ~6 GB free.")
finish(t)

t = tc("TC-EM-BRD-111", "Boardroom session lifecycle (smoke)")
c, b = admin.req("POST", "/api/boardroom/sessions", {"title": "QA test TC-EM-BRD-111"})
sid = (b.get("session_id") or b.get("id") or (b.get("session") or {}).get("id")) if isinstance(b, dict) else None
step(t, "BRD-111.1", "Create a boardroom session", "HTTP 200 with a session id", c == 200 and bool(sid), f"HTTP {c}; id {sid}; {str(b)[:120] if not sid else ''}")
c, b = admin.req("GET", f"/api/boardroom/sessions/{sid}") if sid else (None, None)
step(t, "BRD-111.2", "Retrieve the session", "HTTP 200", c == 200, f"HTTP {c}")
c, b = admin.req("DELETE", f"/api/boardroom/sessions/{sid}") if sid else (None, None)
c2, b2 = admin.req("GET", f"/api/boardroom/sessions/{sid}") if sid else (None, None)
step(t, "BRD-111.3", "Delete the session", "Deleted; subsequent GET 404", c == 200 and c2 == 404, f"DELETE HTTP {c}; GET after delete HTTP {c2}")
finish(t)

# ── Not executed / blocked / prior evidence ──────────────────────────────────────
t = tc("TC-EM-TRN-112", "Live transcription end to end (microphone → transcript → live check)")
step(t, "TRN-112.1", "Speak into Live Transcript; verify transcript, tags and Silent Assistant checks", "See test case", "Not executed",
     "Requires a microphone session; scheduled as a manual test. Backend transcription unit tests pass (TC-EM-UNIT-109).")
finish(t)
t = tc("TC-EM-BKP-113", "Data backup integrity and restore")
rc, out = sh("cd backups && sha256sum -c echomind_data_20260922-135915.tar.gz.sha256 2>&1 | tail -1")
step(t, "BKP-113.1", "Verify the checksum of the latest backup archive", "OK", out.strip().endswith("OK"), out.strip())
step(t, "BKP-113.2", "Restore into a scratch volume and run SQLite integrity_check", "ok", "Not executed",
     "Last executed 2026-09-22 (integrity_check ok, 32 tables — REG-09 NFR-R04). Not repeated today; no scheduled backup since (open action).")
finish(t)
t = tc("TC-EM-EVAL-114", "Golden-question retrieval evaluation (52 questions)")
step(t, "EVAL-114.1", "python3 eval/run_eval.py", "Score not below 49/52 baseline", "Blocked",
     "Blocked by NC-2026-008 — the evaluation corpus is absent; last run 2026-09-22 scored 9/52 by construction (eval/reports/eval_20260922-135322.json).")
finish(t)

json.dump(RESULTS, open(f"{OUT}/results.json", "w"), indent=1)
open(f"{OUT}/execution_log.txt", "w").write("\n".join(LOG) + "\n")
print("\nSUMMARY:", {k: v["result"] for k, v in RESULTS.items()})
