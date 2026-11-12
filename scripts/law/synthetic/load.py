#!/usr/bin/env python3
"""Load the synthetic legal dataset into a PRIVATE copy of the demo (2026-11-12). Never the public demo or production.

    python3 scripts/law/synthetic/load.py --base http://127.0.0.1:8161/api --container echomind-legal-syn-api

1. The four matters of eval/fixtures/synthetic/manifest.json join the copy's demo world through `seed-demo-practice`
   (fictional clients, adverse parties and members; the demo people already exist).
2. Each document is imported by its matter's lead through the product's own import (`…/documents/import`), with the
   file's real type, in manifest order; a `version_of` document is imported as a new version of the earlier one. A title
   already on the matter is skipped, so the load can be run again.
3. Files the product is not expected to accept (CSV, .eml, .xlsx: `import: no`) are sent too, and the answer recorded.
4. It waits until every document is searchable (scans are read by OCR first), then writes
   eval/law/reports/<time>-synthetic-load.json: per document, the import answer, the time to searchable and how it was read.
The copy's demo credentials are read through `docker exec` and never printed.
"""
from __future__ import annotations

import argparse
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "eval" / "fixtures" / "synthetic"
sys.path.insert(0, str(ROOT / "scripts" / "law"))
from demo_load import _credentials  # noqa: E402
from law_client import open_session  # noqa: E402

MIME = {".pdf": "application/pdf", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain", ".md": "text/markdown", ".csv": "text/csv", ".eml": "message/rfc822",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}


def seed(container: str, plan: dict) -> None:
    """seed-demo-practice inside the copy's API container, with the settings its launch script gives the server."""
    r = subprocess.run(["docker", "exec", "-i", "-e", "LEGAL_ALLOW_DEMO_SEED=1", "-e", "ECHOMIND_EDITION=law", "-e", "ECHOMIND_DATA_DIR=/data",
                        "-e", "LEGAL_MASTER_KEY_FILE=/run/secrets/master.key", "-e", "HOME=/tmp", "-w", "/repo/backend", container,
                        "python3", "-m", "app.legal.cli", "seed-demo-practice"], input=json.dumps(plan), capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"seed-demo-practice failed: {r.stderr.strip()[-300:]}")
    print(r.stdout.strip().splitlines()[-1])


def multipart(fields: dict, path: Path) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
             for k, v in fields.items() if v is not None]
    parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
                  f"Content-Type: {MIME.get(path.suffix.lower(), 'application/octet-stream')}\r\n\r\n").encode()
                 + path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="the private copy's API, e.g. http://127.0.0.1:8161/api")
    ap.add_argument("--container", required=True, help="the private copy's API container")
    ap.add_argument("--wait", type=int, default=3600, help="seconds to wait for every document to be searchable")
    ap.add_argument("--only", default="", help="comma-separated matter numbers")
    args = ap.parse_args()
    if not re.match(r"^https?://(127\.0\.0\.1|localhost):\d+/", args.base) or args.container.startswith("echomind-legal-demo"):
        print("refusing: load the synthetic set into a private copy on this machine only", file=sys.stderr)
        return 2
    manifest = json.loads((BASE / "manifest.json").read_text())
    cases = [c for c in manifest["cases"] if not args.only or c["number"] in args.only.split(",")]
    keep = ("number", "title", "client", "practice_area", "jurisdiction", "lead", "members", "parties", "sensitivity")
    seed(args.container, {"team": {}, "matters": [{k: c[k] for k in keep if k in c} for c in cases]})
    creds = _credentials(args.container)
    sessions: dict = {}

    def session(user: str):
        if user not in sessions:
            os.environ["LAW_TOTP_SECRET"] = creds[user]["totp_secret"]
            sessions[user] = open_session(args.base, user, creds[user]["password"])
            os.environ.pop("LAW_TOTP_SECRET", None)
        return sessions[user]

    def get(user: str, path: str) -> dict:
        with session(user).open(urllib.request.Request(args.base + path), timeout=60) as r:
            return json.loads(r.read() or b"{}")

    rows, started = [], {}
    for c in cases:
        user = c["lead"]
        mid = {m["number"]: m["id"] for m in get(user, "/legal/matters")["matters"]}[c["number"]]
        have = {d["title"]: d for d in get(user, f"/legal/matters/{urllib.parse.quote(mid)}/documents")["documents"]}
        ids = {d["file"]: have[d["title"]]["id"] for d in c["documents"] if d["title"] in have}
        for d in c["documents"]:
            path = BASE / d["file"]
            row = {"matter": c["number"], "file": d["file"], "title": d["title"], "render": d["render"], "expected": d["import"]}
            parent = ids.get(d.get("version_of")) if d.get("version_of") else None
            if d["title"] in have and d["import"]:
                row["import"] = "already there"
                rows.append(row)
                continue
            body, ctype = multipart({"title": None if parent else d["title"], "kind": d["kind"], "status": d["status"],
                                     "version_of": parent}, path)
            req = urllib.request.Request(f"{args.base}/legal/matters/{urllib.parse.quote(mid)}/documents/import", data=body,
                                         method="POST", headers={"Content-Type": ctype})
            try:
                with session(user).open(req, timeout=300) as r:
                    made = json.loads(r.read() or b"{}")
                row.update({"import": r.status, "document_id": (made.get("document") or {}).get("id")})
                ids[d["file"]] = row["document_id"]
                started[row["document_id"]] = time.time()
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                row.update({"import": e.code, "detail": detail})
            ok = (row["import"] in (200, 201, 202)) == d["import"]
            print(f"{c['number']}: {'ok ' if ok else 'UNEXPECTED'} {row['import']} {d['file']}"
                  + (f"  ({row.get('detail', '')[:120]})" if "detail" in row else ""), flush=True)
            rows.append(row)

    deadline = time.time() + args.wait
    done: dict = {}
    while True:
        pending = []
        for c in cases:
            mid = {m["number"]: m["id"] for m in get(c["lead"], "/legal/matters")["matters"]}[c["number"]]
            for d in get(c["lead"], f"/legal/matters/{urllib.parse.quote(mid)}/documents")["documents"]:
                v = d.get("current_version") or {}
                st = v.get("index_status")
                if st in ("indexed", "failed") and d["id"] not in done:
                    done[d["id"]] = {"status": st, "secs": round(time.time() - started[d["id"]]) if d["id"] in started else None,
                                     "version": v}
                elif st not in ("indexed", "failed"):
                    pending.append(f"{c['number']}: {d['title']} ({st})")
        if not pending or time.time() > deadline:
            break
        print(f"waiting: {len(pending)} not searchable yet", flush=True)
        time.sleep(20)
    for row in rows:
        if row.get("document_id") in done:
            info = done[row["document_id"]]
            v = info["version"]
            row.update({"index_status": info["status"], "secs_to_searchable": info["secs"],
                        "version": {k: v.get(k) for k in ("number", "pages", "filetype", "ocr_pages", "extraction", "chunks") if k in v}})
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = ROOT / "eval" / "law" / "reports" / f"{stamp}-synthetic-load.json"
    out.write_text(json.dumps({"suite": "synthetic-load", "base": args.base, "documents": rows, "not_searchable": pending}, indent=1))
    failed = [r for r in rows if r.get("index_status") == "failed"]
    unexpected = [r for r in rows if (r.get("import") in (200, 201, 202, "already there")) != r["expected"]]
    print(f"searchable: {sum(r.get('index_status') == 'indexed' for r in rows)}, failed {len(failed)}, pending {len(pending)}, "
          f"unexpected import answers {len(unexpected)} -> {out.relative_to(ROOT)}")
    return 0 if not pending and not failed and not unexpected else 1


if __name__ == "__main__":
    sys.exit(main())
