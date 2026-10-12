#!/usr/bin/env python3
"""Load the fictional demo documents into the EchoMind Law pilot stack's demo matters (D-061).

    python3 scripts/law/demo_load.py                 # through the pilot's nginx, http://localhost:3110/api

Needs the demo world first:
    scripts/law/compose.sh exec -T -e LEGAL_ALLOW_DEMO_SEED=1 backend python3 -m app.legal.cli seed-demo

Signs in as the demo lawyers with the credentials the seed wrote inside the backend's data volume. They are
read through `docker exec` and never printed or written anywhere. Each document is imported once (a title
already on the matter is skipped), then the script waits until every document is searchable. Every document
is fictional (eval/fixtures/law/PROVENANCE.md).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "law"))
from law_client import open_session  # noqa: E402

FIX = ROOT / "eval" / "fixtures" / "law"
GOV = FIX / "governance"
# matter number: (who imports it, [(file, kind, status)])
DEMO = {
    "M-101": ("alice", [
        (FIX / "Master Services Agreement (DEMO).md", "contract", "executed"),
        (FIX / "Advice memo on the MSA (DEMO).md", "memo", "final"),
        (FIX / "Engagement request (DEMO).md", "correspondence", "received"),
        (FIX / "panel" / "Reseller Agreement (DEMO).txt", "contract", "executed"),
        (FIX / "summaries" / "Consultancy Agreement (DEMO).txt", "contract", "executed"),
    ]),
    "M-114": ("alice", [
        (GOV / "Articles of Association (DEMO).txt", "other", "final"),
        (GOV / "Board minutes January 2026 (DEMO).txt", "note", "final"),
        (GOV / "Board procedures policy (DEMO).txt", "other", "final"),
        (GOV / "Delegation of authority (DEMO).txt", "other", "final"),
        (GOV / "IP Assignment - A Moreau (DEMO).txt", "contract", "executed"),
        (GOV / "Register of members (DEMO).txt", "other", "final"),
        (GOV / "Supply Agreement - Northwind (DEMO).txt", "contract", "executed"),
        (FIX / "Supply Agreement (DEMO).md", "contract", "executed"),
    ]),
}


def _credentials(container: str) -> dict:
    out = subprocess.run(["docker", "exec", container, "cat", "/data/legal/demo_credentials.json"],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def _multipart(fields: dict, path: pathlib.Path) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    mime = "text/markdown" if path.suffix == ".md" else "text/plain"
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()]
    parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
                  f"Content-Type: {mime}\r\n\r\n").encode() + path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:3110/api")
    ap.add_argument("--container", default="echomind-law-backend")
    ap.add_argument("--wait", type=int, default=900, help="seconds to wait for indexing")
    args = ap.parse_args()
    creds = _credentials(args.container)
    sessions = {}

    def session(user: str):
        if user not in sessions:
            os.environ["LAW_TOTP_SECRET"] = creds[user]["totp_secret"]
            sessions[user] = open_session(args.base, user, creds[user]["password"])
            os.environ.pop("LAW_TOTP_SECRET", None)
        return sessions[user]

    def get(user: str, path: str) -> dict:
        with session(user).open(urllib.request.Request(args.base + path), timeout=60) as r:
            return json.loads(r.read() or b"{}")

    for number, (user, files) in DEMO.items():
        matters = {m["number"]: m for m in get(user, "/legal/matters")["matters"]}
        mid = matters[number]["id"]
        have = {d["title"] for d in get(user, f"/legal/matters/{urllib.parse.quote(mid)}/documents")["documents"]}
        for path, kind, status in files:
            title = path.stem
            if title in have:
                print(f"{number}: {title} already there")
                continue
            body, ctype = _multipart({"title": title, "kind": kind, "status": status}, path)
            req = urllib.request.Request(f"{args.base}/legal/matters/{urllib.parse.quote(mid)}/documents/import",
                                         data=body, method="POST", headers={"Content-Type": ctype})
            with session(user).open(req, timeout=120) as r:
                r.read()
            print(f"{number}: imported {title} ({kind}, {status})")

    deadline = time.time() + args.wait
    while True:
        pending, failed = [], []
        for number, (user, _) in DEMO.items():
            matters = {m["number"]: m for m in get(user, "/legal/matters")["matters"]}
            for d in get(user, f"/legal/matters/{urllib.parse.quote(matters[number]['id'])}/documents")["documents"]:
                st = (d.get("current_version") or {}).get("index_status")
                (failed if st == "failed" else pending if st != "indexed" else []).append(f"{number}: {d['title']}")
        if not pending or time.time() > deadline:
            break
        time.sleep(10)
    print(f"searchable: all but {len(pending)} pending, {len(failed)} failed")
    for p in pending + failed:
        print("  not searchable:", p)
    return 0 if not pending and not failed else 1


if __name__ == "__main__":
    sys.exit(main())
