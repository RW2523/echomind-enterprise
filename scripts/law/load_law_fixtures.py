#!/usr/bin/env python3
"""Upload the fictional EchoMind Law fixtures (eval/fixtures/law/*.md) into a DEVELOPMENT backend.

    python3 scripts/law/load_law_fixtures.py --base http://127.0.0.1:8110/api

Signs in with ECHOMIND_EVAL_USER / ECHOMIND_EVAL_PASSWORD (defaults: admin and the password in
.law-data/dev_admin_password written by scripts/law/dev_backend.sh). Re-running is safe: the
backend de-duplicates identical content per namespace. Refuses non-local hosts unless
--allow-remote is given, so fixtures cannot be pushed into a shared or production instance by accident.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.parse
import urllib.request
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "eval" / "fixtures" / "law"


def _request(url: str, data: bytes | None = None, headers: dict | None = None, timeout: int = 600) -> dict:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def _multipart(fields: dict, file_field: str, filename: str, content: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
                  f"Content-Type: text/markdown\r\n\r\n").encode() + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8110/api")
    ap.add_argument("--namespace", default="law")
    ap.add_argument("--allow-remote", action="store_true")
    args = ap.parse_args()

    host = urllib.parse.urlparse(args.base).hostname or ""
    if host not in ("127.0.0.1", "localhost", "::1") and not args.allow_remote:
        sys.exit(f"refusing to load fixtures into non-local host {host!r} (use --allow-remote deliberately)")

    user = os.getenv("ECHOMIND_EVAL_USER", "admin")
    password = os.getenv("ECHOMIND_EVAL_PASSWORD")
    if not password:
        pw_file = ROOT / ".law-data" / "dev_admin_password"
        password = pw_file.read_text().strip() if pw_file.exists() else ""
    headers = {}
    if _request(f"{args.base}/auth/config").get("auth_enabled"):
        token = _request(f"{args.base}/auth/login", json.dumps({"username": user, "password": password}).encode(),
                         {"Content-Type": "application/json"}).get("token")
        headers["Authorization"] = f"Bearer {token}"

    files = sorted(p for p in FIXTURES.glob("*.md") if p.name != "PROVENANCE.md")
    for path in files:
        body, ctype = _multipart({"namespace": args.namespace}, "file", path.name, path.read_bytes())
        out = _request(f"{args.base}/docs/upload", body, {**headers, "Content-Type": ctype})
        state = "duplicate (already loaded)" if out.get("duplicate_of") else f"{out.get('chunks')} chunks"
        print(f"{path.name}: {state} doc_id={out.get('doc_id')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
