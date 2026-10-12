#!/usr/bin/env python3
"""Load the demo data into the EchoMind Law pilot stack (D-061, D-064).

    python3 scripts/law/demo_load.py                 # through the pilot's nginx, http://localhost:3110/api

Needs the demo world first:
    scripts/law/compose.sh exec -T -e LEGAL_ALLOW_DEMO_SEED=1 backend python3 -m app.legal.cli seed-demo

Then it adds (D-064), through `seed-demo-practice` in the backend container:
- the practice group and six fictional cases (eval/fixtures/law/synthetic, written by make_synthetic_cases.py);
- M-130, the fictional evaluation corpus (the base data set), with the MSA's signed and draft versions;
- R-001, the reference library of REAL public law (eval/fixtures/law/reference, fetched by fetch_reference.py),
  readable by every lawyer in the demo.

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
CORPUS = ROOT / "eval" / "law" / "corpus"
REF = FIX / "reference"
SYN = FIX / "synthetic"
LAWYERS = ["alice", "bob", "carol", "nakamura", "harper", "sofia", "kenji", "lena", "omar", "theo"]
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


def plan() -> tuple[dict, dict]:
    """The practice plan for seed-demo-practice, and the documents each new matter gets: number -> (lead, files)."""
    syn = json.loads((SYN / "manifest.json").read_text())
    matters, docs = [], {}
    for c in syn["cases"]:
        matters.append({k: c[k] for k in ("number", "title", "client", "practice_area", "jurisdiction", "lead", "members", "parties")})
        docs[c["number"]] = (c["lead"], [(SYN / d["file"], d["kind"], d["status"], None) for d in c["documents"]])
    corpus = json.loads((CORPUS / "manifest.json").read_text())["documents"]
    matters.append({"number": "M-130", "title": "Harrowgate platform — renewal file (base corpus)",
                    "client": {"name": "Voss-Lindqvist AB", "code": "VOSS"}, "practice_area": "Commercial contracts",
                    "jurisdiction": "Delaware", "lead": "alice",
                    "members": {"carol": "paralegal", "harper": "attorney", "nakamura": "reviewer"},
                    "parties": [["Harrowgate Systems Inc.", "adverse"]]})
    docs["M-130"] = ("alice", [(CORPUS.parent / d["file"], d["kind"], d["status"], d.get("version_of"), d["title"], d["key"])
                               for d in corpus if d["matter"] == "eval"])
    docs["M-215"] = ("bob", [(CORPUS.parent / d["file"], d["kind"], d["status"], None, d["title"], d["key"])
                             for d in corpus if d["matter"] == "forbidden"])
    matters.append({"number": "R-001", "title": "Reference library — public law (real, not fictional)",
                    "client": {"name": "Firm knowledge (internal)", "code": "KNOW"}, "practice_area": "Knowledge",
                    "jurisdiction": "England & Wales; United States", "lead": "morgan",
                    "members": {u: "read_only" for u in LAWYERS}, "parties": []})
    docs["R-001"] = ("morgan", [(f, "other", "final", None) for f in sorted(REF.glob("*.txt"))])
    return {"team": syn["team"], "matters": matters}, docs


def _seed_practice(container: str, practice: dict) -> None:
    r = subprocess.run(["docker", "exec", "-i", "-e", "LEGAL_ALLOW_DEMO_SEED=1", container, "python3", "-m", "app.legal.cli",
                        "seed-demo-practice"], input=json.dumps(practice), capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"seed-demo-practice failed: {r.stderr.strip()[-300:]}")
    print(r.stdout.strip().splitlines()[-1])


def _credentials(container: str) -> dict:
    out = subprocess.run(["docker", "exec", container, "cat", "/data/legal/demo_credentials.json"],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def _multipart(fields: dict, path: pathlib.Path) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    fields = {k: v for k, v in fields.items() if v is not None}
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
    practice, more = plan()
    _seed_practice(args.container, practice)
    creds = _credentials(args.container)
    sessions = {}
    work = {n: (u, [(f, k, s, None) for f, k, s in files]) for n, (u, files) in DEMO.items()}
    work.update(more)

    def session(user: str):
        if user not in sessions:
            os.environ["LAW_TOTP_SECRET"] = creds[user]["totp_secret"]
            sessions[user] = open_session(args.base, user, creds[user]["password"])
            os.environ.pop("LAW_TOTP_SECRET", None)
        return sessions[user]

    def get(user: str, path: str) -> dict:
        with session(user).open(urllib.request.Request(args.base + path), timeout=60) as r:
            return json.loads(r.read() or b"{}")

    for number, (user, files) in work.items():
        matters = {m["number"]: m for m in get(user, "/legal/matters")["matters"]}
        mid = matters[number]["id"]
        listed = get(user, f"/legal/matters/{urllib.parse.quote(mid)}/documents")["documents"]
        have = {d["title"]: d for d in listed}
        keys = {}
        for entry in files:
            path, kind, status, version_of = entry[:4]
            title = entry[4] if len(entry) > 4 else path.stem
            key = entry[5] if len(entry) > 5 else None
            parent = keys.get(version_of) if version_of else None
            if title in have and not parent:
                keys[key] = have[title]["id"]
                print(f"{number}: {title} already there")
                continue
            if parent and (have.get(title, {}).get("current_version") or {}).get("number", 1) >= 2:
                print(f"{number}: {title} (new version) already there")
                continue
            body, ctype = _multipart({"title": None if parent else title, "kind": kind, "status": status,
                                      "version_of": parent}, path)
            req = urllib.request.Request(f"{args.base}/legal/matters/{urllib.parse.quote(mid)}/documents/import",
                                         data=body, method="POST", headers={"Content-Type": ctype})
            with session(user).open(req, timeout=120) as r:
                made = json.loads(r.read() or b"{}")
            keys[key] = (made.get("document") or {}).get("id")
            print(f"{number}: imported {title}{' (new version)' if parent else ''} ({kind}, {status})")

    deadline = time.time() + args.wait
    while True:
        pending, failed = [], []
        for number, (user, _) in work.items():
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
