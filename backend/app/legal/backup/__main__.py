"""Backup CLI (see app/legal/backup/__init__.py).

    python -m app.legal.backup keygen  --out KEYFILE
    python -m app.legal.backup create  --out ARCHIVE --key-file KEYFILE
    python -m app.legal.backup verify  --archive ARCHIVE --key-file KEYFILE
    python -m app.legal.backup restore --archive ARCHIVE --key-file KEYFILE --target NEW_DATA_DIR

create uses the data folder and database of this installation (ECHOMIND_DATA_DIR). Keep the backup
key and the master key apart from the archives and from each other.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import BackupError, create, generate_key, restore, verify


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.legal.backup")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("keygen").add_argument("--out", required=True)
    c = sub.add_parser("create")
    c.add_argument("--out", required=True)
    c.add_argument("--key-file", required=True)
    for name in ("verify", "restore"):
        p = sub.add_parser(name)
        p.add_argument("--archive", required=True)
        p.add_argument("--key-file", required=True)
        if name == "restore":
            p.add_argument("--target", required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "keygen":
            print(f"backup key written to {generate_key(args.out)} (mode 600). Store it apart from the archives and the master key.")
            return 0
        if args.cmd == "create":
            result = create(args.out, args.key_file)
        elif args.cmd == "verify":
            m = verify(args.archive, args.key_file)
            result = {"ok": True, "created_at": m["created_at"], "files": len(m["files"]), "schema": m["schema"],
                      "audit": m["audit"], "counts": m["counts"]}
        else:
            result = restore(args.archive, args.key_file, args.target)
    except BackupError as e:
        print(f"backup error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
