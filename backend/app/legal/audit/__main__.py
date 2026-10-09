"""Audit-trail CLI.

    python -m app.legal.audit verify     # recompute the chain and check the anchors; exit 1 if broken
    python -m app.legal.audit anchor     # write today's anchor for the current head now

Uses the database at settings.DB_PATH (ECHOMIND_DATA_DIR).
"""
from __future__ import annotations

import argparse
import json
import sys

from . import chain


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.legal.audit")
    ap.add_argument("cmd", choices=["verify", "anchor"])
    args = ap.parse_args(argv)
    if args.cmd == "verify":
        result = chain.verify()
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    path = chain.anchor_now()
    print(path or "no events yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
