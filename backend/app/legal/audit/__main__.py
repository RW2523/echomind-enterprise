"""Audit-trail CLI.

    python -m app.legal.audit verify           # recompute the chain and check the anchors; exit 1 if broken
    python -m app.legal.audit anchor           # write today's anchor for the current head now
    python -m app.legal.audit forward --once   # send pending events to LEGAL_AUDIT_SYSLOG now (D-069)

Uses the database at settings.DB_PATH (ECHOMIND_DATA_DIR).
"""
from __future__ import annotations

import argparse
import json
import sys

from . import chain, forward


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.legal.audit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    sub.add_parser("anchor")
    sub.add_parser("forward").add_argument("--once", action="store_true", required=True,
                                           help="send what is pending, then exit (the backend forwards continuously)")
    args = ap.parse_args(argv)
    if args.cmd == "verify":
        result = chain.verify()
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    if args.cmd == "forward":
        total = 0
        try:
            forward.destination()
            while n := forward.forward_pending():
                total += n
        except ValueError as e:
            print(f"audit forwarding: {e}", file=sys.stderr)
            return 1
        except OSError as e:
            print(f"audit forwarding: could not send ({type(e).__name__}); unsent events stay pending", file=sys.stderr)
            return 1
        print(f"forwarded: {total}")
        return 0
    path = chain.anchor_now()
    print(path or "no events yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
