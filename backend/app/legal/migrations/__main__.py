"""Law-schema migration CLI.

    python -m app.legal.migrations status
    python -m app.legal.migrations up [--to N]
    python -m app.legal.migrations down --to N     # reverts every applied migration above N

Uses the database at settings.DB_PATH (ECHOMIND_DATA_DIR). Take a backup before `down` on any
non-development database.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from contextlib import closing

from . import runner


def main(argv=None) -> int:
    from ...core.config import settings

    ap = argparse.ArgumentParser(prog="python -m app.legal.migrations")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    up = sub.add_parser("up")
    up.add_argument("--to", type=int, default=None)
    down = sub.add_parser("down")
    down.add_argument("--to", type=int, required=True)
    args = ap.parse_args(argv)

    try:
        with closing(sqlite3.connect(settings.DB_PATH)) as conn:
            if args.cmd == "status":
                for row in runner.status(conn):
                    print(f"{row['version']:04d}  {row['name']:<32} {row['state']}")
            elif args.cmd == "up":
                print("applied:", runner.migrate_up(conn, target=args.to) or "nothing")
            else:
                print("reverted:", runner.migrate_down(conn, target=args.to) or "nothing")
    except runner.MigrationError as e:
        print(f"migration error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
