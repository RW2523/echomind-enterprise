#!/usr/bin/env python3
"""Sign-in details for the FICTIONAL demo people on the EchoMind Law pilot stack (D-061), for the owner to run.

    python3 scripts/law/demo_signin.py            # everyone
    python3 scripts/law/demo_signin.py alice      # one person

Prints each demo person's password and the authenticator code valid right now (codes change every 30 seconds
and each can be used once). They come from the credentials the demo seed wrote inside the pilot backend's data
volume, read through `docker exec`. Demo accounts only: a firm's real people never have a stored key.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from law_client import totp  # noqa: E402

ROLES = {"alice": "attorney, leads M-101 and M-114", "bob": "attorney, leads M-202 and M-215", "carol": "paralegal",
         "morgan": "managing partner", "nakamura": "managing partner, reviewer on M-101", "fiona": "firm administrator",
         "rhea": "records administrator", "iris": "front desk (intake)", "conway": "conflicts team"}


def main() -> int:
    creds = json.loads(subprocess.run(["docker", "exec", "echomind-law-backend", "cat", "/data/legal/demo_credentials.json"],
                                      check=True, capture_output=True, text=True).stdout)
    who = sys.argv[1:] or list(creds)
    left = 30 - int(time.time()) % 30
    print(f"Sign in at https://localhost:3553 (or this server's address on port 3553). Codes valid for {left} more seconds.\n")
    for name in who:
        c = creds[name]
        print(f"{name:<9} {ROLES.get(name, ''):<38} password: {c['password']:<22} code: {totp(c['totp_secret'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
