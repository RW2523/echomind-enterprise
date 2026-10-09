"""EchoMind Law developer CLI.

    LEGAL_ALLOW_DEMO_SEED=1 python -m app.legal.cli seed-demo

seed-demo creates the FICTIONAL demonstration world used by the end-to-end tests and design
reviews (IMPLEMENTATION_PLAN §8): three colleagues and four matters for two clients, including a
highly restricted matter that only its lead can see, a revoked and an expired membership. Demo
passwords and authenticator keys are written to $ECHOMIND_DATA_DIR/legal/demo_credentials.json
(mode 0600). It refuses to run unless LEGAL_ALLOW_DEMO_SEED=1, so it cannot be run against a
firm's real installation by accident. Re-running it changes nothing.
"""
from __future__ import annotations

import json
import os
import secrets
import sys
from datetime import datetime, timedelta, timezone

DEMO_USERS = {  # username: firm roles
    "alice": ["attorney"],
    "bob": ["attorney"],
    "carol": ["paralegal"],
}


def _ts(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def seed_demo() -> dict:
    from ..core import auth
    from ..core.config import settings
    from ..core.db import get_conn, init_db
    from .identity import accounts, mfa
    from .matters import repo
    from .orgs import default_org_id

    init_db()
    org = default_org_id()
    path = os.path.join(settings.DATA_DIR, "legal", "demo_credentials.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)

    creds, ids = {}, {}
    for name, roles in DEMO_USERS.items():
        if auth.get_user_by_username(name):
            raise SystemExit(f"user {name!r} already exists but no demo credentials file was found; refusing to touch it")
        pw = "demo " + secrets.token_urlsafe(12)
        uid = auth.create_user(name, pw)["id"]
        for r in roles:
            accounts.grant_role(uid, r, granted_by="seed-demo")
        secret = mfa.begin_enrollment(uid)
        with get_conn() as conn:
            conn.execute("UPDATE users SET mfa_enabled=1 WHERE id=?", (uid,))
            conn.commit()
        creds[name] = {"password": pw, "totp_secret": secret}
        ids[name] = uid

    alpha = repo.create_client(org, "Voss-Lindqvist AB", "VOSS", ids["alice"])
    alder = repo.create_client(org, "Alder Finch Ltd", "ALDER", ids["alice"])
    beta = repo.create_client(org, "Beta Shipping Corp", "BETA", ids["bob"])
    m101 = repo.create_matter(org, alpha["id"], "M-101", "Harrowgate MSA renewal", ids["alice"],
                              practice_area="Commercial contracts", jurisdiction="Delaware")
    repo.create_matter(org, alder["id"], "M-114", "Series B due diligence", ids["alice"],
                       practice_area="Corporate", jurisdiction="England & Wales", sensitivity="standard")
    m202 = repo.create_matter(org, beta["id"], "M-202", "Board investigation", ids["bob"],
                              practice_area="Investigations", jurisdiction="New York", sensitivity="highly_restricted")
    m215 = repo.create_matter(org, beta["id"], "M-215", "Warehouse lease dispute", ids["bob"],
                              practice_area="Real estate litigation", jurisdiction="New York")
    repo.add_membership(m101["id"], ids["carol"], "paralegal", granted_by=ids["alice"])
    gone = repo.add_membership(m215["id"], ids["carol"], "paralegal", granted_by=ids["bob"])
    repo.revoke_membership(m215["id"], gone["id"], by=ids["bob"], reason="moved to another team")
    repo.add_membership(m202["id"], ids["alice"], "attorney", granted_by=ids["bob"], valid_from=_ts(-60), valid_to=_ts(-2))

    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(creds, fh, indent=2)
    return creds


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] != ["seed-demo"]:
        print(__doc__)
        return 2
    from ..core.edition import is_law
    if not is_law():
        print("seed-demo needs ECHOMIND_EDITION=law", file=sys.stderr)
        return 2
    if (os.getenv("LEGAL_ALLOW_DEMO_SEED") or "") != "1":
        print("refusing: set LEGAL_ALLOW_DEMO_SEED=1 to create fictional demo accounts and matters", file=sys.stderr)
        return 2
    creds = seed_demo()
    print(f"demo world ready: users {', '.join(sorted(creds))}; credentials in $ECHOMIND_DATA_DIR/legal/demo_credentials.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
