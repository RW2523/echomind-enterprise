"""Import a regulatory bundle (see app/legal/regulatory/__init__.py).

    python -m app.legal.regulatory import < bundle.json

Run by scripts/law/regulatory_daily.sh through `docker exec -i`. Refuses unless LEGAL_REGULATORY is on. Prints
counts only and records one audit event (regulatory.import) with the same counts.
"""
from __future__ import annotations

import json
import sys


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] != ["import"]:
        print(__doc__)
        return 2
    from ...core.config import settings
    from ...core.edition import is_law
    if not is_law() or not settings.LEGAL_REGULATORY:
        print("refusing: regulatory updates are off (LEGAL_REGULATORY)", file=sys.stderr)
        return 2
    from ..audit.record import record
    from ..orgs import default_org_id
    from .store import Invalid, import_bundle
    raw = sys.stdin.buffer.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024:
        print("refusing: the bundle is over 16 MB", file=sys.stderr)
        return 1
    org = default_org_id()
    try:
        counts = import_bundle(raw, org)
    except Invalid as e:
        record("regulatory.import", "deny", object_type="regulatory_bundle", reason="invalid_bundle")
        print(f"refusing: {e}", file=sys.stderr)
        return 1
    record("regulatory.import", "allow", object_type="regulatory_bundle", details=counts)
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
