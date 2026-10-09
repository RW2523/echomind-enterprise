"""Organisations (tenants). A Law deployment normally serves one firm: the first boot creates a
default organisation (LEGAL_ORG_NAME) and every account is attached to it. All Law data is scoped
by org_id so a second organisation on the same installation sees nothing of the first."""
from __future__ import annotations

import os
import time
import uuid

from ..core.db import get_conn
from .identity.sessions import _iso


def default_org_id() -> str:
    """The deployment's organisation, created on first use."""
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM organizations ORDER BY created_at LIMIT 1").fetchone()
        if row:
            return row[0]
        oid = str(uuid.uuid4())
        conn.execute("INSERT INTO organizations (id, name, created_at, created_by) VALUES (?,?,?,?)",
                     (oid, os.getenv("LEGAL_ORG_NAME", "Law Firm"), _iso(time.time()), "system"))
        conn.commit()
        return oid


def ensure_default_org() -> str:
    """Create the organisation if needed and attach accounts that have none (called at start-up)."""
    oid = default_org_id()
    with get_conn() as conn:
        conn.execute("UPDATE users SET org_id=? WHERE org_id IS NULL OR org_id=''", (oid,))
        conn.commit()
    return oid
