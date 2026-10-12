"""Importing a regulatory bundle (D-068): every field checked again, new updates stored, then matched to open matters.

The bundle comes from the internet through the host, so nothing in it is trusted: each item must name a known
source, carry an official https link on that source's own site, a valid date and bounded plain text. Licence and
jurisdiction come from the table here, never from the bundle. An item that fails a check is skipped and counted.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import uuid
from datetime import date, timedelta
from typing import Optional
from urllib.parse import urlsplit

from ...core.db import get_conn
from .. import policy
from . import topics

FORMAT = "echomind-law-regulatory/1"
SOURCES = {
    "legislation.gov.uk": {"jurisdiction": "UK", "licence": "OGL-3.0"},
    "federalregister.gov": {"jurisdiction": "US", "licence": "public-domain"},
}
MAX_ITEMS = 5000
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9/._-]{0,99}$")


class Invalid(ValueError):
    pass


def _text(v, limit: int, required: bool = False) -> str:
    if not isinstance(v, str):
        if required:
            raise Invalid("text")
        return ""
    t = "".join(ch if unicodedata.category(ch)[0] != "C" else " " for ch in v)     # no control or format characters
    t = " ".join(t.split())
    if (required and not t) or len(t) > limit:
        raise Invalid("text")
    return t


def clean(item: dict, today: Optional[date] = None) -> dict:
    """One bundle item as stored, or Invalid."""
    if not isinstance(item, dict):
        raise Invalid("item")
    src = item.get("source")
    if src not in SOURCES:
        raise Invalid("source")
    ext = item.get("external_id")
    if not isinstance(ext, str) or not _ID.match(ext):
        raise Invalid("external_id")
    u = urlsplit(item.get("url") if isinstance(item.get("url"), str) else "")
    if u.scheme != "https" or u.hostname != f"www.{src}" or u.port is not None or u.username or u.password:
        raise Invalid("url")
    try:
        published = date.fromisoformat(str(item.get("published") or ""))
    except ValueError:
        raise Invalid("published") from None
    today = today or date.today()
    if not today - timedelta(days=3650) <= published <= today + timedelta(days=60):   # rules are published ahead of time
        raise Invalid("published")
    subjects = item.get("subjects") or []
    if not isinstance(subjects, list) or len(subjects) > 30:
        raise Invalid("subjects")
    return {"source": src, "external_id": ext, "jurisdiction": SOURCES[src]["jurisdiction"], "licence": SOURCES[src]["licence"],
            "title": _text(item.get("title"), 1000, required=True), "summary": _text(item.get("summary"), 4000),
            "kind": _text(item.get("kind"), 80) or "Update", "url": f"https://www.{src}{u.path}" + (f"?{u.query}" if u.query else ""),
            "published": published.isoformat(), "subjects": [s for s in (_text(x, 200) for x in subjects) if s]}


def import_bundle(raw: bytes, org_id: str) -> dict:
    """Store what is new and match it to the org's open matters. Returns counts only."""
    try:
        bundle = json.loads(raw)
    except ValueError:
        raise Invalid("not JSON") from None
    if not isinstance(bundle, dict) or bundle.get("format") != FORMAT or not isinstance(bundle.get("items"), list):
        raise Invalid(f"not a {FORMAT} bundle")
    if len(bundle["items"]) > MAX_ITEMS:
        raise Invalid(f"more than {MAX_ITEMS} items")
    sha = hashlib.sha256(raw).hexdigest()
    now = policy.now_iso()
    counts = {"items": len(bundle["items"]), "skipped": 0, "new": 0, "known": 0, "matches": 0}
    with get_conn() as conn:
        matters = [dict(zip(("id", "practice_area", "jurisdiction"), r)) for r in conn.execute(
            "SELECT id, practice_area, jurisdiction FROM matters WHERE org_id=? AND status='open'", (org_id,))]
        for item in bundle["items"]:
            try:
                u = clean(item)
            except Invalid:
                counts["skipped"] += 1
                continue
            if conn.execute("SELECT 1 FROM legal_regulatory_updates WHERE org_id=? AND source=? AND external_id=?",
                            (org_id, u["source"], u["external_id"])).fetchone():
                counts["known"] += 1
                continue
            uid = str(uuid.uuid4())
            conn.execute("INSERT INTO legal_regulatory_updates (id, org_id, source, external_id, jurisdiction, kind, title, summary, url, published, "
                         "subjects_json, licence, bundle_sha256, imported_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (uid, org_id, u["source"], u["external_id"], u["jurisdiction"], u["kind"], u["title"], u["summary"], u["url"],
                          u["published"], json.dumps(u["subjects"]), u["licence"], sha, now))
            counts["new"] += 1
            for m in matters:
                terms = topics.match(u, m)
                if terms:
                    conn.execute("INSERT INTO legal_regulatory_matches (update_id, matter_id, terms_json, matched_at) VALUES (?,?,?,?)",
                                 (uid, m["id"], json.dumps(terms), now))
                    counts["matches"] += 1
        conn.commit()
    return counts


def rematch(org_id: str, days: int = 90) -> dict:
    """Match the recent updates again, after the terms change: undecided matches follow the new terms (added, updated
    or removed); a match a lawyer has decided is kept as decided. Returns counts only."""
    from datetime import date as _date
    since = (_date.today() - timedelta(days=days)).isoformat()
    now = policy.now_iso()
    counts = {"updates": 0, "added": 0, "removed": 0, "kept_decided": 0}
    with get_conn() as conn:
        matters = [dict(zip(("id", "practice_area", "jurisdiction"), r)) for r in conn.execute(
            "SELECT id, practice_area, jurisdiction FROM matters WHERE org_id=? AND status='open'", (org_id,))]
        for uid, jurisdiction, kind, title, summary, subjects in conn.execute(
                "SELECT id, jurisdiction, kind, title, summary, subjects_json FROM legal_regulatory_updates WHERE org_id=? AND published >= ?",
                (org_id, since)).fetchall():
            counts["updates"] += 1
            u = {"jurisdiction": jurisdiction, "kind": kind, "title": title, "summary": summary, "subjects": json.loads(subjects)}
            have = dict(conn.execute("SELECT matter_id, status FROM legal_regulatory_matches WHERE update_id=?", (uid,)).fetchall())
            for m in matters:
                terms, status = topics.match(u, m), have.get(m["id"])
                if status not in (None, "new"):
                    counts["kept_decided"] += 1
                elif terms and status is None:
                    conn.execute("INSERT INTO legal_regulatory_matches (update_id, matter_id, terms_json, matched_at) VALUES (?,?,?,?)",
                                 (uid, m["id"], json.dumps(terms), now))
                    counts["added"] += 1
                elif terms:
                    conn.execute("UPDATE legal_regulatory_matches SET terms_json=? WHERE update_id=? AND matter_id=?", (json.dumps(terms), uid, m["id"]))
                elif status == "new":
                    conn.execute("DELETE FROM legal_regulatory_matches WHERE update_id=? AND matter_id=? AND status='new'", (uid, m["id"]))
                    counts["removed"] += 1
        conn.commit()
    return counts

