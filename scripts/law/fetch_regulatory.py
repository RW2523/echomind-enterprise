#!/usr/bin/env python3
"""Fetch new public legislation and rules for EchoMind Law's regulatory updates (S4.4, OPT-04; decision D-068).

    python3 scripts/law/fetch_regulatory.py --out BUNDLE.json [--since YYYY-MM-DD]

Runs on the host, never in the Law containers (they stay offline): the daily job (scripts/law/regulatory_daily.sh)
fetches, then imports the bundle into the pilot with `python -m app.legal.cli import-regulatory`. Owner-approved
daily network use (2026-10-12), official public sources only, nothing sent but the request:
- UK: new legislation from legislation.gov.uk (Atom feed), under the Open Government Licence v3.0.
- US: rules and proposed rules from the Federal Register API (federalregister.gov): US Government works, public
  domain (17 U.S.C. 105).
The bundle holds titles, short summaries, dates, kinds, agencies and official links only, never the documents.
Everything in it is untrusted text: the importer checks every field again.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
import sys
import urllib.parse
import urllib.request

from lxml import etree

AGENT = "EchoMind-Law-regulatory-fetch/1.0 (daily, owner-approved)"
MAX_BYTES = 4 * 1024 * 1024
UK_FEED = "https://www.legislation.gov.uk/new/data.feed"
US_API = "https://www.federalregister.gov/api/v1/documents.json"
OGL = ("Contains public sector information licensed under the Open Government Licence v3.0 "
       "(https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).")
SOURCES = {
    "legislation.gov.uk": {"jurisdiction": "UK", "licence": "OGL-3.0", "attribution": OGL},
    "federalregister.gov": {"jurisdiction": "US", "licence": "public-domain",
                            "attribution": "Federal Register, a US Government work in the public domain (17 U.S.C. 105)."},
}
_ATOM = {"a": "http://www.w3.org/2005/Atom", "ukm": "http://www.legislation.gov.uk/namespaces/metadata"}
_UK_KINDS = {"UnitedKingdomPublicGeneralAct": "UK Act", "UnitedKingdomStatutoryInstrument": "UK Statutory Instrument",
             "ScottishAct": "Act of the Scottish Parliament", "ScottishStatutoryInstrument": "Scottish Statutory Instrument",
             "WelshParliamentAct": "Act of Senedd Cymru", "WelshStatutoryInstrument": "Welsh Statutory Instrument",
             "NorthernIrelandAct": "Northern Ireland Act", "NorthernIrelandStatutoryRule": "Northern Ireland Statutory Rule",
             "UnitedKingdomChurchMeasure": "Church Measure", "UnitedKingdomMinisterialOrder": "UK Ministerial Order"}
_US_KINDS = {"Rule": "Final rule", "Proposed Rule": "Proposed rule"}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": AGENT}), timeout=60) as r:
        data = r.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError(f"response over {MAX_BYTES} bytes: {url}")
    return data


def _https(url: str, host: str) -> str:
    """The official https address, or an error: links shown to lawyers must go only to the source."""
    u = urllib.parse.urlsplit((url or "").strip())
    if u.hostname not in (host, "www." + host):
        raise ValueError(f"unexpected link host: {u.hostname}")
    return urllib.parse.urlunsplit(("https", "www." + host, u.path, u.query, ""))


def uk_entries(raw: bytes, since: dt.date) -> tuple[list[dict], bool]:
    """Entries published on or after `since`; and whether older ones were reached (no need for more pages)."""
    root = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False))
    out, older = [], False
    for e in root.findall("a:entry", _ATOM):
        published = (e.findtext("a:published", default="", namespaces=_ATOM) or "")[:10]
        try:
            day = dt.date.fromisoformat(published)
        except ValueError:
            continue
        if day < since:
            older = True
            continue
        ident = (e.findtext("a:id", default="", namespaces=_ATOM) or "").strip()
        link = next((x.get("href") for x in e.findall("a:link", _ATOM) if x.get("rel") is None), None) or ident
        kind = e.find("ukm:DocumentMainType", _ATOM)
        ext = re.sub(r"^https?://(www\.)?legislation\.gov\.uk/(id/)?", "", ident)     # uksi/2026/901, or a correction slip's path
        out.append({"source": "legislation.gov.uk", "external_id": ext, "title": e.findtext("a:title", "", _ATOM),
                    "summary": e.findtext("a:summary", "", _ATOM) or "", "url": _https(link, "legislation.gov.uk"),
                    "published": day.isoformat(), "kind": _UK_KINDS.get(kind.get("Value") if kind is not None else "", "UK legislation"),
                    "subjects": []})
    return out, older


def us_page(raw: bytes) -> tuple[list[dict], str | None]:
    d = json.loads(raw)
    out = []
    for r in d.get("results") or []:
        out.append({"source": "federalregister.gov", "external_id": str(r.get("document_number") or ""), "title": r.get("title") or "",
                    "summary": r.get("abstract") or "", "url": _https(r.get("html_url") or "", "federalregister.gov"),
                    "published": str(r.get("publication_date") or "")[:10], "kind": _US_KINDS.get(r.get("type"), r.get("type") or "Rule"),
                    "subjects": sorted({a.get("name") for a in r.get("agencies") or [] if a.get("name")} | set(r.get("topics") or []))})
    nxt = d.get("next_page_url")
    return out, (_https(nxt, "federalregister.gov") if nxt else None)


def us_first_url(since: dt.date) -> str:
    q = [("per_page", "100"), ("order", "newest"), ("conditions[publication_date][gte]", since.isoformat()),
         ("conditions[type][]", "RULE"), ("conditions[type][]", "PRORULE")]
    q += [("fields[]", f) for f in ("title", "abstract", "html_url", "publication_date", "agencies", "type", "document_number", "topics")]
    return US_API + "?" + urllib.parse.urlencode(q)


def gather(since: dt.date, max_pages: int = 6) -> dict:
    items, errors = [], []
    try:
        for page in range(1, max_pages + 1):
            got, older = uk_entries(fetch(UK_FEED + (f"?page={page}" if page > 1 else "")), since)
            items += got
            if older:
                break
    except Exception as e:                                   # one source failing never stops the other
        errors.append({"source": "legislation.gov.uk", "error": type(e).__name__})
    try:
        url: str | None = us_first_url(since)
        for _ in range(max_pages):
            if not url:
                break
            got, url = us_page(fetch(url))
            items += got
    except Exception as e:
        errors.append({"source": "federalregister.gov", "error": type(e).__name__})
    seen, unique = set(), []
    for it in items:
        key = (it["source"], it["external_id"])
        if it["external_id"] and key not in seen:
            seen.add(key)
            unique.append(it)
    return {"format": "echomind-law-regulatory/1", "fetched_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "since": since.isoformat(), "sources": SOURCES, "errors": errors, "items": unique}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--since", help="first publication date (default: three days ago, so a missed day is caught up)")
    args = ap.parse_args(argv)
    since = dt.date.fromisoformat(args.since) if args.since else dt.date.today() - dt.timedelta(days=3)
    bundle = gather(since)
    data = json.dumps(bundle, indent=1, ensure_ascii=False).encode()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    counts = {s: sum(1 for i in bundle["items"] if i["source"] == s) for s in SOURCES}
    print(json.dumps({"out": str(out), "sha256": hashlib.sha256(data).hexdigest(), "items": counts, "errors": bundle["errors"]}))
    return 1 if len(bundle["errors"]) == len(SOURCES) else 0


if __name__ == "__main__":
    sys.exit(main())
