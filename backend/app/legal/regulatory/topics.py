"""Which open matters a regulatory update may concern (D-068): a plain rule a lawyer can check, never a model.

An update matches a matter when the matter's jurisdiction is in the update's (England & Wales, Scotland and
Northern Ireland are UK; a US state or "United States" is US) and a term of the matter's practice area appears
as a whole word (or its plural) in the update's title, summary, agencies or topics. The match keeps the terms found, which the
page shows as the reason. A practice area with no terms below matches nothing.
"""
from __future__ import annotations

import re
from typing import Optional

_UK = ("england", "wales", "scotland", "northern ireland", "united kingdom", "uk", "great britain", "gb")
_US_STATES = ("alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware", "florida", "georgia",
              "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts",
              "michigan", "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada", "new hampshire", "new jersey",
              "new mexico", "new york", "north carolina", "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island",
              "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont", "virginia", "washington", "west virginia",
              "wisconsin", "wyoming", "district of columbia")
_US = ("united states", "usa", "us", "u.s.", "federal") + _US_STATES

# Practice area (any of its words) -> terms looked for in an update. Specific legal words and phrases only: the first
# real import (D-068) showed generic ones ("commercial", "enforce", "freight", "data") matching fishery quotas, Coast Guard
# safety zones and rail notices. To be reviewed with the pilot's lawyers (P-008); `python -m app.legal.regulatory rematch`
# applies a change to the undecided matches.
AREAS: dict[tuple[str, ...], tuple[str, ...]] = {
    ("employment", "labour", "labor"): ("employment", "employer", "employee", "worker", "minimum wage", "national living wage", "wage",
                                        "pension", "redundancy", "discrimination", "equality act", "whistleblowing", "working time",
                                        "statutory sick pay", "parental leave", "occupational safety", "unfair dismissal", "trade union",
                                        "labor standards", "labour market"),
    ("real estate", "property", "lease", "leasing"): ("lease", "leasehold", "landlord", "tenant", "tenancy", "rent", "renters",
                                                      "housing", "land registration", "mortgage", "conveyancing", "commonhold",
                                                      "real estate", "building safety", "town and country planning", "real property"),
    ("technology", "data", "privacy", "ip"): ("data protection", "personal data", "personal information", "privacy", "cybersecurity",
                                              "cyber security", "artificial intelligence", "breach notification", "information security",
                                              "telecommunications", "electronic communications", "online safety"),
    ("regulatory", "food", "consumer"): ("food", "food safety", "product safety", "consumer protection", "recall", "labelling",
                                         "labeling", "hygiene", "trading standards", "health and safety", "contaminant"),
    ("corporate", "venture", "securities", "m&a"): ("securities", "company law", "companies act", "investment adviser",
                                                    "investment company", "capital markets", "insolvency", "beneficial ownership",
                                                    "financial services", "corporate governance", "shareholder", "crowdfunding",
                                                    "private fund", "prospectus"),
    ("commercial", "contracts"): ("late payment", "sale of goods", "supply of goods", "consumer credit", "arbitration",
                                  "unfair contract terms", "electronic signatures", "commercial contracts", "consumer contracts"),
    ("litigation", "disputes"): ("civil procedure", "court", "tribunal", "arbitration", "judgment", "limitation period", "class action"),
    ("investigations", "white collar"): ("fraud", "bribery", "money laundering", "sanctions", "whistleblower", "whistleblowing",
                                         "anti-corruption", "foreign corrupt practices", "export controls", "false claims"),
}


def jurisdictions_of(text: Optional[str]) -> set[str]:
    """UK and/or US for a matter's jurisdiction text ("England & Wales", "Delaware", "England & Wales; United States")."""
    t = " " + re.sub(r"[^a-z.]+", " ", (text or "").lower()) + " "
    out = set()
    if any(f" {w} " in t for w in _UK):
        out.add("UK")
    if any(f" {w} " in t for w in _US):
        out.add("US")
    return out


def terms_for(practice_area: Optional[str]) -> tuple[str, ...]:
    words = " " + re.sub(r"[^a-z&]+", " ", (practice_area or "").lower()) + " "
    found: list[str] = []
    for keys, terms in AREAS.items():
        if any(f" {k} " in words for k in keys):
            found += [t for t in terms if t not in found]
    return tuple(found)


def match(update: dict, matter: dict) -> list[str]:
    """The practice-area terms found in the update, or [] when it does not concern the matter."""
    if update["jurisdiction"] not in jurisdictions_of(matter.get("jurisdiction")):
        return []
    text = " ".join([update.get("title") or "", update.get("summary") or "", *update.get("subjects", [])]).lower()
    return [t for t in terms_for(matter.get("practice_area")) if re.search(rf"(?<![a-z]){re.escape(t)}s?(?![a-z])", text)]
