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

# Practice area (any of its words) -> terms looked for in an update. Reviewed with the pilot's lawyers (P-items).
AREAS: dict[tuple[str, ...], tuple[str, ...]] = {
    ("employment", "labour", "labor"): ("employment", "employer", "employers", "employee", "employees", "worker", "workers", "wage", "wages",
                                        "minimum wage", "pay", "pension", "pensions", "redundancy", "discrimination", "equality",
                                        "whistleblowing", "working time", "occupational", "labor", "labour", "unfair dismissal"),
    ("real estate", "property", "lease", "leasing"): ("property", "land", "lease", "leasehold", "landlord", "tenant", "tenancy", "rent",
                                                      "renters", "housing", "building", "buildings", "planning", "mortgage", "conveyancing",
                                                      "commonhold", "real estate"),
    ("technology", "data", "privacy", "ip"): ("data", "data protection", "privacy", "personal information", "cyber", "cybersecurity",
                                              "electronic", "digital", "online", "telecommunications", "artificial intelligence",
                                              "information security", "breach notification"),
    ("regulatory", "food", "consumer"): ("food", "food safety", "product safety", "consumer protection", "recall", "recalls", "labelling",
                                         "labeling", "hygiene", "trading standards", "health and safety", "contaminant", "contaminants"),
    ("corporate", "venture", "securities", "m&a"): ("companies", "company", "corporate", "securities", "investment", "investor", "investors",
                                                    "capital markets", "insolvency", "beneficial ownership", "financial services", "shares",
                                                    "partnership", "partnerships"),
    ("commercial", "contracts"): ("contract", "contracts", "commercial", "late payment", "sale of goods", "supply of goods", "consumer credit",
                                  "arbitration", "carriage", "freight"),
    ("litigation", "disputes"): ("civil procedure", "court", "courts", "tribunal", "tribunals", "arbitration", "judgment", "judgments",
                                 "limitation", "costs"),
    ("investigations", "white collar"): ("investigation", "investigations", "enforcement", "fraud", "bribery", "money laundering",
                                         "sanctions", "whistleblowing", "penalties"),
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
