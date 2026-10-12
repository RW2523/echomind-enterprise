-- S4.4 — regulatory updates (OPT-04, D-068). New public legislation and rules, fetched daily on the host from
-- official sources (legislation.gov.uk, OGL; the Federal Register, public domain) and imported as data: titles,
-- short summaries, dates and official links, never the documents. Each update is matched to open matters by
-- jurisdiction and practice-area terms; a lawyer marks a match reviewed or not relevant (with a reason).
CREATE TABLE legal_regulatory_updates(
  id            TEXT PRIMARY KEY,
  org_id        TEXT NOT NULL,
  source        TEXT NOT NULL CHECK(source IN ('legislation.gov.uk','federalregister.gov')),
  external_id   TEXT NOT NULL,
  jurisdiction  TEXT NOT NULL CHECK(jurisdiction IN ('UK','US')),
  kind          TEXT NOT NULL,
  title         TEXT NOT NULL,
  summary       TEXT NOT NULL DEFAULT '',
  url           TEXT NOT NULL,
  published     TEXT NOT NULL,
  subjects_json TEXT NOT NULL DEFAULT '[]',
  licence       TEXT NOT NULL,
  bundle_sha256 TEXT NOT NULL,
  imported_at   TEXT NOT NULL,
  UNIQUE(org_id, source, external_id)
);
CREATE INDEX ix_regulatory_published ON legal_regulatory_updates(org_id, published);

CREATE TABLE legal_regulatory_matches(
  update_id   TEXT NOT NULL REFERENCES legal_regulatory_updates(id),
  matter_id   TEXT NOT NULL REFERENCES matters(id),
  terms_json  TEXT NOT NULL,
  matched_at  TEXT NOT NULL,
  status      TEXT NOT NULL DEFAULT 'new' CHECK(status IN ('new','reviewed','not_relevant')),
  decided_by  TEXT,
  decided_at  TEXT,
  note        TEXT,
  version     INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY(update_id, matter_id)
);
CREATE INDEX ix_regulatory_matches_matter ON legal_regulatory_matches(matter_id, status);
