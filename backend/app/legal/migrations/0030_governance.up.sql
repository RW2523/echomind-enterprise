-- S3.8 — governance checks (DOC-07, D-052): a finding can quote a second document (the approval, the delegation of
-- authority, the articles), shown and hidden with it.
ALTER TABLE legal_dd_issues ADD COLUMN related_json TEXT NOT NULL DEFAULT '[]';
