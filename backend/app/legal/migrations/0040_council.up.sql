-- EchoMind Counsel Intelligence, iteration 1 (owner's request 2026-11-05): the review panel becomes the Council. Which
-- perspectives answer is chosen by rules from the question (panel/run.py `assemble`) when it is asked, and kept with it,
-- so a rerun with the lawyer's facts asks the same ones. Depth: "focused" (answers, one challenge round, the chair) or
-- "deep" (also a rebuttal by the first perspective). Reviews asked before this read as the four default perspectives.
ALTER TABLE legal_panel_reviews ADD COLUMN roles_json TEXT NOT NULL DEFAULT '["deal","risk","litigation","specialist"]';
ALTER TABLE legal_panel_reviews ADD COLUMN depth TEXT NOT NULL DEFAULT 'focused' CHECK(depth IN ('focused','deep'));
