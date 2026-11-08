-- The Council's discussion flow (2026-11-07): where a run has got to, so the page's stepper follows the run itself
-- (answers, challenge, reply, chair) instead of guessing. Only the step and the roles that have finished it are kept,
-- never what anyone wrote: the answers stay in the review's artifact, under its own visibility rules.
ALTER TABLE legal_panel_reviews ADD COLUMN progress_json TEXT NOT NULL DEFAULT '{}';
