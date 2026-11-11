-- What an answer's checks found, kept with it (2026-11-12, evaluation fix list items 2 and 3): the sentences left without a
-- verified source after the support check (the history rebuilt them without it, so a reloaded answer could carry a label
-- the live one did not), and the "is it stated?" check's reading of the question and verdict, shown with a withdrawn answer.
ALTER TABLE legal_research_turns ADD COLUMN unsupported_json TEXT;
ALTER TABLE legal_research_turns ADD COLUMN stated_json TEXT;
