-- Case management (owner's request 2026-11-07): where a matter stands in its pipeline, what has happened on it, and
-- the plan of work. The stages of each track are fixed in code (matters/pipeline.py). Updates are written by the
-- matter's people and never changed: a mistake is corrected by a new update that names the one it corrects. Tasks,
-- key dates and hearings are entered by a person, never set from what a model read (CLAUDE.md rule 7).
ALTER TABLE matters ADD COLUMN description TEXT;
ALTER TABLE matters ADD COLUMN track TEXT NOT NULL DEFAULT 'advisory' CHECK (track IN ('disputes','transactions','advisory'));
ALTER TABLE matters ADD COLUMN stage TEXT;
ALTER TABLE matters ADD COLUMN stage_since TEXT;
UPDATE matters SET track = CASE
    WHEN lower(COALESCE(practice_area,'')) LIKE '%litigat%' OR lower(COALESCE(practice_area,'')) LIKE '%disput%'
      OR lower(COALESCE(practice_area,'')) LIKE '%employment%' OR lower(COALESCE(practice_area,'')) LIKE '%arbitrat%'
      OR lower(COALESCE(practice_area,'')) LIKE '%insolvenc%' THEN 'disputes'
    WHEN lower(COALESCE(practice_area,'')) LIKE '%corporate%' OR lower(COALESCE(practice_area,'')) LIKE '%m&a%'
      OR lower(COALESCE(practice_area,'')) LIKE '%commercial%' OR lower(COALESCE(practice_area,'')) LIKE '%real estate%'
      OR lower(COALESCE(practice_area,'')) LIKE '%property%' OR lower(COALESCE(practice_area,'')) LIKE '%finance%'
      OR lower(COALESCE(practice_area,'')) LIKE '%transaction%' THEN 'transactions'
    ELSE 'advisory' END;
UPDATE matters SET stage = CASE track WHEN 'disputes' THEN 'intake' ELSE 'instruction' END, stage_since = created_at;

CREATE TABLE legal_matter_updates(
  id           TEXT PRIMARY KEY,
  org_id       TEXT NOT NULL,
  matter_id    TEXT NOT NULL REFERENCES matters(id),
  author_id    TEXT NOT NULL,
  kind         TEXT NOT NULL CHECK (kind IN ('note','call','meeting','filing','hearing','order','instruction','milestone','stage')),
  body         TEXT NOT NULL,
  occurred_on  TEXT NOT NULL,
  document_id  TEXT,
  corrects_id  TEXT REFERENCES legal_matter_updates(id),
  stage_from   TEXT,
  stage_to     TEXT,
  created_at   TEXT NOT NULL
);
CREATE INDEX ix_matter_updates ON legal_matter_updates(matter_id, occurred_on, created_at);
-- Never changed or removed; the only change allowed is unlinking a document that records deletion removed.
CREATE TRIGGER matter_updates_append_only BEFORE UPDATE ON legal_matter_updates
  WHEN NEW.body IS NOT OLD.body OR NEW.kind IS NOT OLD.kind OR NEW.author_id IS NOT OLD.author_id
    OR NEW.matter_id IS NOT OLD.matter_id OR NEW.occurred_on IS NOT OLD.occurred_on OR NEW.corrects_id IS NOT OLD.corrects_id
    OR NEW.stage_from IS NOT OLD.stage_from OR NEW.stage_to IS NOT OLD.stage_to OR NEW.created_at IS NOT OLD.created_at
    OR (NEW.document_id IS NOT OLD.document_id AND NEW.document_id IS NOT NULL)
  BEGIN SELECT RAISE(ABORT, 'matter updates are append-only'); END;
CREATE TRIGGER matter_updates_no_delete BEFORE DELETE ON legal_matter_updates
  BEGIN SELECT RAISE(ABORT, 'matter updates are append-only'); END;

CREATE TABLE legal_matter_tasks(
  id           TEXT PRIMARY KEY,
  org_id       TEXT NOT NULL,
  matter_id    TEXT NOT NULL REFERENCES matters(id),
  title        TEXT NOT NULL,
  kind         TEXT NOT NULL CHECK (kind IN ('task','key_date','hearing','filing')),
  due_on       TEXT,
  assignee_id  TEXT,
  stage        TEXT,
  status       TEXT NOT NULL CHECK (status IN ('open','done','cancelled')),
  source       TEXT NOT NULL CHECK (source IN ('person','standard')),
  created_by   TEXT NOT NULL,
  created_at   TEXT NOT NULL,
  updated_at   TEXT NOT NULL,
  version      INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX ix_matter_tasks ON legal_matter_tasks(matter_id, status, due_on);
CREATE TABLE legal_matter_task_events(
  id           TEXT PRIMARY KEY,
  task_id      TEXT NOT NULL REFERENCES legal_matter_tasks(id),
  matter_id    TEXT NOT NULL,
  action       TEXT NOT NULL CHECK (action IN ('created','changed','done','reopened','cancelled')),
  changes_json TEXT NOT NULL DEFAULT '{}',
  by           TEXT NOT NULL,
  at           TEXT NOT NULL
);
CREATE INDEX ix_matter_task_events ON legal_matter_task_events(task_id, at);
CREATE TRIGGER matter_task_events_no_update BEFORE UPDATE ON legal_matter_task_events
  BEGIN SELECT RAISE(ABORT, 'task events are append-only'); END;
CREATE TRIGGER matter_task_events_no_delete BEFORE DELETE ON legal_matter_task_events
  BEGIN SELECT RAISE(ABORT, 'task events are append-only'); END;
