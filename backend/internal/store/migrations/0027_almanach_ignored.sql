-- Almanach answers 202 "ignored" for a note of a repo that is not one of
-- its source repositories: taken, so marked emitted and never resent, but
-- kept by neither side. The flag lets the Teams page count those notes
-- (user, 2026-11-03: the page stayed green while every note of a team was
-- dropped). Rows emitted before carry false: their answer was not read.
ALTER TABLE almanach_notes ADD COLUMN almanach_ignored BOOLEAN NOT NULL DEFAULT false;
-- Final notes are never deleted, so the table only grows; the Teams page
-- counts a handful of ignored rows per poll and must not scan the rest.
CREATE INDEX almanach_notes_ignored_idx ON almanach_notes (almanach_emitted_at) WHERE almanach_ignored;
