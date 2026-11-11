-- The +/- lines of the files the prompt actually carried: lines_added and
-- lines_removed count every reviewable file, size-gated and over-budget ones
-- included, which the model never saw. The metrics page's "lines reviewed"
-- reads these (user, 2026-11-11). NULL on runs from before this column: the
-- stored run detail cannot tell a sent deleted or renamed file from one
-- compression only named, so no backfill.
ALTER TABLE review_runs ADD COLUMN reviewed_lines_added INTEGER;
ALTER TABLE review_runs ADD COLUMN reviewed_lines_removed INTEGER;
