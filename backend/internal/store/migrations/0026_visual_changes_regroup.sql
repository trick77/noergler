-- The model now answers once per change, naming every screenshot that shows
-- it, so screenshots of one change share a sentence and fold into one line
-- (user, 2026-11-02: eight screenshots of one backlink change were listed
-- eight times, each sentence reworded for its screen). Lines stored before
-- were written one screenshot at a time and would never fold; dropping them
-- sends those pairs once more. '' rows (same pixels, nothing visible) stay:
-- grouping changes nothing about them.
DELETE FROM visual_changes WHERE change_text <> '';
