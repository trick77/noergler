-- The model now answers once per change, naming every screenshot that shows
-- it, so screenshots of one change share a sentence and fold into one line
-- (user, 2026-11-02: eight screenshots of one backlink change were listed
-- eight times, each sentence reworded for its screen). Lines stored before
-- were written one screenshot at a time and would never fold; dropping them
-- sends those pairs once more. Only for a PR whose lines fit one call (10
-- pairs): more are re-sent over several pushes, the rest reading "not
-- compared yet" meanwhile, and calls apart never share a sentence anyway.
-- '' rows (same pixels, nothing visible) stay: grouping changes nothing
-- about them.
DELETE FROM visual_changes
WHERE change_text <> ''
  AND pull_request_id IN (
    SELECT pull_request_id FROM visual_changes
    WHERE change_text <> ''
    GROUP BY pull_request_id
    HAVING count(*) <= 10
  );
