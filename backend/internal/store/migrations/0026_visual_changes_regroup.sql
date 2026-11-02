-- The model now answers once per change, naming every screenshot that shows
-- it, so screenshots of one change share a sentence and fold into one line
-- (user, 2026-11-02: eight screenshots of one backlink change were listed
-- eight times, each sentence reworded for its screen). Lines stored before
-- were written one screenshot at a time and would never fold; dropping them
-- sends those pairs once more, on the PR's next review. Open PRs only: a
-- closed one is never reviewed again. And only where the stored lines fit
-- one call (10 pairs), a rough cut: more are re-sent over several pushes,
-- reading "not compared yet" meanwhile, and calls apart never share a
-- sentence anyway. '' rows (same pixels, nothing visible) stay: grouping
-- changes nothing about them.
DELETE FROM visual_changes
WHERE change_text <> ''
  AND pull_request_id IN (
    SELECT v.pull_request_id
    FROM visual_changes v JOIN pull_requests p ON v.pull_request_id = p.id
    WHERE v.change_text <> '' AND p.merged_at IS NULL AND p.deleted_at IS NULL
    GROUP BY v.pull_request_id
    HAVING count(*) <= 10
  );
