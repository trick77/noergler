-- An unparseable response is billed but writes no run row (review_runs holds
-- successes only), so its cost reached neither the per-PR cap nor anything
-- else: a PR the model kept refusing spent past the cap without limit.
--
-- The cost rides on the attempt instead. Set only for a billed outcome with
-- no run; a successful attempt's cost stays on its run, never counted twice.
-- NULL means unpriced, as on review_runs.
ALTER TABLE review_attempts ADD COLUMN cost_nano_usd BIGINT;
