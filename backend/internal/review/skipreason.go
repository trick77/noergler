package review

// SkipReason names a pre-flight exit: a decision the pipeline made before the
// gateway was ever called. inference.Outcome covers what happens after, and
// the two never overlap.
//
// These exist so the dashboard can group skips. The strings are stored in
// review_attempts.reason and are read by operators, so they are stable: adding
// one is free, renaming one silently rewrites history.
//
// Deliberately coarse. The log line still carries the specifics (which author,
// which keyword, how many tokens); this is the bucket that line falls in, and
// a reason per call site would make the grouping useless.
type SkipReason string

const (
	// SkipNone is the zero value: the exit was not a skip. prepare's two
	// hard failures (unparseable payload, diff fetch failed) use it, and no
	// attempt row is written for them.
	SkipNone SkipReason = ""

	SkipNotAutoAuthor    SkipReason = "not_auto_review_author"
	SkipIgnoredAuthor    SkipReason = "ignored_author"
	SkipIgnoredPR        SkipReason = "pr_ignored"
	SkipBranchOptOut     SkipReason = "branch_opt_out"
	SkipNoAgentsMD       SkipReason = "agents_md_missing"
	SkipAgentsMDTooLarge SkipReason = "agents_md_too_large"
	SkipCostCap          SkipReason = "pr_cost_cap"
	SkipNoReviewable     SkipReason = "no_reviewable_files"
	SkipEmptyDiff        SkipReason = "empty_diff"
	SkipHeadUnchanged    SkipReason = "head_unchanged"
	SkipDiffTooLarge     SkipReason = "diff_too_large"
)

// Label is the reason as a dashboard prints it. Unknown values print
// themselves, so a row written by a newer binary is never blank.
func (s SkipReason) Label() string {
	switch s {
	case SkipNotAutoAuthor:
		return "Author not in auto-review authors"
	case SkipIgnoredAuthor:
		return "Ignored author"
	case SkipIgnoredPR:
		return "PR ignored (summary comment removed)"
	case SkipBranchOptOut:
		return "Branch opt-out keyword"
	case SkipNoAgentsMD:
		return "AGENTS.md missing"
	case SkipAgentsMDTooLarge:
		return "AGENTS.md over the token cap"
	case SkipCostCap:
		return "PR cost cap reached"
	case SkipNoReviewable:
		return "No reviewable files"
	case SkipEmptyDiff:
		return "Empty diff"
	case SkipHeadUnchanged:
		return "HEAD unchanged since last review"
	case SkipDiffTooLarge:
		return "Diff too large"
	}
	return string(s)
}
