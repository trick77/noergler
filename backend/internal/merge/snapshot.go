package merge

import (
	"slices"

	"github.com/trick77/noergler/internal/snapshot"
)

// Snapshot reports a target or source line mentioning an unreleased build
// (snapshot.Mentions, the loose rule), in any file. Never resolved or
// offered (user, 2026-11-15: "snapshot darf nicht in den master"): a
// resolution makes the PR mergeable. Base, Head and Tail are left out: a
// resolution never picks the base, and both sides already have Head and
// Tail. A stray match fails safe, nothing pushed.
//
// Its own file, not merge.go: TestAgreesWithGit merges edits of merge.go's
// first lines, so an edit there changes what that test checks.
func (c *Conflict) Snapshot() bool {
	for _, l := range slices.Concat(c.Target, c.Source) {
		if snapshot.Mentions(l) {
			return true
		}
	}
	return false
}
