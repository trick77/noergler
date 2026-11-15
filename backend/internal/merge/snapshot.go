package merge

import (
	"slices"

	"github.com/trick77/noergler/internal/snapshot"
)

// Snapshot reports a target or source line mentioning an unreleased build
// (snapshot.Mentions, the loose rule), in any file. Never resolved or
// offered (user, 2026-11-15: "a snapshot must not go into master"): a
// resolution makes the PR mergeable. Base, Head and Tail are left out: a
// resolution never picks the base, and both sides already have Head and
// Tail. A stray match fails safe, nothing pushed.
func (c *Conflict) Snapshot() bool {
	for _, l := range slices.Concat(c.Target, c.Source) {
		if snapshot.Mentions(l) {
			return true
		}
	}
	return false
}
