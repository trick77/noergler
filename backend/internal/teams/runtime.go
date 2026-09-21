// Package teams is the runtime half of team identity: the per-team clients
// and reviewer, the registry the HTTP routes look slugs up in, and the
// startup reconciliation between teams.yaml and the DB.
//
// The DB half (claims, settings) lives in internal/store. A team's identity
// is its webhook path plus its HMAC secret plus the ownership check; nothing
// here trusts a payload's project key.
package teams

import (
	"context"
	"sync"
	"sync/atomic"

	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/jira"
	"github.com/trick77/noergler/internal/riptide"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// Reviewer is what the runtime needs of a team's review pipeline.
//
// Consumer-side, so the HTTP routes can be tested with a recorder that
// reports which handler the dispatch table picked, without standing up an
// inference client and a store.
type Reviewer interface {
	ReviewPullRequest(ctx context.Context, p *webhook.Payload, skipAuthorCheck bool)
	HandleMention(ctx context.Context, p *webhook.Payload)
	HandleCommentDeleted(ctx context.Context, p *webhook.Payload)
	HandlePRMerged(ctx context.Context, p *webhook.Payload)
	HandlePRDeclined(ctx context.Context, p *webhook.Payload)
	HandlePRDeleted(ctx context.Context, p *webhook.Payload)
	IsAutoReviewAuthor(author string) bool
	SetAuthorLists(auto, ignore []string)
}

// Runtime is one enabled team at request time.
//
// The Bitbucket client, the store and the queue are shared by every team;
// the inference client, Jira and riptide are this team's own.
//
// The team config is copy-on-write: a reader takes one snapshot per request
// with Team(), a writer builds a whole new *config.Team and swaps the
// pointer. Python mutates in place and gets away with it under the GIL; here
// the webhook route reads while the team API writes, so the snapshot is the
// synchronisation.
//
// The swap is a shallow copy of the struct. That is sound because every
// field a writer touches is replaced with a fresh slice header and the
// slices an old snapshot shares are never mutated in place. Do not add a
// writer that appends to a slice already published in a snapshot.
type Runtime struct {
	team atomic.Pointer[config.Team]
	// writeMu serialises the writers. The swap itself is atomic, but a write
	// is a read-modify-write over the whole struct, so two concurrent ones
	// (a settings PUT against an /onboard claim) would otherwise each build
	// on the same snapshot and the later Store would drop the earlier
	// change. Readers never take it.
	writeMu  sync.Mutex
	Reviewer Reviewer
	LLM      *inference.Client
	Jira     *jira.Client
	Riptide  *riptide.Emitter
}

// NewRuntime publishes the first snapshot. The team is copied, so a later
// caller holding the same pointer cannot mutate what readers see.
func NewRuntime(t *config.Team, r Reviewer, llm *inference.Client, jr *jira.Client, rt *riptide.Emitter) *Runtime {
	rtm := &Runtime{Reviewer: r, LLM: llm, Jira: jr, Riptide: rt}
	snapshot := *t
	rtm.team.Store(&snapshot)
	return rtm
}

// Team is the current snapshot.
//
// A caller takes it ONCE per request and uses that one value for every
// decision. Re-reading mid-request would let a concurrent settings write
// land between the ownership check and the exclude check, producing a
// verdict no consistent config ever authorised.
func (r *Runtime) Team() *config.Team { return r.team.Load() }

// ApplyClaims swaps in a new claim set.
//
// The DB is the truth; this makes the in-memory snapshot agree with the
// write that just succeeded. Always called with a freshly re-read
// ListClaims, never with a delta.
func (r *Runtime) ApplyClaims(scopes []config.ProjectScope) {
	r.writeMu.Lock()
	defer r.writeMu.Unlock()
	next := *r.team.Load()
	next.Projects = scopes
	r.team.Store(&next)
}

// ApplySettings swaps in the three lists, then mirrors the two author lists
// onto the live Reviewer.
//
// The Reviewer copies config.Review by value at construction, so without the
// mirror a team's author routing would stay stale until restart. ExcludeRepos
// is deliberately not mirrored: Python does not either, and the only reader
// is the webhook route, off the snapshot.
func (r *Runtime) ApplySettings(s store.TeamSettings) {
	r.writeMu.Lock()
	defer r.writeMu.Unlock()
	next := *r.team.Load()
	next.Review.AutoReviewAuthors = s.AutoReviewAuthors
	next.Review.IgnoreAuthors = s.IgnoreAuthors
	next.Review.ExcludeRepos = s.ExcludeRepos
	r.team.Store(&next)
	r.Reviewer.SetAuthorLists(s.AutoReviewAuthors, s.IgnoreAuthors)
}

// Settings is the current three lists: the baseline a partial PUT merges
// into. Derived, never stored.
func (r *Runtime) Settings() store.TeamSettings {
	t := r.team.Load()
	return store.TeamSettings{
		AutoReviewAuthors: t.Review.AutoReviewAuthors,
		IgnoreAuthors:     t.Review.IgnoreAuthors,
		ExcludeRepos:      t.Review.ExcludeRepos,
	}
}
