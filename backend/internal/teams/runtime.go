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
	"errors"
	"net/http"
	"sync"
	"sync/atomic"

	"github.com/trick77/noergler/internal/almanach"
	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/onboarding"
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
	// ReviewPullRequestStaged is the queued path: the gateway call goes to
	// the inference pool and the posting comes back to the worker. It
	// reports whether that work outlives the call, which is what keeps the
	// PR's queue hold alive across it.
	ReviewPullRequestStaged(ctx context.Context, p *webhook.Payload, team string, skipAuthorCheck bool, sched Scheduler) bool
	// HandleMention reports handoff like the staged review does: a keyword
	// mention stages one, so the PR's hold must outlive the job.
	HandleMention(ctx context.Context, p *webhook.Payload, team string, sched Scheduler) bool
	HandleCommentDeleted(ctx context.Context, p *webhook.Payload)
	// HandleFindingReply answers a reply on one of the bot's finding threads,
	// once per finding, and a reply on the summary as a mention. It decides
	// itself whether the reply qualifies:
	// handled false hands an @mention on to HandleMention. handedOff is the
	// staged review's contract.
	HandleFindingReply(ctx context.Context, p *webhook.Payload, team string, sched Scheduler) (handled, handedOff bool)
	HandlePRMerged(ctx context.Context, p *webhook.Payload)
	HandlePRDeclined(ctx context.Context, p *webhook.Payload)
	HandlePRDeleted(ctx context.Context, p *webhook.Payload)
	// HandleReadyForReview notes open finding threads on the summary when a
	// draft goes ready. No model call.
	HandleReadyForReview(ctx context.Context, p *webhook.Payload)
	IsAutoReviewAuthor(author string) bool
	SetAuthorLists(auto, ignore []string)
	SetUsePRDescription(on bool)
}

// Runtime is one enabled team at request time.
//
// The store and the queue are shared by every team; the inference client and
// riptide are this team's own. Bitbucket and Jira are the shared accounts
// until the team pushes its own (accounts.go).
//
// The team config is copy-on-write: a reader takes one snapshot per request
// with Team(), a writer builds a whole new *config.Team and swaps the
// pointer. The webhook route reads while the team API writes, so the
// snapshot is the synchronisation: mutating in place would let a settings
// write land between the ownership check and the exclude check.
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
	writeMu sync.Mutex
	// settingsMu serialises settings writers across their persist; see
	// UpdateSettings.
	settingsMu sync.Mutex
	Reviewer   Reviewer
	LLM        *inference.Client
	Riptide    *riptide.Emitter
	// Almanach is the team's note receiver, nil without an almanach: block.
	// Set by Start after NewRuntime, like the accounts.
	Almanach *almanach.Emitter
	// accounts is nil for a Runtime built without them (route tests): the
	// team then reads as on the shared accounts.
	accounts *accounts
}

// NewRuntime publishes the first snapshot. The team is copied, so a later
// caller holding the same pointer cannot mutate what readers see.
func NewRuntime(t *config.Team, r Reviewer, llm *inference.Client, rt *riptide.Emitter) *Runtime {
	rtm := &Runtime{Reviewer: r, LLM: llm, Riptide: rt}
	snapshot := *t
	rtm.team.Store(&snapshot)
	return rtm
}

// StartAccounts builds and installs the team's accounts on a Runtime made
// with NewRuntime. Start wires them itself, before the Reviewer that uses
// them exists; this is for a Runtime built around a fake Reviewer.
func (r *Runtime) StartAccounts(ctx context.Context, d AccountDeps, opened map[Kind]Opened) error {
	acc, err := newAccounts(r.Team().Slug, d)
	if err != nil {
		return err
	}
	acc.start(ctx, opened)
	r.accounts = acc
	return nil
}

// Accounts is the state of the team's Bitbucket and Jira accounts.
func (r *Runtime) Accounts() (bb, jr Account) {
	if r.accounts == nil {
		return Account{}, Account{}
	}
	return r.accounts.get(KindBitbucket), r.accounts.get(KindJira)
}

// OwnBot is the team's own Bitbucket client, as onboarding proves and grants
// it, and its @mention handle; nil while the team is on the shared bot.
func (r *Runtime) OwnBot() (onboarding.BotClient, string) {
	bb, _ := r.Accounts()
	if !bb.Own {
		return nil, ""
	}
	return r.accounts.bot, bb.Username
}

// BitbucketBlocked reports whether the team's own Bitbucket account cannot
// be worked with (rejected, undecryptable, another account's username): its
// webhooks are acknowledged and not processed until it pushes a working one.
//
// A rejected token is proven again first, at most every recheckEvery: a
// blocked team sends no Bitbucket call that could clear it otherwise, and a
// token re-enabled in Bitbucket unblocks on the next delivery. The account
// comes back from the same read the verdict is taken on.
func (r *Runtime) BitbucketBlocked(ctx context.Context) (Account, bool) {
	bb, _ := r.Accounts()
	if bb.Own && bb.Status == StatusRejected {
		r.accounts.recheck(ctx, KindBitbucket)
		bb, _ = r.Accounts()
	}
	return bb, bb.blocking()
}

var errNoAccounts = errors.New("team accounts are not set up on this instance")

// PushCredentials proves every part before anything is stored, then
// persists and swaps each in; a part left nil is kept. A *PushError is a
// proof failing, and then nothing changed.
func (r *Runtime) PushCredentials(ctx context.Context, bb *BitbucketPush, jiraToken *string, by string) error {
	if r.accounts == nil {
		return errNoAccounts
	}
	return r.accounts.push(ctx, bb, jiraToken, by)
}

// ResetCredentials goes back to the shared account; false when the team had
// none of that kind.
func (r *Runtime) ResetCredentials(ctx context.Context, kind Kind, by string) (bool, error) {
	if r.accounts == nil {
		return false, errNoAccounts
	}
	return r.accounts.reset(ctx, kind, by)
}

// UnreachableClaims lists the claimed targets the team's own bot is refused
// on (401, 403, 404): what still needs `grant-bot` after the team switched
// bots. Any other failure (a 5xx, a timeout) says nothing about rights and
// lists nothing.
func (r *Runtime) UnreachableClaims(ctx context.Context) []string {
	bot, _ := r.OwnBot()
	if bot == nil {
		return nil
	}
	refused := func(err error) bool {
		switch bitbucket.Status(err) {
		case http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound:
			return true
		}
		return false
	}
	out := []string{}
	for _, p := range r.Team().Projects {
		if p.Repos == nil {
			if refused(bot.GetProject(ctx, p.Key)) {
				out = append(out, p.Key)
			}
			continue
		}
		for _, repo := range p.Repos {
			if refused(bot.GetRepo(ctx, p.Key, repo)) {
				out = append(out, p.Key+"/"+repo)
			}
		}
	}
	return out
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

// UpdateSettings merges onto the current settings, persists the result
// (nil persist skips that), swaps it in, then mirrors the two author lists
// and the PR description switch onto the live Reviewer. A persist error swaps nothing.
//
// settingsMu serialises settings writers from merge to swap, so a partial
// PUT always merges onto the previous one's result: merging on a stale
// baseline lost one of two concurrent PUTs, and an optimistic retry could
// leave the DB holding a stale row when its second persist failed. The
// persist runs outside writeMu, so a slow DB never blocks an /onboard claim
// write; the swap takes the latest snapshot, keeping a claim that landed
// meanwhile. Claims are the only other writer and never touch these lists.
//
// The Reviewer copies config.Review by value at construction, so without the
// mirror a team's author routing would stay stale until restart. ExcludeRepos
// and the branch lists are deliberately not mirrored: the webhook route and
// the team_config tool read them off the snapshot, never the Reviewer's cfg.
func (r *Runtime) UpdateSettings(merge func(store.TeamSettings) store.TeamSettings, persist func(store.TeamSettings) error) error {
	r.settingsMu.Lock()
	defer r.settingsMu.Unlock()
	s := merge(settingsOf(r.team.Load()))
	if persist != nil {
		if err := persist(s); err != nil {
			return err
		}
	}
	r.writeMu.Lock()
	defer r.writeMu.Unlock()
	next := *r.team.Load()
	applySettings(&next, s)
	r.team.Store(&next)
	r.Reviewer.SetAuthorLists(s.AutoReviewAuthors, s.IgnoreAuthors)
	r.Reviewer.SetUsePRDescription(s.UsePRDescription)
	return nil
}

// settingsOf and applySettings are the one mapping between a team's
// settings and the DB row, in each direction.
func settingsOf(t *config.Team) store.TeamSettings {
	return store.TeamSettings{
		AutoReviewAuthors:     t.Review.AutoReviewAuthors,
		IgnoreAuthors:         t.Review.IgnoreAuthors,
		ExcludeRepos:          t.Review.ExcludeRepos,
		ExcludeSourceBranches: t.Review.ExcludeSourceBranches,
		ExcludeTargetBranches: t.Review.ExcludeTargetBranches,
		UsePRDescription:      t.Review.UsePRDescription,
	}
}

func applySettings(t *config.Team, s store.TeamSettings) {
	t.Review.AutoReviewAuthors = s.AutoReviewAuthors
	t.Review.IgnoreAuthors = s.IgnoreAuthors
	t.Review.ExcludeRepos = s.ExcludeRepos
	t.Review.ExcludeSourceBranches = s.ExcludeSourceBranches
	t.Review.ExcludeTargetBranches = s.ExcludeTargetBranches
	t.Review.UsePRDescription = s.UsePRDescription
}
