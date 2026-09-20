package onboarding

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"

	"github.com/trick77/noergler-go/internal/bitbucket"
	"github.com/trick77/noergler-go/internal/config"
)

// HasAdmin proves the caller's token has project admin on the target by
// listing its webhooks, which needs admin there.
//
// It is not a boolean: only 401 and 403 are a plain "no". Any other status
// (404 included) and any transport failure come back as *UpstreamError,
// because that is Bitbucket's fault and not the caller's. The caller aborts
// the whole request on one, never one target.
func HasAdmin(ctx context.Context, admin AdminClient, target Target) (bool, error) {
	if _, err := admin.ListWebhooks(ctx, target.Project, target.Repo); err != nil {
		s := bitbucket.Status(err)
		if s == http.StatusUnauthorized || s == http.StatusForbidden {
			return false, nil
		}
		return false, &UpstreamError{Target: target.Key(), Status: s, Err: err}
	}
	return true, nil
}

// ClaimResult is what ClaimAndOnboard did.
type ClaimResult struct {
	Results []TargetResult
	Text    string
	Healthy bool
	// Claimed are the scopes AddClaims wrote, never nil.
	Claimed []string
	// Claims is the team's full claim list after the write, for the caller to
	// refresh its runtime with. Nil when nothing was written.
	Claims []config.ProjectScope
}

// ClaimAndOnboard takes `projects` with onboard/grant-bot: prove admin per
// target, claim what is proven (all or nothing against other teams), then hook
// exactly those.
//
// team is never mutated. The hook step runs against a local view: the team's
// fresh claims on a real run, and team.Projects plus the proven scopes on a
// dry run, so the ownership check passes for what would have been claimed. The
// caller applies Claims to its own runtime.
//
// A *store.ClaimConflict from AddClaims comes back unwrapped, for the HTTP
// layer's 409. An *UpstreamError from the admin proof aborts everything.
func ClaimAndOnboard(
	ctx context.Context, admin AdminClient, bot BotClient, st ClaimStore,
	team *config.Team, scopes []config.ProjectScope, caller, webhookURL string,
	opts Options, log *slog.Logger,
) (*ClaimResult, error) {
	if log == nil {
		log = slog.New(slog.DiscardHandler)
	}
	var proven []config.ProjectScope
	var failed []TargetResult
	for _, scope := range scopes {
		one := *team
		one.Projects = []config.ProjectScope{scope}
		targets, err := TargetsFor(&one, nil)
		if err != nil {
			return nil, err
		}
		var okRepos []string
		for _, target := range targets {
			ok, err := HasAdmin(ctx, admin, target)
			if err != nil {
				return nil, err
			}
			switch {
			case !ok:
				failed = append(failed, TargetResult{target, "failed", fmt.Sprintf(
					"no project admin on %s with this token; not claimed", target.Key()), []string{}})
			case target.Repo == "":
				proven = append(proven, scope)
			default:
				okRepos = append(okRepos, target.Repo)
			}
		}
		if len(okRepos) > 0 {
			proven = append(proven, config.ProjectScope{Key: scope.Key, Repos: okRepos})
		}
	}

	out := &ClaimResult{Claimed: []string{}}
	// view is what the hook step sees as the team's claims.
	view := *team
	if len(proven) > 0 && !opts.DryRun {
		claimed, err := st.AddClaims(ctx, team.Slug, proven, caller)
		if err != nil {
			return nil, err
		}
		out.Claimed = claimed
		// The write is already durable, so a failed read-back must not make
		// this look like nothing happened: the caller would skip ApplyClaims
		// and the runtime would keep a claim set the DB no longer has. Fall
		// back to the scopes just written, warn, and let the caller apply
		// them; the next restart reconciles from the DB anyway.
		fresh, err := st.ListClaims(ctx, team.Slug)
		if err != nil {
			log.WarnContext(ctx, "claims written but could not be read back; applying the written scopes",
				"team", team.Slug, "error", err)
			fresh = append(append([]config.ProjectScope(nil), team.Projects...), proven...)
		}
		out.Claims = fresh
		// Python's runtime.apply_claims writes onto the same config object the
		// hook step then reads, so the new claims are in scope for it.
		view.Projects = fresh
	} else if opts.DryRun {
		// For the hook step the proven scopes count as claimed even on a dry
		// run, so the Onboarder's ownership check passes. Python copies the
		// team here; the caller's runtime must not gain phantom claims from a
		// request that wrote nothing.
		view.Projects = append(append([]config.ProjectScope(nil), team.Projects...), proven...)
	}

	var targets []Target
	if len(proven) > 0 {
		scoped := view
		scoped.Projects = proven
		var err error
		if targets, err = TargetsFor(&scoped, nil); err != nil {
			return nil, err
		}
	}
	log.InfoContext(ctx, fmt.Sprintf("onboard by=%s action=%s claimed=%v targets=%d dry_run=%t",
		caller, actionFor(opts), out.Claimed, len(targets), opts.DryRun))

	var rows []TargetResult
	if len(targets) > 0 {
		o := New(admin, bot, &view, webhookURL, opts, log)
		_, rows, _, _ = Run(ctx, o, actionFor(opts), targets)
	}
	// Failed rows are prepended: the unprovable targets come first.
	out.Results = append(failed, rows...)
	if out.Results == nil {
		out.Results = []TargetResult{}
	}
	out.Text = RenderResults(out.Results)
	out.Healthy = ResultsHealthy(out.Results)
	return out, nil
}

func actionFor(opts Options) Action {
	if opts.GrantBot {
		return ActionGrantBot
	}
	return ActionOnboard
}

// RemoveResult is what RemoveAndUnclaim did.
type RemoveResult struct {
	Results   []TargetResult
	Text      string
	Healthy   bool
	Unclaimed []string
	PurgedPRs int
	// Claims is the team's claim list after the removal, nil when nothing was
	// written.
	Claims []config.ProjectScope
}

// RemoveAndUnclaim takes `projects` with remove: hooks off, claims gone, every
// PR record of the team on those targets purged (findings cascade). A dry run
// counts only.
//
// A whole-project scope means every claim the team has on that project,
// whether it holds the project or some of its repos. Giving a target up needs
// the same proof as taking it: admin on it.
//
// team is never mutated; the caller applies Claims to its own runtime.
func RemoveAndUnclaim(
	ctx context.Context, admin AdminClient, bot BotClient, st ClaimStore,
	team *config.Team, scopes []config.ProjectScope, caller, webhookURL string,
	opts Options, log *slog.Logger,
) (*RemoveResult, error) {
	if log == nil {
		log = slog.New(slog.DiscardHandler)
	}
	all, err := TargetsFor(team, nil)
	if err != nil {
		return nil, err
	}
	var wanted []Target
	for _, scope := range scopes {
		if scope.Repos == nil {
			var mine []Target
			for _, t := range all {
				if t.Project == scope.Key {
					mine = append(mine, t)
				}
			}
			if len(mine) == 0 {
				return nil, &NoClaim{Msg: "no claim on " + scope.Key}
			}
			wanted = append(wanted, mine...)
			continue
		}
		keys := make([]string, 0, len(scope.Repos))
		for _, r := range scope.Repos {
			keys = append(keys, scope.Key+"/"+r)
		}
		some, err := TargetsFor(team, keys)
		if err != nil {
			return nil, err
		}
		wanted = append(wanted, some...)
	}

	var proven []Target
	results := []TargetResult{}
	for _, target := range wanted {
		ok, err := HasAdmin(ctx, admin, target)
		if err != nil {
			return nil, err
		}
		if ok {
			proven = append(proven, target)
		} else {
			results = append(results, TargetResult{target, "failed", fmt.Sprintf(
				"no project admin on %s with this token; not removed", target.Key()), []string{}})
		}
	}

	// The remove Onboarder takes Python's defaults for everything but the name
	// and the dry run: grant and prune never apply to a removal.
	o := New(admin, bot, team, webhookURL, Options{WebhookName: opts.WebhookName, DryRun: opts.DryRun}, log)
	var hookResults []TargetResult
	if len(proven) > 0 {
		_, hookResults, _, _ = Run(ctx, o, ActionRemove, proven)
	}

	out := &RemoveResult{Unclaimed: []string{}}
	// Purging runs for every hook result, the failed ones included: the claim
	// is being given up either way, so the records go with it.
	for i := range hookResults {
		t := hookResults[i].Target
		var repo *string
		if t.Repo != "" {
			r := t.Repo
			repo = &r
		}
		var n int
		var err error
		if opts.DryRun {
			n, err = st.CountProjectPRs(ctx, team.Slug, t.Project, repo)
			if err != nil {
				return nil, err
			}
			hookResults[i].Detail += fmt.Sprintf("; dry-run: would purge %d PR record(s)", n)
		} else {
			n, err = st.PurgeProject(ctx, team.Slug, t.Project, repo)
			if err != nil {
				return nil, err
			}
			hookResults[i].Detail += fmt.Sprintf("; purged %d PR record(s)", n)
		}
		out.PurgedPRs += n
	}
	results = append(results, hookResults...)

	if len(proven) > 0 && !opts.DryRun {
		drop := make([]config.ProjectScope, 0, len(proven))
		for _, t := range proven {
			if t.Repo == "" {
				drop = append(drop, config.ProjectScope{Key: t.Project})
			} else {
				drop = append(drop, config.ProjectScope{Key: t.Project, Repos: []string{t.Repo}})
			}
		}
		unclaimed, err := st.RemoveClaims(ctx, team.Slug, drop)
		if err != nil {
			return nil, err
		}
		out.Unclaimed = unclaimed
		// A failed read-back here is worse than on the claim path: without an
		// ApplyClaims the runtime still owns the repo, so webhooks for it keep
		// passing the ownership check and write fresh rows against the PR
		// records this request just purged. Subtract what was dropped instead.
		fresh, err := st.ListClaims(ctx, team.Slug)
		if err != nil {
			log.WarnContext(ctx, "claims removed but could not be read back; applying the remainder",
				"team", team.Slug, "error", err)
			fresh = withoutTargets(team.Projects, proven)
		}
		out.Claims = fresh
	}
	log.InfoContext(ctx, fmt.Sprintf("onboard by=%s action=remove unclaimed=%v purged_prs=%d dry_run=%t",
		caller, out.Unclaimed, out.PurgedPRs, opts.DryRun))

	out.Results = results
	out.Text = RenderResults(results)
	out.Healthy = ResultsHealthy(results)
	return out, nil
}

// withoutTargets removes the given targets from a claim list.
//
// Only used when the read-back after a successful RemoveClaims fails: it is a
// local reconstruction of what the DB now holds, deliberately conservative
// (dropping a whole project when any of its targets went) so the runtime
// never keeps reviewing something it no longer owns.
func withoutTargets(scopes []config.ProjectScope, gone []Target) []config.ProjectScope {
	wholeGone := map[string]bool{}
	repoGone := map[string]bool{}
	for _, t := range gone {
		if t.Repo == "" {
			wholeGone[t.Project] = true
		} else {
			repoGone[t.Project+"/"+t.Repo] = true
		}
	}
	out := make([]config.ProjectScope, 0, len(scopes))
	for _, s := range scopes {
		if wholeGone[s.Key] {
			continue
		}
		if s.Repos == nil {
			out = append(out, s)
			continue
		}
		repos := make([]string, 0, len(s.Repos))
		for _, r := range s.Repos {
			if !repoGone[s.Key+"/"+r] {
				repos = append(repos, r)
			}
		}
		if len(repos) > 0 {
			out = append(out, config.ProjectScope{Key: s.Key, Repos: repos})
		}
	}
	return out
}
