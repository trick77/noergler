// Package onboarding puts noergler's webhook on the Bitbucket targets a team
// claims, and takes it off again.
//
// Onboarding = one webhook per target the team holds in `teams.yaml` (or in
// the claims table): a whole-project claim is one project webhook, a `repos:`
// list is one repo webhook per slug. The caller can only narrow that set,
// never widen it.
//
// Every Bitbucket write goes through the team admin's own token (AdminClient);
// the bot's token (BotClient) is only ever used to check that the bot can read
// a target. The admin token lives in one client for the duration of the
// request and is never logged or stored.
//
// This package is the domain layer: no HTTP handlers, no status codes. A
// separate package maps the error types in errors.go onto responses.
package onboarding

import (
	"fmt"
	"strings"

	"github.com/trick77/noergler-go/internal/config"
)

// DefaultWebhookName is the name Bitbucket stores the hook under.
const DefaultWebhookName = "noergler"

// Action selects what Run does with each target.
type Action string

const (
	// ActionStatus reports the status of each target.
	ActionStatus Action = "status"
	// ActionOnboard claims and onboards a project.
	ActionOnboard Action = "onboard"
	// ActionGrantBot grants the bot read access to claimed projects.
	ActionGrantBot Action = "grant-bot"
	// ActionRemove removes and unclaims a project.
	ActionRemove Action = "remove"
)

// Target is a project (Repo == "": one project webhook) or a single repo.
type Target struct {
	Project string
	Repo    string
}

// IsProject reports whether this is a whole-project target.
func (t Target) IsProject() bool { return t.Repo == "" }

// Key is the target as it appears in a `teams.yaml` block: `KEY` or `KEY/repo`.
func (t Target) Key() string {
	if t.Repo == "" {
		return t.Project
	}
	return t.Project + "/" + t.Repo
}

// Label is Key for a repo, and `KEY (project)` for a project, for the tables.
func (t Target) Label() string {
	if t.Repo == "" {
		return t.Project + " (project)"
	}
	return t.Key()
}

// BotPermission is the Bitbucket permission the bot needs on this target.
func (t Target) BotPermission() string {
	if t.Repo == "" {
		return "PROJECT_WRITE"
	}
	return "REPO_WRITE"
}

// Claim is how `teams.yaml` claims a target for the team, and whether the bot
// can read it.
type Claim struct {
	Owned bool
	// Kind is "whole", "repos" or "none": how the team holds the *project*.
	Kind       string
	BotCanRead bool
}

// TargetResult is the outcome of one onboard/grant-bot/remove step.
//
// Detail is appended to after Run returns: RemoveAndUnclaim adds the purged-PR
// count to every hook result, so these must be handled by pointer or index,
// never by value copy.
type TargetResult struct {
	Target Target
	// Status is "ok", "failed" or "skipped".
	Status string
	Detail string
	Diff   []string
}

// StatusRow is one line of the status table.
type StatusRow struct {
	Target     Target
	Owned      bool
	BotCanRead bool
	// Webhook is "ok", "missing", "stale: …", "foreign: …", "HTTP <code>",
	// or why the target is not owned.
	Webhook string
	// Stray are this instance's repo-level hooks under a project target.
	Stray []string
	// Foreign are same-named hooks pointing at another noergler instance.
	Foreign []string
}

// TargetsFor returns the team's targets, optionally narrowed to subset
// (`KEY` or `KEY/repo`, as they appear in the team block). A nil subset means
// all of them; an empty non-nil subset means none.
func TargetsFor(team *config.Team, subset []string) ([]Target, error) {
	var targets []Target
	for _, scope := range team.Projects {
		if scope.Repos == nil {
			targets = append(targets, Target{Project: scope.Key})
			continue
		}
		for _, repo := range scope.Repos {
			targets = append(targets, Target{Project: scope.Key, Repo: repo})
		}
	}
	if subset == nil {
		return targets, nil
	}
	// byKey keeps insertion order for the `known:` list, as Python's dict does.
	byKey := make(map[string]Target, len(targets))
	known := make([]string, 0, len(targets))
	for _, t := range targets {
		if _, seen := byKey[t.Key()]; !seen {
			known = append(known, t.Key())
		}
		byKey[t.Key()] = t
	}
	var unknown []string
	for _, s := range subset {
		if _, ok := byKey[s]; !ok {
			unknown = append(unknown, s)
		}
	}
	if len(unknown) > 0 {
		return nil, &UnknownTarget{Msg: fmt.Sprintf(
			"not in team %s's teams.yaml block: %s; known: %s",
			team.Slug, strings.Join(unknown, ", "), strings.Join(known, ", "))}
	}
	out := make([]Target, 0, len(subset))
	for _, s := range subset {
		out = append(out, byKey[s])
	}
	return out, nil
}

// claimKind reports how the team holds the project: "whole", "repos" or "none".
func claimKind(team *config.Team, project string) string {
	for _, p := range team.Projects {
		if p.Key == project {
			if p.Repos == nil {
				return "whole"
			}
			return "repos"
		}
	}
	return "none"
}

// notOwnedReason explains why a target is not the team's to onboard.
func notOwnedReason(target Target, claim Claim) string {
	if target.IsProject() && claim.Kind == "repos" {
		return fmt.Sprintf(
			"teams.yaml lists specific repos of %s for this team; "+
				"onboard those, or ask the noergler admin to claim the whole project", target.Project)
	}
	return "not owned by this team in teams.yaml; ask the noergler admin"
}

// wholeClaimReason explains why a repo webhook is refused under a whole-project
// claim: the project hook already delivers every event.
func wholeClaimReason(target Target) string {
	return fmt.Sprintf(
		"teams.yaml claims all of %s for this team: one project webhook "+
			"covers it, a repo webhook next to it would deliver every event twice", target.Project)
}
