package teams

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"github.com/trick77/noergler-go/internal/config"
	"github.com/trick77/noergler-go/internal/store"
)

// ClaimStore is what reconciliation needs of the store.
type ClaimStore interface {
	ListAllClaims(ctx context.Context) (map[string][]config.ProjectScope, error)
	GetAllSettings(ctx context.Context) (map[string]store.TeamSettings, error)
	AddClaims(ctx context.Context, teamSlug string, scopes []config.ProjectScope, claimedBy string) ([]string, error)
	PutSettings(ctx context.Context, teamSlug string, t store.TeamSettings, updatedBy string) error
}

// seededBy is the audit value on a row written from the config file.
const seededBy = "teams.yaml"

// Reconcile settles teams.yaml against the DB, in place, before any Runtime
// exists.
//
// The DB wins: a slug it knows carries the claims and the three lists, and
// teams.yaml's projects: block is ignored entirely. A slug it does not know
// is seeded from teams.yaml once. A seed that collides with another team's
// claims disables that team alone and skips its settings step, exactly as
// Python's `continue` does.
//
// It MUST run before the per-team loop builds any Reviewer: review.New
// copies config.Review by value into an unexported field, so a Reviewer
// constructed first would hold the teams.yaml author lists for the life of
// the process, with no setter for anything but the two author lists.
//
// A failure to read either table is a shared-layer fault and comes back as
// err: nothing works without them, and it aborts boot. A per-team seed
// conflict never does.
func Reconcile(ctx context.Context, db ClaimStore, teams map[string]*config.Team, order []string, log *slog.Logger) (map[string]string, error) {
	claims, err := db.ListAllClaims(ctx)
	if err != nil {
		return nil, fmt.Errorf("read team claims: %w", err)
	}
	settings, err := db.GetAllSettings(ctx)
	if err != nil {
		return nil, fmt.Errorf("read team settings: %w", err)
	}

	seedErrors := map[string]string{}
	// order, not a map range: with two teams seeding overlapping claims the
	// earlier one wins and the later is disabled, and which is which must not
	// depend on Go's map iteration.
	for _, slug := range order {
		team, ok := teams[slug]
		if !ok {
			continue
		}
		reason, err := reconcileClaims(ctx, db, slug, team, claims, log)
		if err != nil {
			return nil, err
		}
		if reason != "" {
			seedErrors[slug] = reason
			// Python continues here, so the team's settings are left alone.
			continue
		}
		if err := reconcileSettings(ctx, db, slug, team, settings); err != nil {
			return nil, err
		}
	}
	return seedErrors, nil
}

// reconcileClaims returns a disable reason, or "" when the team is fine.
//
// Only a ClaimConflict disables one team. Python wraps just that exception;
// any other add_claims failure propagates out of the lifespan and aborts
// boot, because it means the claims table is unusable rather than contested.
func reconcileClaims(ctx context.Context, db ClaimStore, slug string, team *config.Team, claims map[string][]config.ProjectScope, log *slog.Logger) (string, error) {
	if scopes, known := claims[slug]; known {
		team.Projects = scopes
		return "", nil
	}
	if len(team.Projects) == 0 {
		log.InfoContext(ctx, "no claims yet (claim via POST /onboard)", "team", slug)
		return "", nil
	}
	added, err := db.AddClaims(ctx, slug, team.Projects, seededBy)
	if err != nil {
		var conflict *store.ClaimConflict
		if errors.As(err, &conflict) {
			return "teams.yaml seed: " + conflict.Error(), nil
		}
		return "", fmt.Errorf("seed claims for team %s: %w", slug, err)
	}
	log.InfoContext(ctx, "claims seeded from teams.yaml", "team", slug, "n", len(added))
	return "", nil
}

// reconcileSettings writes the config's lists to the DB the first time a
// team is seen and reads them back on every later boot.
//
// The seed branch effectively always fires on a team's first boot, because
// exclude_repos defaults to ["*-infra"]. That is how the default row comes to
// exist, independently of the column default.
// A failed write aborts boot: Python's put_settings here is unguarded, so it
// raises out of the lifespan. An unusable settings table is a shared-layer
// fault, not one team's.
func reconcileSettings(ctx context.Context, db ClaimStore, slug string, team *config.Team, settings map[string]store.TeamSettings) error {
	if s, known := settings[slug]; known {
		team.Review.AutoReviewAuthors = s.AutoReviewAuthors
		team.Review.IgnoreAuthors = s.IgnoreAuthors
		team.Review.ExcludeRepos = s.ExcludeRepos
		return nil
	}
	r := team.Review
	if len(r.AutoReviewAuthors) == 0 && len(r.IgnoreAuthors) == 0 && len(r.ExcludeRepos) == 0 {
		return nil
	}
	s := store.TeamSettings{
		AutoReviewAuthors: r.AutoReviewAuthors,
		IgnoreAuthors:     r.IgnoreAuthors,
		ExcludeRepos:      r.ExcludeRepos,
	}
	if err := db.PutSettings(ctx, slug, s, seededBy); err != nil {
		return fmt.Errorf("seed settings for team %s: %w", slug, err)
	}
	return nil
}
