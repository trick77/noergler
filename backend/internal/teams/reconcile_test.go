package teams

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"reflect"
	"testing"

	"github.com/trick77/noergler-go/internal/config"
	"github.com/trick77/noergler-go/internal/store"
)

func quietLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// fakeClaimStore records what reconciliation wrote.
type fakeClaimStore struct {
	claims   map[string][]config.ProjectScope
	settings map[string]store.TeamSettings

	claimsErr   error
	settingsErr error
	addErr      error
	putErr      error

	added []string // "slug:by"
	put   []string // "slug:by"
}

func (f *fakeClaimStore) ListAllClaims(context.Context) (map[string][]config.ProjectScope, error) {
	return f.claims, f.claimsErr
}

func (f *fakeClaimStore) GetAllSettings(context.Context) (map[string]store.TeamSettings, error) {
	return f.settings, f.settingsErr
}

func (f *fakeClaimStore) AddClaims(_ context.Context, slug string, _ []config.ProjectScope, by string) ([]string, error) {
	if f.addErr != nil {
		return nil, f.addErr
	}
	f.added = append(f.added, slug+":"+by)
	return []string{slug}, nil
}

func (f *fakeClaimStore) PutSettings(_ context.Context, slug string, _ store.TeamSettings, by string) error {
	if f.putErr != nil {
		return f.putErr
	}
	f.put = append(f.put, slug+":"+by)
	return nil
}

func teamWith(slug string, projects []config.ProjectScope, r config.Review) *config.Team {
	return &config.Team{Slug: slug, Projects: projects, Review: r}
}

// The DB is the truth: a slug it knows carries the claims, and teams.yaml's
// projects block is ignored entirely, even when it lists more.
func TestReconcile_DBWinsOverYAML(t *testing.T) {
	db := &fakeClaimStore{
		claims: map[string][]config.ProjectScope{
			"platform": {{Key: "FROMDB"}},
		},
		settings: map[string]store.TeamSettings{
			"platform": {AutoReviewAuthors: []string{"db-alice"}, ExcludeRepos: []string{"*-db"}},
		},
	}
	team := teamWith("platform", []config.ProjectScope{{Key: "FROMYAML"}}, config.Review{
		AutoReviewAuthors: []string{"yaml-bob"},
		ExcludeRepos:      []string{"*-infra"},
	})
	teams := map[string]*config.Team{"platform": team}

	seedErrs, err := Reconcile(context.Background(), db, teams, []string{"platform"}, quietLogger())
	if err != nil || len(seedErrs) != 0 {
		t.Fatalf("Reconcile = %v, %v", seedErrs, err)
	}
	if got := team.Projects; !reflect.DeepEqual(got, []config.ProjectScope{{Key: "FROMDB"}}) {
		t.Errorf("projects = %+v, want the DB's", got)
	}
	if got := team.Review.AutoReviewAuthors; !reflect.DeepEqual(got, []string{"db-alice"}) {
		t.Errorf("auto_review_authors = %v, want the DB's", got)
	}
	if got := team.Review.ExcludeRepos; !reflect.DeepEqual(got, []string{"*-db"}) {
		t.Errorf("exclude_repos = %v, want the DB's", got)
	}
	if len(db.added) != 0 || len(db.put) != 0 {
		t.Errorf("wrote to a DB that already knew the team: added=%v put=%v", db.added, db.put)
	}
}

// A slug the DB does not know is seeded once, under claimed_by=teams.yaml.
func TestReconcile_YAMLSeedsOnce(t *testing.T) {
	db := &fakeClaimStore{claims: map[string][]config.ProjectScope{}, settings: map[string]store.TeamSettings{}}
	team := teamWith("platform", []config.ProjectScope{{Key: "PLAT"}}, config.Review{
		ExcludeRepos: []string{"*-infra"},
	})

	if _, err := Reconcile(context.Background(), db, map[string]*config.Team{"platform": team}, []string{"platform"}, quietLogger()); err != nil {
		t.Fatalf("Reconcile: %v", err)
	}
	if !reflect.DeepEqual(db.added, []string{"platform:teams.yaml"}) {
		t.Errorf("added = %v, want one teams.yaml seed", db.added)
	}
	// exclude_repos defaults to *-infra, so the settings seed effectively
	// always fires on a team's first boot. That is how the default row exists.
	if !reflect.DeepEqual(db.put, []string{"platform:teams.yaml"}) {
		t.Errorf("put = %v, want one teams.yaml settings seed", db.put)
	}
}

// Three empty lists write nothing.
func TestReconcile_NoSettingsSeedWhenEverythingEmpty(t *testing.T) {
	db := &fakeClaimStore{claims: map[string][]config.ProjectScope{}, settings: map[string]store.TeamSettings{}}
	team := teamWith("platform", nil, config.Review{})

	if _, err := Reconcile(context.Background(), db, map[string]*config.Team{"platform": team}, []string{"platform"}, quietLogger()); err != nil {
		t.Fatalf("Reconcile: %v", err)
	}
	if len(db.put) != 0 || len(db.added) != 0 {
		t.Errorf("wrote for a team with nothing to seed: added=%v put=%v", db.added, db.put)
	}
}

// A seed conflict disables that team alone AND skips its settings step,
// matching Python's `continue`.
func TestReconcile_SeedConflictDisablesOneTeamAndSkipsItsSettings(t *testing.T) {
	repo := "svc"
	db := &fakeClaimStore{
		claims:   map[string][]config.ProjectScope{},
		settings: map[string]store.TeamSettings{},
		addErr:   &store.ClaimConflict{Project: "PLAT", Repo: &repo, OtherTeam: "payments"},
	}
	team := teamWith("platform", []config.ProjectScope{{Key: "PLAT"}}, config.Review{
		ExcludeRepos: []string{"*-infra"},
	})

	seedErrs, err := Reconcile(context.Background(), db, map[string]*config.Team{"platform": team}, []string{"platform"}, quietLogger())
	if err != nil {
		t.Fatalf("a seed conflict must not abort boot: %v", err)
	}
	want := "teams.yaml seed: PLAT/svc is claimed by team payments"
	if seedErrs["platform"] != want {
		t.Errorf("reason = %q, want %q", seedErrs["platform"], want)
	}
	if len(db.put) != 0 {
		t.Errorf("settings step ran for a team whose seed conflicted: %v", db.put)
	}
}

// Any other claim-write failure is the claims table being unusable, which is
// a shared-layer fault and aborts boot.
func TestReconcile_NonConflictSeedFailureAbortsBoot(t *testing.T) {
	db := &fakeClaimStore{
		claims:   map[string][]config.ProjectScope{},
		settings: map[string]store.TeamSettings{},
		addErr:   errors.New("connection refused"),
	}
	team := teamWith("platform", []config.ProjectScope{{Key: "PLAT"}}, config.Review{})

	if _, err := Reconcile(context.Background(), db, map[string]*config.Team{"platform": team}, []string{"platform"}, quietLogger()); err == nil {
		t.Error("a broken claims table must abort boot, not disable one team")
	}
}

func TestReconcile_UnreadableTablesAbortBoot(t *testing.T) {
	for _, c := range []struct {
		name string
		db   *fakeClaimStore
	}{
		{"claims", &fakeClaimStore{claimsErr: errors.New("boom")}},
		{"settings", &fakeClaimStore{claims: map[string][]config.ProjectScope{}, settingsErr: errors.New("boom")}},
	} {
		t.Run(c.name, func(t *testing.T) {
			if _, err := Reconcile(context.Background(), c.db, nil, nil, quietLogger()); err == nil {
				t.Error("want an error that aborts boot")
			}
		})
	}
}

// Seeding order follows app.Order, not Go's map iteration: with two teams
// contending for a project the earlier one must always win.
func TestReconcile_SeedsInConfigOrder(t *testing.T) {
	db := &fakeClaimStore{claims: map[string][]config.ProjectScope{}, settings: map[string]store.TeamSettings{}}
	teams := map[string]*config.Team{
		"alpha": teamWith("alpha", []config.ProjectScope{{Key: "A"}}, config.Review{}),
		"beta":  teamWith("beta", []config.ProjectScope{{Key: "B"}}, config.Review{}),
		"gamma": teamWith("gamma", []config.ProjectScope{{Key: "G"}}, config.Review{}),
	}
	order := []string{"gamma", "alpha", "beta"}

	if _, err := Reconcile(context.Background(), db, teams, order, quietLogger()); err != nil {
		t.Fatalf("Reconcile: %v", err)
	}
	want := []string{"gamma:teams.yaml", "alpha:teams.yaml", "beta:teams.yaml"}
	if !reflect.DeepEqual(db.added, want) {
		t.Errorf("seed order = %v, want %v", db.added, want)
	}
}
