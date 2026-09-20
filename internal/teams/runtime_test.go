package teams

import (
	"reflect"
	"sync"
	"testing"

	"github.com/trick77/noergler-go/internal/config"
	"github.com/trick77/noergler-go/internal/review"
	"github.com/trick77/noergler-go/internal/store"
)

func newTestRuntime(t *config.Team) *Runtime {
	return NewRuntime(t, review.New(review.Options{Config: t.Review}), nil, nil, nil)
}

// The webhook route reads the snapshot while the team API writes it. Without
// the atomic pointer this is the race -race reports.
func TestRuntime_SnapshotIsRaceFreeUnderConcurrentWrites(t *testing.T) {
	rt := newTestRuntime(&config.Team{
		Slug:     "platform",
		Projects: []config.ProjectScope{{Key: "PLAT"}},
		Review:   config.Review{ExcludeRepos: []string{"*-infra"}},
	})

	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 200; j++ {
				team := rt.Team()
				team.Owns("PLAT", "svc")
				team.ReviewsRepo("PLAT", "svc")
			}
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for j := 0; j < 200; j++ {
			rt.ApplyClaims([]config.ProjectScope{{Key: "OTHER"}})
			rt.ApplySettings(store.TeamSettings{ExcludeRepos: []string{"*-test"}})
		}
	}()
	wg.Wait()
}

// A snapshot taken before a write keeps answering from the old config: that
// is what lets the webhook route decide ownership and exclusion from one
// consistent view.
func TestRuntime_SnapshotIsStable(t *testing.T) {
	rt := newTestRuntime(&config.Team{
		Slug:     "platform",
		Projects: []config.ProjectScope{{Key: "PLAT"}},
	})
	before := rt.Team()

	rt.ApplyClaims([]config.ProjectScope{{Key: "OTHER"}})

	if !before.Owns("PLAT", "svc") {
		t.Error("an older snapshot must keep its claims")
	}
	if rt.Team().Owns("PLAT", "svc") {
		t.Error("the new snapshot must carry the new claims")
	}
}

// ApplySettings swaps the three lists AND mirrors the two author lists onto
// the live Reviewer. Forgetting the mirror leaves author routing stale until
// restart, which is the bug this test exists for.
func TestRuntime_ApplySettingsMirrorsAuthorListsOntoReviewer(t *testing.T) {
	rt := newTestRuntime(&config.Team{
		Slug:   "platform",
		Review: config.Review{AutoReviewAuthors: []string{"bob"}},
	})
	if rt.Reviewer.IsAutoReviewAuthor("alice") {
		t.Fatal("alice is not in the initial allow list")
	}

	rt.ApplySettings(store.TeamSettings{
		AutoReviewAuthors: []string{"alice"},
		IgnoreAuthors:     []string{"ci-bot"},
		ExcludeRepos:      []string{"*-test"},
	})

	if !rt.Reviewer.IsAutoReviewAuthor("alice") {
		t.Error("the reviewer did not see the new allow list")
	}
	if rt.Reviewer.IsAutoReviewAuthor("ci-bot") {
		t.Error("the reviewer did not see the new ignore list")
	}
	if got := rt.Team().Review.ExcludeRepos; !reflect.DeepEqual(got, []string{"*-test"}) {
		t.Errorf("exclude_repos = %v, want the new list on the snapshot", got)
	}
}

func TestRuntime_SettingsRoundTripsTheSnapshot(t *testing.T) {
	rt := newTestRuntime(&config.Team{Slug: "platform"})
	want := store.TeamSettings{
		AutoReviewAuthors: []string{"alice"},
		IgnoreAuthors:     []string{"ci-bot"},
		ExcludeRepos:      []string{"*-infra"},
	}
	rt.ApplySettings(want)
	if got := rt.Settings(); !reflect.DeepEqual(got, want) {
		t.Errorf("Settings() = %+v, want %+v", got, want)
	}
}

func TestRegistry_LookupDistinguishesUnknownFromDisabled(t *testing.T) {
	g := &Registry{
		enabled:  map[string]*Runtime{"platform": newTestRuntime(&config.Team{Slug: "platform"})},
		disabled: map[string]string{"payments": "LLM check failed"},
		log:      quietLogger(),
	}

	if _, _, ok := g.Lookup("platform"); !ok {
		t.Error("an enabled team must resolve")
	}
	if rt, reason, ok := g.Lookup("payments"); ok || rt != nil || reason == "" {
		t.Errorf("a disabled team must carry a reason: (%v, %q, %v)", rt, reason, ok)
	}
	if _, reason, ok := g.Lookup("nope"); ok || reason != "" {
		t.Errorf("an unknown team must carry no reason: (%q, %v)", reason, ok)
	}
}

func TestRegistry_StatusIsSortedSlugsOnly(t *testing.T) {
	g := &Registry{
		enabled: map[string]*Runtime{
			"zeta":  newTestRuntime(&config.Team{Slug: "zeta"}),
			"alpha": newTestRuntime(&config.Team{Slug: "alpha"}),
		},
		disabled: map[string]string{"yankee": "boom", "bravo": "boom"},
		log:      quietLogger(),
	}
	enabled, disabled := g.Status()
	if !reflect.DeepEqual(enabled, []string{"alpha", "zeta"}) {
		t.Errorf("enabled = %v, want sorted", enabled)
	}
	if !reflect.DeepEqual(disabled, []string{"bravo", "yankee"}) {
		t.Errorf("disabled = %v, want sorted", disabled)
	}
}
