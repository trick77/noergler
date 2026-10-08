package teams

import (
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/review"
	"github.com/trick77/noergler/internal/store"
)

func newTestRuntime(t *config.Team) *Runtime {
	return NewRuntime(t, review.New(review.Options{Config: t.Review}), nil, nil, nil)
}

// The webhook route reads the snapshot while the team API writes it. Without
// the atomic pointer this is the race -race reports.
func TestRuntime_SnapshotIsRaceFreeUnderConcurrentWrites(_ *testing.T) {
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
			setSettings(rt, store.TeamSettings{ExcludeRepos: []string{"*-test"}})
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

// UpdateSettings swaps the three lists AND mirrors the two author lists onto
// the live Reviewer. Forgetting the mirror leaves author routing stale until
// restart, which is the bug this test exists for.
func TestRuntime_UpdateSettingsMirrorsAuthorListsOntoReviewer(t *testing.T) {
	rt := newTestRuntime(&config.Team{
		Slug:   "platform",
		Review: config.Review{AutoReviewAuthors: []string{"bob"}},
	})
	if rt.Reviewer.IsAutoReviewAuthor("alice") {
		t.Fatal("alice is not in the initial allow list")
	}

	setSettings(rt, store.TeamSettings{
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
	setSettings(rt, want)
	var got store.TeamSettings
	_ = rt.UpdateSettings(func(cur store.TeamSettings) store.TeamSettings { got = cur; return cur }, nil)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("baseline = %+v, want %+v", got, want)
	}
}

// Two partial PUTs: the second merges onto the first's result, not onto the
// baseline both started from. Merging outside the lock dropped one field,
// and the DB and the runtime could keep different requests' lists.
func TestRuntime_ConcurrentPartialUpdatesKeepBothFields(t *testing.T) {
	rt := newTestRuntime(&config.Team{Slug: "platform"})
	var persisted []store.TeamSettings
	var mu sync.Mutex
	persist := func(s store.TeamSettings) error {
		mu.Lock()
		persisted = append(persisted, s)
		mu.Unlock()
		return nil
	}

	done := make(chan struct{})
	var race sync.Once
	err := rt.UpdateSettings(func(cur store.TeamSettings) store.TeamSettings {
		cur.AutoReviewAuthors = []string{"alice"}
		return cur
	}, func(s store.TeamSettings) error {
		// Once: a retry persists again and must not start a third writer.
		race.Do(func() {
			go func() {
				defer close(done)
				_ = rt.UpdateSettings(func(cur store.TeamSettings) store.TeamSettings {
					cur.IgnoreAuthors = []string{"ci-bot"}
					return cur
				}, persist)
			}()
			time.Sleep(20 * time.Millisecond) // the second update is now racing this one
		})
		return persist(s)
	})
	if err != nil {
		t.Fatal(err)
	}
	<-done

	want := store.TeamSettings{AutoReviewAuthors: []string{"alice"}, IgnoreAuthors: []string{"ci-bot"}}
	if got := settingsOf(rt.Team()); !reflect.DeepEqual(got, want) {
		t.Errorf("runtime = %+v, want %+v", got, want)
	}
	if last := persisted[len(persisted)-1]; !reflect.DeepEqual(last, want) {
		t.Errorf("last persisted = %+v, want %+v", last, want)
	}
}

// The DB write happens outside writeMu: a hung PUT must not block the team's
// /onboard claim writes behind it. The claim that lands meanwhile survives.
func TestRuntime_PersistDoesNotHoldTheWriteLock(t *testing.T) {
	rt := newTestRuntime(&config.Team{Slug: "platform"})
	var claim sync.Once
	err := rt.UpdateSettings(func(cur store.TeamSettings) store.TeamSettings {
		cur.IgnoreAuthors = []string{"ci-bot"}
		return cur
	}, func(store.TeamSettings) error {
		// Once: the claim swaps the snapshot, so the update retries and
		// persists again.
		claim.Do(func() {
			done := make(chan struct{})
			go func() {
				rt.ApplyClaims([]config.ProjectScope{{Key: "PLAT"}})
				close(done)
			}()
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Error("ApplyClaims blocked behind the settings persist")
			}
		})
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !rt.Team().Owns("PLAT", "svc") {
		t.Error("the claim written during the persist was lost")
	}
	if got := rt.Team().Review.IgnoreAuthors; !reflect.DeepEqual(got, []string{"ci-bot"}) {
		t.Errorf("ignore_authors = %v, want the update", got)
	}
}

// A failed persist leaves the snapshot alone.
func TestRuntime_FailedPersistDoesNotSwap(t *testing.T) {
	rt := newTestRuntime(&config.Team{Slug: "platform", Review: config.Review{ExcludeRepos: []string{"*-infra"}}})
	err := rt.UpdateSettings(func(store.TeamSettings) store.TeamSettings {
		return store.TeamSettings{ExcludeRepos: []string{"*-test"}}
	}, func(store.TeamSettings) error { return errors.New("db down") })
	if err == nil {
		t.Fatal("want the persist error")
	}
	if got := rt.Team().Review.ExcludeRepos; !reflect.DeepEqual(got, []string{"*-infra"}) {
		t.Errorf("exclude_repos = %v, want the old list", got)
	}
}

func setSettings(rt *Runtime, s store.TeamSettings) {
	_ = rt.UpdateSettings(func(store.TeamSettings) store.TeamSettings { return s }, nil)
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
