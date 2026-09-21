package config

import (
	"strings"
	"testing"
)

func TestQueue_Defaults(t *testing.T) {
	e := newEnv(t)
	app := e.mustLoad()

	if got := app.Queue.InferenceConcurrency; got != 6 {
		t.Errorf("InferenceConcurrency = %d, want 6", got)
	}
	if got := app.Queue.InferenceConcurrencyPerTeam; got != 2 {
		t.Errorf("InferenceConcurrencyPerTeam = %d, want 2", got)
	}
}

func TestQueue_Overrides(t *testing.T) {
	e := newEnv(t)
	e.set("REVIEW_INFERENCE_CONCURRENCY", "10", "REVIEW_INFERENCE_CONCURRENCY_PER_TEAM", "3")
	app := e.mustLoad()

	if got := app.Queue.InferenceConcurrency; got != 10 {
		t.Errorf("InferenceConcurrency = %d, want 10", got)
	}
	if got := app.Queue.InferenceConcurrencyPerTeam; got != 3 {
		t.Errorf("InferenceConcurrencyPerTeam = %d, want 3", got)
	}
}

// Zero has to fail at load. A pool of zero is an unbuffered channel, which
// would boot clean and then block every review forever.
func TestQueue_ZeroFailsAtLoad(t *testing.T) {
	for _, name := range []string{"REVIEW_INFERENCE_CONCURRENCY", "REVIEW_INFERENCE_CONCURRENCY_PER_TEAM"} {
		t.Run(name, func(t *testing.T) {
			e := newEnv(t)
			e.set(name, "0")
			_, err := e.load()
			if err == nil {
				t.Fatalf("%s=0 loaded clean", name)
			}
			if !strings.Contains(err.Error(), name) {
				t.Errorf("error does not name %s: %v", name, err)
			}
		})
	}
}

func TestQueue_NegativeFailsAtLoad(t *testing.T) {
	e := newEnv(t)
	e.set("REVIEW_INFERENCE_CONCURRENCY", "-1")
	if _, err := e.load(); err == nil {
		t.Fatal("a negative pool size loaded clean")
	}
}

// A per-team cap above the global one never binds, so it is a setting that
// silently does nothing. Fail rather than let it reach production.
func TestQueue_PerTeamAboveGlobalFailsAtLoad(t *testing.T) {
	e := newEnv(t)
	e.set("REVIEW_INFERENCE_CONCURRENCY", "6", "REVIEW_INFERENCE_CONCURRENCY_PER_TEAM", "8")

	_, err := e.load()
	if err == nil {
		t.Fatal("per-team above global loaded clean")
	}
	if !strings.Contains(err.Error(), "REVIEW_INFERENCE_CONCURRENCY_PER_TEAM") {
		t.Errorf("error does not name the offending var: %v", err)
	}
}

// Equal is legal: it means the per-team cap never binds in practice, which
// is a deliberate choice (one team may use the whole pool), not a typo.
func TestQueue_PerTeamEqualToGlobalIsAllowed(t *testing.T) {
	e := newEnv(t)
	e.set("REVIEW_INFERENCE_CONCURRENCY", "4", "REVIEW_INFERENCE_CONCURRENCY_PER_TEAM", "4")
	app := e.mustLoad()

	if app.Queue.InferenceConcurrencyPerTeam != 4 {
		t.Errorf("per-team = %d, want 4", app.Queue.InferenceConcurrencyPerTeam)
	}
}
