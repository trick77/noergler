package teams

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/logging"
	"github.com/trick77/noergler/internal/store"
)

// bufLogger captures the alerted startup lines so their exact shape can be
// asserted; Splunk matches on the rendered message.
func bufLogger() (*slog.Logger, *bytes.Buffer) {
	var buf bytes.Buffer
	return slog.New(logging.NewHandler(&buf, slog.LevelDebug, "test")), &buf
}

// appWith builds a one-team App whose templates resolve and whose LLM would
// fail, so Start stops before touching a gateway unless a test says otherwise.
func appWith(teams map[string]*config.Team, order []string, disabled map[string]string) *config.App {
	if disabled == nil {
		disabled = map[string]string{}
	}
	return &config.App{Teams: teams, Order: order, Disabled: disabled}
}

func reviewCfg() config.Review {
	return config.Review{
		ReviewPromptTemplate:  "prompts/review.txt",
		MentionPromptTemplate: "prompts/mention.txt",
	}
}

func emptyStore() *fakeClaimStore {
	return &fakeClaimStore{
		claims:   map[string][]config.ProjectScope{},
		settings: map[string]store.TeamSettings{},
	}
}

// A missing prompt template disables that team. Python raises
// FileNotFoundError inside LLMClient.__init__, which _start_team catches.
func TestStart_MissingTemplateDisablesTheTeam(t *testing.T) {
	team := &config.Team{Slug: "platform", Review: reviewCfg()}
	log, _ := bufLogger()

	_, reason := Start(context.Background(), appWith(nil, nil, nil), team, Deps{
		Log:      log,
		ReadFile: func(string) ([]byte, error) { return nil, errors.New("no such file") },
	})
	if !strings.Contains(reason, "prompt template not found") {
		t.Errorf("reason = %q, want a missing-template disable", reason)
	}
}

// One team's failure never takes the instance down, and never stops the next
// team from starting.
func TestBoot_OneTeamsFaultDisablesThatTeamOnly(t *testing.T) {
	broken := &config.Team{Slug: "broken", Review: reviewCfg()}
	// The second team is disabled for the same reason; what matters is that
	// the loop reaches it at all and that boot returns a registry.
	other := &config.Team{Slug: "other", Review: reviewCfg()}
	log, buf := bufLogger()

	g, err := Boot(context.Background(), appWith(
		map[string]*config.Team{"broken": broken, "other": other},
		[]string{"broken", "other"}, nil,
	), Deps{
		Claims:   emptyStore(),
		Log:      log,
		ReadFile: func(string) ([]byte, error) { return nil, errors.New("no such file") },
	})
	if err != nil {
		t.Fatalf("a per-team fault must not abort boot: %v", err)
	}
	_, disabled := g.Status()
	if len(disabled) != 2 {
		t.Errorf("disabled = %v, want both teams", disabled)
	}
	for _, slug := range []string{"broken", "other"} {
		want := "team_disabled team=" + slug + " reason="
		if !strings.Contains(buf.String(), want) {
			t.Errorf("missing alerted line %q", want)
		}
	}
}

// A team whose config failed to resolve is disabled before startup is tried,
// and carries the config layer's reason.
func TestBoot_CarriesConfigDisabledTeams(t *testing.T) {
	log, _ := bufLogger()
	g, err := Boot(context.Background(), appWith(
		map[string]*config.Team{}, nil,
		map[string]string{"payments": "unknown key: base_url"},
	), Deps{Claims: emptyStore(), Log: log})
	if err != nil {
		t.Fatalf("Boot: %v", err)
	}
	if _, reason, ok := g.Lookup("payments"); ok || reason != "unknown key: base_url" {
		t.Errorf("Lookup = (%q, %v), want the config reason", reason, ok)
	}
}

// A seed conflict disables the team without ever calling Start.
func TestBoot_SeedConflictDisablesBeforeStartup(t *testing.T) {
	db := emptyStore()
	db.addErr = &store.ClaimConflict{Project: "PLAT", OtherTeam: "payments"}
	team := &config.Team{
		Slug:     "platform",
		Projects: []config.ProjectScope{{Key: "PLAT"}},
		Review:   reviewCfg(),
	}
	log, buf := bufLogger()

	g, err := Boot(context.Background(), appWith(
		map[string]*config.Team{"platform": team}, []string{"platform"}, nil,
	), Deps{
		Claims: db,
		Log:    log,
		ReadFile: func(string) ([]byte, error) {
			t.Error("Start must not run for a team whose seed conflicted")
			return nil, errors.New("unreachable")
		},
	})
	if err != nil {
		t.Fatalf("Boot: %v", err)
	}
	_, reason, _ := g.Lookup("platform")
	if !strings.Contains(reason, "teams.yaml seed:") {
		t.Errorf("reason = %q, want the seed conflict", reason)
	}
	if !strings.Contains(buf.String(), "team_disabled team=platform") {
		t.Error("missing the alerted team_disabled line")
	}
}

// The three alerted lines keep Python's rendering: the lists live inside the
// message, in Python list repr, because the Splunk alert matches on it.
func TestBoot_TeamsReadyLineRendering(t *testing.T) {
	log, buf := bufLogger()
	team := &config.Team{Slug: "platform", Review: reviewCfg()}

	if _, err := Boot(context.Background(), appWith(
		map[string]*config.Team{"platform": team}, []string{"platform"}, nil,
	), Deps{
		Claims:   emptyStore(),
		Log:      log,
		ReadFile: func(string) ([]byte, error) { return nil, errors.New("no such file") },
	}); err != nil {
		t.Fatalf("Boot: %v", err)
	}
	want := `teams_ready enabled=[] disabled=['platform']`
	if !strings.Contains(buf.String(), want) {
		t.Errorf("log missing %q\ngot: %s", want, buf.String())
	}
	if !strings.Contains(buf.String(), "no team is enabled") {
		t.Error("an all-disabled config must log the no-team error")
	}
}

// A panic in one team's startup is contained, not fatal.
func TestBoot_PanicInOneTeamIsContained(t *testing.T) {
	log, _ := bufLogger()
	team := &config.Team{Slug: "platform", Review: reviewCfg()}

	g, err := Boot(context.Background(), appWith(
		map[string]*config.Team{"platform": team}, []string{"platform"}, nil,
	), Deps{
		Claims:   emptyStore(),
		Log:      log,
		ReadFile: func(string) ([]byte, error) { panic("boom") },
	})
	if err != nil {
		t.Fatalf("a panicking team must not abort boot: %v", err)
	}
	_, reason, _ := g.Lookup("platform")
	if !strings.Contains(reason, "startup failed: panic: boom") {
		t.Errorf("reason = %q, want the contained panic", reason)
	}
}

// A shared-layer fault is not a per-team fault: it aborts boot.
func TestBoot_UnreadableClaimsTableAbortsBoot(t *testing.T) {
	log, _ := bufLogger()
	db := emptyStore()
	db.claimsErr = errors.New("connection refused")

	if _, err := Boot(context.Background(), appWith(nil, nil, nil), Deps{Claims: db, Log: log}); err == nil {
		t.Error("want boot to abort when the claims table is unreadable")
	}
}
