package config

import (
	"strings"
	"testing"
)

func TestInstance_EmptyRequiredIsFatal(t *testing.T) {
	e := newEnv(t)
	e.set("BITBUCKET_TOKEN", "  ")
	if _, err := e.load(); err == nil || !strings.Contains(err.Error(), "BITBUCKET_TOKEN is required") {
		t.Errorf("err = %v", err)
	}
}

// A misspelled boolean used to read as false; now it fails the boot.
func TestInstance_BadBooleanIsFatal(t *testing.T) {
	e := newEnv(t)
	e.set("REVIEW_REQUIRE_AGENTS_MD", "ture")
	if _, err := e.load(); err == nil || !strings.Contains(err.Error(), "REVIEW_REQUIRE_AGENTS_MD") {
		t.Errorf("err = %v", err)
	}
	e.set("REVIEW_REQUIRE_AGENTS_MD", "off")
	if app := e.mustLoad(); app.Review.RequireAgentsMD {
		t.Error("off read as true")
	}
}

// A bad LOG_LEVEL is one more fault in the same report, not a startup of its
// own before the rest are read.
func TestInstance_BadLogLevelReportedWithTheRest(t *testing.T) {
	e := newEnv(t)
	e.set("LOG_LEVEL", "verbose", "REVIEW_REQUIRE_AGENTS_MD", "ture")
	_, err := e.load()
	if err == nil || !strings.Contains(err.Error(), "LOG_LEVEL") || !strings.Contains(err.Error(), "REVIEW_REQUIRE_AGENTS_MD") {
		t.Errorf("err = %v, want both faults", err)
	}
	e.set("LOG_LEVEL", "WARNING", "REVIEW_REQUIRE_AGENTS_MD", "on")
	e.mustLoad()
}

// Blank is a setting for these: no opt-out keyword, no excluded repos, no AC
// prefixes. Unset keeps the default.
func TestInstance_BlankTurnsOffWhatBlankMeansOff(t *testing.T) {
	e := newEnv(t)
	app := e.mustLoad()
	if app.Review.OptOutBranchKeyword != "noergloff" || len(app.Review.ExcludeRepos) != 1 ||
		len(app.Jira.AcceptanceCriteriaPrefixes) == 0 {
		t.Fatalf("defaults = %q %v %v", app.Review.OptOutBranchKeyword, app.Review.ExcludeRepos,
			app.Jira.AcceptanceCriteriaPrefixes)
	}
	e.set("REVIEW_OPT_OUT_BRANCH_KEYWORD", "", "REVIEW_EXCLUDE_REPOS", " ", "JIRA_ACCEPTANCE_CRITERIA_PREFIXES", "")
	app = e.mustLoad()
	if app.Review.OptOutBranchKeyword != "" || len(app.Review.ExcludeRepos) != 0 ||
		len(app.Jira.AcceptanceCriteriaPrefixes) != 0 {
		t.Errorf("blank = %q %v %v", app.Review.OptOutBranchKeyword, app.Review.ExcludeRepos,
			app.Jira.AcceptanceCriteriaPrefixes)
	}
}

func TestTeams_DuplicateKeyDisablesTheTeam(t *testing.T) {
	e := newEnv(t)
	e.teams(strings.Replace(twoTeams, "      max_pr_cost_usd: 8.5\n", "      max_pr_cost_usd: 8.5\n      max_pr_cost_usd: 1\n", 1))
	e.payments()
	app := e.mustLoad()
	if got := app.Disabled["payments"]; !strings.HasPrefix(got, "review.max_pr_cost_usd: Duplicate key") {
		t.Errorf("disabled = %v", app.Disabled)
	}
	e.teams(strings.Replace(twoTeams, "    name: \"Payments\"\n", "    name: \"Payments\"\n    slug: other\n", 1))
	app = e.mustLoad()
	if got := app.Disabled["payments"]; !strings.HasPrefix(got, "slug: Duplicate key") {
		t.Errorf("disabled = %v", app.Disabled)
	}
}
