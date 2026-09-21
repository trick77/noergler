package config

import (
	"strings"
	"testing"
)

func TestInstance_EmptyRequiredIsFatal(t *testing.T) {
	e := newEnv(t)
	e.set("BITBUCKET_TOKEN", "  ")
	if _, err := e.load(); err == nil || !strings.Contains(err.Error(), "Environment variable BITBUCKET_TOKEN is empty") {
		t.Errorf("err = %v", err)
	}
}

func TestStripChatSuffix(t *testing.T) {
	for in, want := range map[string]string{
		"https://h/v1":                   "https://h/v1",
		"https://h/v1/":                  "https://h/v1",
		"https://h/v1/chat/completions":  "https://h/v1",
		"https://h/v1/chat/completions/": "https://h/v1",
	} {
		if got := stripChatSuffix(in); got != want {
			t.Errorf("stripChatSuffix(%q) = %q, want %q", in, got, want)
		}
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
