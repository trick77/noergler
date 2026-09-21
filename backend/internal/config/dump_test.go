package config

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/logging"
)

func TestDump_MasksSecretsAndKeepsTheSectionHeaders(t *testing.T) {
	e := newEnv(t)
	e.teams(twoTeams)
	e.payments()
	e.set("BITBUCKET_TOKEN", "secret-bb-token", "JIRA_TOKEN", "secret-jira-token",
		"DATABASE_URL", "postgresql://noergler:secret-db-password@localhost/noergler",
		"TEAM_PLATFORM_WEBHOOK_SECRET", "secret-webhook", "TEAM_PLATFORM_OPENAI_API_KEY", "secret-api-key",
		"TEAM_PAYMENTS_RIPTIDE_TOKEN", "secret-riptide-token", "REVIEW_AUTO_REVIEW_AUTHORS", "alice,bob", "SERVER_PORT", "9090")
	app := e.mustLoad()
	app.Disabled["broken"] = "TEAM_BROKEN_OPENAI_API_KEY is not set"

	var buf bytes.Buffer
	Dump(app, slog.New(logging.NewHandler(&buf, slog.LevelInfo, "test")))
	text := buf.String()

	for _, secret := range []string{"secret-bb-token", "secret-webhook", "secret-api-key", "secret-jira-token",
		"secret-riptide-token", "secret-db-password", "pay-secret", "pay-key", "pay-riptide"} {
		if strings.Contains(text, secret) {
			t.Errorf("secret %q in the dump", secret)
		}
	}
	for _, want := range []string{
		"[config.bitbucket]", "[config.llm]", "[config.review]", "[config.jira]", "[config.server]", "[config.database]",
		"[config.teams.platform]", "[config.teams.platform.llm]", "[config.teams.platform.riptide] disabled",
		"[config.teams.payments.riptide]", "https://bb.example.com", "gpt-5.5", "['alice', 'bob']", "9090",
		"projects = ['PLAT']", "projects = ['PAY/{billing,ledger}']",
		"[config.teams.broken] DISABLED: TEAM_BROKEN_OPENAI_API_KEY is not set",
		"  api_key = ***", "  token = ***", "  url = ***", "  webhook_secret = ***",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("missing %q", want)
		}
	}
	if !strings.Contains(text, `"log_level":"error"`) {
		t.Error("a disabled team is an error line")
	}
}
