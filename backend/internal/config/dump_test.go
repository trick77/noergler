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
		"[config.queue]", "  inference_concurrency = 6", "  inference_concurrency_per_team = 2",
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

// A bare 0 reads as "no context window" when it means the limit is read from
// the gateway. The dump runs before any team starts, so this line is all the
// operator sees until inference logs the resolved value.
func TestDump_ContextWindowZeroSaysWhereTheWindowComesFrom(t *testing.T) {
	e := newEnv(t)
	e.teams(twoTeams)
	e.payments()
	app := e.mustLoad()

	var buf bytes.Buffer
	Dump(app, slog.New(logging.NewHandler(&buf, slog.LevelInfo, "test")))
	text := buf.String()

	if !strings.Contains(text, "context_window = from gateway") {
		t.Error("an unset context window should name the gateway as its source")
	}
	if strings.Contains(text, "context_window = 0") {
		t.Error("a bare 0 reads as no window at all")
	}
	if !strings.Contains(text, "reasoning_effort = model balanced") {
		t.Error("an unset level should say the model's balanced level applies")
	}
}

// The diff cap defaults to 0, and a bare 0 reads as "refuses every diff" -
// which is exactly what the field used to mean, so it has to say otherwise.
func TestDump_ZeroDiffCapSaysUnlimited(t *testing.T) {
	e := newEnv(t)
	e.teams(twoTeams)
	e.payments()
	app := e.mustLoad()

	var buf bytes.Buffer
	Dump(app, slog.New(logging.NewHandler(&buf, slog.LevelInfo, "test")))
	text := buf.String()

	if !strings.Contains(text, "max_diff_bytes = unlimited") {
		t.Error("an unset diff cap should print as unlimited")
	}
	if strings.Contains(text, "max_diff_bytes = 0") {
		t.Error("a bare 0 reads as a cap of zero bytes")
	}
	// The per-file cap is a real number and stays one.
	if !strings.Contains(text, "max_file_bytes = 1048576") {
		t.Error("the file cap should print its value")
	}
}

func TestDump_ExplicitDiffCapIsPrintedAsTheNumber(t *testing.T) {
	e := newEnv(t)
	e.teams(twoTeams)
	e.payments()
	e.set("BITBUCKET_MAX_DIFF_BYTES", "2048")
	app := e.mustLoad()

	var buf bytes.Buffer
	Dump(app, slog.New(logging.NewHandler(&buf, slog.LevelInfo, "test")))

	if !strings.Contains(buf.String(), "max_diff_bytes = 2048") {
		t.Error("an explicit cap should print its value")
	}
}

func TestDump_ExplicitContextWindowIsPrintedAsTheNumber(t *testing.T) {
	e := newEnv(t)
	e.teams(twoTeams)
	e.payments()
	e.set("OPENAI_CONTEXT_WINDOW", "2000000")
	app := e.mustLoad()

	var buf bytes.Buffer
	Dump(app, slog.New(logging.NewHandler(&buf, slog.LevelInfo, "test")))

	if text := buf.String(); !strings.Contains(text, "context_window = 2000000") {
		t.Error("an explicit window is printed as the number it is")
	}
}

// The dump format keeps a float's decimal point. Go's default verb drops it,
// so max_pr_cost_usd printed as 5 and any search expecting a decimal missed.
func TestRender_FloatsKeepTheDecimalPoint(t *testing.T) {
	cases := map[float64]string{
		5.0:  "5.0",
		0.5:  "0.5",
		2.25: "2.25",
		0.0:  "0.0",
	}
	for in, want := range cases {
		if got := render(in); got != want {
			t.Errorf("render(%v) = %q, want %q", in, got, want)
		}
	}
}
