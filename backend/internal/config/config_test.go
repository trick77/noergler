package config

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/lang"
)

const minimalTeams = `
teams:
  - slug: platform
    webhook_secret_env: TEAM_PLATFORM_WEBHOOK_SECRET
    projects:
      - key: PLAT
    inference:
      api_key_env: TEAM_PLATFORM_OPENAI_API_KEY
`

const twoTeams = minimalTeams + `
  - slug: payments
    name: "Payments"
    webhook_secret_env: TEAM_PAYMENTS_WEBHOOK_SECRET
    projects:
      - key: PAY
        repos: [billing, ledger]
    inference:
      api_key_env: TEAM_PAYMENTS_OPENAI_API_KEY
      model: gpt-5.5
      reasoning_effort: medium
      context_window: 1200000
    review:
      auto_review_authors: [alice]
      ignore_authors: [renovate-bot]
      exclude_source_branches: [dependabot/*]
      exclude_target_branches: [release/*]
      max_pr_cost_usd: 8.5
    jira:
      acceptance_criteria_prefixes: [AC, DoD]
    riptide:
      url: https://riptide-payments.example.com/
      token_env: TEAM_PAYMENTS_RIPTIDE_TOKEN
    almanach:
      url: https://almanach.example.com/
      token_env: TEAM_PAYMENTS_ALMANACH_TOKEN
`

type env struct {
	t    *testing.T
	vars map[string]string
	path string
}

func newEnv(t *testing.T) *env {
	t.Helper()
	path := filepath.Join(t.TempDir(), "teams.yaml")
	e := &env{t: t, path: path, vars: map[string]string{
		"BITBUCKET_URL":                "https://bb.example.com",
		"BITBUCKET_TOKEN":              "tok",
		"BITBUCKET_USERNAME":           "bot",
		"LLMWIRE_LITELLM_BASE_URL":     "https://llm.example.com/v1",
		"LLMWIRE_LITELLM_MODELS":       "gpt-5.5=ai-gateway-gpt-5.5,gpt-5.4=ai-gateway-gpt-5.4",
		"OPENAI_MODEL":                 "gpt-5.5",
		"JIRA_URL":                     "https://jira.example.com",
		"JIRA_TOKEN":                   "jira-tok",
		"DATABASE_URL":                 "postgresql://u:p@localhost/db",
		"NOERGLER_SECRETS_KEY":         strings.Repeat("ab", 32),
		"TEAMS_CONFIG":                 path,
		"TEAM_PLATFORM_WEBHOOK_SECRET": "plat-secret",
		"TEAM_PLATFORM_OPENAI_API_KEY": "plat-key",
	}}
	e.teams(minimalTeams)
	return e
}

func (e *env) set(kv ...string) {
	for i := 0; i+1 < len(kv); i += 2 {
		e.vars[kv[i]] = kv[i+1]
	}
}
func (e *env) unset(names ...string) {
	for _, n := range names {
		delete(e.vars, n)
	}
}
func (e *env) teams(text string) {
	if err := os.WriteFile(e.path, []byte(text), 0o600); err != nil {
		e.t.Fatal(err)
	}
}
func (e *env) lookup(name string) (string, bool) { v, ok := e.vars[name]; return v, ok }
func (e *env) payments() {
	e.set("TEAM_PAYMENTS_WEBHOOK_SECRET", "pay-secret", "TEAM_PAYMENTS_OPENAI_API_KEY", "pay-key",
		"TEAM_PAYMENTS_RIPTIDE_TOKEN", "pay-riptide", "TEAM_PAYMENTS_ALMANACH_TOKEN", "pay-almanach")
}
func (e *env) load() (*App, error) { return Load(e.lookup) }
func (e *env) mustLoad() *App {
	e.t.Helper()
	app, err := e.load()
	if err != nil {
		e.t.Fatalf("load: %v", err)
	}
	return app
}

// --- instance env ------------------------------------------------------------

func TestInstance_RequiredVarsAndDefaults(t *testing.T) {
	e := newEnv(t)
	app := e.mustLoad()
	// Unset means the model's balanced level: no level name is a default
	// in code, since which ones exist is the model's profile's business.
	if app.LLM.ReasoningEffort != "" || app.Teams["platform"].LLM.ReasoningEffort != "" {
		t.Errorf("reasoning effort default: %q, want unset", app.LLM.ReasoningEffort)
	}
	if app.Review.OptOutBranchKeyword != "noergloff" || app.Review.MaxComments != 25 || app.Review.MaxPRCostUSD != 5.0 {
		t.Errorf("review defaults: %+v", app.Review)
	}
	if app.Review.Language != lang.DE || app.Teams["platform"].Review.Language != lang.DE {
		t.Errorf("language default = %q, want de", app.Review.Language)
	}
	if !app.Review.TicketComplianceCheck || !app.Review.DiffAllowDynamicContext || !app.Review.RequireAgentsMD {
		t.Errorf("bool defaults: %+v", app.Review)
	}
	// The AGENTS.md budget: the warn threshold is advisory, the max is the
	// gate that skips a review outright, so both are pinned.
	if app.Review.AgentsMDMaxTokens != 7000 || app.Review.AgentsMDWarnTokens != 4000 {
		t.Errorf("AGENTS.md token defaults: %+v", app.Review)
	}
	if app.Review.MaxFileLines != 1000 {
		t.Errorf("max file lines default = %d, want 1000", app.Review.MaxFileLines)
	}
	if !reflect.DeepEqual(app.Review.ExcludeRepos, []string{"*-infra"}) || len(app.Review.AutoReviewAuthors) != 0 {
		t.Errorf("list defaults: %+v", app.Review)
	}
	if app.Server.Host != "0.0.0.0" || app.Server.Port != 8080 || app.Trust.Threshold != 256000 {
		t.Errorf("server/trust defaults: %+v %+v", app.Server, app.Trust)
	}
	// The diff is uncapped by default and one file body is capped at 1 MiB.
	// A diff's size is mostly files IsReviewable discards, so capping it
	// refuses a PR over bytes that never become resident.
	if app.Bitbucket.MaxDiffBytes != 0 || app.Bitbucket.MaxFileBytes != 1024*1024 {
		t.Errorf("byte cap defaults: %+v", app.Bitbucket)
	}
	if app.LLM.APIKey != "" || app.Teams["platform"].LLM.APIKey != "plat-key" || app.Teams["platform"].WebhookSecret != "plat-secret" {
		t.Error("instance has no key; the team's come from its env vars")
	}
	if !reflect.DeepEqual(app.Jira.AcceptanceCriteriaPrefixes, DefaultAcceptanceCriteriaPrefixes) {
		t.Errorf("prefixes: %v", app.Jira.AcceptanceCriteriaPrefixes)
	}
}

// Unset is "Last First"; blank set explicitly is off; a bad or first-less
// pattern aborts boot rather than never naming anyone.
func TestInstance_DisplayNamePattern(t *testing.T) {
	e := newEnv(t)
	if re := e.mustLoad().Bitbucket.DisplayNamePattern; re == nil || re.String() != DefaultDisplayNamePattern {
		t.Errorf("default = %v", re)
	}
	e.set("BITBUCKET_DISPLAY_NAME_PATTERN", " ")
	if re := e.mustLoad().Bitbucket.DisplayNamePattern; re != nil {
		t.Errorf("blank = %v, want off", re)
	}
	e.set("BITBUCKET_DISPLAY_NAME_PATTERN", `^(?P<first>\S+) (?P<last>\S+)$`)
	if re := e.mustLoad().Bitbucket.DisplayNamePattern; re == nil || re.FindStringSubmatch("Jan Saner")[1] != "Jan" {
		t.Errorf("override = %v", re)
	}
	for _, bad := range []string{`(?P<first>`, `^(\S+)`} {
		e.set("BITBUCKET_DISPLAY_NAME_PATTERN", bad)
		if _, err := e.load(); err == nil || !strings.Contains(err.Error(), "BITBUCKET_DISPLAY_NAME_PATTERN") {
			t.Errorf("%q: err = %v", bad, err)
		}
	}
}

func TestInstance_MissingRequiredIsFatalAndNamesTheVariable(t *testing.T) {
	for _, name := range []string{"OPENAI_MODEL", "LLMWIRE_LITELLM_BASE_URL", "LLMWIRE_LITELLM_MODELS", "DATABASE_URL", "BITBUCKET_URL", "NOERGLER_SECRETS_KEY"} {
		e := newEnv(t)
		e.unset(name)
		_, err := e.load()
		if err == nil || !strings.Contains(err.Error(), name+" is required") {
			t.Errorf("%s: err = %v", name, err)
		}
	}
}

func TestInstance_SecretsKeyMustBe32HexBytes(t *testing.T) {
	for _, bad := range []string{"not-hex", strings.Repeat("ab", 16), strings.Repeat("ab", 33)} {
		e := newEnv(t)
		e.set("NOERGLER_SECRETS_KEY", bad)
		if _, err := e.load(); err == nil || !strings.Contains(err.Error(), "NOERGLER_SECRETS_KEY") {
			t.Errorf("%q: err = %v", bad, err)
		}
	}
	if key := newEnv(t).mustLoad().SecretsKey; len(key) != 32 {
		t.Errorf("valid key parsed to %d bytes", len(key))
	}
}

func TestInstance_ParseRules(t *testing.T) {
	e := newEnv(t)
	e.set("REVIEW_DIFF_EXTRA_LINES_BEFORE", " 5 ", "REVIEW_DIFF_EXTRA_LINES_AFTER", "4",
		"REVIEW_TICKET_COMPLIANCE_CHECK", "False", "REVIEW_REQUIRE_AGENTS_MD", "YES", "REVIEW_DIFF_ALLOW_DYNAMIC_CONTEXT", "1",
		"REVIEW_IGNORE_AUTHORS", "os-jenkins-bb, renovate,,", "REVIEW_MAX_PR_COST_USD", "8.50",
		"OPENAI_REASONING_EFFORT", "HIGH", "REVIEW_OPT_OUT_BRANCH_KEYWORD", "skipme",
		"NOERGLER_PUBLIC_URL", "https://n.example.com/")
	app := e.mustLoad()
	r := app.Review
	if r.DiffExtraLinesBefore != 5 || r.DiffExtraLinesAfter != 4 || r.TicketComplianceCheck || !r.RequireAgentsMD || !r.DiffAllowDynamicContext {
		t.Errorf("parsed: %+v", r)
	}
	if !reflect.DeepEqual(r.IgnoreAuthors, []string{"os-jenkins-bb", "renovate"}) || r.MaxPRCostUSD != 8.5 || r.OptOutBranchKeyword != "skipme" {
		t.Errorf("parsed: %+v", r)
	}
	if app.LLM.ReasoningEffort != "high" || app.LLM.BaseURL != "https://llm.example.com/v1" || app.Server.PublicURL != "https://n.example.com" {
		t.Errorf("llm/server: %+v %+v", app.LLM, app.Server)
	}
}

func TestInstance_BadIntAbortsBoot(t *testing.T) {
	e := newEnv(t)
	e.set("REVIEW_MAX_COMMENTS", "many", "SERVER_PORT", "8o80")
	_, err := e.load()
	if err == nil || !strings.Contains(err.Error(), "REVIEW_MAX_COMMENTS") || !strings.Contains(err.Error(), "SERVER_PORT") {
		t.Errorf("err = %v", err)
	}
}

func TestInstance_ByteCapsOverridden(t *testing.T) {
	e := newEnv(t)
	e.set("BITBUCKET_MAX_DIFF_BYTES", "2048", "BITBUCKET_MAX_FILE_BYTES", "512")
	app := e.mustLoad()
	if app.Bitbucket.MaxDiffBytes != 2048 || app.Bitbucket.MaxFileBytes != 512 {
		t.Errorf("byte caps: %+v", app.Bitbucket)
	}
}

// A bad value must not kill the process at startup.
func TestInstance_BadByteCapAbortsBoot(t *testing.T) {
	e := newEnv(t)
	e.set("BITBUCKET_MAX_DIFF_BYTES", "10MiB")
	_, err := e.load()
	if err == nil || !strings.Contains(err.Error(), "BITBUCKET_MAX_DIFF_BYTES") {
		t.Errorf("err = %v", err)
	}
}

// A zero or negative per-file cap would boot clean and then reject every file,
// so it has to fail at load rather than at the first review.
func TestInstance_NonPositiveByteCapAbortsBoot(t *testing.T) {
	for _, bad := range []string{"0", "-1"} {
		e := newEnv(t)
		e.set("BITBUCKET_MAX_FILE_BYTES", bad)
		_, err := e.load()
		if err == nil || !strings.Contains(err.Error(), "BITBUCKET_MAX_FILE_BYTES") {
			t.Errorf("%s: err = %v, want a load failure", bad, err)
		}
	}
}

// 0 on the DIFF cap is unlimited, not "refuse everything". It is the default,
// so a validator that rejected it the way the file cap does would refuse every
// PR out of the box.
func TestInstance_ZeroDiffCapIsUnlimited(t *testing.T) {
	e := newEnv(t)
	e.set("BITBUCKET_MAX_DIFF_BYTES", "0")
	app := e.mustLoad()
	if app.Bitbucket.MaxDiffBytes != 0 {
		t.Errorf("max_diff_bytes = %d, want 0 (unlimited)", app.Bitbucket.MaxDiffBytes)
	}
}

// Negative is still a typo, not a third meaning.
func TestInstance_NegativeDiffCapAbortsBoot(t *testing.T) {
	e := newEnv(t)
	e.set("BITBUCKET_MAX_DIFF_BYTES", "-1")
	_, err := e.load()
	if err == nil || !strings.Contains(err.Error(), "BITBUCKET_MAX_DIFF_BYTES") {
		t.Errorf("err = %v, want a load failure", err)
	}
}

// A negative comment cap boots clean and then panics every review in post,
// after inference is paid. 0 stays legal: summary only, no inline comments.
func TestInstance_NegativeMaxCommentsAbortsBoot(t *testing.T) {
	e := newEnv(t)
	e.set("REVIEW_MAX_COMMENTS", "-1")
	_, err := e.load()
	if err == nil || !strings.Contains(err.Error(), "REVIEW_MAX_COMMENTS") {
		t.Errorf("err = %v, want a load failure", err)
	}
	e.set("REVIEW_MAX_COMMENTS", "0")
	if app := e.mustLoad(); app.Review.MaxComments != 0 {
		t.Errorf("max_comments = %d, want 0", app.Review.MaxComments)
	}
}

// A cap of 0 reads as "no cap" but skips every auto-review after the first
// priced run; negative is a typo. Both fail at load.
func TestInstance_NonPositiveCostCapAbortsBoot(t *testing.T) {
	for _, bad := range []string{"0", "-1", "NaN"} {
		e := newEnv(t)
		e.set("REVIEW_MAX_PR_COST_USD", bad)
		_, err := e.load()
		if err == nil || !strings.Contains(err.Error(), "REVIEW_MAX_PR_COST_USD") {
			t.Errorf("%s: err = %v, want a load failure", bad, err)
		}
	}
}

func TestInstance_LanguageIsDeOrEn(t *testing.T) {
	e := newEnv(t)
	e.set("REVIEW_LANGUAGE", "EN")
	if app := e.mustLoad(); app.Review.Language != lang.EN {
		t.Errorf("language = %q, want en", app.Review.Language)
	}
	e = newEnv(t)
	e.set("REVIEW_LANGUAGE", "fr")
	if _, err := e.load(); err == nil || !strings.Contains(err.Error(), "REVIEW_LANGUAGE") {
		t.Errorf("err = %v, want a load failure", err)
	}
}

func TestTeam_LanguageOverride(t *testing.T) {
	e := newEnv(t)
	e.teams(strings.Replace(twoTeams, "      max_pr_cost_usd: 8.5\n", "      max_pr_cost_usd: 8.5\n      language: en\n", 1))
	e.payments()
	app := e.mustLoad()
	if got := app.Teams["payments"].Review.Language; got != lang.EN {
		t.Errorf("payments language = %q, want en", got)
	}
	if got := app.Teams["platform"].Review.Language; got != lang.DE {
		t.Errorf("platform language = %q, want the instance de", got)
	}
}

// A team's blank level overrides an instance level back to the model's
// balanced one.
func TestTeams_BlankEffortOverridesToUnset(t *testing.T) {
	e := newEnv(t)
	e.set("OPENAI_REASONING_EFFORT", "high")
	e.teams(strings.Replace(twoTeams, "reasoning_effort: medium", "reasoning_effort: ''", 1))
	e.payments()
	app := e.mustLoad()
	if got := app.Teams["payments"].LLM.ReasoningEffort; got != "" {
		t.Errorf("payments effort = %q, want unset", got)
	}
	if got := app.Teams["platform"].LLM.ReasoningEffort; got != "high" {
		t.Errorf("platform effort = %q, want the instance's", got)
	}
}

// A blank level is the unset one, not an error: the model's balanced level.
func TestInstance_BlankReasoningEffortIsUnset(t *testing.T) {
	e := newEnv(t)
	e.set("OPENAI_REASONING_EFFORT", "  ")
	app, err := e.load()
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if app.LLM.ReasoningEffort != "" {
		t.Errorf("ReasoningEffort = %q, want unset", app.LLM.ReasoningEffort)
	}
}

func TestInstance_TeamsConfigPathDefault(t *testing.T) {
	e := newEnv(t)
	e.unset("TEAMS_CONFIG")
	app, err := LoadInstance(e.lookup)
	if err != nil || app.TeamsConfigPath != "teams.yaml" {
		t.Errorf("path = %v, err = %v", app, err)
	}
}

// --- teams.yaml: file-level faults abort -------------------------------------

func fileError(t *testing.T, e *env, want string) {
	t.Helper()
	_, err := e.load()
	var fe *TeamsFileError
	if !errors.As(err, &fe) || !strings.Contains(err.Error(), want) {
		t.Errorf("err = %v, want TeamsFileError containing %q", err, want)
	}
}

func TestTeamsFile_Faults(t *testing.T) {
	e := newEnv(t)
	e.set("TEAMS_CONFIG", filepath.Join(filepath.Dir(e.path), "nope.yaml"))
	fileError(t, e, "not found")

	e = newEnv(t)
	e.teams("teams: [unclosed")
	fileError(t, e, "not valid YAML")

	e = newEnv(t)
	e.teams("foo: bar")
	fileError(t, e, "top-level `teams:`")

	e = newEnv(t)
	e.teams("teams: []")
	fileError(t, e, "non-empty")

	e = newEnv(t)
	e.teams("teams:\n  - just a string\n")
	fileError(t, e, "teams[0] must be a mapping")

	e = newEnv(t)
	e.set("TEAMS_CONFIG", filepath.Dir(e.path))
	fileError(t, e, "cannot be read")

	e = newEnv(t)
	e.teams(minimalTeams + strings.Replace(minimalTeams, "teams:\n", "", 1))
	fileError(t, e, "duplicate slug")
}

func TestTeamsFile_SluglessBlocksAreDisabledNotDuplicates(t *testing.T) {
	e := newEnv(t)
	e.teams("teams:\n  - {webhook_secret_env: A, projects: [{key: X}], inference: {api_key_env: B}}\n" +
		"  - {webhook_secret_env: A, projects: [{key: Y}], inference: {api_key_env: B}}\n" +
		strings.Replace(minimalTeams, "teams:\n", "", 1))
	app := e.mustLoad()
	if !reflect.DeepEqual(app.Order, []string{"platform"}) {
		t.Errorf("enabled = %v", app.Order)
	}
	if len(app.Disabled) != 2 {
		t.Fatalf("disabled = %v", app.Disabled)
	}
	for _, slug := range []string{"teams[0]", "teams[1]"} {
		if !strings.HasPrefix(app.Disabled[slug], "slug: Field required") {
			t.Errorf("%s: %q", slug, app.Disabled[slug])
		}
	}
}

// A team that fails validation keeps its display name: the dashboard lists
// disabled teams and would otherwise print the slug for this one only.
func TestTeamsFile_DisabledTeamKeepsItsName(t *testing.T) {
	e := newEnv(t)
	e.teams("teams:\n  - {slug: hotspot, name: Hotspot, bogus: 1}\n")
	app := e.mustLoad()
	if _, ok := app.Disabled["hotspot"]; !ok {
		t.Fatalf("disabled = %v, want hotspot", app.Disabled)
	}
	if got := app.Names["hotspot"]; got != "Hotspot" {
		t.Errorf("Names[hotspot] = %q, want Hotspot", got)
	}
}

// --- teams.yaml: team-level faults disable that team only --------------------

// OwnModel compares values, not keys: a team that names the instance's own
// model and effort still runs on the default.
func TestTeams_OwnModelComparesWithTheInstance(t *testing.T) {
	cases := []struct {
		inference string
		want      bool
	}{
		{"{api_key_env: TEAM_PLATFORM_OPENAI_API_KEY}", false},
		{"{api_key_env: TEAM_PLATFORM_OPENAI_API_KEY, model: gpt-5.5}", false},
		{"{api_key_env: TEAM_PLATFORM_OPENAI_API_KEY, model: gpt-5.4}", true},
		{"{api_key_env: TEAM_PLATFORM_OPENAI_API_KEY, reasoning_effort: Medium}", true},
		{"{api_key_env: TEAM_PLATFORM_OPENAI_API_KEY, context_window: 1000}", false},
	}
	for _, c := range cases {
		e := newEnv(t)
		e.teams("teams:\n  - {slug: platform, webhook_secret_env: TEAM_PLATFORM_WEBHOOK_SECRET, projects: [{key: PLAT}], inference: " + c.inference + "}\n")
		app := e.mustLoad()
		if got := app.Teams["platform"].OwnModel; got != c.want {
			t.Errorf("%s: OwnModel = %v, want %v", c.inference, got, c.want)
		}
	}
}

func TestTeams_FullBlockResolvesOverridesOnTopOfInstanceDefaults(t *testing.T) {
	e := newEnv(t)
	e.teams(twoTeams)
	e.payments()
	e.set("OPENAI_MODEL", "gpt-5.4", "REVIEW_MAX_COMMENTS", "7", "REVIEW_IGNORE_AUTHORS", "os-jenkins-bb, renovate")
	app := e.mustLoad()
	if len(app.Disabled) != 0 || !reflect.DeepEqual(app.Order, []string{"platform", "payments"}) {
		t.Fatalf("disabled = %v, order = %v", app.Disabled, app.Order)
	}
	plat := app.Teams["platform"]
	if plat.Name != "platform" || plat.LLM.Model != "gpt-5.4" || plat.LLM.APIKey != "plat-key" || plat.LLM.BaseURL != "https://llm.example.com/v1" {
		t.Errorf("platform: %+v", plat)
	}
	if plat.Review.MaxComments != 7 || len(plat.Review.AutoReviewAuthors) != 0 || !reflect.DeepEqual(plat.Review.IgnoreAuthors, []string{"os-jenkins-bb", "renovate"}) {
		t.Errorf("platform review: %+v", plat.Review)
	}
	if !reflect.DeepEqual(plat.Jira.AcceptanceCriteriaPrefixes, DefaultAcceptanceCriteriaPrefixes) || plat.Riptide != nil || plat.Almanach != nil {
		t.Errorf("platform jira/riptide/almanach: %+v %+v %+v", plat.Jira, plat.Riptide, plat.Almanach)
	}

	pay := app.Teams["payments"]
	if pay.Name != "Payments" || pay.WebhookSecret != "pay-secret" {
		t.Errorf("payments: %+v", pay)
	}
	if !reflect.DeepEqual(pay.Projects, []ProjectScope{{Key: "PAY", Repos: []string{"billing", "ledger"}}}) {
		t.Errorf("projects: %+v", pay.Projects)
	}
	if pay.LLM.Model != "gpt-5.5" || pay.LLM.ReasoningEffort != "medium" || pay.LLM.ContextWindow != 1_200_000 || pay.LLM.APIKey != "pay-key" || pay.LLM.BaseURL != "https://llm.example.com/v1" {
		t.Errorf("payments llm: %+v", pay.LLM)
	}
	if !reflect.DeepEqual(pay.Review.AutoReviewAuthors, []string{"alice"}) || !reflect.DeepEqual(pay.Review.IgnoreAuthors, []string{"renovate-bot"}) ||
		pay.Review.MaxPRCostUSD != 8.5 || pay.Review.MaxComments != 7 ||
		!reflect.DeepEqual(pay.Review.ExcludeSourceBranches, []string{"dependabot/*"}) ||
		!reflect.DeepEqual(pay.Review.ExcludeTargetBranches, []string{"release/*"}) {
		t.Errorf("payments review: %+v", pay.Review)
	}
	if pay.Jira.URL != "https://jira.example.com" || pay.Jira.Token != "jira-tok" || !reflect.DeepEqual(pay.Jira.AcceptanceCriteriaPrefixes, []string{"AC", "DoD"}) {
		t.Errorf("payments jira: %+v", pay.Jira)
	}
	if !reflect.DeepEqual(pay.Riptide, &Riptide{URL: "https://riptide-payments.example.com", Token: "pay-riptide"}) {
		t.Errorf("payments riptide: %+v", pay.Riptide)
	}
	if !reflect.DeepEqual(pay.Almanach, &Almanach{URL: "https://almanach.example.com", Token: "pay-almanach"}) {
		t.Errorf("payments almanach: %+v", pay.Almanach)
	}
	if app.AlmanachLimits.NoteMaxChars != 6000 {
		t.Errorf("ALMANACH_NOTE_MAX_CHARS default = %d, want 6000", app.AlmanachLimits.NoteMaxChars)
	}
	if plat.OwnModel || !pay.OwnModel {
		t.Errorf("OwnModel: platform %v, payments %v; want false, true", plat.OwnModel, pay.OwnModel)
	}
	// Instance lists are not shared with the team's copies.
	plat.Review.ExcludeRepos[0] = "changed"
	if app.Review.ExcludeRepos[0] == "changed" {
		t.Error("team review lists alias the instance lists")
	}
}

func TestTeams_FaultDisablesOnlyThatTeam(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(string) string
		reason string
	}{
		{"missing key var", func(s string) string {
			return strings.ReplaceAll(s, "TEAM_PAYMENTS_OPENAI_API_KEY", "TEAM_PAYMENTS_MISSING")
		},
			"inference.api_key_env: environment variable TEAM_PAYMENTS_MISSING is not set"},
		{"missing secret var", func(s string) string {
			return strings.ReplaceAll(s, "TEAM_PAYMENTS_WEBHOOK_SECRET", "TEAM_PAYMENTS_MISSING")
		},
			"webhook_secret_env: environment variable TEAM_PAYMENTS_MISSING is not set"},
		{"missing riptide var", func(s string) string {
			return strings.ReplaceAll(s, "TEAM_PAYMENTS_RIPTIDE_TOKEN", "TEAM_PAYMENTS_MISSING")
		},
			"riptide.token_env: environment variable TEAM_PAYMENTS_MISSING is not set"},
		{"no webhook_secret_env", func(s string) string {
			return strings.Replace(s, "    webhook_secret_env: TEAM_PAYMENTS_WEBHOOK_SECRET\n", "", 1)
		}, "webhook_secret_env: Field required"},
		{"no api_key_env", func(s string) string {
			return strings.Replace(s, "      api_key_env: TEAM_PAYMENTS_OPENAI_API_KEY\n", "", 1)
		},
			"inference.api_key_env: Field required"},
		{"empty repos", func(s string) string { return strings.Replace(s, "repos: [billing, ledger]", "repos: []", 1) },
			"projects.0.repos: Value error, repos must list at least one slug"},
		{"base_url in team", func(s string) string {
			return strings.Replace(s, "      model: gpt-5.5\n", "      model: gpt-5.5\n      base_url: https://x\n", 1)
		}, "inference.base_url: Extra inputs are not permitted"},
		{"review template in team", func(s string) string {
			return strings.Replace(s, "      max_pr_cost_usd: 8.5\n", "      max_pr_cost_usd: 8.5\n      review_prompt_template: x\n", 1)
		}, "review.review_prompt_template: no longer read: the prompts are compiled into the binary"},
		{"mention template in team", func(s string) string {
			return strings.Replace(s, "      max_pr_cost_usd: 8.5\n", "      max_pr_cost_usd: 8.5\n      mention_prompt_template: x\n", 1)
		}, "review.mention_prompt_template: no longer read: the prompts are compiled into the binary"},
		{"review typo", func(s string) string {
			return strings.Replace(s, "      max_pr_cost_usd: 8.5\n", "      max_pr_cost_usd: 8.5\n      max_comment: 3\n", 1)
		}, "review.max_comment: Extra inputs are not permitted"},
		{"riptide without token_env", func(s string) string {
			return strings.Replace(s, "      token_env: TEAM_PAYMENTS_RIPTIDE_TOKEN\n", "", 1)
		}, "riptide.token_env: Field required"},
		{"riptide without url", func(s string) string {
			return strings.Replace(s, "      url: https://riptide-payments.example.com/\n", "", 1)
		}, "riptide.url: Field required"},
		{"riptide empty url", func(s string) string {
			return strings.Replace(s, "url: https://riptide-payments.example.com/", "url: ''", 1)
		}, "riptide.url must be non-empty"},
		{"missing almanach var", func(s string) string {
			return strings.ReplaceAll(s, "TEAM_PAYMENTS_ALMANACH_TOKEN", "TEAM_PAYMENTS_MISSING")
		}, "almanach.token_env: environment variable TEAM_PAYMENTS_MISSING is not set"},
		{"almanach without token_env", func(s string) string {
			return strings.Replace(s, "      token_env: TEAM_PAYMENTS_ALMANACH_TOKEN\n", "", 1)
		}, "almanach.token_env: Field required"},
		{"almanach without url", func(s string) string {
			return strings.Replace(s, "      url: https://almanach.example.com/\n", "", 1)
		}, "almanach.url: Field required"},
		{"almanach empty url", func(s string) string {
			return strings.Replace(s, "url: https://almanach.example.com/", "url: ''", 1)
		}, "almanach.url must be non-empty"},
		{"almanach typo", func(s string) string {
			return strings.Replace(s, "      token_env: TEAM_PAYMENTS_ALMANACH_TOKEN\n", "      token_env: TEAM_PAYMENTS_ALMANACH_TOKEN\n      note_max_chars: 1\n", 1)
		}, "almanach.note_max_chars: Extra inputs are not permitted"},
		{"bad slug", func(s string) string { return strings.Replace(s, "slug: payments", "slug: Payments", 1) },
			"slug: Value error, slug 'Payments' must match"},
		{"negative max_comments", func(s string) string {
			return strings.Replace(s, "      max_pr_cost_usd: 8.5\n", "      max_pr_cost_usd: 8.5\n      max_comments: -1\n", 1)
		}, "review.max_comments: Input should be greater than or equal to 0"},
		{"zero max_pr_cost_usd", func(s string) string {
			return strings.Replace(s, "max_pr_cost_usd: 8.5", "max_pr_cost_usd: 0", 1)
		}, "review.max_pr_cost_usd: Input should be greater than 0"},
		{"unsupported language", func(s string) string {
			return strings.Replace(s, "      max_pr_cost_usd: 8.5\n", "      max_pr_cost_usd: 8.5\n      language: fr\n", 1)
		}, "review.language: Input should be 'de' or 'en'"},
		{"bad int", func(s string) string { return strings.Replace(s, "context_window: 1200000", "context_window: lots", 1) },
			"inference.context_window: Input should be a valid integer"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := newEnv(t)
			e.teams(tc.mutate(twoTeams))
			e.payments()
			app := e.mustLoad()
			if !reflect.DeepEqual(app.Order, []string{"platform"}) {
				t.Fatalf("enabled = %v", app.Order)
			}
			if len(app.Disabled) != 1 {
				t.Fatalf("disabled = %v", app.Disabled)
			}
			for slug, got := range app.Disabled {
				if strings.ToLower(slug) != "payments" {
					t.Errorf("keyed by the raw slug even when invalid, got %q", slug)
				}
				if !strings.HasPrefix(got, tc.reason) {
					t.Errorf("reason = %q, want prefix %q", got, tc.reason)
				}
			}
		})
	}
}

func TestTeams_EmptySecretValueDisablesTheTeam(t *testing.T) {
	e := newEnv(t)
	e.teams(twoTeams)
	e.payments()
	e.set("TEAM_PAYMENTS_OPENAI_API_KEY", "   ")
	app := e.mustLoad()
	want := map[string]string{"payments": "inference.api_key_env: environment variable TEAM_PAYMENTS_OPENAI_API_KEY is empty"}
	if !reflect.DeepEqual(app.Disabled, want) {
		t.Errorf("disabled = %v", app.Disabled)
	}
}

func TestTeams_ProjectsAreAnOptionalSeed(t *testing.T) {
	e := newEnv(t)
	e.teams(strings.Replace(twoTeams, "      - key: PAY\n        repos: [billing, ledger]\n", "      []\n", 1))
	e.payments()
	app := e.mustLoad()
	if len(app.Disabled) != 0 || len(app.Teams["payments"].Projects) != 0 {
		t.Errorf("disabled = %v, projects = %v", app.Disabled, app.Teams["payments"].Projects)
	}
	// Two blocks naming the same project both load: claims are the DB's.
	e.teams(strings.ReplaceAll(twoTeams, "key: PAY", "key: PLAT"))
	app = e.mustLoad()
	if len(app.Teams) != 2 {
		t.Errorf("teams = %v", app.Order)
	}
	// A block without projects: at all.
	e.teams(strings.Replace(minimalTeams, "    projects:\n      - key: PLAT\n", "", 1))
	app = e.mustLoad()
	if len(app.Disabled) != 0 || app.Teams["platform"].Projects == nil {
		t.Errorf("disabled = %v, projects = %v", app.Disabled, app.Teams["platform"].Projects)
	}
}

func TestTeam_Ownership(t *testing.T) {
	e := newEnv(t)
	e.teams(twoTeams)
	e.payments()
	app := e.mustLoad()
	plat, pay := app.Teams["platform"], app.Teams["payments"]
	if !plat.Owns("PLAT", "anything") || !pay.Owns("PAY", "billing") || pay.Owns("PAY", "other") || plat.Owns("NOPE", "x") {
		t.Error("ownership")
	}
	if !plat.ReviewsRepo("PLAT", "api") || plat.ReviewsRepo("PLAT", "cluster-infra") || plat.ReviewsRepo("PLAT", "Cluster-INFRA") {
		t.Error("exclude_repos carve-out, case-insensitive")
	}
	pay.Projects = []ProjectScope{{Key: "PAY", Repos: []string{"pay-infra"}}}
	if !pay.ReviewsRepo("PAY", "pay-infra") || !pay.ClaimsRepoExplicitly("PAY", "pay-infra") || pay.ClaimsRepoExplicitly("PAY", "x") {
		t.Error("an explicitly claimed repo is never carved out")
	}
}

// Source first, case-insensitive, `*` crosses `/`; "" when nothing matches.
func TestTeam_ExcludedBranch(t *testing.T) {
	team := &Team{Review: Review{
		ExcludeSourceBranches: []string{"Dependabot/*", "wip-?"},
		ExcludeTargetBranches: []string{"release/*", "refs/heads/hotfix/*"},
	}}
	cases := []struct{ from, to, want string }{
		{"dependabot/npm/lodash", "master", "exclude_source_branches"},
		{"DEPENDABOT/x", "master", "exclude_source_branches"},
		{"wip-1", "master", "exclude_source_branches"},
		{"feature/x", "release/2026/10", "exclude_target_branches"},
		{"dependabot/x", "release/1", "exclude_source_branches"},
		{"feature/dependabot", "master", ""},
		{"wip-12", "releases", ""},
		// A full-ref pattern matches as its short form.
		{"feature/x", "hotfix/9", "exclude_target_branches"},
	}
	for _, c := range cases {
		if got := team.ExcludedBranch(c.from, c.to); got != c.want {
			t.Errorf("ExcludedBranch(%q, %q) = %q, want %q", c.from, c.to, got, c.want)
		}
	}
	if (&Team{}).ExcludedBranch("anything", "anything") != "" {
		t.Error("empty lists exclude nothing")
	}
}

// The instance lists seed a team that sets none; blank is empty.
func TestReview_ExcludeBranchEnv(t *testing.T) {
	e := newEnv(t)
	e.teams(minimalTeams)
	app := e.mustLoad()
	if len(app.Review.ExcludeSourceBranches) != 0 || len(app.Review.ExcludeTargetBranches) != 0 {
		t.Errorf("defaults = %v / %v, want empty", app.Review.ExcludeSourceBranches, app.Review.ExcludeTargetBranches)
	}
	e.set("REVIEW_EXCLUDE_SOURCE_BRANCHES", "dependabot/*,renovate/*", "REVIEW_EXCLUDE_TARGET_BRANCHES", "release/*")
	app = e.mustLoad()
	plat := app.Teams["platform"]
	if !reflect.DeepEqual(plat.Review.ExcludeSourceBranches, []string{"dependabot/*", "renovate/*"}) ||
		!reflect.DeepEqual(plat.Review.ExcludeTargetBranches, []string{"release/*"}) {
		t.Errorf("team = %v / %v", plat.Review.ExcludeSourceBranches, plat.Review.ExcludeTargetBranches)
	}
	plat.Review.ExcludeSourceBranches[0] = "changed"
	if app.Review.ExcludeSourceBranches[0] == "changed" {
		t.Error("team branch lists alias the instance lists")
	}
}

func TestFnMatch(t *testing.T) {
	cases := []struct {
		pattern, name string
		want          bool
	}{
		{"*-infra", "cluster-infra", true}, {"*-infra", "infra", false}, {"*", "", true}, {"*", "a/b", true},
		{"a?c", "abc", true}, {"a?c", "ac", false}, {"[abc]x", "bx", true}, {"[!abc]x", "bx", false},
		{"[a-c]x", "cx", true}, {"[a-c]x", "dx", false}, {"[", "[", true}, {"[]]", "]", true},
		{"lib-*-core", "lib-a-b-core", true}, {"exact", "exact", true}, {"exact", "exactly", false},
	}
	for _, tc := range cases {
		if got := FnMatch(tc.pattern, tc.name); got != tc.want {
			t.Errorf("FnMatch(%q, %q) = %v", tc.pattern, tc.name, got)
		}
	}
}

func TestModelLabel(t *testing.T) {
	if ModelLabel("gpt-5.5", "high") != "gpt-5.5-high" || ModelLabel("m", "") != "m" {
		t.Error("label")
	}
}
