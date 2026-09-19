// Package config is the two-layer configuration: the instance environment
// (what is physically one thing: the Bitbucket account, the Jira user, the
// database, the gateway, plus the defaults for every team-overridable knob)
// and teams.yaml (one block per team, secrets referenced by env var name).
//
// One team's bad block never affects another: a per-team fault disables that
// team (reason logged with team=<slug>) and the instance still boots. Only
// file-level faults (missing file, unparseable, zero teams, duplicate slug)
// and instance env faults abort startup.
package config

import (
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// RequiredWebhookEvents are the Bitbucket events the webhook dispatches on.
// Shared with onboarding, which registers exactly these.
var RequiredWebhookEvents = []string{
	"pr:opened",
	"pr:from_ref_updated",
	"pr:comment:added",
	"pr:comment:deleted",
	"pr:merged",
	"pr:declined",
	"pr:deleted",
}

// Env var names of the llmwire gateway contract (read by llmwire itself
// through Config.Lookup; held here so they are dumped and validated at boot).
const (
	GatewayBaseURLEnv = "LLMWIRE_LITELLM_BASE_URL"
	GatewayModelsEnv  = "LLMWIRE_LITELLM_MODELS"
)

// Bitbucket is the shared service account.
type Bitbucket struct {
	BaseURL  string
	Token    string
	Username string
}

// LLM is the inference setup: instance-wide gateway, per-team key and model.
type LLM struct {
	// Model is an llmwire profile id (gpt-5.5), never a gateway alias: the
	// alias is the operator's, in LLMWIRE_LITELLM_MODELS.
	Model string
	// APIKey is empty on the instance: every team brings its own.
	APIKey string
	// BaseURL is the gateway host (LLMWIRE_LITELLM_BASE_URL), instance-wide.
	BaseURL string
	// GatewayModels is LLMWIRE_LITELLM_MODELS verbatim.
	GatewayModels string
	// ReasoningEffort is mandatory: noergler needs a reasoning-capable model.
	// Normalised (trimmed, lower-cased); the allowed set is the profile's and
	// is checked at inference startup.
	ReasoningEffort string
	// ContextWindow in tokens; 0 = read max_input_tokens from the gateway.
	ContextWindow int
}

// Review holds the per-team review knobs with the REVIEW_* defaults.
type Review struct {
	AutoReviewAuthors               []string
	IgnoreAuthors                   []string
	ExcludeRepos                    []string
	MaxComments                     int
	MaxFileLines                    int
	DiffExtraLinesBefore            int
	DiffExtraLinesAfter             int
	DiffMaxExtraLinesDynamicContext int
	DiffAllowDynamicContext         bool
	ReviewPromptTemplate            string
	MentionPromptTemplate           string
	TicketComplianceCheck           bool
	RequireAgentsMD                 bool
	AgentsMDWarnTokens              int
	AgentsMDMaxTokens               int
	AgentsMDCustomLink              string
	OptOutBranchKeyword             string
	MaxPRCostUSD                    float64
}

// DefaultAcceptanceCriteriaPrefixes is the built-in prefix list.
var DefaultAcceptanceCriteriaPrefixes = []string{
	"AC", "AK", "Acceptance Criteria", "Acceptance Criterion",
	"Akzeptanzkriterium", "Akzeptanzkriterien", "DoD", "Req",
}

// Jira is the shared read-only user plus the per-team prefixes.
type Jira struct {
	URL                        string
	Token                      string
	AcceptanceCriteriaPrefixes []string
}

// Server is the listener plus the public URL onboarding writes into webhooks.
type Server struct {
	Host string
	Port int
	// PublicURL is how Bitbucket reaches this instance. Only POST /onboard
	// needs it; empty disables that endpoint, nothing else. Never guessed
	// from Host headers.
	PublicURL string
}

// Database is the connection string.
type Database struct {
	URL string
}

// Riptide is a team's forwarding target; nil on the team = forwarding off.
type Riptide struct {
	URL   string
	Token string
}

// Trust is the context-window trust curve (instance only).
type Trust struct {
	HeadroomTokens int
	Threshold      int
	Tail           float64
}

// UsableContextBudget turns an advertised window into a per-call budget.
// Below the threshold: the window minus a flat headroom. Above it: the
// threshold plus only Tail of the excess, because large advertised windows
// are the least trustworthy (many endpoints 413 below them). Examples with
// T=256k, Tail=0.5, headroom 16k: 128k->112k, 272k->264k, 512k->384k,
// 1.05M->653k. Never below 2000.
func (t Trust) UsableContextBudget(window int) int {
	var usable int
	if window <= t.Threshold {
		usable = window - t.HeadroomTokens
	} else {
		usable = t.Threshold + int(float64(window-t.Threshold)*t.Tail)
	}
	return max(2000, usable)
}

// Team is a fully resolved team: secrets read, defaults merged.
//
// Projects and the author/exclude lists are the team's own to change through
// the DB (team_claims, team_settings); at startup they are loaded from there
// and the values here only seed an empty DB.
type Team struct {
	Slug          string
	Name          string
	WebhookSecret string
	Projects      []ProjectScope
	LLM           LLM
	Review        Review
	Jira          Jira
	Riptide       *Riptide
}

// ProjectScope is a Bitbucket project a team owns, optionally narrowed to
// some repos (nil = the whole project).
type ProjectScope struct {
	Key   string
	Repos []string
}

// Owns reports whether the scope covers the repo.
func (p ProjectScope) Owns(projectKey, repoSlug string) bool {
	if projectKey != p.Key {
		return false
	}
	if p.Repos == nil {
		return true
	}
	for _, r := range p.Repos {
		if r == repoSlug {
			return true
		}
	}
	return false
}

// String renders the scope the way the config dump shows it.
func (p ProjectScope) String() string {
	if p.Repos == nil {
		return p.Key
	}
	return p.Key + "/{" + strings.Join(p.Repos, ",") + "}"
}

// Owns reports whether any claimed scope covers the repo.
func (t *Team) Owns(projectKey, repoSlug string) bool {
	for _, p := range t.Projects {
		if p.Owns(projectKey, repoSlug) {
			return true
		}
	}
	return false
}

// ClaimsRepoExplicitly: the repo is named in a repos: list, not just covered
// by a whole-project claim. A deliberate per-repo claim wins over
// exclude_repos globs.
func (t *Team) ClaimsRepoExplicitly(projectKey, repoSlug string) bool {
	for _, p := range t.Projects {
		if p.Key == projectKey && p.Repos != nil && p.Owns(projectKey, repoSlug) {
			return true
		}
	}
	return false
}

// ReviewsRepo: owned, and not carved out by review.exclude_repos (an
// explicitly claimed repo is never carved out).
func (t *Team) ReviewsRepo(projectKey, repoSlug string) bool {
	if !t.Owns(projectKey, repoSlug) {
		return false
	}
	if t.ClaimsRepoExplicitly(projectKey, repoSlug) {
		return true
	}
	return !ExcludesRepo(t.Review.ExcludeRepos, repoSlug)
}

// ExcludesRepo matches the slug against the globs, both lower-cased, with
// fnmatch semantics.
func ExcludesRepo(patterns []string, repoSlug string) bool {
	slug := strings.ToLower(repoSlug)
	for _, p := range patterns {
		if FnMatch(strings.ToLower(p), slug) {
			return true
		}
	}
	return false
}

// ModelLabel is `<model>-<effort>`, the string stored per run and shown in
// the summary.
func ModelLabel(model, effort string) string {
	if effort != "" {
		return model + "-" + effort
	}
	return model
}

// TeamSlugRE is the slug format; the path segment, the DB key and the log
// field are all this string.
var TeamSlugRE = regexp.MustCompile(`^[a-z0-9][a-z0-9-]*$`)

// TeamEnvPrefix is `TEAM_<SLUG>_`, the naming convention for a team's secret
// env vars. Not enforced by the loader.
func TeamEnvPrefix(slug string) string {
	return "TEAM_" + strings.ReplaceAll(strings.ToUpper(slug), "-", "_") + "_"
}

// App is the instance layer plus every resolved team.
type App struct {
	Bitbucket       Bitbucket
	LLM             LLM
	Review          Review
	Jira            Jira
	Server          Server
	Database        Database
	Trust           Trust
	TeamsConfigPath string
	// Teams are the enabled teams by slug in file order; Disabled the rest
	// with the reason. A slug is in exactly one of the two.
	Teams    map[string]*Team
	Order    []string
	Disabled map[string]string
}

// --- instance env ------------------------------------------------------------

type envReader struct {
	lookup func(string) (string, bool)
	errs   []string
}

func (e *envReader) get(name, def string, required bool) string {
	v, ok := e.lookup(name)
	if !ok {
		if required {
			e.errs = append(e.errs, fmt.Sprintf("Environment variable %s is not set", name))
		}
		return def
	}
	// A required var set to nothing (an unfilled .env.example line) is the
	// same misconfiguration as an unset one, caught at boot rather than at
	// the first API call. Python let it through to the connectivity check.
	if required && strings.TrimSpace(v) == "" {
		e.errs = append(e.errs, fmt.Sprintf("Environment variable %s is empty", name))
	}
	return v
}

func (e *envReader) str(name, def string) string    { return e.get(name, def, false) }
func (e *envReader) required(name string) string    { return e.get(name, "", true) }
func (e *envReader) list(name, def string) []string { return commaList(e.get(name, def, false)) }
func (e *envReader) boolean(name, def string) bool  { return parseBool(e.get(name, def, false)) }
func (e *envReader) integer(name, def string) int   { return e.intOr(name, e.get(name, def, false)) }
func (e *envReader) float(name, def string) float64 { return e.floatOr(name, e.get(name, def, false)) }
func (e *envReader) intOr(name, raw string) int     { v, err := parseInt(raw); e.fail(name, err); return v }
func (e *envReader) floatOr(name, raw string) float64 {
	v, err := parseFloat(raw)
	e.fail(name, err)
	return v
}
func (e *envReader) fail(name string, err error) {
	if err != nil {
		e.errs = append(e.errs, name+": "+err.Error())
	}
}

func commaList(s string) []string {
	out := []string{}
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// parseBool is the Python rule: true/1/yes, case-insensitive; anything else
// is false, never an error.
func parseBool(s string) bool {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "true", "1", "yes":
		return true
	}
	return false
}

func parseInt(s string) (int, error) {
	v, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil {
		return 0, fmt.Errorf("invalid literal for int(): %q", s)
	}
	return v, nil
}

func parseFloat(s string) (float64, error) {
	v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return 0, fmt.Errorf("could not convert string to float: %q", s)
	}
	return v, nil
}

// normalizeEffort trims and lower-cases; empty is refused because noergler
// needs a reasoning-capable model and an empty value would silently disable
// reasoning.
func normalizeEffort(v string) (string, error) {
	s := strings.ToLower(strings.TrimSpace(v))
	if s == "" {
		return "", fmt.Errorf("reasoning_effort is required (noergler needs a reasoning-capable model); set the profile's effort level")
	}
	return s, nil
}

// LoadInstance reads layer 1 from lookup (os.LookupEnv in production). Every
// fault is collected and reported at once: a bad int and a missing var are
// both boot failures.
func LoadInstance(lookup func(string) (string, bool)) (*App, error) {
	e := &envReader{lookup: lookup}
	app := &App{
		Bitbucket: Bitbucket{
			BaseURL:  e.required("BITBUCKET_URL"),
			Token:    e.required("BITBUCKET_TOKEN"),
			Username: e.required("BITBUCKET_USERNAME"),
		},
		LLM: LLM{
			Model:         e.required("OPENAI_MODEL"),
			BaseURL:       stripChatSuffix(e.required(GatewayBaseURLEnv)),
			GatewayModels: e.required(GatewayModelsEnv),
			ContextWindow: e.integer("OPENAI_CONTEXT_WINDOW", "0"),
		},
		Review: Review{
			AutoReviewAuthors:               e.list("REVIEW_AUTO_REVIEW_AUTHORS", ""),
			IgnoreAuthors:                   e.list("REVIEW_IGNORE_AUTHORS", ""),
			ExcludeRepos:                    e.list("REVIEW_EXCLUDE_REPOS", "*-infra"),
			MaxComments:                     e.integer("REVIEW_MAX_COMMENTS", "25"),
			MaxFileLines:                    e.integer("REVIEW_MAX_FILE_LINES", "1000"),
			DiffExtraLinesBefore:            e.integer("REVIEW_DIFF_EXTRA_LINES_BEFORE", "3"),
			DiffExtraLinesAfter:             e.integer("REVIEW_DIFF_EXTRA_LINES_AFTER", "2"),
			DiffMaxExtraLinesDynamicContext: e.integer("REVIEW_DIFF_MAX_EXTRA_LINES_DYNAMIC_CONTEXT", "10"),
			DiffAllowDynamicContext:         e.boolean("REVIEW_DIFF_ALLOW_DYNAMIC_CONTEXT", "true"),
			ReviewPromptTemplate:            e.str("REVIEW_PROMPT_TEMPLATE", "prompts/review.txt"),
			MentionPromptTemplate:           e.str("REVIEW_MENTION_PROMPT_TEMPLATE", "prompts/mention.txt"),
			TicketComplianceCheck:           e.boolean("REVIEW_TICKET_COMPLIANCE_CHECK", "true"),
			RequireAgentsMD:                 e.boolean("REVIEW_REQUIRE_AGENTS_MD", "true"),
			AgentsMDWarnTokens:              e.integer("REVIEW_AGENTS_MD_WARN_TOKENS", "4000"),
			AgentsMDMaxTokens:               e.integer("REVIEW_AGENTS_MD_MAX_TOKENS", "7000"),
			AgentsMDCustomLink:              e.str("REVIEW_AGENTS_MD_CUSTOM_LINK", ""),
			OptOutBranchKeyword:             e.str("REVIEW_OPT_OUT_BRANCH_KEYWORD", "noergloff"),
			MaxPRCostUSD:                    e.float("REVIEW_MAX_PR_COST_USD", "5.00"),
		},
		Jira: Jira{
			URL:                        e.required("JIRA_URL"),
			Token:                      e.required("JIRA_TOKEN"),
			AcceptanceCriteriaPrefixes: e.list("JIRA_ACCEPTANCE_CRITERIA_PREFIXES", strings.Join(DefaultAcceptanceCriteriaPrefixes, ",")),
		},
		Server: Server{
			Host:      e.str("SERVER_HOST", "0.0.0.0"),
			Port:      e.integer("SERVER_PORT", "8080"),
			PublicURL: strings.TrimRight(e.str("NOERGLER_PUBLIC_URL", ""), "/"),
		},
		Database: Database{URL: e.required("DATABASE_URL")},
		Trust: Trust{
			HeadroomTokens: e.integer("CONTEXT_WINDOW_HEADROOM_TOKENS", "16000"),
			Threshold:      e.integer("CONTEXT_TRUST_THRESHOLD", "256000"),
			Tail:           e.float("CONTEXT_TRUST_TAIL", "0.5"),
		},
		TeamsConfigPath: e.str("TEAMS_CONFIG", "teams.yaml"),
		Teams:           map[string]*Team{},
		Disabled:        map[string]string{},
	}
	effort, err := normalizeEffort(e.str("OPENAI_REASONING_EFFORT", "high"))
	if err != nil {
		e.errs = append(e.errs, "OPENAI_REASONING_EFFORT: "+err.Error())
	}
	app.LLM.ReasoningEffort = effort
	if len(e.errs) > 0 {
		return nil, fmt.Errorf("%s", strings.Join(e.errs, "; "))
	}
	return app, nil
}

// Load is the instance env plus every team from TEAMS_CONFIG.
func Load(lookup func(string) (string, bool)) (*App, error) {
	app, err := LoadInstance(lookup)
	if err != nil {
		return nil, err
	}
	teams, order, disabled, err := LoadTeams(app.TeamsConfigPath, app, lookup)
	if err != nil {
		return nil, err
	}
	app.Teams, app.Order, app.Disabled = teams, order, disabled
	return app, nil
}

// OSLookup is os.LookupEnv, the production lookup.
func OSLookup(name string) (string, bool) { return os.LookupEnv(name) }

// stripChatSuffix removes a user-supplied /chat/completions (llmwire appends
// it) and trailing slashes on either side of it.
func stripChatSuffix(u string) string {
	u = strings.TrimRight(u, "/")
	u = strings.TrimSuffix(u, "/chat/completions")
	return strings.TrimRight(u, "/")
}
