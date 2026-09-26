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
	// MaxFileBytes caps one file body at the socket; over it, that file falls
	// back to diff-only. MaxDiffBytes caps a whole PR or compare diff the same
	// way, but defaults to 0 = unlimited: it refuses the PR outright, and a
	// diff's size is mostly files IsReviewable is about to discard, so the
	// number it measures is not the number that ends up resident. GOMEMLIMIT
	// bounds the process; this only exists for a pod too small to trust it.
	MaxDiffBytes int
	MaxFileBytes int
}

// LLM is the inference setup: instance-wide gateway, per-team key and model.
type LLM struct {
	// Model is an llmwire profile id, never a gateway alias: the alias is the
	// operator's, in LLMWIRE_LITELLM_MODELS.
	Model string
	// APIKey is empty on the instance: every team brings its own.
	APIKey string
	// BaseURL is the gateway host (LLMWIRE_LITELLM_BASE_URL), instance-wide.
	BaseURL string
	// GatewayModels is LLMWIRE_LITELLM_MODELS verbatim.
	GatewayModels string
	// ReasoningEffort is the operator's reasoning level, a product knob shown
	// in every run's model label. Empty means the model's balanced level.
	// Normalised (trimmed, lower-cased) and not otherwise validated here: the
	// allowed set is the profile's, checked at team startup.
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

// ClaimsRepoExplicitly reports whether the repo is named in a repos list, not just covered
// by a whole-project claim. A deliberate per-repo claim wins over exclude_repos globs.
func (t *Team) ClaimsRepoExplicitly(projectKey, repoSlug string) bool {
	for _, p := range t.Projects {
		if p.Key == projectKey && p.Repos != nil && p.Owns(projectKey, repoSlug) {
			return true
		}
	}
	return false
}

// ReviewsRepo reports whether the repo is owned and not carved out by review.exclude_repos.
// An explicitly claimed repo is never carved out.
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
// FnMatch: `*` crosses `/`, so a pattern needs no path segments.
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

// Queue sizes the review queue's inference pool.
//
// Prepare (Bitbucket fetches, prompt assembly) and post stay on the single
// worker, so every Bitbucket call is still one at a time. Only the gateway
// call overlaps, which is what a 200s+ wait was blocking the worker for.
//
// Both are instance-wide on purpose. A per-team value is not a per-team knob:
// it applies to every team, so teams.yaml gets no say and the override table
// is untouched. A team has no basis to tune a number whose only effect is on
// other teams.
type Queue struct {
	// InferenceConcurrency bounds in-flight inference process-wide. The
	// per-team cap nests inside it, so the total can never become
	// teams * InferenceConcurrencyPerTeam.
	InferenceConcurrency int
	// InferenceConcurrencyPerTeam stops one team's burst taking every slot.
	InferenceConcurrencyPerTeam int
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
	Queue           Queue
	TeamsConfigPath string
	// Teams are the enabled teams by slug in file order; Disabled the rest
	// with the reason. A slug is in exactly one of the two.
	Teams    map[string]*Team
	Order    []string
	Disabled map[string]string
	// Names is the display name of every team that has one, disabled teams
	// included, by slug.
	Names map[string]string
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
	// the first API call.
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

// positive is integer plus a floor of 1. A zero or negative per-file byte cap
// would boot clean and then refuse every file, so it has to fail at load.
func (e *envReader) positive(name, def string) int {
	v := e.integer(name, def)
	if v < 1 {
		e.errs = append(e.errs, fmt.Sprintf("%s: must be a positive integer, got %d", name, v))
	}
	return v
}

// nonNegative is integer plus a floor of 0, for a cap where 0 means unlimited
// rather than "refuse everything". Negative still fails at load: it is a typo,
// not a third meaning.
func (e *envReader) nonNegative(name, def string) int {
	v := e.integer(name, def)
	if v < 0 {
		e.errs = append(e.errs, fmt.Sprintf("%s: must be zero (unlimited) or a positive integer, got %d", name, v))
	}
	return v
}

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

// parseBool accepts true/1/yes, case-insensitive; anything else is false,
// never an error.
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

// normalizeEffort trims and lower-cases. Empty is the unset level: the
// model's balanced one, not reasoning switched off.
func normalizeEffort(v string) string {
	return strings.ToLower(strings.TrimSpace(v))
}

// LoadInstance reads layer 1 from lookup (os.LookupEnv in production). Every
// fault is collected and reported at once: a bad int and a missing var are
// both boot failures.
func LoadInstance(lookup func(string) (string, bool)) (*App, error) {
	e := &envReader{lookup: lookup}
	app := &App{
		Bitbucket: Bitbucket{
			BaseURL:      e.required("BITBUCKET_URL"),
			Token:        e.required("BITBUCKET_TOKEN"),
			Username:     e.required("BITBUCKET_USERNAME"),
			MaxDiffBytes: e.nonNegative("BITBUCKET_MAX_DIFF_BYTES", "0"),
			MaxFileBytes: e.positive("BITBUCKET_MAX_FILE_BYTES", "1048576"),
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
		Queue: Queue{
			InferenceConcurrency:        e.positive("REVIEW_INFERENCE_CONCURRENCY", "6"),
			InferenceConcurrencyPerTeam: e.positive("REVIEW_INFERENCE_CONCURRENCY_PER_TEAM", "2"),
		},
		TeamsConfigPath: e.str("TEAMS_CONFIG", "teams.yaml"),
		Teams:           map[string]*Team{},
		Disabled:        map[string]string{},
	}
	app.LLM.ReasoningEffort = normalizeEffort(e.str("OPENAI_REASONING_EFFORT", ""))
	// A per-team cap above the global one never binds, so it reads as a
	// setting that does nothing. Fail rather than let the misconfiguration
	// survive to production unnoticed.
	if q := app.Queue; q.InferenceConcurrencyPerTeam > q.InferenceConcurrency {
		e.errs = append(e.errs, fmt.Sprintf(
			"REVIEW_INFERENCE_CONCURRENCY_PER_TEAM (%d) must not exceed REVIEW_INFERENCE_CONCURRENCY (%d)",
			q.InferenceConcurrencyPerTeam, q.InferenceConcurrency))
	}
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
	teams, order, disabled, names, err := LoadTeams(app.TeamsConfigPath, app, lookup)
	if err != nil {
		return nil, err
	}
	app.Teams, app.Order, app.Disabled, app.Names = teams, order, disabled, names
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
