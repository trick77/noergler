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
	"os"
	"regexp"
	"strings"

	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/secretbox"
	kitenv "github.com/trick77/ship/kit/env"
	"github.com/trick77/ship/kit/logging"
)

// RequiredWebhookEvents are the Bitbucket events the webhook dispatches on.
// Shared with onboarding, which registers exactly these.
var RequiredWebhookEvents = []string{
	"pr:opened",
	"pr:from_ref_updated",
	"pr:modified",
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
	// DisplayNamePattern extracts a commenter's first name from their
	// displayName (group "first", "last" optional) for finding replies; nil
	// = off. Instance-wide: one Bitbucket, one directory format.
	DisplayNamePattern *regexp.Regexp
}

// DefaultDisplayNamePattern reads "Last First (CODE)": the first name is the
// last word before the bracket, so a multi-word surname works ("Van Simpson
// Brian-David (BFD)" is Brian-David) and an unhyphenated double first name
// does not ("Saner Jan Peter (SJP)" is Peter, accepted). No bracket, no name.
// A "First Last" directory overrides it, or blanks it to switch names off.
// Unicode-aware: RE2's \s and \w are ASCII.
const DefaultDisplayNamePattern = `^(?P<last>.+?)[\s\p{Zs}]+(?P<first>[\pL\pM'-]+)[\s\p{Zs}]*\(`

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
	// ContextWindow in tokens; 0 = the gateway's max_input_tokens, else the
	// llmwire profile's input limit.
	ContextWindow int
}

// Review holds the per-team review knobs with the REVIEW_* defaults.
type Review struct {
	AutoReviewAuthors []string
	IgnoreAuthors     []string
	ExcludeRepos      []string
	// Branch-name globs: a PR from a matching source or into a matching
	// target branch is never auto-reviewed; an @mention still is.
	ExcludeSourceBranches           []string
	ExcludeTargetBranches           []string
	MaxComments                     int
	MaxFileLines                    int
	DiffExtraLinesBefore            int
	DiffExtraLinesAfter             int
	DiffMaxExtraLinesDynamicContext int
	DiffAllowDynamicContext         bool
	TicketComplianceCheck           bool
	RequireAgentsMD                 bool
	AgentsMDWarnTokens              int
	AgentsMDMaxTokens               int
	AgentsMDCustomLink              string
	OptOutBranchKeyword             string
	MaxPRCostUSD                    float64
	// Language is the default for the prose noergler posts; a user's own
	// choice (user_settings) wins over it.
	Language lang.Lang
	// UsePRDescription feeds the PR title and description to the model.
	// Team API only (no env var, no teams.yaml key); the DB row wins.
	UsePRDescription bool
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
	// PublicURL is this instance's base URL, reachable by Bitbucket (POST
	// /onboard writes webhooks to it) and by reviewers' browsers (the
	// summary footer links each run's dashboard page under it). Empty
	// disables /onboard and the footer link, nothing else. Never guessed
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

// Almanach is a team's Almanach, the living wiki its merged PRs' notes go
// to; nil on the team = no note call, no row, nothing forwarded. Opt-in per
// team, like riptide. The token is the team's identity at Almanach.
type Almanach struct {
	URL   string
	Token string
}

// AlmanachLimits is the instance's cap on a note (ALMANACH_NOTE_MAX_CHARS).
// Instance only: the cap bounds a row in the shared database and the
// payload a shared Almanach reads; a team has no business raising it.
type AlmanachLimits struct {
	// NoteMaxChars is the rune count of the serialized note over which the
	// whole note is dropped, never truncated.
	NoteMaxChars int
}

// Trust is the context-window trust curve (instance only).
type Trust struct {
	HeadroomTokens int
	Threshold      int
	Tail           float64
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
	// OwnModel: the team's model or reasoning effort differs from the
	// instance's, so its model label is not the instance default.
	OwnModel bool
	Review   Review
	Jira     Jira
	Riptide  *Riptide
	Almanach *Almanach
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
	return !MatchesAny(t.Review.ExcludeRepos, repoSlug)
}

// ExcludedBranch names the list that carves the PR out of auto-review by
// its branches (displayId, e.g. `feature/x`), source first; "" = neither.
// `*` crosses `/`, so `release/*` covers `release/a/b`.
// A pattern written as a full ref (`refs/heads/release/*`, as toRef.id
// shows it) matches as its short form.
func (t *Team) ExcludedBranch(fromBranch, toBranch string) string {
	if matchesBranch(t.Review.ExcludeSourceBranches, fromBranch) {
		return "exclude_source_branches"
	}
	if matchesBranch(t.Review.ExcludeTargetBranches, toBranch) {
		return "exclude_target_branches"
	}
	return ""
}

func matchesBranch(patterns []string, branch string) bool {
	branch = strings.ToLower(branch)
	for _, p := range patterns {
		if FnMatch(strings.TrimPrefix(strings.ToLower(p), "refs/heads/"), branch) {
			return true
		}
	}
	return false
}

// MatchesAny reports whether any glob matches the name, both lower-cased,
// with FnMatch: `*` crosses `/`, so a pattern needs no path segments.
func MatchesAny(patterns []string, name string) bool {
	name = strings.ToLower(name)
	for _, p := range patterns {
		if FnMatch(strings.ToLower(p), name) {
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
	Bitbucket Bitbucket
	LLM       LLM
	Review    Review
	Jira      Jira
	Server    Server
	Database  Database
	Trust     Trust
	Queue     Queue
	Almanach  AlmanachLimits
	// SecretsKey seals the credentials teams push (NOERGLER_SECRETS_KEY).
	// Required even before any team pushes one: a key that appears only
	// later is a deploy step nobody remembers.
	SecretsKey      []byte
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

// positive is an int with a floor of 1. A zero or negative per-file byte cap
// would boot clean and then refuse every file, so it has to fail at load.
func positive(r *kitenv.Reader, name string, def int) int {
	v := r.Int(name, def)
	if v < 1 {
		r.Fail("%s: must be a positive integer, got %d", name, v)
	}
	return v
}

// nonNegative is an int with a floor of 0, for a cap where 0 means unlimited
// rather than "refuse everything". Negative still fails at load: it is a typo,
// not a third meaning.
func nonNegative(r *kitenv.Reader, name string, def int) int {
	v := r.Int(name, def)
	if v < 0 {
		r.Fail("%s: must be zero (unlimited) or a positive integer, got %d", name, v)
	}
	return v
}

// setEmpty reports a variable set to blank. kit's reader takes blank as unset
// (the default); for the few knobs whose blank is a setting of its own (no
// opt-out keyword, no excluded repos) the caller turns that back into "off".
func setEmpty(lookup func(string) (string, bool), name string) bool {
	v, ok := lookup(name)
	return ok && strings.TrimSpace(v) == ""
}

// blankOff is r.String, but blank set explicitly is "", not def.
func blankOff(r *kitenv.Reader, lookup func(string) (string, bool), name, def string) string {
	if setEmpty(lookup, name) {
		return ""
	}
	return r.String(name, def)
}

// blankNone is r.List, but blank set explicitly is the empty list, not def.
func blankNone(r *kitenv.Reader, lookup func(string) (string, bool), name string, def []string) []string {
	if setEmpty(lookup, name) {
		return []string{}
	}
	return r.List(name, def)
}

// displayNamePattern compiles BITBUCKET_DISPLAY_NAME_PATTERN; blank set
// explicitly is off (nil). A pattern without a "first" group would boot clean
// and never name anyone, so it fails at load.
func displayNamePattern(r *kitenv.Reader, lookup func(string) (string, bool)) *regexp.Regexp {
	const name = "BITBUCKET_DISPLAY_NAME_PATTERN"
	v := blankOff(r, lookup, name, DefaultDisplayNamePattern)
	if v == "" {
		return nil
	}
	re, err := regexp.Compile(v)
	if err != nil {
		r.Fail("%s: %v", name, err)
		return nil
	}
	if re.SubexpIndex("first") < 0 {
		r.Fail("%s: needs a named group (?P<first>...)", name)
		return nil
	}
	return re
}

// normalizeEffort trims and lower-cases. Empty is the unset level: the
// model's balanced one, not reasoning switched off.
func normalizeEffort(v string) string {
	return strings.ToLower(strings.TrimSpace(v))
}

// LoadInstance reads layer 1 from lookup (os.LookupEnv in production). Every
// fault is collected and reported at once: a bad int and a missing var are
// both boot failures, and so is a boolean that is not one.
func LoadInstance(lookup func(string) (string, bool)) (*App, error) {
	r := kitenv.New(func(name string) string { v, _ := lookup(name); return v })
	app := &App{
		Bitbucket: Bitbucket{
			BaseURL:      r.Required("BITBUCKET_URL"),
			Token:        r.Required("BITBUCKET_TOKEN"),
			Username:     r.Required("BITBUCKET_USERNAME"),
			MaxDiffBytes: nonNegative(r, "BITBUCKET_MAX_DIFF_BYTES", 0),
			MaxFileBytes: positive(r, "BITBUCKET_MAX_FILE_BYTES", 1048576),
		},
		LLM: LLM{
			Model:         r.Required("OPENAI_MODEL"),
			BaseURL:       r.Required(GatewayBaseURLEnv),
			GatewayModels: r.Required(GatewayModelsEnv),
			ContextWindow: r.Int("OPENAI_CONTEXT_WINDOW", 0),
		},
		Review: Review{
			AutoReviewAuthors:               r.List("REVIEW_AUTO_REVIEW_AUTHORS", []string{}),
			IgnoreAuthors:                   r.List("REVIEW_IGNORE_AUTHORS", []string{}),
			ExcludeRepos:                    blankNone(r, lookup, "REVIEW_EXCLUDE_REPOS", []string{"*-infra"}),
			ExcludeSourceBranches:           r.List("REVIEW_EXCLUDE_SOURCE_BRANCHES", []string{}),
			ExcludeTargetBranches:           r.List("REVIEW_EXCLUDE_TARGET_BRANCHES", []string{}),
			MaxComments:                     r.Int("REVIEW_MAX_COMMENTS", 25),
			MaxFileLines:                    r.Int("REVIEW_MAX_FILE_LINES", 1000),
			DiffExtraLinesBefore:            r.Int("REVIEW_DIFF_EXTRA_LINES_BEFORE", 3),
			DiffExtraLinesAfter:             r.Int("REVIEW_DIFF_EXTRA_LINES_AFTER", 2),
			DiffMaxExtraLinesDynamicContext: r.Int("REVIEW_DIFF_MAX_EXTRA_LINES_DYNAMIC_CONTEXT", 10),
			DiffAllowDynamicContext:         r.Bool("REVIEW_DIFF_ALLOW_DYNAMIC_CONTEXT", true),
			TicketComplianceCheck:           r.Bool("REVIEW_TICKET_COMPLIANCE_CHECK", true),
			UsePRDescription:                true,
			RequireAgentsMD:                 r.Bool("REVIEW_REQUIRE_AGENTS_MD", true),
			AgentsMDWarnTokens:              r.Int("REVIEW_AGENTS_MD_WARN_TOKENS", 4000),
			AgentsMDMaxTokens:               r.Int("REVIEW_AGENTS_MD_MAX_TOKENS", 7000),
			AgentsMDCustomLink:              r.String("REVIEW_AGENTS_MD_CUSTOM_LINK", ""),
			OptOutBranchKeyword:             blankOff(r, lookup, "REVIEW_OPT_OUT_BRANCH_KEYWORD", "noergloff"),
			MaxPRCostUSD:                    r.Float("REVIEW_MAX_PR_COST_USD", 5.00),
		},
		Jira: Jira{
			URL:   r.Required("JIRA_URL"),
			Token: r.Required("JIRA_TOKEN"),
			AcceptanceCriteriaPrefixes: blankNone(r, lookup, "JIRA_ACCEPTANCE_CRITERIA_PREFIXES",
				DefaultAcceptanceCriteriaPrefixes),
		},
		Server: Server{
			Host:      r.String("SERVER_HOST", "0.0.0.0"),
			Port:      r.Int("SERVER_PORT", 8080),
			PublicURL: strings.TrimRight(r.String("NOERGLER_PUBLIC_URL", ""), "/"),
		},
		Database: Database{URL: r.Required("DATABASE_URL")},
		Trust: Trust{
			HeadroomTokens: r.Int("CONTEXT_WINDOW_HEADROOM_TOKENS", 16000),
			Threshold:      r.Int("CONTEXT_TRUST_THRESHOLD", 256000),
			Tail:           r.Float("CONTEXT_TRUST_TAIL", 0.5),
		},
		Queue: Queue{
			InferenceConcurrency:        positive(r, "REVIEW_INFERENCE_CONCURRENCY", 6),
			InferenceConcurrencyPerTeam: positive(r, "REVIEW_INFERENCE_CONCURRENCY_PER_TEAM", 2),
		},
		Almanach:        AlmanachLimits{NoteMaxChars: positive(r, "ALMANACH_NOTE_MAX_CHARS", 6000)},
		TeamsConfigPath: r.String("TEAMS_CONFIG", "teams.yaml"),
		Teams:           map[string]*Team{},
		Disabled:        map[string]string{},
	}
	app.Bitbucket.DisplayNamePattern = displayNamePattern(r, lookup)
	if raw := r.Required("NOERGLER_SECRETS_KEY"); raw != "" {
		key, err := secretbox.ParseKey(raw)
		if err != nil {
			r.Fail("NOERGLER_SECRETS_KEY: %v", err)
		}
		app.SecretsKey = key
	}
	if l, ok := lang.Parse(r.String("REVIEW_LANGUAGE", string(lang.Default))); ok {
		app.Review.Language = l
	} else {
		r.Fail("REVIEW_LANGUAGE: must be de or en")
	}
	app.LLM.ReasoningEffort = normalizeEffort(r.String("OPENAI_REASONING_EFFORT", ""))
	// Logging is set up before this runs (at info on a bad value, see
	// withLog); the level is checked here so its fault is reported with the
	// rest rather than one restart earlier.
	if _, err := logging.ParseLevel(r.String("LOG_LEVEL", "")); err != nil {
		r.Fail("LOG_LEVEL: %v", err)
	}
	// A per-team cap above the global one never binds, so it reads as a
	// setting that does nothing. Fail rather than let the misconfiguration
	// survive to production unnoticed.
	if q := app.Queue; q.InferenceConcurrencyPerTeam > q.InferenceConcurrency {
		r.Fail("REVIEW_INFERENCE_CONCURRENCY_PER_TEAM (%d) must not exceed REVIEW_INFERENCE_CONCURRENCY (%d)",
			q.InferenceConcurrencyPerTeam, q.InferenceConcurrency)
	}
	// sortAndLimit slices to it: negative panics every review in post.
	if app.Review.MaxComments < 0 {
		r.Fail("REVIEW_MAX_COMMENTS: must be zero or a positive integer, got %d", app.Review.MaxComments)
	}
	// The cap check is cumulative >= limit: 0 reads as "no cap" but skips
	// every auto-review after the first priced run.
	if v := app.Review.MaxPRCostUSD; v <= 0 {
		r.Fail("REVIEW_MAX_PR_COST_USD: must be greater than 0, got %v", v)
	}
	if err := r.Err(); err != nil {
		return nil, err
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
