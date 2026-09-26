package config

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// TeamsFileError indicates that teams.yaml as a whole is unusable and startup must abort.
type TeamsFileError struct{ Msg string }

func (e *TeamsFileError) Error() string { return e.Msg }

// TeamError indicates that a single team's block is unusable and that team is disabled.
type TeamError struct{ Msg string }

func (e *TeamError) Error() string { return e.Msg }

func teamErrorf(format string, args ...any) error {
	return &TeamError{Msg: fmt.Sprintf(format, args...)}
}

// readTeamsFile parses teams.yaml into raw block nodes. File-level faults are
// TeamsFileError: the instance must not start without a usable file.
func readTeamsFile(path string) ([]*yaml.Node, error) {
	// G304: the teams file path is operator configuration, not request data.
	data, err := os.ReadFile(path) //nolint:gosec // G304
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, &TeamsFileError{Msg: fmt.Sprintf("teams file %s not found. noergler does not start without teams: "+
				"set TEAMS_CONFIG to a teams.yaml (see teams.example.yaml).", path)}
		}
		// e.g. a directory: a compose bind-mount of a missing host file
		// creates a directory in its place.
		return nil, &TeamsFileError{Msg: fmt.Sprintf("teams file %s cannot be read: %v", path, err)}
	}
	var doc yaml.Node
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return nil, &TeamsFileError{Msg: fmt.Sprintf("teams file %s is not valid YAML: %v", path, err)}
	}
	root := &doc
	if root.Kind == yaml.DocumentNode && len(root.Content) > 0 {
		root = root.Content[0]
	}
	if root.Kind != yaml.MappingNode {
		return nil, &TeamsFileError{Msg: fmt.Sprintf("teams file %s must be a mapping with a top-level `teams:` list", path)}
	}
	teams := mappingGet(root, "teams")
	if teams == nil {
		return nil, &TeamsFileError{Msg: fmt.Sprintf("teams file %s must be a mapping with a top-level `teams:` list", path)}
	}
	if teams.Kind != yaml.SequenceNode || len(teams.Content) == 0 {
		return nil, &TeamsFileError{Msg: fmt.Sprintf("teams file %s: `teams:` must be a non-empty list", path)}
	}
	for i, item := range teams.Content {
		if item.Kind != yaml.MappingNode {
			return nil, &TeamsFileError{Msg: fmt.Sprintf("teams file %s: teams[%d] must be a mapping", path, i)}
		}
	}
	return teams.Content, nil
}

func mappingGet(m *yaml.Node, key string) *yaml.Node {
	for i := 0; i+1 < len(m.Content); i += 2 {
		if m.Content[i].Value == key {
			return m.Content[i+1]
		}
	}
	return nil
}

// rawSlug is the block's slug as written, or "" when absent or not a scalar.
func rawSlug(block *yaml.Node) string {
	n := mappingGet(block, "slug")
	if n == nil || n.Kind != yaml.ScalarNode {
		return ""
	}
	return n.Value
}

// rawName is the block's display name as written, or "" when absent or not a
// scalar. Read off the raw block so a team that fails validation keeps its
// name: the dashboard lists disabled teams too.
func rawName(block *yaml.Node) string {
	n := mappingGet(block, "name")
	if n == nil || n.Kind != yaml.ScalarNode {
		return ""
	}
	return n.Value
}

// LoadTeams resolves every team in the file. Returns enabled teams, their
// file order, disabled slugs with the reason, and the display name of every
// team that has one, disabled teams included.
//
// Duplicate slugs abort (the fault has no single owner). Ownership of
// projects is not checked here: claims live in the DB. Every disabled team
// is logged as `team_disabled team=<slug> reason=...` with the team bound.
func LoadTeams(path string, instance *App, lookup func(string) (string, bool)) (enabled map[string]*Team, order []string, disabled, names map[string]string, err error) {
	blocks, err := readTeamsFile(path)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	counts := map[string]int{}
	for _, b := range blocks {
		if s := rawSlug(b); s != "" {
			counts[s]++
		}
	}
	var dupes []string
	for s, n := range counts {
		if n > 1 {
			dupes = append(dupes, s)
		}
	}
	if len(dupes) > 0 {
		sort.Strings(dupes)
		return nil, nil, nil, nil, &TeamsFileError{Msg: fmt.Sprintf("teams file %s: duplicate slug(s) %v", path, dupes)}
	}

	enabled = map[string]*Team{}
	disabled = map[string]string{}
	names = map[string]string{}
	var disabledOrder []string
	for i, b := range blocks {
		slug := rawSlug(b)
		if slug == "" {
			slug = fmt.Sprintf("teams[%d]", i)
		}
		if name := rawName(b); name != "" {
			names[slug] = name
		}
		team, err := ResolveTeam(b, instance, lookup)
		if err != nil {
			disabled[slug] = err.Error()
			disabledOrder = append(disabledOrder, slug)
			continue
		}
		enabled[slug] = team
		order = append(order, slug)
	}
	for _, slug := range disabledOrder {
		// Bound, not just in the message: Splunk extracts `team` as a field.
		slog.Error("team_disabled team="+slug+" reason="+disabled[slug], "team", slug)
	}
	return enabled, order, disabled, names, nil
}

// --- block validation --------------------------------------------------------
//
// Strict: an unknown key disables the team, so that a knob named in the
// wrong place (base_url in a team block, a typo in a review field) is loud
// rather than silently ignored. Message shape is fixed to match the
// validation output, which the operators' runbooks quote.

type teamBlock struct {
	slug             string
	name             string
	webhookSecretEnv string
	projects         []ProjectScope
	inference        struct {
		apiKeyEnv       string
		model           *string
		reasoningEffort *string
		contextWindow   *int
	}
	review  map[string]any // validated key -> typed value
	jira    *[]string
	riptide *struct{ url, tokenEnv string }
}

type blockErrors struct{ list []string }

func (b *blockErrors) add(path, msg string) { b.list = append(b.list, path+": "+msg) }

// checkKeys refuses unknown and duplicate keys under path. yaml.v3 keeps a
// duplicate mapping key in the node tree without complaint; left alone, one
// occurrence would silently win.
func (b *blockErrors) checkKeys(m *yaml.Node, path string, allowed ...string) {
	ok := map[string]bool{}
	for _, a := range allowed {
		ok[a] = true
	}
	seen := map[string]bool{}
	for i := 0; i+1 < len(m.Content); i += 2 {
		k := m.Content[i].Value
		switch {
		case seen[k]:
			b.add(joinPath(path, k), "Duplicate key")
		case !ok[k]:
			b.add(joinPath(path, k), "Extra inputs are not permitted")
		}
		seen[k] = true
	}
}

func joinPath(parts ...string) string {
	var out []string
	for _, p := range parts {
		if p != "" {
			out = append(out, p)
		}
	}
	return strings.Join(out, ".")
}

func (b *blockErrors) scalar(m *yaml.Node, path, key string, required bool) (string, bool) {
	n := mappingGet(m, key)
	if n == nil {
		if required {
			b.add(joinPath(path, key), "Field required")
		}
		return "", false
	}
	if n.Kind != yaml.ScalarNode {
		b.add(joinPath(path, key), "Input should be a valid string")
		return "", false
	}
	return n.Value, true
}

func (b *blockErrors) intField(m *yaml.Node, path, key string) *int {
	n := mappingGet(m, key)
	if n == nil {
		return nil
	}
	var v int
	if n.Kind != yaml.ScalarNode || n.Decode(&v) != nil {
		b.add(joinPath(path, key), "Input should be a valid integer")
		return nil
	}
	return &v
}

func (b *blockErrors) floatField(m *yaml.Node, path, key string) *float64 {
	n := mappingGet(m, key)
	if n == nil {
		return nil
	}
	var v float64
	if n.Kind != yaml.ScalarNode || n.Decode(&v) != nil {
		b.add(joinPath(path, key), "Input should be a valid number")
		return nil
	}
	return &v
}

func (b *blockErrors) boolField(m *yaml.Node, path, key string) *bool {
	n := mappingGet(m, key)
	if n == nil {
		return nil
	}
	var v bool
	if n.Kind != yaml.ScalarNode || n.Decode(&v) != nil {
		b.add(joinPath(path, key), "Input should be a valid boolean")
		return nil
	}
	return &v
}

func (b *blockErrors) stringField(m *yaml.Node, path, key string) *string {
	v, ok := b.scalar(m, path, key, false)
	if !ok {
		return nil
	}
	return &v
}

func (b *blockErrors) listField(m *yaml.Node, path, key string) *[]string {
	n := mappingGet(m, key)
	if n == nil {
		return nil
	}
	var v []string
	if n.Kind != yaml.SequenceNode || n.Decode(&v) != nil {
		b.add(joinPath(path, key), "Input should be a valid list")
		return nil
	}
	if v == nil {
		v = []string{}
	}
	return &v
}

// reviewOverrideKeys are the team-overridable review knobs: every Review
// field minus the two prompt templates (instance-only by decision).
var reviewOverrideKeys = map[string]string{
	"auto_review_authors": "list", "ignore_authors": "list", "exclude_repos": "list",
	"max_comments": "int", "max_file_lines": "int",
	"diff_extra_lines_before": "int", "diff_extra_lines_after": "int",
	"diff_max_extra_lines_dynamic_context": "int", "diff_allow_dynamic_context": "bool",
	"ticket_compliance_check": "bool", "require_agents_md": "bool",
	"agents_md_warn_tokens": "int", "agents_md_max_tokens": "int",
	"agents_md_custom_link": "string", "opt_out_branch_keyword": "string",
	"max_pr_cost_usd": "float",
}

func parseBlock(m *yaml.Node) (*teamBlock, error) {
	var e blockErrors
	tb := &teamBlock{}
	e.checkKeys(m, "", "slug", "name", "webhook_secret_env", "projects", "inference", "review", "jira", "riptide")

	if slug, ok := e.scalar(m, "", "slug", true); ok {
		if !TeamSlugRE.MatchString(slug) {
			e.add("slug", fmt.Sprintf("Value error, slug '%s' must match %s", slug, TeamSlugRE.String()))
		}
		tb.slug = slug
	}
	if name := e.stringField(m, "", "name"); name != nil {
		tb.name = *name
	}
	tb.webhookSecretEnv, _ = e.scalar(m, "", "webhook_secret_env", true)

	if pn := mappingGet(m, "projects"); pn != nil {
		if pn.Kind != yaml.SequenceNode {
			e.add("projects", "Input should be a valid list")
		} else {
			tb.projects = []ProjectScope{}
			for i, item := range pn.Content {
				path := fmt.Sprintf("projects.%d", i)
				if item.Kind != yaml.MappingNode {
					e.add(path, "Input should be a valid dictionary")
					continue
				}
				e.checkKeys(item, path, "key", "repos")
				scope := ProjectScope{}
				if key, ok := e.scalar(item, path, "key", true); ok {
					if strings.TrimSpace(key) == "" {
						e.add(path+".key", "Value error, project key must be non-empty")
					}
					scope.Key = strings.TrimSpace(key)
				}
				if repos := e.listField(item, path, "repos"); repos != nil {
					cleaned := []string{}
					for _, r := range *repos {
						if r = strings.TrimSpace(r); r != "" {
							cleaned = append(cleaned, r)
						}
					}
					if len(cleaned) == 0 {
						e.add(path+".repos", "Value error, repos must list at least one slug when present")
					}
					scope.Repos = cleaned
				}
				tb.projects = append(tb.projects, scope)
			}
		}
	}

	if in := mappingGet(m, "inference"); in == nil {
		e.add("inference", "Field required")
	} else if in.Kind != yaml.MappingNode {
		e.add("inference", "Input should be a valid dictionary")
	} else {
		// base_url is deliberately absent: the gateway is instance-wide.
		e.checkKeys(in, "inference", "api_key_env", "model", "reasoning_effort", "context_window")
		tb.inference.apiKeyEnv, _ = e.scalar(in, "inference", "api_key_env", true)
		tb.inference.model = e.stringField(in, "inference", "model")
		tb.inference.reasoningEffort = e.stringField(in, "inference", "reasoning_effort")
		tb.inference.contextWindow = e.intField(in, "inference", "context_window")
	}

	if rv := mappingGet(m, "review"); rv != nil {
		if rv.Kind != yaml.MappingNode {
			e.add("review", "Input should be a valid dictionary")
		} else {
			tb.review = map[string]any{}
			reviewKeys := make([]string, 0, len(reviewOverrideKeys))
			for k := range reviewOverrideKeys {
				reviewKeys = append(reviewKeys, k)
			}
			e.checkKeys(rv, "review", reviewKeys...)
			for i := 0; i+1 < len(rv.Content); i += 2 {
				key := rv.Content[i].Value
				kind, ok := reviewOverrideKeys[key]
				if !ok {
					continue
				}
				var v any
				switch kind {
				case "list":
					if p := e.listField(rv, "review", key); p != nil {
						v = *p
					}
				case "int":
					if p := e.intField(rv, "review", key); p != nil {
						v = *p
					}
				case "bool":
					if p := e.boolField(rv, "review", key); p != nil {
						v = *p
					}
				case "float":
					if p := e.floatField(rv, "review", key); p != nil {
						v = *p
					}
				case "string":
					if p := e.stringField(rv, "review", key); p != nil {
						v = *p
					}
				}
				if v != nil {
					tb.review[key] = v
				}
			}
		}
	}

	if jn := mappingGet(m, "jira"); jn != nil {
		if jn.Kind != yaml.MappingNode {
			e.add("jira", "Input should be a valid dictionary")
		} else {
			e.checkKeys(jn, "jira", "acceptance_criteria_prefixes")
			tb.jira = e.listField(jn, "jira", "acceptance_criteria_prefixes")
		}
	}

	if rn := mappingGet(m, "riptide"); rn != nil {
		if rn.Kind != yaml.MappingNode {
			e.add("riptide", "Input should be a valid dictionary")
		} else {
			e.checkKeys(rn, "riptide", "url", "token_env")
			url, _ := e.scalar(rn, "riptide", "url", true)
			tok, _ := e.scalar(rn, "riptide", "token_env", true)
			tb.riptide = &struct{ url, tokenEnv string }{url, tok}
		}
	}

	if len(e.list) > 0 {
		return nil, &TeamError{Msg: strings.Join(e.list, "; ")}
	}
	return tb, nil
}

// secretFromEnv resolves a *_env reference. Empty is rejected: an empty
// inference key or webhook secret is a misconfiguration, never a value.
func secretFromEnv(field, variable string, lookup func(string) (string, bool)) (string, error) {
	if strings.TrimSpace(variable) == "" {
		return "", teamErrorf("%s must name an environment variable", field)
	}
	value, ok := lookup(variable)
	if !ok {
		return "", teamErrorf("%s: environment variable %s is not set", field, variable)
	}
	if strings.TrimSpace(value) == "" {
		return "", teamErrorf("%s: environment variable %s is empty", field, variable)
	}
	return value, nil
}

// ResolveTeam turns one raw block into a Team: secrets read, defaults merged
// (team value → instance value → built-in). Any fault is a TeamError; the
// caller disables the team and keeps going.
func ResolveTeam(block *yaml.Node, instance *App, lookup func(string) (string, bool)) (*Team, error) {
	tb, err := parseBlock(block)
	if err != nil {
		return nil, err
	}
	webhookSecret, err := secretFromEnv("webhook_secret_env", tb.webhookSecretEnv, lookup)
	if err != nil {
		return nil, err
	}
	apiKey, err := secretFromEnv("inference.api_key_env", tb.inference.apiKeyEnv, lookup)
	if err != nil {
		return nil, err
	}

	llm := instance.LLM
	llm.APIKey = apiKey
	if tb.inference.model != nil {
		llm.Model = *tb.inference.model
	}
	if tb.inference.reasoningEffort != nil {
		llm.ReasoningEffort = normalizeEffort(*tb.inference.reasoningEffort)
	}
	if tb.inference.contextWindow != nil {
		llm.ContextWindow = *tb.inference.contextWindow
	}

	review := instance.Review
	review.AutoReviewAuthors = append([]string{}, instance.Review.AutoReviewAuthors...)
	review.IgnoreAuthors = append([]string{}, instance.Review.IgnoreAuthors...)
	review.ExcludeRepos = append([]string{}, instance.Review.ExcludeRepos...)
	for key, v := range tb.review {
		switch key {
		case "auto_review_authors":
			review.AutoReviewAuthors = v.([]string)
		case "ignore_authors":
			review.IgnoreAuthors = v.([]string)
		case "exclude_repos":
			review.ExcludeRepos = v.([]string)
		case "max_comments":
			review.MaxComments = v.(int)
		case "max_file_lines":
			review.MaxFileLines = v.(int)
		case "diff_extra_lines_before":
			review.DiffExtraLinesBefore = v.(int)
		case "diff_extra_lines_after":
			review.DiffExtraLinesAfter = v.(int)
		case "diff_max_extra_lines_dynamic_context":
			review.DiffMaxExtraLinesDynamicContext = v.(int)
		case "diff_allow_dynamic_context":
			review.DiffAllowDynamicContext = v.(bool)
		case "ticket_compliance_check":
			review.TicketComplianceCheck = v.(bool)
		case "require_agents_md":
			review.RequireAgentsMD = v.(bool)
		case "agents_md_warn_tokens":
			review.AgentsMDWarnTokens = v.(int)
		case "agents_md_max_tokens":
			review.AgentsMDMaxTokens = v.(int)
		case "agents_md_custom_link":
			review.AgentsMDCustomLink = v.(string)
		case "opt_out_branch_keyword":
			review.OptOutBranchKeyword = v.(string)
		case "max_pr_cost_usd":
			review.MaxPRCostUSD = v.(float64)
		}
	}

	jira := Jira{
		URL:                        instance.Jira.URL,
		Token:                      instance.Jira.Token,
		AcceptanceCriteriaPrefixes: append([]string{}, instance.Jira.AcceptanceCriteriaPrefixes...),
	}
	if tb.jira != nil {
		jira.AcceptanceCriteriaPrefixes = *tb.jira
	}

	var riptide *Riptide
	if tb.riptide != nil {
		if strings.TrimSpace(tb.riptide.url) == "" {
			return nil, teamErrorf("riptide.url must be non-empty")
		}
		token, err := secretFromEnv("riptide.token_env", tb.riptide.tokenEnv, lookup)
		if err != nil {
			return nil, err
		}
		riptide = &Riptide{URL: strings.TrimRight(strings.TrimSpace(tb.riptide.url), "/"), Token: token}
	}

	name := tb.name
	if name == "" {
		name = tb.slug
	}
	projects := tb.projects
	if projects == nil {
		projects = []ProjectScope{}
	}
	return &Team{
		Slug:          tb.slug,
		Name:          name,
		WebhookSecret: webhookSecret,
		Projects:      projects,
		LLM:           llm,
		Review:        review,
		Jira:          jira,
		Riptide:       riptide,
	}, nil
}
