package config

import (
	"fmt"
	"log/slog"
	"sort"
	"strconv"
	"strings"
)

// Dump logs the effective configuration, one line per field, secrets masked
// as ***. Section headers are fixed so runbooks and Splunk
// searches still match.
//
// Two fields deliberately do not render their raw value, because 0 means the
// opposite of what it reads as: context_window is 0 "from gateway" (the limit
// comes from max_input_tokens at team startup, and the resolved value never
// appeared in this dump at all), and max_diff_bytes is 0 "unlimited" (a cap of
// zero would refuse every diff, which is what the field used to mean).
func Dump(app *App, log *slog.Logger) {
	section(log, "config.bitbucket", kv{"base_url", app.Bitbucket.BaseURL}, kv{"token", mask}, kv{"username", app.Bitbucket.Username},
		kv{"max_diff_bytes", byteCap(app.Bitbucket.MaxDiffBytes)}, kv{"max_file_bytes", app.Bitbucket.MaxFileBytes})
	llmSection(log, "config.llm", app.LLM)
	reviewSection(log, "config.review", app.Review)
	jiraSection(log, "config.jira", app.Jira)
	section(log, "config.server", kv{"host", app.Server.Host}, kv{"port", app.Server.Port}, kv{"public_url", app.Server.PublicURL})
	section(log, "config.database", kv{"url", mask})
	section(log, "config.trust", kv{"headroom_tokens", app.Trust.HeadroomTokens}, kv{"threshold", app.Trust.Threshold}, kv{"tail", app.Trust.Tail})
	section(log, "config.queue", kv{"inference_concurrency", app.Queue.InferenceConcurrency},
		kv{"inference_concurrency_per_team", app.Queue.InferenceConcurrencyPerTeam})
	log.Info(fmt.Sprintf("[config.teams] path = %s", app.TeamsConfigPath))
	for _, slug := range app.Order {
		team := app.Teams[slug]
		log.Info(fmt.Sprintf("[config.teams.%s] name = %s", slug, team.Name))
		log.Info("  webhook_secret = ***")
		scopes := make([]string, len(team.Projects))
		for i, p := range team.Projects {
			scopes[i] = p.String()
		}
		log.Info(fmt.Sprintf("  projects = %s", quotedList(scopes)))
		llmSection(log, "config.teams."+slug+".llm", team.LLM)
		reviewSection(log, "config.teams."+slug+".review", team.Review)
		jiraSection(log, "config.teams."+slug+".jira", team.Jira)
		if team.Riptide != nil {
			section(log, "config.teams."+slug+".riptide", kv{"url", team.Riptide.URL}, kv{"token", mask})
		} else {
			log.Info(fmt.Sprintf("[config.teams.%s.riptide] disabled", slug))
		}
	}
	for _, slug := range sortedKeys(app.Disabled) {
		log.Error(fmt.Sprintf("[config.teams.%s] DISABLED: %s", slug, app.Disabled[slug]))
	}
}

const mask = "***"

type kv struct {
	k string
	v any
}

func section(log *slog.Logger, label string, fields ...kv) {
	log.Info(fmt.Sprintf("[%s]", label))
	for _, f := range fields {
		log.Info(fmt.Sprintf("  %s = %v", f.k, render(f.v)))
	}
}

func render(v any) string {
	switch x := v.(type) {
	case []string:
		return quotedList(x)
	case bool:
		if x {
			return "True"
		}
		return "False"
	case float64:
		// The dump format keeps the decimal point: 5.0, not 5.
		// FormatFloat with -1 drops it, so add it back for whole numbers.
		s := strconv.FormatFloat(x, 'f', -1, 64)
		if !strings.ContainsAny(s, ".eE") {
			s += ".0"
		}
		return s
	default:
		return fmt.Sprint(v)
	}
}

// quotedList renders a list of strings as ['a', 'b'], single-quoted: that is
// the shape the existing Splunk searches match on.
func quotedList(items []string) string {
	quoted := make([]string, len(items))
	for i, s := range items {
		quoted[i] = "'" + s + "'"
	}
	return "[" + strings.Join(quoted, ", ") + "]"
}

// contextWindow renders the configured window. 0 is not a window: it means
// the limit comes from the gateway's max_input_tokens, resolved per team at
// startup and logged there.
func contextWindow(n int) string {
	if n == 0 {
		return "from gateway"
	}
	return fmt.Sprint(n)
}

// reasoningEffort renders a level. Empty is not "no reasoning": the model's
// balanced level applies, resolved per team at startup and shown in its
// model label.
func reasoningEffort(s string) string {
	if s == "" {
		return "model balanced"
	}
	return s
}

// byteCap renders a byte cap. 0 is not a cap of zero, which would refuse every
// body; it means no cap at all, which is the diff default.
func byteCap(n int) string {
	if n <= 0 {
		return "unlimited"
	}
	return fmt.Sprint(n)
}

func llmSection(log *slog.Logger, label string, l LLM) {
	section(log, label,
		kv{"model", l.Model}, kv{"api_key", mask}, kv{"base_url", l.BaseURL},
		kv{"gateway_models", l.GatewayModels}, kv{"reasoning_effort", reasoningEffort(l.ReasoningEffort)},
		kv{"context_window", contextWindow(l.ContextWindow)})
}

func reviewSection(log *slog.Logger, label string, r Review) {
	section(log, label,
		kv{"auto_review_authors", r.AutoReviewAuthors}, kv{"ignore_authors", r.IgnoreAuthors},
		kv{"exclude_repos", r.ExcludeRepos}, kv{"max_comments", r.MaxComments},
		kv{"max_file_lines", r.MaxFileLines}, kv{"diff_extra_lines_before", r.DiffExtraLinesBefore},
		kv{"diff_extra_lines_after", r.DiffExtraLinesAfter},
		kv{"diff_max_extra_lines_dynamic_context", r.DiffMaxExtraLinesDynamicContext},
		kv{"diff_allow_dynamic_context", r.DiffAllowDynamicContext},
		kv{"review_prompt_template", r.ReviewPromptTemplate}, kv{"mention_prompt_template", r.MentionPromptTemplate},
		kv{"ticket_compliance_check", r.TicketComplianceCheck}, kv{"require_agents_md", r.RequireAgentsMD},
		kv{"agents_md_warn_tokens", r.AgentsMDWarnTokens}, kv{"agents_md_max_tokens", r.AgentsMDMaxTokens},
		kv{"agents_md_custom_link", r.AgentsMDCustomLink}, kv{"opt_out_branch_keyword", r.OptOutBranchKeyword},
		kv{"max_pr_cost_usd", r.MaxPRCostUSD})
}

func jiraSection(log *slog.Logger, label string, j Jira) {
	section(log, label, kv{"url", j.URL}, kv{"token", mask}, kv{"acceptance_criteria_prefixes", j.AcceptanceCriteriaPrefixes})
}

func sortedKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
