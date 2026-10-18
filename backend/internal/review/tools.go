package review

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/store"
)

// mentionAsk is who asked what on which PR: what the tools may act on. It
// comes from the webhook, never from a model argument, so no call can name
// another team, PR or user.
type mentionAsk struct {
	key   store.PRKey
	prTag string
	// user is the commenter's Bitbucket name, text their comment with the
	// mention stripped.
	user, text string
	// switched is set when set_my_language stored a language this turn:
	// the posting stage confirms it in that language.
	switched lang.Lang
}

// mentionTools are the tools a Q&A answer may call: two reads and the
// commenter's own language. Every one runs off the worker, so none calls
// Bitbucket; each returns an explicit struct, never a config or a row.
func (r *Reviewer) mentionTools(ask *mentionAsk) []inference.MentionTool {
	return []inference.MentionTool{
		{
			Name: "team_config",
			Description: "noergler's effective review settings for this team, and whether the asking user " +
				"is on the auto-review or ignore list. Use it for questions about how noergler is configured.",
			Run: func(ctx context.Context, _ string) string { return toolJSON(r.teamConfig(ctx, ask.user)) },
		},
		{
			Name: "pr_status",
			Description: "What noergler knows about this pull request: whether and when it was reviewed, the " +
				"latest review outcomes with their skip reasons, and its cost against the cap. Use it for " +
				"questions like why the PR was or was not reviewed.",
			Run: func(ctx context.Context, _ string) string { return toolJSON(r.prStatus(ctx, ask.key)) },
		},
		{
			Name: "set_my_language",
			Description: "Set the language noergler writes to the asking user in, for their PRs and their " +
				"questions. Only when the user asks for it in their own comment.",
			Parameters: map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"required":             []any{"language"},
				"properties": map[string]any{
					"language": map[string]any{"type": "string", "enum": []any{"de", "en"}},
				},
			},
			Run: func(ctx context.Context, args string) string { return toolJSON(r.setMyLanguage(ctx, ask, args)) },
		},
	}
}

func toolJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return `{"error": "unavailable"}`
	}
	return string(b)
}

// teamConfigView is what team_config returns. An allowlist: a field added
// to config.Review is not exposed until it is added here. The author lists
// are counts, as on the public dashboard: the names sit behind the team
// secret, and a commenter is not that.
type teamConfigView struct {
	Team                  string `json:"team"`
	Model                 string `json:"model"`
	DefaultLanguage       string `json:"default_language"`
	MaxPRCostUSD          string `json:"max_pr_cost_usd"`
	MaxComments           int    `json:"max_comments"`
	MaxFileLines          int    `json:"max_file_lines"`
	RequireAgentsMD       bool   `json:"require_agents_md"`
	AgentsMDWarnTokens    int    `json:"agents_md_warn_tokens"`
	AgentsMDMaxTokens     int    `json:"agents_md_max_tokens"`
	TicketComplianceCheck bool   `json:"ticket_compliance_check"`
	JiraEnabled           bool   `json:"jira_enabled"`
	UsePRDescription      bool   `json:"use_pr_description"`
	OptOutBranchKeyword   string `json:"opt_out_branch_keyword"`
	AutoReviewAuthors     int    `json:"auto_review_authors_count"`
	IgnoreAuthors         int    `json:"ignore_authors_count"`
	Asker                 struct {
		Language     string `json:"language"`
		AutoReviewed bool   `json:"auto_reviewed"`
		Ignored      bool   `json:"ignored"`
	} `json:"asking_user"`
}

func (r *Reviewer) teamConfig(ctx context.Context, user string) teamConfigView {
	auto, ignore := r.authorLists()
	v := teamConfigView{
		Team:                  r.TeamSlug,
		Model:                 r.llm.Label(),
		DefaultLanguage:       string(r.cfg.Language.Or(lang.Default)),
		MaxPRCostUSD:          fmt.Sprintf("%.3f", r.cfg.MaxPRCostUSD),
		MaxComments:           r.cfg.MaxComments,
		MaxFileLines:          r.cfg.MaxFileLines,
		RequireAgentsMD:       r.cfg.RequireAgentsMD,
		AgentsMDWarnTokens:    r.cfg.AgentsMDWarnTokens,
		AgentsMDMaxTokens:     r.cfg.AgentsMDMaxTokens,
		TicketComplianceCheck: r.cfg.TicketComplianceCheck,
		JiraEnabled:           r.jiraEnabled(),
		UsePRDescription:      r.usePRDescription.Load(),
		OptOutBranchKeyword:   r.cfg.OptOutBranchKeyword,
		AutoReviewAuthors:     len(auto),
		IgnoreAuthors:         len(ignore),
	}
	v.Asker.Language = string(r.languageFor(ctx, user))
	v.Asker.AutoReviewed, v.Asker.Ignored = r.autoReviewDecision(user)
	return v
}

// prStatusView is what pr_status returns. Skip reasons are the stable enum
// values, not dashboard labels: the model words them in the user's language.
type prStatusView struct {
	Known              bool              `json:"known"`
	LastReviewedCommit string            `json:"last_reviewed_commit,omitempty"`
	Ignored            bool              `json:"ignored_summary_deleted"`
	Merged             bool              `json:"merged"`
	Declined           bool              `json:"declined"`
	Runs               int               `json:"review_runs"`
	LastRun            *prStatusRun      `json:"last_run,omitempty"`
	Attempts           []prStatusAttempt `json:"latest_outcomes"`
	// CostUSD is nil when unpriced: never "0.000" for an unknown.
	CostUSD *string `json:"pr_cost_usd"`
	CapUSD  string  `json:"pr_cost_cap_usd"`
}

type prStatusRun struct {
	At          string `json:"at"`
	Kind        string `json:"kind"`
	Incremental bool   `json:"incremental"`
	Findings    int    `json:"findings_posted"`
	Model       string `json:"model"`
}

type prStatusAttempt struct {
	At         string `json:"at"`
	Kind       string `json:"kind"`
	Outcome    string `json:"outcome"`
	SkipReason string `json:"skip_reason,omitempty"`
}

func (r *Reviewer) prStatus(ctx context.Context, key store.PRKey) prStatusView {
	v := prStatusView{CapUSD: fmt.Sprintf("%.3f", r.cfg.MaxPRCostUSD), Attempts: []prStatusAttempt{}}
	st := safeDB(ctx, r.log, "PRStatus", func() (*store.PRStatus, error) { return r.store.PRStatus(ctx, key) })
	if st == nil {
		return v
	}
	v.Known, v.Runs = st.Known, st.Runs
	if st.LastReviewedCommit != nil {
		v.LastReviewedCommit = shortSHA(*st.LastReviewedCommit, 10)
	}
	v.Ignored, v.Merged, v.Declined = st.IgnoredAt != nil, st.MergedAt != nil, st.DeclinedAt != nil
	if lr := st.LastRun; lr != nil {
		v.LastRun = &prStatusRun{At: lr.At.UTC().Format(time.RFC3339), Kind: string(lr.Kind),
			Incremental: lr.Incremental, Findings: lr.Findings, Model: lr.ModelLabel}
	}
	for _, a := range st.Attempts {
		v.Attempts = append(v.Attempts, prStatusAttempt{At: a.At.UTC().Format(time.RFC3339), Kind: string(a.Kind),
			Outcome: a.Outcome, SkipReason: a.Reason})
	}
	if cost := safeDB(ctx, r.log, "PRCost", func() (*int64, error) { return r.store.PRCost(ctx, key) }); cost != nil {
		s := fmt.Sprintf("%.3f", float64(*cost)/nanoPerUSD)
		v.CostUSD = &s
	}
	return v
}

// languageWord is a language named in the commenter's own text. Unicode
// boundaries: RE2's \b is ASCII.
var languageWord = map[lang.Lang]*regexp.Regexp{
	lang.EN: regexp.MustCompile(`(?i)(?:^|[^\pL\pN_])(?:english|englisch|en)(?:[^\pL\pN_]|$)`),
	lang.DE: regexp.MustCompile(`(?i)(?:^|[^\pL\pN_])(?:deutsch|german|de)(?:[^\pL\pN_]|$)`),
}

// setMyLanguage is the model proposing a switch and code deciding it: the
// commenter's own comment must name the language. The model reads the diff
// and the description, so a call it was steered into by either finds no
// such word and changes nothing.
func (r *Reviewer) setMyLanguage(ctx context.Context, ask *mentionAsk, args string) map[string]string {
	var in struct {
		Language string `json:"language"`
	}
	l, ok := lang.Lang(""), false
	if json.Unmarshal([]byte(args), &in) == nil {
		l, ok = lang.Parse(in.Language)
	}
	if !ok || !languageWord[l].MatchString(ask.text) {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: language switch for %s rejected: the comment does not ask for %q",
			ask.prTag, ask.user, strings.TrimSpace(in.Language)))
		return map[string]string{"result": "rejected: the user's comment does not ask for this language"}
	}
	if err := r.store.SetUserLanguage(ctx, ask.user, l); err != nil {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: language for %s not stored: %v", ask.prTag, ask.user, err))
		return map[string]string{"result": "failed: could not be saved, tell the user to try again later"}
	}
	r.log.InfoContext(ctx, fmt.Sprintf("%s: language set to %s for %s", ask.prTag, l, ask.user))
	ask.switched = l
	return map[string]string{"result": "ok", "write_the_answer_in": string(l)}
}
