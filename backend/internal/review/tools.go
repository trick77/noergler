package review

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"strings"
	"time"

	"github.com/trick77/noergler/internal/diff"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/render"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/ship/kit/logging"
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
	// from and to are the PR's source and target branch.
	from, to string
	// lang is the asker's language as resolved for this answer.
	lang lang.Lang
	// switched is set when set_my_language accepted a switch this turn:
	// the posting stage stores and confirms it.
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
			Run: func(context.Context, string) string { return toolJSON(r.teamConfig(ask)) },
		},
		{
			Name: "pr_status",
			Description: "What noergler knows about this pull request: whether and when it was reviewed, the " +
				"latest review outcomes with their skip reasons, and its cost against the cap. Use it for " +
				"questions like why the PR was or was not reviewed.",
			Run: func(ctx context.Context, _ string) string { return toolJSON(r.prStatus(ctx, ask)) },
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
// carry the names (decided): Bitbucket user names, which any commenter on
// the team's PRs can see in Bitbucket anyway, and "who is auto-reviewed"
// is the question this tool exists for.
//
// Never in it: env var names, keys, tokens, the webhook secret, the gateway
// host and alias, the riptide URL, disable reasons.
type teamConfigView struct {
	Team            string `json:"team"`
	Model           string `json:"model"`
	DefaultLanguage string `json:"default_language"`
	// Projects are the claims, "KEY" or "KEY/{repo,...}"; absent when the
	// Reviewer has no live team view.
	Projects     []string `json:"projects,omitempty"`
	ExcludeRepos []string `json:"exclude_repos"`
	// A PR from a matching source or into a matching target branch is not
	// auto-reviewed; a mention still is.
	ExcludeSourceBranches []string `json:"exclude_source_branches"`
	ExcludeTargetBranches []string `json:"exclude_target_branches"`
	// AutoReviewAuthors empty means every author is auto-reviewed; the
	// ignore list wins over it.
	AutoReviewAuthors   []string `json:"auto_review_authors"`
	IgnoreAuthors       []string `json:"ignore_authors"`
	OptOutBranchKeyword string   `json:"opt_out_branch_keyword"`
	MaxPRCostUSD        string   `json:"max_pr_cost_usd"`
	MaxComments         int      `json:"max_comments"`
	MaxFileLines        int      `json:"max_file_lines"`
	RequireAgentsMD     bool     `json:"require_agents_md"`
	AgentsMDWarnTokens  int      `json:"agents_md_warn_tokens"`
	AgentsMDMaxTokens   int      `json:"agents_md_max_tokens"`
	AgentsMDCustomLink  string   `json:"agents_md_custom_link,omitempty"`
	// The same reason disputed in this many PRs within this many days gets
	// an AGENTS.md proposal in the finding thread. The knobs, never the
	// signals themselves.
	DisputeSignalWindowDays int      `json:"dispute_signal_window_days"`
	DisputeSignalThreshold  int      `json:"dispute_signal_threshold_prs"`
	UsePRDescription        bool     `json:"use_pr_description"`
	TicketComplianceCheck   bool     `json:"ticket_compliance_check"`
	JiraEnabled             bool     `json:"jira_enabled"`
	JiraACPrefixes          []string `json:"jira_acceptance_criteria_prefixes,omitempty"`
	RiptideEnabled          bool     `json:"riptide_enabled"`
	Diff                    struct {
		ExtraLinesBefore       int  `json:"context_lines_before"`
		ExtraLinesAfter        int  `json:"context_lines_after"`
		DynamicContext         bool `json:"dynamic_context"`
		MaxDynamicContextLines int  `json:"max_dynamic_context_lines"`
	} `json:"diff_context"`
	// NeverReviewed is built in, the same for every team.
	NeverReviewed struct {
		Dirs       []string       `json:"dirs"`
		Files      []string       `json:"files"`
		Extensions []string       `json:"extensions"`
		SizeGated  map[string]int `json:"reviewed_only_up_to_changed_lines"`
	} `json:"never_reviewed_builtin"`
	Asker struct {
		Language     string `json:"language"`
		AutoReviewed bool   `json:"auto_reviewed"`
		Ignored      bool   `json:"ignored"`
	} `json:"asking_user"`
}

func (r *Reviewer) teamConfig(ask *mentionAsk) teamConfigView {
	auto, ignore := r.authorLists()
	v := teamConfigView{
		Team:                    r.TeamSlug,
		Model:                   r.llm.Label(),
		DefaultLanguage:         string(r.cfg.Language.Or(lang.Default)),
		ExcludeRepos:            nonNilNames(r.cfg.ExcludeRepos),
		ExcludeSourceBranches:   nonNilNames(r.cfg.ExcludeSourceBranches),
		ExcludeTargetBranches:   nonNilNames(r.cfg.ExcludeTargetBranches),
		AutoReviewAuthors:       nonNilNames(auto),
		IgnoreAuthors:           nonNilNames(ignore),
		OptOutBranchKeyword:     r.cfg.OptOutBranchKeyword,
		MaxPRCostUSD:            usd3(r.cfg.MaxPRCostUSD),
		MaxComments:             r.cfg.MaxComments,
		MaxFileLines:            r.cfg.MaxFileLines,
		DisputeSignalWindowDays: r.cfg.DisputeSignalWindowDays,
		DisputeSignalThreshold:  r.cfg.DisputeSignalThreshold,
		RequireAgentsMD:         r.cfg.RequireAgentsMD,
		AgentsMDWarnTokens:      r.cfg.AgentsMDWarnTokens,
		AgentsMDMaxTokens:       r.cfg.AgentsMDMaxTokens,
		AgentsMDCustomLink:      r.cfg.AgentsMDCustomLink,
		UsePRDescription:        r.usePRDescription.Load(),
		TicketComplianceCheck:   r.cfg.TicketComplianceCheck,
		JiraEnabled:             r.jiraEnabled(),
		RiptideEnabled:          r.riptide != nil && r.riptide.Enabled(),
	}
	v.Diff.ExtraLinesBefore, v.Diff.ExtraLinesAfter = r.cfg.DiffExtraLinesBefore, r.cfg.DiffExtraLinesAfter
	v.Diff.DynamicContext, v.Diff.MaxDynamicContextLines = r.cfg.DiffAllowDynamicContext, r.cfg.DiffMaxExtraLinesDynamicContext
	// The claims and the exclude lists change at runtime (the team API); the
	// live snapshot has them, the Reviewer's copy of cfg does not.
	if r.team != nil {
		if t := r.team(); t != nil {
			for _, p := range t.Projects {
				v.Projects = append(v.Projects, p.String())
			}
			v.ExcludeRepos = nonNilNames(t.Review.ExcludeRepos)
			v.ExcludeSourceBranches = nonNilNames(t.Review.ExcludeSourceBranches)
			v.ExcludeTargetBranches = nonNilNames(t.Review.ExcludeTargetBranches)
			if v.JiraEnabled {
				v.JiraACPrefixes = nonNilNames(t.Jira.AcceptanceCriteriaPrefixes)
			}
		}
	}
	skip := diff.BuiltinSkipRules()
	v.NeverReviewed.Dirs, v.NeverReviewed.Files = skip.Dirs, skip.Files
	v.NeverReviewed.Extensions, v.NeverReviewed.SizeGated = skip.Extensions, skip.SizeGated
	// Resolved once for the answer: no second read, no second log line.
	v.Asker.Language = string(ask.lang.Or(r.cfg.Language.Or(lang.Default)))
	v.Asker.AutoReviewed, v.Asker.Ignored = r.autoReviewDecision(ask.user)
	return v
}

// nonNilNames is a copy of names that marshals as [], never null: an empty
// list is "nobody", not "unknown".
func nonNilNames(names []string) []string {
	return append([]string{}, names...)
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
	// The branches, and the team list that keeps their pushes from being
	// auto-reviewed: that skip is decided at the webhook and leaves no
	// attempt, so this is the only place the model can learn of it.
	SourceBranch         string `json:"source_branch"`
	TargetBranch         string `json:"target_branch"`
	AutoReviewExcludedBy string `json:"auto_review_excluded_by,omitempty"`
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

// prStatus answers pr_status. A DB fault says so rather than reading as
// "never seen": the model would tell the user noergler has no record.
func (r *Reviewer) prStatus(ctx context.Context, ask *mentionAsk) any {
	key := ask.key
	v := prStatusView{CapUSD: usd3(r.cfg.MaxPRCostUSD), Attempts: []prStatusAttempt{},
		SourceBranch: ask.from, TargetBranch: ask.to}
	// The live snapshot: the lists change at runtime, cfg is a boot copy.
	if r.team != nil {
		if t := r.team(); t != nil {
			v.AutoReviewExcludedBy = t.ExcludedBranch(ask.from, ask.to)
		}
	}
	st, err := r.store.PRStatus(ctx, key, r.TeamSlug)
	if err != nil || st == nil {
		r.log.WarnContext(ctx, "DB operation failed: PRStatus", logging.Err(err))
		return map[string]string{"error": "noergler's records are unavailable right now; say so, do not guess"}
	}
	v.Known, v.Runs = st.Known, st.Runs
	if st.LastReviewedCommit != nil {
		v.LastReviewedCommit = render.ShortSHA(*st.LastReviewedCommit, 10)
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
		s := inference.FormatUSD3(*cost)
		v.CostUSD = &s
	}
	return v
}

// usd3 is a configured dollar amount as the 3-decimal string every USD
// figure uses, through the same rounding as the cost log and the dashboard.
func usd3(usd float64) string { return inference.FormatUSD3(int64(math.Round(usd * nanoPerUSD))) }

// languageWord is a language named in the commenter's own text: the full
// word only. "en" and "de" are path segments and locale tags ("i18n/en/",
// "de-DE") far more often than a request, so they count only as the exact
// keyword, never here. Unicode boundaries: RE2's \b is ASCII.
// switchWords is the other half of a request: a word about answering, the
// language or a change. "Does the German locale fallback work?" names a
// language but asks for nothing, so a call the model was steered into by
// the diff, or misread the question for, changes nothing.
var switchWords = regexp.MustCompile(`(?i)(?:^|[^\pL\pN_])(?:` +
	`answers?|answering|reply|replies|respond|write|writes|talk|speak|switch|language|from now on|` +
	`antwort\pL*|schreib\pL*|sprich|rede|wechsl?\pL*|umstell\pL*|sprache|ab jetzt|künftig` +
	`)(?:[^\pL\pN_]|$)`)

var languageWord = map[lang.Lang]*regexp.Regexp{
	lang.EN: regexp.MustCompile(`(?i)(?:^|[^\pL\pN_])(?:english|englisch|englische[mn]?)(?:[^\pL\pN_]|$)`),
	lang.DE: regexp.MustCompile(`(?i)(?:^|[^\pL\pN_])(?:deutsch|deutsche[mn]?|german)(?:[^\pL\pN_]|$)`),
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
	if !ok || !languageWord[l].MatchString(ask.text) || !asksForSwitch(ask.text) {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: language switch for %s rejected: the comment does not ask for %q",
			ask.prTag, ask.user, strings.TrimSpace(in.Language)))
		return map[string]string{"result": "rejected: the user's comment does not ask for this language"}
	}
	// Stored by the posting stage, with the confirmation: a later round
	// that fails, or a repo given up meanwhile, must not switch a user's
	// language without telling them.
	ask.switched = l
	return map[string]string{"result": "ok", "write_the_answer_in": string(l)}
}

// shortRequestWords is how short a comment naming a language may be and
// still count as asking for it without a switch word: "English please",
// "auf Englisch bitte". A question about a language is longer than that.
const shortRequestWords = 4

// asksForSwitch is the request half of the guard.
func asksForSwitch(text string) bool {
	return switchWords.MatchString(text) || len(strings.Fields(text)) <= shortRequestWords
}
