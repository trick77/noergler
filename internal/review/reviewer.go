package review

import (
	"context"
	"fmt"
	"log/slog"
	"regexp"
	"strings"

	"github.com/trick77/noergler-go/internal/config"
	"github.com/trick77/noergler-go/internal/jira"
	"github.com/trick77/noergler-go/internal/store"
)

// fileFetchConcurrency bounds how many full file bodies are in flight at
// once; each one stays resident until the review is rendered.
const fileFetchConcurrency = 4

// maxPreviouslyPostedFindings caps the prompt's previously-posted block by
// count; the token budget caps it again by size.
const maxPreviouslyPostedFindings = 50

// bytesPerTokenCeiling is an upper bound on bytes per token for diff text;
// real code diffs sit at 3-4. Used to reject a cumulative diff by length
// before paying to tokenize it, since tokenizing expands the text in RAM.
const bytesPerTokenCeiling = 8

// reviewKeywords route a mention to a full review rather than Q&A.
var reviewKeywords = map[string]bool{
	"review": true, "review this": true, "re-review": true, "rereview": true,
}

// jiraTicketRE extracts a ticket key from a branch name or PR title.
//
// Python is `\b([A-Z]{2,10}-\d{1,7})\b` and both classes differ in RE2: its
// \b is ASCII where Python's is Unicode, and its \d is ASCII where Python's
// matches Arabic-Indic digits. Probed in both directions against the venv:
// a naive port matches "üABC-123" where Python does not, and misses
// "ABC-١٢٣" where Python matches. This form agrees with Python on all ten
// probe cases.
//
// The boundary characters are consumed by the group-less alternatives, so
// only the captured group is read.
var jiraTicketRE = regexp.MustCompile(`(?:^|[^\pL\pN_])([A-Z]{2,10}-\p{Nd}{1,7})(?:[^\pL\pN_]|$)`)

// Reviewer reviews pull requests for one team.
type Reviewer struct {
	TeamSlug string

	bitbucket BitbucketClient
	llm       InferenceClient
	store     Store
	jira      JiraClient
	riptide   RiptideEmitter
	tokens    TokenCounter
	cfg       config.Review
	// template is the review prompt, mentionTmpl the Q&A prompt. Both are
	// loaded once at startup and resident for the team.
	template    string
	mentionTmpl string
	log         *slog.Logger
}

// Options builds a Reviewer. Bitbucket, the store, the inference client, the
// token counter and the prompt template are required; Jira and riptide are
// optional and nil disables that feature for the team.
type Options struct {
	TeamSlug  string
	Bitbucket BitbucketClient
	LLM       InferenceClient
	Store     Store
	Jira      JiraClient
	Riptide   RiptideEmitter
	Tokens    TokenCounter
	Config    config.Review
	// Template is the review prompt; MentionTemplate is the Q&A prompt.
	Template        string
	MentionTemplate string
	Log             *slog.Logger
}

// New builds a Reviewer for one team.
func New(opt Options) *Reviewer {
	return &Reviewer{
		TeamSlug:    opt.TeamSlug,
		bitbucket:   opt.Bitbucket,
		llm:         opt.LLM,
		store:       opt.Store,
		jira:        opt.Jira,
		riptide:     opt.Riptide,
		tokens:      opt.Tokens,
		cfg:         opt.Config,
		template:    opt.Template,
		mentionTmpl: opt.MentionTemplate,
		log:         opt.Log,
	}
}

// jiraEnabled reports whether this team has a Jira client.
//
// A typed nil in the interface would pass a plain != nil check, so New's
// callers pass an untyped nil and this stays the single place that asks.
func (r *Reviewer) jiraEnabled() bool { return r.jira != nil }

// IsAutoReviewAuthor reports whether a PR by this author is reviewed
// automatically.
//
// The ignore list wins over the allow list. An @mention bypasses both, which
// is why the caller passes skipAuthorCheck: a bot's PR can still be reviewed
// on request.
func (r *Reviewer) IsAutoReviewAuthor(author string) bool {
	for _, ignored := range r.cfg.IgnoreAuthors {
		if ignored == author {
			return false
		}
	}
	if len(r.cfg.AutoReviewAuthors) == 0 {
		return true
	}
	for _, allowed := range r.cfg.AutoReviewAuthors {
		if allowed == author {
			return true
		}
	}
	return false
}

func (r *Reviewer) isIgnoredAuthor(name string) bool {
	for _, ignored := range r.cfg.IgnoreAuthors {
		if ignored == name {
			return true
		}
	}
	return false
}

// extractTicketID reads a Jira key off the branch name, then the PR title.
func extractTicketID(branch, title string) string {
	for _, s := range []string{branch, title} {
		if m := jiraTicketRE.FindStringSubmatch(s); m != nil {
			return m[1]
		}
	}
	return ""
}

// formatTicketBlock renders one ticket for the prompt's ticket context.
func formatTicketBlock(t *jira.Ticket, heading string) []string {
	lines := []string{fmt.Sprintf("### %s: [%s](%s)", heading, t.Key, t.URL)}
	lines = append(lines, "**Title:** "+t.Title)
	if t.IssueType != "" || t.Status != "" {
		var parts []string
		if t.IssueType != "" {
			parts = append(parts, "**Type:** "+t.IssueType)
		}
		if t.Status != "" {
			parts = append(parts, "**Status:** "+t.Status)
		}
		lines = append(lines, strings.Join(parts, " · "))
	}
	if t.Description != "" {
		lines = append(lines, "**Description:** "+t.Description)
	}
	if len(t.Labels) > 0 {
		lines = append(lines, "**Labels:** "+strings.Join(t.Labels, ", "))
	}
	if t.AcceptanceCriteria != "" {
		lines = append(lines, "**Acceptance criteria:** "+t.AcceptanceCriteria)
	}
	if len(t.Subtasks) > 0 {
		lines = append(lines, "**Subtasks:**")
		for _, st := range t.Subtasks {
			lines = append(lines, "- "+st)
		}
	}
	return lines
}

// renderTicketContext builds the prompt block for a ticket and its parent.
//
// Python re-fetches the ticket here (reviewer.py:359) having already fetched
// it with its parent; AGENTS.md pins fetching once, so this formats what the
// caller already holds.
func renderTicketContext(ticket, parent *jira.Ticket) string {
	if ticket == nil {
		return ""
	}
	var blocks []string
	if parent != nil {
		blocks = append(blocks, formatTicketBlock(parent, "Parent ticket")...)
		blocks = append(blocks, "")
		blocks = append(blocks, formatTicketBlock(ticket, "Sub-task")...)
	} else {
		blocks = append(blocks, formatTicketBlock(ticket, "Jira ticket")...)
	}
	return strings.Join(blocks, "\n")
}

// fetchRepoInstructions reads AGENTS.md from the PR branch, falling back to
// the target branch.
func (r *Reviewer) fetchRepoInstructions(ctx context.Context, project, repo, fromCommit, toDisplayID string) string {
	for _, ref := range []string{fromCommit, toDisplayID} {
		if ref == "" {
			continue
		}
		content, err := r.bitbucket.FetchFileContent(ctx, project, repo, ref, "AGENTS.md")
		if err != nil {
			r.log.DebugContext(ctx, "AGENTS.md not found at ref "+ref)
			continue
		}
		if strings.TrimSpace(content) != "" {
			r.log.InfoContext(ctx, "Loaded AGENTS.md from ref "+ref)
			return content
		}
	}
	return ""
}

// prKey builds the store key.
func prKey(project, repo string, prID int) store.PRKey {
	return store.PRKey{Project: project, Repo: repo, PRID: prID}
}

// shortSHA truncates a commit for log lines and notices.
func shortSHA(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}
