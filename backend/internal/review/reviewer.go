package review

import (
	"context"
	"fmt"
	"log/slog"
	"regexp"
	"slices"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/jira"
	"github.com/trick77/noergler/internal/prompts"
	"github.com/trick77/noergler/internal/webhook"
)

// fileFetchConcurrency bounds how many full file bodies are in flight at
// once; each one stays resident until the prompt is assembled.
const fileFetchConcurrency = 4

// maxPreviouslyPostedFindings caps the prompt's previously-posted block by
// count; the token budget caps it again by size.
const maxPreviouslyPostedFindings = 50

// bytesPerTokenCeiling is an upper bound on bytes per token for diff text;
// real code diffs sit at 3-4. Used to reject a cumulative-diff part by length
// before paying to tokenize it, since tokenizing expands the text in RAM.
const bytesPerTokenCeiling = 8

// reviewKeywords route a mention to a full review rather than Q&A.
var reviewKeywords = map[string]bool{
	"review": true, "review this": true, "re-review": true, "rereview": true,
}

// jiraTicketRE extracts a ticket key from a branch name or PR title.
//
// Both character classes must be Unicode-aware, and RE2's are ASCII: with a
// plain `\b([A-Z]{2,10}-\d{1,7})\b` the pattern matches "üABC-123", where ü
// is a word character and there is no boundary, and misses "ABC-١٢٣". The
// boundary is therefore spelled out and the digits are \p{Nd}
// (TestExtractTicketIDIsPinned).
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
	// almanach is the team's Almanach, nil or disabled when the team has no
	// almanach: block: then no note call is made and no row written.
	almanach         AlmanachEmitter
	almanachTmpl     string
	almanachMaxChars int
	// almanachInflight is the PRs whose note call is on the pool right now,
	// so a merge that lands meanwhile stamps the row and leaves the emit to
	// the round. Single process: a map is enough.
	almanachInflight inflightSet
	tokens           TokenCounter
	cfg              config.Review
	// The two author lists live here rather than in cfg because the team API
	// rewrites them at runtime (PUT /teams/{slug}/settings) while the review
	// worker may be reading them. Everything else in cfg is fixed at startup.
	// ExcludeRepos is deliberately not mirrored here: the webhook route reads
	// it off the team config snapshot, and the Reviewer never asks.
	authorsMu         sync.RWMutex
	autoReviewAuthors []string
	ignoreAuthors     []string
	// usePRDescription is rewritten by the same API, so it is atomic too.
	usePRDescription atomic.Bool
	// template is the review prompt, mentionTmpl the Q&A prompt; both the
	// compiled-in ones unless a test overrides them.
	template    string
	mentionTmpl string
	// replyTmpl answers a reply on a finding's thread.
	replyTmpl string
	// displayName extracts the author's first name for a finding reply;
	// nil = names off.
	displayName *regexp.Regexp
	log         *slog.Logger
	// owns is the live claim check: a repo can be given up while its
	// review is queued or in inference.
	owns func(project, repo string) bool
	// team is the live snapshot behind owns, for what team_config shows.
	team func() *config.Team
	// publicURL is the instance's base URL (NOERGLER_PUBLIC_URL), for the
	// footer's link to the run page; empty prints no link.
	publicURL string
}

// Options builds a Reviewer. Bitbucket, the store, the inference client and
// the token counter are required; Jira and riptide are optional and nil
// disables that feature for the team.
type Options struct {
	TeamSlug  string
	Bitbucket BitbucketClient
	LLM       InferenceClient
	Store     Store
	Jira      JiraClient
	Riptide   RiptideEmitter
	// Almanach is the team's note receiver; nil or disabled = no feed.
	// AlmanachNoteMaxChars is the instance cap on a serialized note.
	Almanach             AlmanachEmitter
	AlmanachNoteMaxChars int
	Tokens               TokenCounter
	Config               config.Review
	// Template, MentionTemplate, FindingReplyTemplate and AlmanachTemplate
	// override the compiled-in prompts, for tests only: production leaves
	// them empty and the prompts are never configurable.
	Template             string
	MentionTemplate      string
	FindingReplyTemplate string
	AlmanachTemplate     string
	// DisplayNamePattern is config.Bitbucket's; nil = names off.
	DisplayNamePattern *regexp.Regexp
	Log                *slog.Logger
	// Owns reports whether the team still claims the repo; nil claims all.
	Owns func(project, repo string) bool
	// Team is the live team snapshot (claims, exclude_repos, Jira, riptide),
	// read by team_config only; nil leaves those fields out.
	Team func() *config.Team
	// PublicURL is config.Server.PublicURL, trailing slash already trimmed.
	PublicURL string
}

// New builds a Reviewer for one team.
func New(opt Options) *Reviewer {
	r := &Reviewer{
		TeamSlug:          opt.TeamSlug,
		bitbucket:         opt.Bitbucket,
		llm:               opt.LLM,
		store:             opt.Store,
		jira:              opt.Jira,
		riptide:           opt.Riptide,
		almanach:          opt.Almanach,
		almanachTmpl:      orDefault(opt.AlmanachTemplate, prompts.AlmanachNote),
		almanachMaxChars:  opt.AlmanachNoteMaxChars,
		tokens:            opt.Tokens,
		cfg:               opt.Config,
		autoReviewAuthors: opt.Config.AutoReviewAuthors,
		ignoreAuthors:     opt.Config.IgnoreAuthors,
		template:          orDefault(opt.Template, prompts.Review),
		mentionTmpl:       orDefault(opt.MentionTemplate, prompts.Mention),
		replyTmpl:         orDefault(opt.FindingReplyTemplate, prompts.FindingReply),
		displayName:       opt.DisplayNamePattern,
		log:               opt.Log,
		owns:              opt.Owns,
		team:              opt.Team,
		publicURL:         opt.PublicURL,
	}
	r.usePRDescription.Store(opt.Config.UsePRDescription)
	return r
}

// orDefault is s, or the compiled-in prompt when s is empty: no Reviewer can
// send a blank prompt.
func orDefault(s, compiled string) string {
	if s == "" {
		return compiled
	}
	return s
}

// SetUsePRDescription switches the PR title and description in or out of
// the prompt for every later review and mention.
func (r *Reviewer) SetUsePRDescription(on bool) { r.usePRDescription.Store(on) }

// SetAuthorLists replaces the two author lists.
//
// The team API writes them while the review worker may be reading them, so
// they live behind a lock instead of in cfg
// (TestSetAuthorListsIsSafeUnderConcurrentReads). Only these two fields are
// mirrored onto the live Reviewer; INVARIANTS.md: exclude_repos is not.
func (r *Reviewer) SetAuthorLists(auto, ignore []string) {
	r.authorsMu.Lock()
	defer r.authorsMu.Unlock()
	r.autoReviewAuthors, r.ignoreAuthors = auto, ignore
}

// authorLists takes one snapshot of both lists. The slices are replaced
// wholesale by SetAuthorLists and never mutated in place, so the caller may
// range over them after the lock is dropped.
func (r *Reviewer) authorLists() (auto, ignore []string) {
	r.authorsMu.RLock()
	defer r.authorsMu.RUnlock()
	return r.autoReviewAuthors, r.ignoreAuthors
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
	autoReview, _ := r.autoReviewDecision(author)
	return autoReview
}

func (r *Reviewer) isIgnoredAuthor(name string) bool {
	_, ignore := r.authorLists()
	return slices.Contains(ignore, name)
}

// autoReviewDecision answers both "is this author auto-reviewed" and "which
// list decided" from ONE snapshot.
//
// Asking IsAutoReviewAuthor and then isIgnoredAuthor takes two, and a
// settings write landing between them would let the skip reason describe a
// different list state than the decision did, which is the misreporting this
// is meant to remove. Same rule as teams.Runtime: one snapshot per request.
func (r *Reviewer) autoReviewDecision(author string) (autoReview, ignored bool) {
	auto, ignore := r.authorLists()
	if slices.Contains(ignore, author) {
		return false, true
	}
	if len(auto) == 0 {
		return true, false
	}
	return slices.Contains(auto, author), false
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
// The caller already holds the ticket and its parent, so this only formats
// them. Re-fetching here would double the Jira traffic per review, which
// a test pins against ("Jira fetched once per review").
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

// maxPRDescriptionRunes caps the PR description in the prompt, the same
// figure as the Jira description cap: long ones are mostly templates and
// changelogs, and Bitbucket allows several times this.
const maxPRDescriptionRunes = 5000

// renderPRDescription builds the prompt block for the PR's own title and
// description. No description renders nothing: the title alone restates
// what the branch and the ticket already say.
func renderPRDescription(title, description string) string {
	description = strings.TrimSpace(description)
	if description == "" {
		return ""
	}
	if capped := truncateRunes(description, maxPRDescriptionRunes); capped != description {
		description = capped + "..."
	}
	return "**Title:** " + title + "\n**Description:**\n" + description
}

// prDescription is the PR description block for this run, or "" when the
// team switched it off or the PR has none. Logged either way: it changes
// what the model is told the PR is for.
func (r *Reviewer) prDescription(ctx context.Context, prTag string, pr webhook.PullRequest) string {
	if !r.usePRDescription.Load() {
		r.log.InfoContext(ctx, prTag+": PR description not sent (disabled by team)")
		return ""
	}
	block := renderPRDescription(pr.Title, pr.Description)
	if block == "" {
		r.log.InfoContext(ctx, prTag+": PR description not sent (empty)")
		return ""
	}
	// Runes, not tokens: the review's composition line already counts it,
	// and a second tokenizer pass here would run on the single worker.
	r.log.InfoContext(ctx, fmt.Sprintf("%s: PR description sent (%d runes)", prTag, len([]rune(block))))
	return block
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

// shortSHA truncates a commit for log lines and notices.
func shortSHA(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}
