package render

import (
	"fmt"
	"regexp"
	"strings"
)

// Link is a title and URL pair for the further-reading list.
type Link struct {
	Title string
	URL   string
}

// agentsMDFurtherReading are the curated references surfaced in the
// "AGENTS.md too large" skip summary. Publication dates were verified
// 2026-05; replace if the URLs rot or better-dated sources appear.
var agentsMDFurtherReading = []Link{
	{
		Title: "Upsun — The research is in: your AGENTS.md is probably too long (2026-02-23)",
		URL:   "https://developer.upsun.com/posts/ai/agents-md-less-is-more",
	},
	{
		Title: "Augment Code — A good AGENTS.md is a model upgrade. A bad one is worse than no docs at all (2026-04-22)",
		URL:   "https://www.augmentcode.com/blog/how-to-write-good-agents-dot-md-files",
	},
	{
		Title: `Caveman — Claude Code skill, "why use many token when few token do trick"`,
		URL:   "https://github.com/juliusbrussee/caveman",
	},
}

// markdownLink matches [Title](URL).
//
// The URL group is greedy so a URL containing ")" (Wikipedia-style
// Foo_(bar)) still matches: anchoring to end of string forces the final ")"
// to be the closer.
var markdownLink = regexp.MustCompile(`^\[([^\]]+)\]\((.+)\)$`)

// ParseCustomLink parses "[Title](URL)" markdown or a bare URL. The second
// return is false when the input is empty or blank.
func ParseCustomLink(raw string) (Link, bool) {
	value := strings.TrimSpace(raw)
	if value == "" {
		return Link{}, false
	}
	if m := markdownLink.FindStringSubmatch(value); m != nil {
		return Link{Title: strings.TrimSpace(m[1]), URL: strings.TrimSpace(m[2])}, true
	}
	return Link{Title: value, URL: value}, true
}

// OptOutBranchSummary is the skip summary for a branch carrying the opt-out
// keyword. No LLM or Jira call was made.
func OptOutBranchSummary(keyword, branch string) string {
	return "### Review skipped — opt-out keyword in branch name 🛑\n\n" +
		fmt.Sprintf("The source branch `%s` contains the opt-out keyword ", branch) +
		fmt.Sprintf("`%s`, so the auto-review was skipped. No LLM or Jira ", keyword) +
		"calls were made for this PR.\n\n" +
		"**What to do**\n" +
		"- Rename the branch to remove the keyword and push again; the " +
		"next webhook event will trigger a full review.\n" +
		"- Or @mention the bot explicitly in a comment if you want a " +
		"one-off review on this branch — mentions are not muted.\n\n" +
		"**Configure**\n" +
		"- The keyword is set via `REVIEW_OPT_OUT_BRANCH_KEYWORD` on the " +
		"noergler service (empty string disables the feature).\n"
}

// AgentsMDMissingSummary is the skip summary when AGENTS.md is required and
// absent from both the PR branch and the target branch.
func AgentsMDMissingSummary() string {
	return "### Review skipped — no `AGENTS.md` found 🛑\n\n" +
		"This repository has no `AGENTS.md` on the PR branch or the target branch. " +
		"Project-specific review guidelines are **vital** for producing targeted, " +
		"high-signal feedback — without them the reviewer falls back to generic nits, " +
		"so the review was not run.\n\n" +
		"**What to do**\n" +
		"- Add an `AGENTS.md` file to the repository root describing project " +
		"conventions, forbidden patterns, and areas the reviewer should focus on.\n" +
		"- Push the file to the PR branch (or merge it into the target branch) and " +
		"the next webhook event on this PR will trigger a full review.\n\n" +
		"**Opt-out**\n" +
		"- To review PRs without an `AGENTS.md`, set `require_agents_md: false` " +
		"for this team in the noergler `teams.yaml` (or `REVIEW_REQUIRE_AGENTS_MD=false` " +
		"as the instance default) and restart.\n"
}

// DiffTooLargeSummary is the skip summary when the PR diff exceeds the byte
// cap. limit is in bytes.
func DiffTooLargeSummary(limit int) string {
	return "### Review skipped — diff too large 🛑\n\n" +
		fmt.Sprintf("The PR diff exceeds %d MiB, more than fits in one ", limit/(1024*1024)) +
		"review even after compression, so the review was not run.\n\n" +
		"**What to do**\n" +
		"- Split the change into smaller PRs, or move generated files, vendored " +
		"code and lockfile churn into their own PR.\n" +
		"- Push the smaller change and the next webhook event on this PR will " +
		"trigger a full review.\n"
}

// AgentsMDTooLargeSummary is the skip summary when AGENTS.md is over the hard
// token limit. customLink, when non-empty, is prepended to the further
// reading list.
func AgentsMDTooLargeSummary(tokens, limit int, customLink string) string {
	links := make([]Link, 0, len(agentsMDFurtherReading)+1)
	if l, ok := ParseCustomLink(customLink); ok {
		links = append(links, l)
	}
	links = append(links, agentsMDFurtherReading...)

	var fr strings.Builder
	for i, l := range links {
		if i > 0 {
			fr.WriteString("\n")
		}
		fmt.Fprintf(&fr, "- [%s](%s)", l.Title, l.URL)
	}

	return "### Review skipped — `AGENTS.md` too large 🛑\n\n" +
		fmt.Sprintf("`AGENTS.md` weighs in at ~%s tokens, exceeding the configured ", Fmt(tokens)) +
		fmt.Sprintf("hard limit of %s tokens. Oversized agent instructions crowd out ", Fmt(limit)) +
		"the actual diff, degrade review quality (context rot), and inflate " +
		"inference cost — so the review was not run.\n\n" +
		"**What to do**\n" +
		"- Trim `AGENTS.md`: drop sections that are there for humans to skim, " +
		"remove duplicated README content, replace prose with terse rules.\n" +
		"- Move detailed reference material into separate files and link to " +
		"them; keep the main file lean.\n" +
		"- Push the slimmer file and the next webhook event on this PR will " +
		"trigger a full review.\n\n" +
		"**Configure**\n" +
		"- Raise the limit via `REVIEW_AGENTS_MD_MAX_TOKENS` on the noergler " +
		"service, or set it to `0` to disable the hard cut-off entirely " +
		"(the soft warning via `REVIEW_AGENTS_MD_WARN_TOKENS` remains).\n\n" +
		"**Further reading**\n" +
		fr.String() + "\n"
}
