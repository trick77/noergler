package inference

import (
	"strings"

	"github.com/trick77/noergler/internal/diff"
	"github.com/trick77/noergler/internal/lang"
)

// NoTicketContext is what {ticket_context} becomes when no ticket is linked.
//
// The mention template's default differs by one word ("available" against
// "provided"). The two are kept distinct rather than unified: both are
// model-facing text, pinned byte for byte.
const (
	NoTicketContext        = "No ticket context provided."
	NoTicketContextMention = "No ticket context available."
)

// NoPRDescription is what {pr_description} becomes when the PR has no
// description or the team switched it off. One text for both templates.
const NoPRDescription = "No PR description provided."

// Mention prompt placeholders. The file group goes into {diff} here, not
// {files} as in the review template.
const (
	PlaceholderQuestion = "{question}"
	PlaceholderDiff     = "{diff}"
)

// MentionTooLargeReply is what a mention gets when the PR does not fit,
// whether the pre-flight or the endpoint said so.
func MentionTooLargeReply(l lang.Lang) string {
	return l.Pick("Dieser PR ist zu gross, um die Frage im Kontextfenster des Modells zu beantworten.",
		"This PR is too large to answer within the model's context window.")
}

// MentionEmptyReply is what an empty model answer becomes, so a mention is
// never silently unanswered.
func MentionEmptyReply(l lang.Lang) string {
	return l.Pick("Ich konnte diesen PR nicht verarbeiten, um deine Frage zu beantworten.",
		"I couldn't process this PR to answer your question.")
}

// MentionPromptRequest is one mention prompt's inputs.
type MentionPromptRequest struct {
	Template         string
	Question         string
	Files            []diff.FileReviewData
	RepoInstructions string
	TicketContext    string
	PRDescription    string
	// Language is the asker's; the zero value writes German.
	Language lang.Lang
}

// RenderMentionPrompt assembles the Q&A prompt.
//
// One substitution pass, for the same reason AssembleReviewPrompt uses one:
// a question or a ticket description containing "{diff}" must stay literal
// rather than expanding into the file group.
func RenderMentionPrompt(req MentionPromptRequest) string {
	ticketContext := req.TicketContext
	if ticketContext == "" {
		ticketContext = NoTicketContextMention
	}
	return strings.NewReplacer(
		PlaceholderQuestion, req.Question,
		PlaceholderRepoInstructions, req.RepoInstructions,
		PlaceholderTicketContext, ticketContext,
		PlaceholderPRDescription, orNoPRDescription(req.PRDescription),
		PlaceholderDiff, RenderFileGroup(req.Files),
		PlaceholderOutputLanguage, OutputLanguageBlock(req.Language),
	).Replace(req.Template)
}

// PromptBreakdown is the per-section input token count the summary footnote
// renders. RepoInstructions doubles as the AGENTS.md figure on the scope line.
type PromptBreakdown struct {
	Template         int
	RepoInstructions int
	Files            int
	// The three below are log-only: the footnote does not render them, the
	// prompt-composition line does. Without them a run's input cost could
	// not be attributed past the files.
	CumulativeDiff   int
	PreviouslyPosted int
	Ticket           int
	PRDescription    int
}

// AssembleRequest is everything the review prompt is built from.
type AssembleRequest struct {
	Template         string
	Files            []diff.FileReviewData
	RepoInstructions string

	OtherModifiedPaths []string
	DeletedFilePaths   []string
	RenamedFilePaths   []string

	TicketContext         string
	TicketComplianceCheck bool
	// PRDescription is the rendered title and description; empty renders
	// NoPRDescription.
	PRDescription    string
	CrossFileContext string
	CumulativePRDiff string
	// CumulativePartial: part of the PR is missing from CumulativePRDiff.
	CumulativePartial bool
	// CumulativeOmitted names PR files left out of CumulativePRDiff for
	// budget.
	CumulativeOmitted []string
	PreviouslyPosted  []PostedFinding
	// Language is the PR author's; the zero value writes German. Evals pin
	// English: their scoring matches English keywords.
	Language lang.Lang
}

// AssembledPrompt is the rendered prompt plus what the caller needs for the
// fit check and the summary footnote.
type AssembledPrompt struct {
	Prompt    string
	Breakdown PromptBreakdown
	// PromptTokens is system message + prompt + schema, which is what the
	// gateway bills as input and therefore what the fit check must weigh.
	PromptTokens int
}

// CountFunc counts tokens in a string.
type CountFunc func(string) int

// AssembleReviewPrompt renders the review prompt and counts it.
//
// The client takes an already-assembled prompt, so the assembly layer lives
// here.
//
// Every placeholder is substituted in ONE pass. Substituting sequentially
// with {files} last would protect file content from being rescanned but would
// leave an earlier block, a ticket description or an AGENTS.md containing
// "{files}", expanded into the real file group. One pass protects every block
// equally; TestPlaceholderInABlockStaysLiteral and injection_test.go pin it.
// Never text/template: file content contains JSON braces.
func AssembleReviewPrompt(req AssembleRequest, count CountFunc) AssembledPrompt {
	ticketContext := req.TicketContext
	if ticketContext == "" {
		ticketContext = NoTicketContext
	}

	compliance := ""
	if req.TicketComplianceCheck && req.TicketContext != "" {
		compliance = ComplianceInstructions
	}

	files := RenderFileGroup(req.Files)
	supplementary := RenderSupplementaryContext(req.OtherModifiedPaths, req.DeletedFilePaths, req.RenamedFilePaths)
	if req.CrossFileContext != "" {
		supplementary = strings.TrimSpace(supplementary + "\n\n" + req.CrossFileContext)
	}
	if supplementary != "" {
		files = files + "\n\n" + supplementary
	}

	cumulative := RenderCumulativePRDiff(req.CumulativePRDiff, req.CumulativePartial, req.CumulativeOmitted)
	posted := RenderPreviouslyPostedFindings(req.PreviouslyPosted)

	prompt := strings.NewReplacer(
		PlaceholderFiles, files,
		PlaceholderCumulativePRDiff, cumulative,
		PlaceholderPreviouslyPosted, posted,
		PlaceholderRepoInstructions, req.RepoInstructions,
		PlaceholderTicketContext, ticketContext,
		PlaceholderComplianceInstructions, compliance,
		PlaceholderPRDescription, orNoPRDescription(req.PRDescription),
		PlaceholderOutputLanguage, OutputLanguageBlock(req.Language),
	).Replace(req.Template)

	breakdown := PromptBreakdown{
		Template: count(strings.NewReplacer(
			PlaceholderFiles, "",
			PlaceholderRepoInstructions, "",
		).Replace(req.Template)),
		Files: 0,
	}
	if req.RepoInstructions != "" {
		breakdown.RepoInstructions = count(req.RepoInstructions)
	}
	for _, f := range req.Files {
		breakdown.Files += count(FormatFileEntry(f))
	}
	if cumulative != "" {
		breakdown.CumulativeDiff = count(cumulative)
	}
	if posted != "" {
		breakdown.PreviouslyPosted = count(posted)
	}
	if req.TicketContext != "" {
		breakdown.Ticket = count(req.TicketContext)
	}
	if req.PRDescription != "" {
		breakdown.PRDescription = count(req.PRDescription)
	}

	// Count everything the request actually carries: the system message, the
	// user prompt and the strict JSON schema bound via response_format. All
	// three are billed as input.
	total := count(ReviewSystemMessage) + count(prompt) + count(SchemaJSON())

	return AssembledPrompt{Prompt: prompt, Breakdown: breakdown, PromptTokens: total}
}

func orNoPRDescription(s string) string {
	if s == "" {
		return NoPRDescription
	}
	return s
}
