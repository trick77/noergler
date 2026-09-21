package inference

import (
	"strings"

	"github.com/trick77/noergler/internal/diff"
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

// Mention prompt placeholders. The file group goes into {diff} here, not
// {files} as in the review template.
const (
	PlaceholderQuestion = "{question}"
	PlaceholderDiff     = "{diff}"
)

// MentionTooLargeReply is what a mention gets when the PR does not fit,
// whether the pre-flight or the endpoint said so.
const MentionTooLargeReply = "This PR is too large to answer within the model's context window."

// MentionEmptyReply is what an empty model answer becomes, so a mention is
// never silently unanswered.
const MentionEmptyReply = "I couldn't process this PR to answer your question."

// MentionPromptRequest is one mention prompt's inputs.
type MentionPromptRequest struct {
	Template         string
	Question         string
	Files            []diff.FileReviewData
	RepoInstructions string
	TicketContext    string
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
		PlaceholderDiff, RenderFileGroup(req.Files),
	).Replace(req.Template)
}

// PromptBreakdown is the per-section input token count the summary footnote
// renders. RepoInstructions doubles as the AGENTS.md figure on the scope line.
type PromptBreakdown struct {
	Template         int
	RepoInstructions int
	Files            int
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
	CrossFileContext      string
	CumulativePRDiff      string
	PreviouslyPosted      []PostedFinding
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
// equally, and PromptInjectionLiteral pins it.
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

	prompt := strings.NewReplacer(
		PlaceholderFiles, files,
		PlaceholderCumulativePRDiff, RenderCumulativePRDiff(req.CumulativePRDiff),
		PlaceholderPreviouslyPosted, RenderPreviouslyPostedFindings(req.PreviouslyPosted),
		PlaceholderRepoInstructions, req.RepoInstructions,
		PlaceholderTicketContext, ticketContext,
		PlaceholderComplianceInstructions, compliance,
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

	// Count everything the request actually carries: the system message, the
	// user prompt and the strict JSON schema bound via response_format. All
	// three are billed as input.
	total := count(ReviewSystemMessage) + count(prompt) + count(SchemaJSON())

	return AssembledPrompt{Prompt: prompt, Breakdown: breakdown, PromptTokens: total}
}
