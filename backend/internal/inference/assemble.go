package inference

import (
	"strings"

	"github.com/trick77/noergler/internal/diff"
)

// NoTicketContext is what {ticket_context} becomes when no ticket is linked.
//
// The mention template's default differs by one word ("available" against
// "provided"). Both are reproduced rather than unified: they are model-facing
// text pinned to the Python.
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
// whether the pre-flight or the endpoint said so (llm_client.py:1183).
const MentionTooLargeReply = "This PR is too large to answer within the model's context window."

// MentionEmptyReply is what an empty model answer becomes, so a mention is
// never silently unanswered (llm_client.py:1200).
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
// Python does this inside review_diff (llm_client.py:1027); in Go the client
// takes an already-assembled prompt, so the layer lives here.
//
// Every placeholder is substituted in ONE pass. Python substitutes
// sequentially with {files} last, which protects file content from being
// rescanned but leaves an earlier block, a ticket description or an AGENTS.md
// containing "{files}", expanded into the real file group. One pass protects
// every block equally. That is stricter than Python on purpose: the hole is
// not worth reproducing, and PromptInjectionLiteral pins it.
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
	// three are billed as input (llm_client.py:1081).
	total := count(ReviewSystemMessage) + count(prompt) + count(SchemaJSON())

	return AssembledPrompt{Prompt: prompt, Breakdown: breakdown, PromptTokens: total}
}
