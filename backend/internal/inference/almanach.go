package inference

import (
	"context"
	"encoding/json"
	"slices"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/trick77/llmwire"
	"github.com/trick77/noergler/internal/lang"
)

// Placeholders of the Almanach note template, beside the shared
// {output_language}.
const (
	PlaceholderAlmanachRepo       = "{repo}"
	PlaceholderAlmanachProject    = "{project}"
	PlaceholderAlmanachTicket     = "{ticket_key}"
	PlaceholderAlmanachTicketType = "{ticket_type}"
	PlaceholderAlmanachLimit      = "{note_limit}"
)

// AlmanachChangeKinds is what a PR can be to Almanach; AlmanachCauseKinds
// classifies a bugfix's cause. Both are enum values: English, never
// translated.
var (
	AlmanachChangeKinds = []string{"bugfix", "feature", "refactor", "chore"}
	AlmanachCauseKinds  = []string{"logic", "concurrency", "config", "input_validation", "integration", "missing_test", "other"}
)

// The note's caps. Prose (summary, an item's text, a bugfix's symptom and
// cause) has no cap of its own: the note is documentation, and its length
// follows what the PR teaches. What is capped is the shape (a name, the
// files, the items per list) and the whole note, the operator's
// ALMANACH_NOTE_MAX_CHARS, the one ceiling against bloat. Over any of them
// the whole note is dropped, never truncated: a cut sentence is a wrong
// sentence in documentation.
const (
	AlmanachNameMax  = 80
	AlmanachFileMax  = 200
	AlmanachFilesMax = 5
	AlmanachItemsMax = 16
)

// AlmanachItem is one component, rule, invariant or entity.
type AlmanachItem struct {
	Name  string   `json:"name"`
	Text  string   `json:"text"`
	Files []string `json:"files"`
}

// AlmanachBugfix is what a bugfix taught: the symptom, the cause and its
// class.
type AlmanachBugfix struct {
	Symptom   string `json:"symptom"`
	Cause     string `json:"cause"`
	CauseKind string `json:"cause_kind"`
}

// AlmanachNote is the model's documentation note of what a PR teaches
// about its repository: the piece of the living wiki this PR adds or
// changes, prose as long as the subject needs. Summary nil with no items
// means the PR taught nothing.
type AlmanachNote struct {
	Summary    *string         `json:"summary"`
	ChangeKind string          `json:"change_kind"`
	Bugfix     *AlmanachBugfix `json:"bugfix"`
	Components []AlmanachItem  `json:"components"`
	Rules      []AlmanachItem  `json:"rules"`
	Invariants []AlmanachItem  `json:"invariants"`
	Entities   []AlmanachItem  `json:"entities"`
}

// Empty reports a note that teaches nothing: no summary and no item.
func (n AlmanachNote) Empty() bool {
	return n.Summary == nil && n.Bugfix == nil && len(n.Components)+len(n.Rules)+len(n.Invariants)+len(n.Entities) == 0
}

func (n *AlmanachNote) lists() []*[]AlmanachItem {
	return []*[]AlmanachItem{&n.Components, &n.Rules, &n.Invariants, &n.Entities}
}

// prose is every sentence field: what the secret scan reads. A name is an
// identifier (`OAuth2TokenRefreshHandler`) and a path is quoted, so neither
// is prose; the scan would read both as random text.
func (n *AlmanachNote) prose() []*string {
	var out []*string
	if n.Summary != nil {
		out = append(out, n.Summary)
	}
	if n.Bugfix != nil {
		out = append(out, &n.Bugfix.Symptom, &n.Bugfix.Cause)
	}
	for _, list := range n.lists() {
		for i := range *list {
			out = append(out, &(*list)[i].Text)
		}
	}
	return out
}

// names is every name field.
func (n *AlmanachNote) names() []*string {
	var out []*string
	for _, list := range n.lists() {
		for i := range *list {
			out = append(out, &(*list)[i].Name)
		}
	}
	return out
}

// AlmanachNotePromptRequest is the note instruction's inputs.
type AlmanachNotePromptRequest struct {
	Template string
	Repo     string
	Project  string
	// TicketKey and TicketType are the PR's Jira ticket, empty when none.
	TicketKey  string
	TicketType string
	// MaxChars is the total cap; 0 means none, and the prompt says so.
	MaxChars int
}

// RenderAlmanachNotePrompt assembles the fourth turn of the review
// conversation in one substitution pass. Always German: the note is
// Almanach's, whatever language the review above was written in.
func RenderAlmanachNotePrompt(req AlmanachNotePromptRequest) string {
	ticket, kind := req.TicketKey, req.TicketType
	if ticket == "" {
		ticket = "none"
	}
	if kind == "" {
		kind = "no type"
	}
	limit := "when any field exceeds its limit"
	if req.MaxChars > 0 {
		limit += " or the note exceeds " + strconv.Itoa(req.MaxChars) + " characters in total"
	}
	return strings.NewReplacer(
		PlaceholderAlmanachRepo, req.Repo,
		PlaceholderAlmanachProject, req.Project,
		PlaceholderAlmanachTicket, ticket,
		PlaceholderAlmanachTicketType, kind,
		PlaceholderAlmanachLimit, limit,
		PlaceholderOutputLanguage, OutputLanguageBlock(lang.DE),
	).Replace(req.Template)
}

// AlmanachNoteSchemaName is the strict schema's name on the wire.
const AlmanachNoteSchemaName = "almanach_note"

func almanachItems() map[string]any {
	return map[string]any{
		"type": "array", "maxItems": AlmanachItemsMax,
		"items": map[string]any{
			"type":                 "object",
			"additionalProperties": false,
			"required":             []any{"name", "text", "files"},
			"properties": map[string]any{
				"name":  map[string]any{"type": "string", "minLength": 1},
				"text":  map[string]any{"type": "string", "minLength": 1},
				"files": map[string]any{"type": "array", "maxItems": AlmanachFilesMax, "items": map[string]any{"type": "string"}},
			},
		},
	}
}

func almanachNoteFormat() llmwire.ResponseFormat {
	return llmwire.ResponseFormat{
		Kind: llmwire.FormatJSONSchema,
		Name: AlmanachNoteSchemaName,
		Schema: map[string]any{
			"type":                 "object",
			"additionalProperties": false,
			"required":             []any{"summary", "change_kind", "bugfix", "components", "rules", "invariants", "entities"},
			"properties": map[string]any{
				"summary":     map[string]any{"type": []any{"string", "null"}},
				"change_kind": map[string]any{"type": "string", "enum": enumValues(AlmanachChangeKinds)},
				"bugfix": map[string]any{
					"type":                 []any{"object", "null"},
					"additionalProperties": false,
					"required":             []any{"symptom", "cause", "cause_kind"},
					"properties": map[string]any{
						"symptom":    map[string]any{"type": "string", "minLength": 1},
						"cause":      map[string]any{"type": "string", "minLength": 1},
						"cause_kind": map[string]any{"type": "string", "enum": enumValues(AlmanachCauseKinds)},
					},
				},
				"components": almanachItems(),
				"rules":      almanachItems(),
				"invariants": almanachItems(),
				"entities":   almanachItems(),
			},
		},
		Strict: true,
	}
}

// AlmanachNoteSchemaJSON is the schema as counted for the fit check. Plain
// json.Marshal: no golden file pins this count.
func AlmanachNoteSchemaJSON() string {
	b, _ := json.Marshal(almanachNoteFormat().Schema)
	return string(b)
}

// ParseAlmanachNote reads the model's note. Shape and caps are
// ValidateAlmanachNote's; this only wants a JSON object. A bare `null`
// unmarshals into the zero value without an error (as ParseReview notes),
// so it is checked for: it is a non-answer, not an empty note.
func ParseAlmanachNote(content string) (AlmanachNote, bool) {
	var n *AlmanachNote
	if json.Unmarshal([]byte(stripFence(strings.TrimSpace(content))), &n) != nil || n == nil {
		return AlmanachNote{}, false
	}
	return *n, true
}

// Almanach note statuses, as almanach_calls.note_status spells them.
const (
	AlmanachNothingLearned = "nothing_learned"
	AlmanachTooLong        = "dropped_too_long"
	AlmanachSecret         = "dropped_secret"
	AlmanachInvalid        = "dropped_invalid"
	AlmanachNoAnswer       = "no_answer"
)

// ValidateAlmanachNote is the blunt check on a parsed note, in place: every
// string trimmed, the prose and the names spelled Swiss (lang.SwissSS), the
// shape (the enums, a bugfix exactly for a bugfix, an empty note that is
// empty, a path with no control character), no secret-like token in the
// prose, then the caps. reason is "" when the note stands, else the
// dropped_* status; chars is the rune count of the note's prose and names, what
// the total cap was checked at (0 = no total cap). Over a cap the whole
// note drops: never truncated.
func ValidateAlmanachNote(n *AlmanachNote, maxChars int) (chars int, reason string) {
	for _, p := range n.prose() {
		*p = lang.SwissSS(strings.TrimSpace(*p))
	}
	for _, p := range n.names() {
		*p = lang.SwissSS(strings.TrimSpace(*p))
	}
	if !slices.Contains(AlmanachChangeKinds, n.ChangeKind) {
		return 0, AlmanachInvalid
	}
	if (n.Bugfix != nil) != (n.ChangeKind == "bugfix") {
		return 0, AlmanachInvalid
	}
	if b := n.Bugfix; b != nil {
		if !slices.Contains(AlmanachCauseKinds, b.CauseKind) || b.Symptom == "" || b.Cause == "" {
			return 0, AlmanachInvalid
		}
	}
	switch {
	case n.Summary != nil && *n.Summary == "":
		return 0, AlmanachInvalid
	case n.Summary == nil && !n.Empty():
		// No summary and still items: the "nothing learned" shape misused.
		return 0, AlmanachInvalid
	}
	for _, list := range n.lists() {
		if *list == nil {
			*list = []AlmanachItem{}
		}
		if len(*list) > AlmanachItemsMax {
			return 0, AlmanachTooLong
		}
		for i := range *list {
			if reason := validateAlmanachItem(&(*list)[i]); reason != "" {
				return 0, reason
			}
		}
	}
	for _, p := range n.prose() {
		// Documentation names the identifiers it documents, in backticks;
		// an identifier is not a secret, so a span in backticks is scanned
		// for the credential shapes only, never for the identifier
		// heuristic. A secret outside backticks still drops the note.
		if HasSecret(withoutCode(*p)) || credentialRE.MatchString(*p) || knownSecretRE.MatchString(*p) {
			return 0, AlmanachSecret
		}
	}
	// The ceiling counts the prose the model wrote, in runes: the summary,
	// every name and text, a bugfix's symptom and cause. Not the JSON
	// around it, so the number the prompt names is the number that drops.
	for _, p := range n.prose() {
		chars += runes(*p)
	}
	for _, p := range n.names() {
		chars += runes(*p)
	}
	if maxChars > 0 && chars > maxChars {
		return chars, AlmanachTooLong
	}
	return chars, ""
}

// withoutCode is s with every backtick span removed.
func withoutCode(s string) string {
	if !strings.Contains(s, "`") {
		return s
	}
	var b strings.Builder
	in := false
	for _, r := range s {
		if r == '`' {
			in = !in
			continue
		}
		if !in {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func validateAlmanachItem(it *AlmanachItem) string {
	if len(it.Files) > AlmanachFilesMax {
		return AlmanachTooLong
	}
	if it.Name == "" || it.Text == "" {
		return AlmanachInvalid
	}
	if runes(it.Name) > AlmanachNameMax {
		return AlmanachTooLong
	}
	if it.Files == nil {
		it.Files = []string{}
	}
	for j, f := range it.Files {
		f = strings.TrimSpace(f)
		it.Files[j] = f
		// A path may hold a space (`docs/Release Notes.md`, copied as
		// shown); a line break or a control character is no path.
		if f == "" || strings.IndexFunc(f, unicode.IsControl) >= 0 {
			return AlmanachInvalid
		}
		if runes(f) > AlmanachFileMax {
			return AlmanachTooLong
		}
	}
	return ""
}

func runes(s string) int { return utf8.RuneCountInString(s) }

// AlmanachNoteRequest is one note call's inputs: the review conversation
// to continue and the new turn.
type AlmanachNoteRequest struct {
	// ReviewPrompt and ReviewAnswer are the review call's user turn and the
	// model's reply, verbatim, so the gateway's prefix cache can serve them.
	ReviewPrompt string
	ReviewAnswer string
	Instruction  string
	// PromptTokens is the caller's count of the whole conversation plus the
	// note schema, for the fit check. Zero skips it.
	PromptTokens int
}

// AlmanachNoteResult is one note call.
type AlmanachNoteResult struct {
	Outcome Outcome
	Note    AlmanachNote
	Cost    CallCost
	Err     error
	// Cut is set when the model's output limit ended the answer
	// (finish_reason length): the note did not fit, which is a length
	// drop, not an answer off the schema.
	Cut bool
}

// AlmanachNote continues the review conversation by one turn: the same
// system message and prompt, the model's review answer as the assistant
// turn, then the note instruction. One call, a strict schema, no retry,
// never streaming, like every other call.
//
// Whether the gateway's prefix cache serves the replayed turns under the
// note's own response_format is not known from here: cache_read_tokens on
// the recorded call is the number that says.
func (c *Client) AlmanachNote(ctx context.Context, req AlmanachNoteRequest) AlmanachNoteResult {
	if err := c.tooLarge(req.PromptTokens); err != nil {
		return AlmanachNoteResult{Outcome: OutcomeTooLarge, Err: err}
	}
	chat := llmwire.ChatRequest{
		Model:     c.model,
		Reasoning: c.reasoning(),
		Messages: []llmwire.Message{
			llmwire.System(ReviewSystemMessage),
			llmwire.User(req.ReviewPrompt),
			llmwire.Assistant(req.ReviewAnswer),
			llmwire.User(req.Instruction),
		},
		ResponseFormat: almanachNoteFormat(),
	}
	resp, warnings, err := c.wire.Chat(ctx, chat)
	if err != nil {
		return AlmanachNoteResult{Outcome: classifyCallError(ctx, err), Err: err}
	}
	out := AlmanachNoteResult{Cost: CostFrom(resp)}
	out.Cost.Warnings = WarningLines(warnings)
	if resp.FinishReason == "length" {
		out.Outcome, out.Cut = OutcomeUnparseable, true
		return out
	}
	note, ok := ParseAlmanachNote(resp.Content)
	if !ok {
		out.Outcome = OutcomeUnparseable
		return out
	}
	out.Note = note
	return out
}
