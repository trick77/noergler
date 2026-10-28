package inference

import (
	"context"
	"encoding/json"
	"slices"
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
	PlaceholderAlmanachMaxChars   = "{note_max_chars}"
)

// AlmanachChangeKinds is what a PR can be to Almanach; AlmanachCauseKinds
// classifies a bugfix's cause. Both are enum values: English, never
// translated.
var (
	AlmanachChangeKinds = []string{"bugfix", "feature", "refactor", "chore"}
	AlmanachCauseKinds  = []string{"logic", "concurrency", "config", "input_validation", "integration", "missing_test", "other"}
)

// The note's caps, in runes. Over any of them the whole note is dropped,
// never truncated: a cut sentence is a wrong sentence in a knowledge base.
// The total is the operator's (ALMANACH_NOTE_MAX_CHARS).
const (
	AlmanachSummaryMax = 600
	AlmanachNameMax    = 80
	AlmanachTextMax    = 300
	AlmanachFileMax    = 200
	AlmanachFilesMax   = 5
	AlmanachItemsMax   = 8
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

// AlmanachNote is the model's scratch note of what a PR teaches about its
// repository. Summary nil with no items means the PR taught nothing.
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

// Prose is every human-readable field, for the caller's spelling pass and
// the secret scan. Files are not prose: a path is quoted, never written.
func (n *AlmanachNote) Prose() []*string {
	var out []*string
	if n.Summary != nil {
		out = append(out, n.Summary)
	}
	if n.Bugfix != nil {
		out = append(out, &n.Bugfix.Symptom, &n.Bugfix.Cause)
	}
	for _, list := range [][]AlmanachItem{n.Components, n.Rules, n.Invariants, n.Entities} {
		for i := range list {
			out = append(out, &list[i].Name, &list[i].Text)
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
	MaxChars   int
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
	return strings.NewReplacer(
		PlaceholderAlmanachRepo, req.Repo,
		PlaceholderAlmanachProject, req.Project,
		PlaceholderAlmanachTicket, ticket,
		PlaceholderAlmanachTicketType, kind,
		PlaceholderAlmanachMaxChars, itoa(req.MaxChars),
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
	kinds := make([]any, len(AlmanachChangeKinds))
	for i, k := range AlmanachChangeKinds {
		kinds[i] = k
	}
	causes := make([]any, len(AlmanachCauseKinds))
	for i, k := range AlmanachCauseKinds {
		causes[i] = k
	}
	return llmwire.ResponseFormat{
		Kind: llmwire.FormatJSONSchema,
		Name: AlmanachNoteSchemaName,
		Schema: map[string]any{
			"type":                 "object",
			"additionalProperties": false,
			"required":             []any{"summary", "change_kind", "bugfix", "components", "rules", "invariants", "entities"},
			"properties": map[string]any{
				"summary":     map[string]any{"type": []any{"string", "null"}},
				"change_kind": map[string]any{"type": "string", "enum": kinds},
				"bugfix": map[string]any{
					"type":                 []any{"object", "null"},
					"additionalProperties": false,
					"required":             []any{"symptom", "cause", "cause_kind"},
					"properties": map[string]any{
						"symptom":    map[string]any{"type": "string", "minLength": 1},
						"cause":      map[string]any{"type": "string", "minLength": 1},
						"cause_kind": map[string]any{"type": "string", "enum": causes},
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
// ValidateAlmanachNote's; this only wants JSON of the right keys.
func ParseAlmanachNote(content string) (AlmanachNote, bool) {
	var n AlmanachNote
	if json.Unmarshal([]byte(stripFence(strings.TrimSpace(content))), &n) != nil {
		return AlmanachNote{}, false
	}
	return n, true
}

// Almanach note statuses, as almanach_calls.note_status spells them.
const (
	AlmanachWritten        = "written"
	AlmanachSuperseded     = "superseded"
	AlmanachNothingLearned = "nothing_learned"
	AlmanachTooLong        = "dropped_too_long"
	AlmanachSecret         = "dropped_secret"
	AlmanachInvalid        = "dropped_invalid"
	AlmanachNoAnswer       = "no_answer"
)

// ValidateAlmanachNote is the blunt check on a parsed note: shape (the
// enums, a bugfix exactly for a bugfix, an empty note that is empty), no
// secret-like token in its prose, then the caps. It trims every string
// first. reason is "" when the note stands, else the dropped_* status;
// chars is the rune count of the serialized note, what the total cap was
// checked at. Over a cap the whole note drops: never truncated.
func ValidateAlmanachNote(n *AlmanachNote, maxChars int) (chars int, reason string) {
	for _, p := range n.Prose() {
		*p = strings.TrimSpace(*p)
	}
	if !slices.Contains(AlmanachChangeKinds, n.ChangeKind) {
		return 0, AlmanachInvalid
	}
	if (n.Bugfix != nil) != (n.ChangeKind == "bugfix") {
		return 0, AlmanachInvalid
	}
	if n.Bugfix != nil {
		if !slices.Contains(AlmanachCauseKinds, n.Bugfix.CauseKind) || n.Bugfix.Symptom == "" || n.Bugfix.Cause == "" {
			return 0, AlmanachInvalid
		}
		if runes(n.Bugfix.Symptom) > AlmanachTextMax || runes(n.Bugfix.Cause) > AlmanachTextMax {
			return 0, AlmanachTooLong
		}
	}
	if n.Summary != nil {
		if *n.Summary == "" {
			return 0, AlmanachInvalid
		}
		if runes(*n.Summary) > AlmanachSummaryMax {
			return 0, AlmanachTooLong
		}
	} else if !n.Empty() {
		// No summary and still items: the "nothing learned" shape misused.
		return 0, AlmanachInvalid
	}
	for _, list := range [][]AlmanachItem{n.Components, n.Rules, n.Invariants, n.Entities} {
		if len(list) > AlmanachItemsMax {
			return 0, AlmanachTooLong
		}
		for i := range list {
			it := &list[i]
			if it.Name == "" || it.Text == "" || len(it.Files) > AlmanachFilesMax {
				if len(it.Files) > AlmanachFilesMax {
					return 0, AlmanachTooLong
				}
				return 0, AlmanachInvalid
			}
			if runes(it.Name) > AlmanachNameMax || runes(it.Text) > AlmanachTextMax {
				return 0, AlmanachTooLong
			}
			if it.Files == nil {
				it.Files = []string{}
			}
			for j, f := range it.Files {
				f = strings.TrimSpace(f)
				it.Files[j] = f
				if f == "" || strings.IndexFunc(f, unicode.IsSpace) >= 0 {
					return 0, AlmanachInvalid
				}
				if runes(f) > AlmanachFileMax {
					return 0, AlmanachTooLong
				}
			}
		}
	}
	for _, p := range n.Prose() {
		if HasSecret(*p) {
			return 0, AlmanachSecret
		}
	}
	for _, list := range []*[]AlmanachItem{&n.Components, &n.Rules, &n.Invariants, &n.Entities} {
		if *list == nil {
			*list = []AlmanachItem{}
		}
	}
	b, err := json.Marshal(n)
	if err != nil {
		return 0, AlmanachInvalid
	}
	chars = utf8.RuneCount(b)
	if maxChars > 0 && chars > maxChars {
		return chars, AlmanachTooLong
	}
	return chars, ""
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
}

// AlmanachNote continues the review conversation by one turn: the same
// system message and prompt, the model's review answer as the assistant
// turn, then the note instruction. One call, a strict schema, no retry,
// never streaming, like every other call.
func (c *Client) AlmanachNote(ctx context.Context, req AlmanachNoteRequest) AlmanachNoteResult {
	if req.PromptTokens > 0 && c.Ready() {
		if ceiling := c.FitCeiling(); req.PromptTokens > ceiling {
			return AlmanachNoteResult{Outcome: OutcomeTooLarge, Err: &TooLargeError{PromptTokens: req.PromptTokens, Ceiling: ceiling}}
		}
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
	note, ok := ParseAlmanachNote(resp.Content)
	if !ok {
		out.Outcome = OutcomeUnparseable
		return out
	}
	out.Note = note
	return out
}
