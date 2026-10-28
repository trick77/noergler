package inference

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func str(s string) *string { return &s }

func goodNote() AlmanachNote {
	return AlmanachNote{
		Summary:    str("Der Datei-Adapter liest den Mandanten aus der Kopfzeile."),
		ChangeKind: "bugfix",
		Bugfix:     &AlmanachBugfix{Symptom: "Dateien landeten beim falschen Mandanten.", Cause: "Stiller Rückfall auf den Standardmandanten.", CauseKind: "input_validation"},
		Components: []AlmanachItem{{Name: "Datei-Adapter", Text: "Liest Sammeldateien ein.", Files: []string{"intake/file/reader.go"}}},
		Rules:      []AlmanachItem{},
		Invariants: []AlmanachItem{},
		Entities:   []AlmanachItem{},
	}
}

// One pass, German always, the ticket spelled out or "none".
func TestRenderAlmanachNotePromptIsOnePassAndGerman(t *testing.T) {
	got := RenderAlmanachNotePrompt(AlmanachNotePromptRequest{
		Template: "R={repo} P={project} T={ticket_key} K={ticket_type} M={note_limit} {output_language}",
		Repo:     "api", Project: "PLAT", TicketKey: "PLAT-7 {repo}", TicketType: "Bug", MaxChars: 6000,
	})
	if !strings.HasPrefix(got, "R=api P=PLAT T=PLAT-7 {repo} K=Bug M=when any field exceeds its limit or the note exceeds 6000 characters in total ") ||
		!strings.Contains(got, "Swiss Standard German") {
		t.Errorf("got %q", got)
	}
	none := RenderAlmanachNotePrompt(AlmanachNotePromptRequest{Template: "{ticket_key} ({ticket_type}) {note_limit}"})
	if none != "none (no type) when any field exceeds its limit" {
		t.Errorf("no ticket and no total cap renders %q", none)
	}
}

func TestValidateAlmanachNote(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*AlmanachNote)
		max    int
		want   string
	}{
		{"good", func(*AlmanachNote) {}, 0, ""},
		// Prose has no cap of its own: a long text or summary passes under
		// the total.
		{"long text passes", func(n *AlmanachNote) { n.Components[0].Text = strings.Repeat("ä", 3000) }, 0, ""},
		{"long summary passes", func(n *AlmanachNote) { n.Summary = str(strings.Repeat("x", 3000)) }, 0, ""},
		{"name over", func(n *AlmanachNote) { n.Components[0].Name = strings.Repeat("x", AlmanachNameMax+1) }, 0, AlmanachTooLong},
		// The total counts runes of the serialized note, so ä is one.
		{"total over by prose", func(n *AlmanachNote) { n.Components[0].Text = strings.Repeat("ä", 3000) }, 3000, AlmanachTooLong},
		{"too many items", func(n *AlmanachNote) {
			for i := 0; i <= AlmanachItemsMax; i++ {
				n.Rules = append(n.Rules, AlmanachItem{Name: "r", Text: "t"})
			}
		}, 0, AlmanachTooLong},
		{"too many files", func(n *AlmanachNote) { n.Components[0].Files = []string{"a", "b", "c", "d", "e", "f"} }, 0, AlmanachTooLong},
		{"total over", func(*AlmanachNote) {}, 40, AlmanachTooLong},
		{"unknown kind", func(n *AlmanachNote) { n.ChangeKind = "hotfix" }, 0, AlmanachInvalid},
		{"bugfix without the object", func(n *AlmanachNote) { n.Bugfix = nil }, 0, AlmanachInvalid},
		{"feature with a bugfix object", func(n *AlmanachNote) { n.ChangeKind = "feature" }, 0, AlmanachInvalid},
		{"unknown cause", func(n *AlmanachNote) { n.Bugfix.CauseKind = "cosmic" }, 0, AlmanachInvalid},
		{"empty name", func(n *AlmanachNote) { n.Components[0].Name = "  " }, 0, AlmanachInvalid},
		// A path copied as shown may hold a space; a line break is no path.
		{"path with a space", func(n *AlmanachNote) { n.Components[0].Files = []string{"docs/Release Notes.md"} }, 0, ""},
		{"path with a newline", func(n *AlmanachNote) { n.Components[0].Files = []string{"a.go\nb.go"} }, 0, AlmanachInvalid},
		{"no summary but items", func(n *AlmanachNote) { n.Summary = nil; n.Bugfix = nil; n.ChangeKind = "chore" }, 0, AlmanachInvalid},
		{"secret in prose", func(n *AlmanachNote) { n.Components[0].Text = "Token aB3dE5fG7hI9jK1lM3nO5pQ7rS9tU gesetzt." }, 0, AlmanachSecret},
		{"secret in the summary", func(n *AlmanachNote) { n.Summary = str("Setzt aB3dE5fG7hI9jK1lM3nO5pQ7rS9tU als Default.") }, 0, AlmanachSecret},
		// Identifiers and paths are not prose: the class-shift heuristic
		// reads a CamelCase name with a digit as random text.
		{"camel-case name is not prose", func(n *AlmanachNote) { n.Components[0].Name = "OAuth2TokenRefreshHandler9Service" }, 0, ""},
		{"hash-like path is not prose", func(n *AlmanachNote) {
			n.Components[0].Files = []string{"vendor/a1B2c3D4e5F6g7H8i9J0k1L2m3N4/x.go"}
		}, 0, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			n := goodNote()
			c.mutate(&n)
			chars, got := ValidateAlmanachNote(&n, c.max)
			if got != c.want {
				t.Errorf("reason = %q, want %q", got, c.want)
			}
			if c.want == "" && chars == 0 {
				t.Error("a standing note reports its size")
			}
		})
	}

	// Nothing learned: a legal shape of its own, and Empty says so.
	empty := AlmanachNote{ChangeKind: "chore"}
	if _, reason := ValidateAlmanachNote(&empty, 0); reason != "" || !empty.Empty() {
		t.Errorf("empty note: reason %q empty %v", reason, empty.Empty())
	}
	// Validation trims and normalises nil lists, so the stored JSON is stable.
	n := goodNote()
	n.Rules = nil
	n.Components[0].Name = "  Datei-Adapter  "
	n.Components[0].Text = "Liest Dateien, Grösse egal; `straße` bleibt Code."
	n.Summary = str("Der Adapter misst die Größe.")
	if _, reason := ValidateAlmanachNote(&n, 0); reason != "" || n.Rules == nil || n.Components[0].Name != "Datei-Adapter" {
		t.Errorf("normalisation: reason %q rules %v name %q", reason, n.Rules, n.Components[0].Name)
	}
	if *n.Summary != "Der Adapter misst die Grösse." || n.Components[0].Text != "Liest Dateien, Grösse egal; `straße` bleibt Code." {
		t.Errorf("Swiss spelling on prose, code spans kept: %q / %q", *n.Summary, n.Components[0].Text)
	}
}

func TestParseAlmanachNote(t *testing.T) {
	raw, _ := json.Marshal(goodNote())
	if n, ok := ParseAlmanachNote("```json\n" + string(raw) + "\n```"); !ok || n.ChangeKind != "bugfix" || n.Bugfix == nil {
		t.Errorf("fenced note: %+v %v", n, ok)
	}
	if _, ok := ParseAlmanachNote("no note"); ok {
		t.Error("prose parsed as a note")
	}
	if _, ok := ParseAlmanachNote("null"); ok {
		t.Error("a bare null parsed as a note: it is a non-answer")
	}
}

// The note call continues the review conversation: the same system message
// and prompt, the review answer as the assistant turn, the instruction last,
// under the strict note schema.
func TestAlmanachNoteCallContinuesTheConversation(t *testing.T) {
	raw, _ := json.Marshal(goodNote())
	f := &fakeGateway{chatBody: chatBody(string(raw))}
	c := reviewClient(t, f)
	got := c.AlmanachNote(context.Background(), AlmanachNoteRequest{
		ReviewPrompt: "the review prompt", ReviewAnswer: `{"overview":"x"}`, Instruction: "write the note",
	})
	if got.Outcome != OutcomeOK || got.Note.ChangeKind != "bugfix" || !got.Cost.Answered {
		t.Fatalf("got %+v", got)
	}
	var sent struct {
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
		ResponseFormat struct {
			JSONSchema struct {
				Name   string `json:"name"`
				Strict bool   `json:"strict"`
			} `json:"json_schema"`
		} `json:"response_format"`
	}
	if err := json.Unmarshal(f.lastChatBody, &sent); err != nil {
		t.Fatal(err)
	}
	roles := make([]string, len(sent.Messages))
	for i, m := range sent.Messages {
		roles[i] = m.Role
	}
	if strings.Join(roles, ",") != "system,user,assistant,user" {
		t.Errorf("roles = %v, want system,user,assistant,user", roles)
	}
	if sent.Messages[0].Content != ReviewSystemMessage || sent.Messages[1].Content != "the review prompt" ||
		sent.Messages[2].Content != `{"overview":"x"}` || sent.Messages[3].Content != "write the note" {
		t.Errorf("messages = %+v", sent.Messages)
	}
	if sent.ResponseFormat.JSONSchema.Name != AlmanachNoteSchemaName || !sent.ResponseFormat.JSONSchema.Strict {
		t.Errorf("response_format = %+v, want the strict %s schema", sent.ResponseFormat, AlmanachNoteSchemaName)
	}

	t.Run("unparseable", func(t *testing.T) {
		c := reviewClient(t, &fakeGateway{chatBody: chatBody("sure")})
		if got := c.AlmanachNote(context.Background(), AlmanachNoteRequest{Instruction: "i"}); got.Outcome != OutcomeUnparseable {
			t.Errorf("Outcome = %v, want unparseable", got.Outcome)
		}
	})
	t.Run("pre-flight too large", func(t *testing.T) {
		f := &fakeGateway{}
		c := reviewClient(t, f)
		got := c.AlmanachNote(context.Background(), AlmanachNoteRequest{Instruction: "i", PromptTokens: c.FitCeiling() + 1})
		if got.Outcome != OutcomeTooLarge || f.gotChat != 0 {
			t.Errorf("Outcome = %v, calls = %d: must refuse before calling the model", got.Outcome, f.gotChat)
		}
	})
}

// The review's raw answer survives on the result, verbatim: it is the
// assistant turn the note call replays.
func TestReviewResultKeepsTheRawAnswer(t *testing.T) {
	body := `{"overview":"o","strengths":[],"security_performance":"None notable.","test_coverage":"n","verdict":{"decision":"approve","rationale":"r"},"findings":[],"fixed":[],"pre_existing":[],"compliance_requirements":null}`
	c := reviewClient(t, &fakeGateway{chatBody: chatBody(body)})
	got := c.Review(context.Background(), ReviewRequest{Prompt: "p"})
	if got.Raw != body {
		t.Errorf("Raw = %q, want the model's content verbatim", got.Raw)
	}
}
