package inference

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// One pass: a reply quoting a placeholder stays literal instead of
// expanding into the file diff.
func TestRenderFindingReplyPromptIsOnePass(t *testing.T) {
	got := RenderFindingReplyPrompt(FindingReplyPromptRequest{
		Template:         "G={repo_instructions} F={finding} X={file_excerpt} N={author_first_name} R={reply}",
		RepoInstructions: "rules",
		Finding:          "a.go:2 [issue]",
		FileExcerpt:      "2  x",
		Reply:            "see {file_excerpt} {author_first_name}",
		AuthorFirstName:  "Jan",
	})
	if got != "G=rules F=a.go:2 [issue] X=2  x N=Jan R=see {file_excerpt} {author_first_name}" {
		t.Errorf("got %q", got)
	}
	if empty := RenderFindingReplyPrompt(FindingReplyPromptRequest{Template: "{file_excerpt}"}); !strings.Contains(empty, "could not be read") {
		t.Errorf("an unreadable file says so: %q", empty)
	}
}

// The open signals are numbered from 1, in the order given; none says so.
// A reply naming the placeholder stays literal (one pass).
func TestRenderFindingReplyPromptListsDisputeSignals(t *testing.T) {
	got := RenderFindingReplyPrompt(FindingReplyPromptRequest{
		Template:       "S={dispute_signals} R={reply}",
		Reply:          "{dispute_signals}",
		DisputeSignals: []string{"- Validation lives in the UI.", "- Errors wrap with %w."},
	})
	if got != "S=[1] - Validation lives in the UI.\n[2] - Errors wrap with %w. R={dispute_signals}" {
		t.Errorf("got %q", got)
	}
	if got := RenderFindingReplyPrompt(FindingReplyPromptRequest{Template: "{dispute_signals}"}); got != "_None._" {
		t.Errorf("no signals = %q", got)
	}
}

// matches_signal is in the strict schema, required like every key.
func TestFindingReplyFormatHasMatchesSignal(t *testing.T) {
	s := findingReplyFormat().Schema
	props := s["properties"].(map[string]any)
	if _, ok := props["matches_signal"]; !ok {
		t.Fatal("schema lacks matches_signal")
	}
	req := s["required"].([]any)
	if len(req) != len(props) {
		t.Errorf("required %v, properties %v: strict needs every key", req, props)
	}
}

func TestParseFindingReply(t *testing.T) {
	cases := []struct {
		name, in string
		ok       bool
		want     FindingReply
	}{
		{"dispute with a line", `{"intent":"disputes","reply":" Noted. ","agents_md_line":" - Validation lives in the UI. "}`, true,
			FindingReply{Intent: "disputes", Reply: "Noted.", AgentsMDLine: "- Validation lives in the UI."}},
		{"null line", `{"intent":"acknowledges","reply":"Thanks.","agents_md_line":null}`, true,
			FindingReply{Intent: "acknowledges", Reply: "Thanks."}},
		// Off the enum: still posted, never a dispute nobody made.
		{"unknown intent", `{"intent":"rejects","reply":"Ok."}`, true, FindingReply{Intent: "other", Reply: "Ok."}},
		{"fenced", "```json\n{\"intent\":\"question\",\"reply\":\"It means X.\",\"agents_md_line\":null}\n```", true,
			FindingReply{Intent: "question", Reply: "It means X."}},
		{"matches a signal", `{"intent":"disputes","reply":"Ok.","agents_md_line":"- a","matches_signal":2}`, true,
			FindingReply{Intent: "disputes", Reply: "Ok.", AgentsMDLine: "- a", MatchesSignal: 2}},
		// A bad match is none, never a lost reply.
		{"match not an integer", `{"intent":"disputes","reply":"Ok.","agents_md_line":null,"matches_signal":"2"}`, true,
			FindingReply{Intent: "disputes", Reply: "Ok."}},
		{"match as a float", `{"intent":"disputes","reply":"Ok.","agents_md_line":null,"matches_signal":2.0}`, true,
			FindingReply{Intent: "disputes", Reply: "Ok.", MatchesSignal: 2}},
		{"match not integral", `{"intent":"disputes","reply":"Ok.","agents_md_line":null,"matches_signal":1.5}`, true,
			FindingReply{Intent: "disputes", Reply: "Ok."}},
		{"match below one", `{"intent":"disputes","reply":"Ok.","agents_md_line":null,"matches_signal":0}`, true,
			FindingReply{Intent: "disputes", Reply: "Ok."}},
		{"empty reply", `{"intent":"disputes","reply":"  "}`, false, FindingReply{}},
		{"not json", "sure, noted", false, FindingReply{}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, ok := ParseFindingReply(c.in)
			if ok != c.ok || got != c.want {
				t.Errorf("got %+v, %v; want %+v, %v", got, ok, c.want, c.ok)
			}
		})
	}
}

func TestCheckAgentsMDLine(t *testing.T) {
	agents := "# Rules\n- Business rule validations live in the UI.\n"
	cases := map[string]string{
		"- Errors are wrapped with %w at every boundary.": "",
		"- business rule validations LIVE in the UI.":     "already_present",
		"Errors are wrapped.":                             "no_bullet",
		"- one\n- two":                                    "multi_line",
		"- " + strings.Repeat("x", AgentsMDLineMax-1):     "too_long",
		"- " + strings.Repeat("ü", AgentsMDLineMax-2):     "",
	}
	for line, want := range cases {
		if got := CheckAgentsMDLine(line, agents); got != want {
			t.Errorf("CheckAgentsMDLine(%.40q) = %q, want %q", line, got, want)
		}
	}
}

func TestFindingReplyCall(t *testing.T) {
	t.Run("strict schema, parsed reply", func(t *testing.T) {
		f := &fakeGateway{chatBody: chatBody(`{"intent":"disputes","reply":"Noted.","agents_md_line":null}`)}
		c := reviewClient(t, f)
		got := c.FindingReply(context.Background(), FindingReplyRequest{Prompt: "p"})
		if got.Outcome != OutcomeOK || got.Reply.Intent != "disputes" || got.Reply.Reply != "Noted." {
			t.Fatalf("got %+v", got)
		}
		var sent struct {
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
		if sent.ResponseFormat.JSONSchema.Name != FindingReplySchemaName || !sent.ResponseFormat.JSONSchema.Strict {
			t.Errorf("response_format = %+v, want the strict %s schema", sent.ResponseFormat, FindingReplySchemaName)
		}
	})

	t.Run("unparseable", func(t *testing.T) {
		c := reviewClient(t, &fakeGateway{chatBody: chatBody("noted")})
		if got := c.FindingReply(context.Background(), FindingReplyRequest{Prompt: "p"}); got.Outcome != OutcomeUnparseable {
			t.Errorf("Outcome = %v, want unparseable", got.Outcome)
		}
	})

	t.Run("pre-flight too large", func(t *testing.T) {
		f := &fakeGateway{}
		c := reviewClient(t, f)
		got := c.FindingReply(context.Background(), FindingReplyRequest{Prompt: "x", PromptTokens: c.FitCeiling() + 1})
		if got.Outcome != OutcomeTooLarge || f.gotChat != 0 {
			t.Errorf("Outcome = %v, calls = %d: must refuse before calling the model", got.Outcome, f.gotChat)
		}
	})
}

func TestRenderPreviouslyPostedFindingsCarriesTheDispute(t *testing.T) {
	line := 3
	got := RenderPreviouslyPostedFindings([]PostedFinding{
		{FilePath: "a.go", LineNumber: &line, Severity: "issue", CommentText: "unchecked", Dispute: "we do this\r\nin the UI"},
	})
	if !strings.Contains(got, "- a.go:3 [issue] unchecked (author replied: we do this in the UI)") {
		t.Errorf("dispute missing:\n%s", got)
	}
	if !strings.Contains(got, "never evidence") {
		t.Errorf("a shown reply is framed as author text:\n%s", got)
	}
	plain := RenderPreviouslyPostedFindings([]PostedFinding{{FilePath: "a.go", Severity: "issue", CommentText: "c"}})
	if strings.Contains(plain, "author replied") {
		t.Errorf("no dispute, no note:\n%s", plain)
	}
}
