package review

import (
	"context"
	"slices"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
)

// One reader, three outcomes: the sides, a nudge for the letters, or the
// Q&A's to read.
func TestReadPickReply(t *testing.T) {
	for _, c := range []struct {
		text string
		n    int
		kind pickRead
		want []bool
	}{
		// The sides.
		{"T", 1, pickSides, []bool{true}},
		{"s", 1, pickSides, []bool{false}},
		{"`S`", 1, pickSides, []bool{false}},
		{"**T** bitte", 1, pickSides, []bool{true}},
		{"nimm S", 1, pickSides, []bool{false}},
		{"T S", 2, pickSides, []bool{true, false}},
		{"1T 2S", 2, pickSides, []bool{true, false}},
		{"2: S, 1: T", 2, pickSides, []bool{true, false}},
		{"1 s 2 t", 2, pickSides, []bool{false, true}},
		{"Nimm 1: T, 2: S, 3: T, 4: S, 5: T bitte", 5, pickSides, []bool{true, false, true, false, true}},
		{"T.", 1, pickSides, []bool{true}},
		{"1. S", 1, pickSides, []bool{false}},
		// Reads like sides, settles none: asked again.
		{"Lets go!", 1, pickNudge, nil},
		{"Let's go!", 1, pickNudge, nil},
		{"Let’s go!", 1, pickNudge, nil},
		{"mach's", 1, pickNudge, nil},
		{"ja", 1, pickNudge, nil},
		{"ok bitte", 1, pickNudge, nil},
		{"T", 2, pickNudge, nil},
		{"T S", 1, pickNudge, nil},
		{"1T S", 2, pickNudge, nil},
		{"1T 1S", 2, pickNudge, nil},
		{"3T 1S", 2, pickNudge, nil},
		// The Q&A's: a version, a no, a put-off, a question, a command,
		// words beside the letters.
		{"1.0.31", 1, pickOther, nil},
		{"31", 1, pickOther, nil},
		{"nimm 1.0.31", 1, pickOther, nil},
		{"T (1.0.31)", 1, pickOther, nil},
		{"S, die 1.0.29", 1, pickOther, nil},
		{"Das passt schon, du kannst 1.0.31 verwenden", 1, pickOther, nil},
		{"nicht T", 1, pickOther, nil},
		{"T, mach ich später", 1, pickOther, nil},
		{"T oder S?", 1, pickOther, nil},
		{"review", 1, pickOther, nil},
		{"explain the difference", 1, pickOther, nil},
		{"it's T", 2, pickOther, nil},
		{"that’s T", 1, pickOther, nil},
		{"👍", 1, pickOther, nil},
		{"", 1, pickOther, nil},
		{"T", 0, pickOther, nil},
	} {
		kind, got := readPickReply(c.text, c.n)
		if kind != c.kind || !slices.Equal(got, c.want) {
			t.Errorf("readPickReply(%q, %d) = %d %v, want %d %v", c.text, c.n, kind, got, c.kind, c.want)
		}
	}
}

// Words code does not read go to the Q&A, which has pick_conflict_side
// there; the model's sides settle the merge, its sentence is not posted.
func TestConflictPickByTheModel(t *testing.T) {
	repo := pickRepo(t)
	h := conflictHarness(t, repo)
	question := askPick(t, h)
	var tool string
	h.llm.onMention = func(_ context.Context, tools []string, run func(name, args string) string) string {
		if !slices.Contains(tools, "pick_conflict_side") || slices.Contains(tools, "resolve_merge_conflict") {
			t.Errorf("tools = %v", tools)
		}
		tool = run("pick_conflict_side", `{"sides":["T"]}`)
		return "MODEL SENTENCE"
	}

	answerPick(h, question, "alice", "Das passt schon, du kannst 1.0.31 verwenden")

	if !strings.Contains(tool, `"ok"`) {
		t.Fatalf("tool = %s", tool)
	}
	if got := repo.show("feature/thing:build.gradle"); !strings.Contains(got, "client:1.0.31") {
		t.Fatalf("build.gradle = %q", got)
	}
	for _, r := range h.bb.Replies {
		if strings.Contains(r.Text, "MODEL SENTENCE") {
			t.Fatal("the model's sentence was posted")
		}
	}
	if len(h.llm.Conflicts) != 1 {
		t.Errorf("%d conflict calls, want the request's only", len(h.llm.Conflicts))
	}
}

// The model's sides on a reply that puts it off, or of the wrong count:
// code's veto stands, nothing pushed, the answer asks for the letters.
func TestConflictPickByTheModelRefused(t *testing.T) {
	for name, c := range map[string]struct{ text, args, want string }{
		"put off":     {"mach ich später selbst", `{"sides":["T"]}`, "rejected: the user's reply"},
		"wrong count": {"die neuere bitte", `{"sides":["T","S"]}`, "asks about 1 spots, 2 sides"},
		"bad side":    {"die neuere bitte", `{"sides":["B"]}`, "neither T nor S"},
	} {
		t.Run(name, func(t *testing.T) {
			repo := pickRepo(t)
			h := conflictHarness(t, repo)
			question := askPick(t, h)
			before := repo.run(repo.bare, "rev-parse", "feature/thing")
			var tool string
			h.llm.onMention = func(_ context.Context, _ []string, run func(name, args string) string) string {
				tool = run("pick_conflict_side", c.args)
				return "Alles klar."
			}

			answerPick(h, question, "alice", c.text)

			if !strings.Contains(tool, c.want) {
				t.Fatalf("tool = %s", tool)
			}
			if got := repo.run(repo.bare, "rev-parse", "feature/thing"); got != before {
				t.Fatal("pushed despite the refusal")
			}
			last := h.bb.Replies[len(h.bb.Replies)-1].Text
			if !strings.Contains(last, "Alles klar.") || !strings.Contains(last, "Reply **T** or **S** here.") {
				t.Errorf("a refused pick does not say what settles it:\n%s", last)
			}
		})
	}
}

// The model answers without calling the tool, maybe promising the merge:
// the author's answer still ends with what settles it; anyone else's not.
func TestConflictPickNoToolCall(t *testing.T) {
	for author, want := range map[string]bool{"alice": true, "bob": false} {
		t.Run(author, func(t *testing.T) {
			h := conflictHarness(t, pickRepo(t))
			question := askPick(t, h)
			h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "Mache ich, ich nehme 1.0.31."}

			answerPick(h, question, author, "nimm die neuere")

			last := h.bb.Replies[len(h.bb.Replies)-1].Text
			if got := strings.Contains(last, "Reply **T** or **S** here."); got != want {
				t.Errorf("hint %t, want %t:\n%s", got, want, last)
			}
		})
	}
}

// A question the model would see cut: no tool, it never picks a spot it
// did not see; the hint stays.
func TestConflictPickToolNotOnACutQuestion(t *testing.T) {
	h := conflictHarness(t, pickRepo(t))
	question := askPick(t, h)
	h.st.conflictChoice.QuestionRunes = maxParentRunes + 1
	h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "ok"}

	answerPick(h, question, "alice", "nimm die neuere")

	for _, tool := range h.llm.Mentions[len(h.llm.Mentions)-1].Tools {
		if tool.Name == "pick_conflict_side" {
			t.Fatal("the tool was offered on a question the model sees cut")
		}
	}
	if last := h.bb.Replies[len(h.bb.Replies)-1].Text; !strings.Contains(last, "Reply **T** or **S** here.") {
		t.Errorf("reply = %q", last)
	}
}

// At the cost cap the model is off, the letters are free: the author's
// worded answer is pointed to them, no call, no banner alone.
func TestConflictPickAtTheCap(t *testing.T) {
	h := conflictHarness(t, pickRepo(t))
	question := askPick(t, h)
	h.st.prCost = ptri(6_000_000_000)
	n := len(h.bb.Replies)

	answerPick(h, question, "alice", "nimm die neuere")

	if len(h.llm.Mentions) != 0 || len(h.bb.Replies) != n+1 || h.bb.Replies[n].Text != "Reply **T** or **S** here." {
		t.Fatalf("mentions %d, replies %+v", len(h.llm.Mentions), h.bb.Replies[n:])
	}

	answerPick(h, question, "alice", "T")

	if len(h.st.Fixes) != 1 || h.st.Fixes[0].Outcome != store.FixCommitted {
		t.Errorf("fixes = %+v: the letters settle it at the cap too", h.st.Fixes)
	}
}

// Without tools (no room for them) the model may promise a merge nobody
// makes: the answer says what settles it.
func TestConflictPickToolsOff(t *testing.T) {
	h := conflictHarness(t, pickRepo(t))
	question := askPick(t, h)
	h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "Mache ich.", ToolsOff: "no room"}

	answerPick(h, question, "alice", "nimm die neuere")

	last := h.bb.Replies[len(h.bb.Replies)-1].Text
	if !strings.Contains(last, "Mache ich.") || !strings.Contains(last, "Reply **T** or **S** here.") {
		t.Fatalf("reply = %q", last)
	}
}

// The tool exists only directly under the open question.
func TestConflictPickToolOnlyUnderTheQuestion(t *testing.T) {
	h := conflictHarness(t, pickRepo(t))
	askPick(t, h)
	h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "ok"}

	answerPick(h, 4242, "alice", "nimm die neuere")

	for _, tool := range h.llm.Mentions[len(h.llm.Mentions)-1].Tools {
		if tool.Name == "pick_conflict_side" {
			t.Fatal("the tool was offered off the question")
		}
	}
}
