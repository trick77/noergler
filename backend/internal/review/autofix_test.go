package review

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// fixHarness is replyHarness with the author asking noergler to apply the
// finding, in English, and a model that writes a valid one-line change.
func fixHarness(t *testing.T) (*harness, *webhook.Payload) {
	t.Helper()
	h, p := replyHarness(t)
	h.st.languages = map[string]lang.Lang{"alice": lang.EN}
	p.Comment.Text = "fix it"
	h.llm.reply = inference.FindingReplyResult{Outcome: inference.OutcomeOK, Reply: inference.FindingReply{
		Intent: inference.IntentRequestsFix, Reply: "On it.",
	}, Cost: inference.CallCost{NanoUSD: ptri(1_000_000), Answered: true}}
	h.llm.fix = inference.FindingFixResult{Outcome: inference.OutcomeOK, Fix: inference.FindingFix{
		Feasible: true, Old: "func new() {}", New: "func new() { validate() }", CommitSubject: "Validate the business rule.",
	}}
	h.llm.fix.Cost = inference.CallCost{NanoUSD: ptri(2_000_000), Answered: true}
	return h, p
}

func TestFixRequestCommitsAndConfirms(t *testing.T) {
	h, p := fixHarness(t)

	h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

	if len(h.llm.Replies) != 1 || h.llm.Replies[0].Fix == nil {
		t.Fatalf("model calls %d, fix request %v: want one reply call carrying the fix call", len(h.llm.Replies), h.llm.Replies)
	}
	// The team's conventions (AGENTS.md) ride the fix prompt: the change
	// must read like the team's own code.
	if fp := h.llm.Replies[0].Fix.Prompt; !strings.Contains(fp, "func new() {}") || !strings.Contains(fp, "fix it") ||
		!strings.Contains(fp, "fix loop redirection") || !strings.Contains(fp, "Be terse.") {
		t.Errorf("fix prompt lacks the code, the request, the commit style or AGENTS.md:\n%s", fp)
	}
	if len(h.bb.Edits) != 1 {
		t.Fatalf("edits = %+v, want one commit", h.bb.Edits)
	}
	e := h.bb.Edits[0]
	want := editedFile{Project: "PROJ", Repo: "my-repo", Branch: "feature/thing", Path: "a.go", Source: "src1",
		Message: "validate the business rule\n\nhttps://bb.test/projects/PROJ/repos/my-repo/pull-requests/42/overview?commentId=501",
		Content: "package a\nfunc new() { validate() }\n"}
	if e != want {
		t.Errorf("edit = %+v\nwant   %+v", e, want)
	}
	if len(h.bb.Replies) != 1 || h.bb.Replies[0].Line != 900 {
		t.Fatalf("replies = %+v, want one under the author's comment", h.bb.Replies)
	}
	body := h.bb.Replies[0].Text
	for _, s := range []string{"Applied in [`c0ffee12`](https://bb.test/projects/PROJ/repos/my-repo/commits/c0ffee1234567890)",
		"> validate the business rule", "Please review the change carefully before you merge."} {
		if !strings.Contains(body, s) {
			t.Errorf("reply lacks %q:\n%s", s, body)
		}
	}
	if len(h.st.Fixes) != 1 || h.st.Fixes[0] != (store.FindingFix{FindingID: 11, TeamSlug: "payments", Outcome: store.FixCommitted, CommitSHA: "c0ffee1234567890"}) {
		t.Errorf("fixes = %+v", h.st.Fixes)
	}
	if len(h.st.Replied) != 1 || h.st.Replied[0].Dispute != nil {
		t.Errorf("replied = %+v, want the finding marked answered, no dispute", h.st.Replied)
	}
	if n := len(h.st.BotReplies); n != 1 || h.st.BotReplies[0].FindingID != 11 {
		t.Errorf("bot replies = %+v, want the confirmation kept with finding 11", h.st.BotReplies)
	}
	kinds := []string{}
	for _, c := range h.st.CommentCalls {
		kinds = append(kinds, c.Kind)
	}
	if strings.Join(kinds, ",") != "reply,fix" {
		t.Errorf("recorded calls = %v, want reply then fix", kinds)
	}
}

// Every refusal gets its own answer in the thread and its own row; nothing
// is committed but by a fix that passed.
func TestFixRequestAnswersEveryRefusal(t *testing.T) {
	cases := map[string]struct {
		tweak   func(*harness)
		outcome string
		say     string
		edits   int
	}{
		"no write permission": {func(h *harness) {
			h.bb.editErr = &bitbucket.StatusError{Method: "PUT", Path: "/browse/a.go", Status: 403}
		}, store.FixNoWrite, "no write permission", 1},
		"token rejected": {func(h *harness) {
			h.bb.editErr = &bitbucket.StatusError{Method: "PUT", Path: "/browse/a.go", Status: 401}
		}, store.FixNoWrite, "no write permission", 1},
		"branch moved": {func(h *harness) {
			h.bb.editErr = &bitbucket.StatusError{Method: "PUT", Path: "/browse/a.go", Status: 409}
		}, store.FixConflict, "branch has moved", 1},
		"hook refuses": {func(h *harness) {
			h.bb.editErr = &bitbucket.StatusError{Method: "PUT", Path: "/browse/a.go", Status: 400}
		}, store.FixRefused, "Bitbucket refused the commit", 1},
		"not feasible": {func(h *harness) {
			h.llm.fix.Fix = inference.FindingFix{Reason: "It needs the generator run on docs/source.md."}
		}, store.FixNotFeasible, "It needs the generator run", 0},
		"old not in the file": {func(h *harness) {
			h.llm.fix.Fix.Old = "func old() {}"
		}, store.FixRejected, "can't change this safely", 0},
		"fix call failed": {func(h *harness) {
			h.llm.fix = inference.FindingFixResult{Outcome: inference.OutcomeTimedOut}
		}, store.FixError, "Writing the fix failed", 0},
		"file unreadable": {func(h *harness) {
			h.bb.fileErr["src1:a.go"] = &bitbucket.StatusError{Method: "GET", Path: "/raw/a.go", Status: 500}
		}, store.FixNotFeasible, "could not be read", 0},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			h, p := fixHarness(t)
			c.tweak(h)

			h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

			if len(h.bb.Edits) != c.edits {
				t.Errorf("edits = %d, want %d", len(h.bb.Edits), c.edits)
			}
			if len(h.bb.Replies) != 1 || !strings.Contains(h.bb.Replies[0].Text, c.say) {
				t.Errorf("replies = %+v, want one saying %q", h.bb.Replies, c.say)
			}
			if len(h.st.Fixes) != 1 || h.st.Fixes[0].Outcome != c.outcome || h.st.Fixes[0].CommitSHA != "" {
				t.Errorf("fixes = %+v, want one %s without a commit", h.st.Fixes, c.outcome)
			}
		})
	}
}

// A thread answered already: a fix request is still heard, anything else
// is not answered again and stamps nothing (user, 2026-11-13).
func TestFixRequestOnAThreadAnsweredAlready(t *testing.T) {
	t.Run("fix request", func(t *testing.T) {
		h, p := fixHarness(t)
		at := time.Now()
		h.st.byComment[501].RepliedAt = &at

		h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

		if len(h.bb.Edits) != 1 || len(h.bb.Replies) != 1 || len(h.st.Replied) != 0 {
			t.Errorf("edits %d, replies %d, marked %d: want the commit and its answer, no new stamp",
				len(h.bb.Edits), len(h.bb.Replies), len(h.st.Replied))
		}
	})
	t.Run("anything else", func(t *testing.T) {
		h, p := replyHarness(t)
		at := time.Now()
		h.st.byComment[501].RepliedAt = &at

		h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

		if len(h.llm.Replies) != 1 || len(h.bb.Replies) != 0 || len(h.st.Replied) != 0 || len(h.st.Occurrences) != 0 {
			t.Errorf("model %d, replies %d, marked %d, occurrences %d: want one call and nothing else",
				len(h.llm.Replies), len(h.bb.Replies), len(h.st.Replied), len(h.st.Occurrences))
		}
	})
}

// The author's reply under noergler's own answer on a finding's thread is a
// reply on that finding: a fix request commits, anything else is answered
// as before, without a second stamp. Someone else there is a Q&A.
func TestReplyUnderNoerglersFindingAnswer(t *testing.T) {
	under := func(h *harness, p *webhook.Payload) {
		at := time.Now()
		h.st.byComment[501].RepliedAt = &at
		h.st.botReplies = map[int]*store.BotReply{950: {CommentID: 950, Kind: store.BotReplyReply, FindingID: 11}}
		h.st.skipState = &store.SkipState{}
		p.CommentParentID = 950
	}
	t.Run("fix request", func(t *testing.T) {
		h, p := fixHarness(t)
		under(h, p)

		h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

		if len(h.bb.Edits) != 1 || len(h.bb.Replies) != 1 || h.bb.Replies[0].Line != 900 {
			t.Errorf("edits %d, replies %+v: want the commit, answered under the author's reply", len(h.bb.Edits), h.bb.Replies)
		}
	})
	t.Run("a question", func(t *testing.T) {
		h, p := replyHarness(t)
		under(h, p)
		h.llm.reply.Reply = inference.FindingReply{Intent: "question", Reply: "It means the input is never checked."}

		h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

		if len(h.bb.Replies) != 1 || len(h.st.Replied) != 0 || len(h.bb.Edits) != 0 {
			t.Errorf("replies %d, marked %d, edits %d: want the answer, nothing stamped", len(h.bb.Replies), len(h.st.Replied), len(h.bb.Edits))
		}
	})
}

func TestNearestLine(t *testing.T) {
	lines := []string{"a", "  x := 1", "b", "x := 1", "c"}
	for _, c := range []struct {
		want string
		near int
		got  int
	}{{"x := 1", 1, 2}, {"x := 1", 5, 4}, {"y", 3, 0}, {"", 3, 0}} {
		if n := nearestLine(lines, c.want, c.near); n != c.got {
			t.Errorf("nearestLine(%q, %d) = %d, want %d", c.want, c.near, n, c.got)
		}
	}
}
