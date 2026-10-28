package review

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

func goodNoteResult() inference.AlmanachNoteResult {
	s := "Der Adapter liest den Mandanten aus der Kopfzeile."
	return inference.AlmanachNoteResult{
		Outcome: inference.OutcomeOK,
		Note: inference.AlmanachNote{
			Summary: &s, ChangeKind: "bugfix",
			Bugfix:     &inference.AlmanachBugfix{Symptom: "Falscher Mandant.", Cause: "Stiller Rückfall.", CauseKind: "input_validation"},
			Components: []inference.AlmanachItem{{Name: "Datei-Adapter", Text: "Liest Sammeldateien, Grösse egal.", Files: []string{"intake/file/reader.go"}}},
		},
		Cost: inference.CallCost{Answered: true, Model: "m"},
	}
}

// withAlmanach turns the feed on for the harness and gives the fake model
// a note to answer with.
func withAlmanach(h *harness) {
	h.al.enabled = true
	h.llm.note = goodNoteResult()
}

// After a review posts, one more turn continues its conversation: the same
// prompt, the model's review answer, the note instruction. The note lands
// in the store with the run id, the ticket and the posted-findings facts;
// the call is recorded in Almanach's ledger, not the PR's; and the review
// was posted before the spawned call ran.
func TestAlmanachNoteFollowsTheReview(t *testing.T) {
	h := newHarness(t, withAlmanach)
	h.llm.review.Raw = `{"overview":"the answer"}`
	sched := &deferredScheduler{}
	if !h.r.ReviewPullRequestStaged(context.Background(), prPayload(webhook.EventOpened), "platform", false, sched) {
		t.Fatal("not handed off")
	}
	sched.finish()
	if len(sched.spawned) != 1 || sched.spawned[0].name != "almanach note" {
		t.Fatalf("spawned = %+v, want one almanach note", sched.spawned)
	}
	if len(h.llm.Notes) != 0 || len(h.st.Notes) != 0 {
		t.Fatal("the note call ran on the worker, before the spawn")
	}
	if len(h.bb.Posted) == 0 {
		t.Fatal("the review was not posted before the note was spawned")
	}

	sched.spawned[0].fn(sched.spawned[0].ctx)

	if len(h.llm.Notes) != 1 {
		t.Fatalf("note calls = %d, want 1", len(h.llm.Notes))
	}
	req := h.llm.Notes[0]
	if req.ReviewPrompt != h.llm.Reviews[0].Prompt || req.ReviewAnswer != `{"overview":"the answer"}` {
		t.Errorf("the note does not continue the review conversation: %+v", req)
	}
	if !strings.Contains(req.Instruction, "my-repo") || !strings.Contains(req.Instruction, "Swiss Standard German") {
		t.Errorf("instruction = %q", req.Instruction)
	}
	if req.PromptTokens <= h.llm.Reviews[0].PromptTokens {
		t.Errorf("note tokens %d must exceed the review's %d: the answer and the turn are counted", req.PromptTokens, h.llm.Reviews[0].PromptTokens)
	}
	if len(h.st.Notes) != 1 {
		t.Fatalf("notes stored = %d, want 1", len(h.st.Notes))
	}
	n := h.st.Notes[0]
	if n.RunID != 1 || n.ChangeKind != "bugfix" || n.ModelLabel != h.llm.Label() || n.TeamSlug != "payments" {
		t.Errorf("stored note = %+v", n)
	}
	var stored inference.AlmanachNote
	if err := json.Unmarshal(n.Note, &stored); err != nil || stored.Components[0].Text != "Liest Sammeldateien, Grösse egal." {
		t.Errorf("stored note json = %s (%v): Swiss spelling expected", n.Note, err)
	}
	if len(h.st.AlmanachCalls) != 1 || h.st.AlmanachCalls[0].NoteStatus != store.NoteWritten || h.st.AlmanachCalls[0].Outcome != "ok" {
		t.Errorf("call rows = %+v", h.st.AlmanachCalls)
	}
	if len(h.st.CommentCalls) != 0 {
		t.Error("the note call reached the PR's ledger")
	}
	if len(h.al.Emitted) != 0 {
		t.Error("a note was sent before the merge")
	}
}

// No almanach block, no call; a failed review, no call; a conversation
// that does not fit, no call and a log line.
func TestAlmanachNoteGates(t *testing.T) {
	t.Run("off", func(t *testing.T) {
		h := newHarness(t, nil)
		h.llm.review.Raw = "x"
		h.r.ReviewPullRequestStaged(context.Background(), prPayload(webhook.EventOpened), "platform", false, &inlineScheduler{})
		if len(h.llm.Notes) != 0 {
			t.Error("note call without an almanach block")
		}
	})
	t.Run("review failed", func(t *testing.T) {
		h := newHarness(t, withAlmanach)
		h.llm.review = inference.ReviewResult{Outcome: inference.OutcomeUnparseable, Review: inference.ParsedReview{Summary: inference.NewReviewSummary()}, Raw: "garbage"}
		h.r.ReviewPullRequestStaged(context.Background(), prPayload(webhook.EventOpened), "platform", false, &inlineScheduler{})
		if len(h.llm.Notes) != 0 {
			t.Error("note call after a failed review")
		}
	})
	t.Run("does not fit", func(t *testing.T) {
		h := newHarness(t, withAlmanach)
		h.llm.review.Raw = strings.Repeat("x", 4*(h.llm.window))
		h.r.ReviewPullRequestStaged(context.Background(), prPayload(webhook.EventOpened), "platform", false, &inlineScheduler{})
		if len(h.llm.Notes) != 0 {
			t.Error("note call over the fit ceiling")
		}
	})
}

// What the model sends back is checked, not trusted: a note over a cap or
// with a secret drops whole, an empty one is "nothing learned", an
// unparseable answer is invalid, no answer is no answer. Every outcome
// leaves a call row saying so, and only a standing note a row.
func TestAlmanachNoteStatuses(t *testing.T) {
	// Over the harness's total cap (6000): prose has no cap of its own.
	long := strings.Repeat("ä", 6001)
	cases := []struct {
		name   string
		tweak  func(*inference.AlmanachNoteResult)
		status string
		stored int
	}{
		{"too long", func(r *inference.AlmanachNoteResult) { r.Note.Components[0].Text = long }, inference.AlmanachTooLong, 0},
		{"secret", func(r *inference.AlmanachNoteResult) { r.Note.Components[0].Text = "aB3dE5fG7hI9jK1lM3nO5pQ7rS9tU" }, inference.AlmanachSecret, 0},
		{"invalid shape", func(r *inference.AlmanachNoteResult) { r.Note.ChangeKind = "feature" }, inference.AlmanachInvalid, 0},
		{"nothing learned", func(r *inference.AlmanachNoteResult) { r.Note = inference.AlmanachNote{ChangeKind: "chore"} }, inference.AlmanachNothingLearned, 0},
		{"unparseable", func(r *inference.AlmanachNoteResult) {
			*r = inference.AlmanachNoteResult{Outcome: inference.OutcomeUnparseable, Cost: inference.CallCost{Answered: true}}
		}, inference.AlmanachInvalid, 0},
		{"cut by the output limit", func(r *inference.AlmanachNoteResult) {
			*r = inference.AlmanachNoteResult{Outcome: inference.OutcomeUnparseable, Cut: true, Cost: inference.CallCost{Answered: true}}
		}, inference.AlmanachTooLong, 0},
		{"no answer", func(r *inference.AlmanachNoteResult) {
			*r = inference.AlmanachNoteResult{Outcome: inference.OutcomeTimedOut}
		}, inference.AlmanachNoAnswer, 0},
		{"written", func(*inference.AlmanachNoteResult) {}, store.NoteWritten, 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := newHarness(t, withAlmanach)
			h.llm.review.Raw = "x"
			c.tweak(&h.llm.note)
			h.r.ReviewPullRequestStaged(context.Background(), prPayload(webhook.EventOpened), "platform", false, &inlineScheduler{})
			if len(h.st.AlmanachCalls) != 1 || h.st.AlmanachCalls[0].NoteStatus != c.status {
				t.Fatalf("call rows = %+v, want one with status %q", h.st.AlmanachCalls, c.status)
			}
			if len(h.st.Notes) != c.stored {
				t.Errorf("notes stored = %d, want %d", len(h.st.Notes), c.stored)
			}
		})
	}
}

// A store that refuses the note is neither the model's fault nor a closed
// PR: the ledger says store_failed, and the call still has its row.
func TestStoreFaultOnTheNoteIsItsOwnStatus(t *testing.T) {
	h := newHarness(t, withAlmanach)
	h.llm.review.Raw = "x"
	h.st.noteErr = errors.New("db hiccup")
	h.r.ReviewPullRequestStaged(context.Background(), prPayload(webhook.EventOpened), "platform", false, &inlineScheduler{})
	if len(h.st.AlmanachCalls) != 1 || h.st.AlmanachCalls[0].NoteStatus != store.NoteStoreFailed {
		t.Fatalf("call rows = %+v, want one with status store_failed", h.st.AlmanachCalls)
	}
	if len(h.st.Notes) != 0 {
		t.Error("a note was recorded as written")
	}
}

// Two rounds of one PR can be on the pool at once; the first to finish
// must not make the merge believe the second is done.
func TestInflightCountsRounds(t *testing.T) {
	var s inflightSet
	k := store.PRKey{Project: "P", Repo: "r", PRID: 1}
	s.add(k)
	s.add(k)
	s.remove(k)
	if !s.has(k) {
		t.Fatal("one round finished, the other is still in flight")
	}
	s.remove(k)
	if s.has(k) {
		t.Fatal("both rounds finished")
	}
	s.remove(k)
	if s.has(k) {
		t.Fatal("a stray remove must not go negative")
	}
}

// A round that finds its PR merged sends its own note and nothing else:
// the team's catch-up drain is the merge's job on the worker, not the
// pool's.
func TestRoundEmitsOnlyItsOwnNote(t *testing.T) {
	h := newHarness(t, withAlmanach)
	h.llm.review.Raw = "x"
	h.al.refuse = true
	h.r.ReviewPullRequestStaged(context.Background(), prPayload(webhook.EventOpened), "platform", false, &inlineScheduler{})
	h.r.HandlePRMerged(context.Background(), prPayload(webhook.EventMerged))
	h.al.refuse = false
	h.al.Emitted = nil

	second := prPayload(webhook.EventOpened)
	second.PullRequest.ID = 43
	sched := &deferredScheduler{}
	h.r.ReviewPullRequestStaged(context.Background(), second, "platform", false, sched)
	sched.finish()
	merged := prPayload(webhook.EventMerged)
	merged.PullRequest.ID = 43
	h.r.HandlePRMerged(context.Background(), merged)
	if len(h.al.Emitted) != 0 {
		t.Fatalf("the merge emitted while the note was in flight: %+v", h.al.Emitted)
	}
	sched.spawned[0].fn(sched.spawned[0].ctx)
	if len(h.al.Emitted) != 1 || h.al.Emitted[0].PRID != 43 {
		t.Errorf("emitted = %+v, want PR 43 alone; PR 42 waits for the next merge", h.al.Emitted)
	}
}

// A repo the team gave up while the note call ran gets no row: the spawned
// writer re-asks ownership, as post does.
func TestAlmanachNoteReasksOwnership(t *testing.T) {
	h := newHarness(t, withAlmanach)
	h.llm.review.Raw = "x"
	sched := &deferredScheduler{}
	h.r.ReviewPullRequestStaged(context.Background(), prPayload(webhook.EventOpened), "platform", false, sched)
	sched.finish()
	h.r.owns = func(string, string) bool { return false }
	sched.spawned[0].fn(sched.spawned[0].ctx)
	if len(h.st.Notes) != 0 {
		t.Error("a note was written for a repo the team no longer claims")
	}
}

// The merge stamps the note final and sends the team's pending notes on,
// with the team's claims for the project; Almanach's 2xx marks the round
// emitted; a redelivered merge sends nothing again. No model call.
func TestMergedFinalizesAndEmitsTheNote(t *testing.T) {
	h := newHarness(t, withAlmanach)
	h.llm.review.Raw = "x"
	h.r.team = func() *config.Team {
		return &config.Team{Projects: []config.ProjectScope{{Key: "PROJ"}, {Key: "OTHER", Repos: []string{"x"}}}}
	}
	h.r.ReviewPullRequestStaged(context.Background(), prPayload(webhook.EventOpened), "platform", false, &inlineScheduler{})
	if len(h.st.Notes) != 1 {
		t.Fatalf("notes = %d", len(h.st.Notes))
	}
	calls := len(h.llm.Notes)

	p := prPayload(webhook.EventMerged)
	p.PullRequest.Properties = &webhook.Properties{MergeCommit: &webhook.MergeCommit{ID: "merge1"}}
	h.r.HandlePRMerged(context.Background(), p)

	if len(h.st.Finalized) != 1 || h.st.Finalized[0].MergeCommit != "merge1" || h.st.Finalized[0].SourceCommit != "src1" {
		t.Fatalf("finalized = %+v", h.st.Finalized)
	}
	if len(h.al.Emitted) != 1 {
		t.Fatalf("emitted = %d, want 1", len(h.al.Emitted))
	}
	got := h.al.Emitted[0]
	if got.PRKey != "PROJ/my-repo#42" || got.MergeCommitSHA != "merge1" || got.ChangeKind != "bugfix" || got.RunID != 1 {
		t.Errorf("note = %+v", got)
	}
	if len(got.Claims) != 1 || got.Claims[0].ProjectKey != "PROJ" || got.Claims[0].RepoSlug != "" {
		t.Errorf("claims = %+v, want the whole-project claim on PROJ", got.Claims)
	}
	if len(h.st.EmittedRuns) != 1 || h.st.EmittedRuns[0] != got.RunID {
		t.Errorf("emitted stamp = %v, want the sent run id", h.st.EmittedRuns)
	}
	if len(h.llm.Notes) != calls {
		t.Error("the merge made a model call")
	}

	h.r.HandlePRMerged(context.Background(), p)
	if len(h.al.Emitted) != 1 {
		t.Error("a redelivered merge sent the note again")
	}
}

// Almanach refusing a note leaves it pending; the team's next merge sends
// it along with the new one, oldest first.
func TestRefusedNoteGoesWithTheNextMerge(t *testing.T) {
	h := newHarness(t, withAlmanach)
	h.llm.review.Raw = "x"
	h.al.refuse = true
	first := prPayload(webhook.EventOpened)
	h.r.ReviewPullRequestStaged(context.Background(), first, "platform", false, &inlineScheduler{})
	h.r.HandlePRMerged(context.Background(), prPayload(webhook.EventMerged))
	if len(h.al.Emitted) != 1 || len(h.st.EmittedRuns) != 0 {
		t.Fatalf("refused note: emitted %d stamped %v", len(h.al.Emitted), h.st.EmittedRuns)
	}

	h.al.refuse = false
	second := prPayload(webhook.EventOpened)
	second.PullRequest.ID = 43
	h.r.ReviewPullRequestStaged(context.Background(), second, "platform", false, &inlineScheduler{})
	merged := prPayload(webhook.EventMerged)
	merged.PullRequest.ID = 43
	h.r.HandlePRMerged(context.Background(), merged)
	if len(h.al.Emitted) != 3 || h.al.Emitted[1].PRID != 42 || h.al.Emitted[2].PRID != 43 {
		t.Fatalf("emitted = %+v, want the old note first, then the new", h.al.Emitted)
	}
	if len(h.st.EmittedRuns) != 2 {
		t.Errorf("stamped = %v, want both", h.st.EmittedRuns)
	}
}

// A merge that lands while this PR's note call is still in flight stamps
// the row and leaves the emit to the round: the newer note is seconds away,
// and it sends itself once it sees the stamp.
func TestMergeDuringTheNoteCallDefersTheEmit(t *testing.T) {
	h := newHarness(t, withAlmanach)
	h.llm.review.Raw = "x"
	sched := &deferredScheduler{}
	h.r.ReviewPullRequestStaged(context.Background(), prPayload(webhook.EventOpened), "platform", false, sched)
	sched.finish()

	h.r.HandlePRMerged(context.Background(), prPayload(webhook.EventMerged))
	if len(h.st.Finalized) != 1 {
		t.Fatal("the merge did not finalize")
	}
	if len(h.al.Emitted) != 0 {
		t.Fatal("the merge emitted while the note was in flight")
	}

	sched.spawned[0].fn(sched.spawned[0].ctx)
	if len(h.al.Emitted) != 1 || len(h.st.EmittedRuns) != 1 {
		t.Errorf("the round did not emit the finalized note: emitted %d stamped %v", len(h.al.Emitted), h.st.EmittedRuns)
	}
}

// Declined and deleted forget the note; the store keeps a final one.
func TestDeclinedAndDeletedDropTheNote(t *testing.T) {
	for _, ev := range []string{webhook.EventDeclined, webhook.EventDeleted} {
		h := newHarness(t, withAlmanach)
		p := prPayload(ev)
		if ev == webhook.EventDeclined {
			h.r.HandlePRDeclined(context.Background(), p)
		} else {
			h.r.HandlePRDeleted(context.Background(), p)
		}
		if h.st.NoteDeletes != 1 {
			t.Errorf("%s: DeleteAlmanachNote called %d times, want 1", ev, h.st.NoteDeletes)
		}
		if len(h.llm.Notes) != 0 || len(h.al.Emitted) != 0 {
			t.Errorf("%s: a call or an emit on a closed PR", ev)
		}
	}
	h := newHarness(t, nil)
	h.r.HandlePRDeclined(context.Background(), prPayload(webhook.EventDeclined))
	if h.st.NoteDeletes != 0 {
		t.Error("a team without almanach touched the notes table")
	}
}
