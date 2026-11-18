package review

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
)

func TestReadPicks(t *testing.T) {
	for _, c := range []struct {
		text string
		n    int
		want []bool
	}{
		{"T", 1, []bool{true}},
		{"s", 1, []bool{false}},
		{"`S`", 1, []bool{false}},
		{"**T** bitte", 1, []bool{true}},
		{"nimm S", 1, []bool{false}},
		{"T S", 2, []bool{true, false}},
		{"1T 2S", 2, []bool{true, false}},
		{"2: S, 1: T", 2, []bool{true, false}},
		{"1 s 2 t", 2, []bool{false, true}},
		{"Nimm 1: T, 2: S, 3: T, 4: S, 5: T bitte", 5, []bool{true, false, true, false, true}},
	} {
		got, ok := readPicks(c.text, c.n)
		if !ok || !slices.Equal(got, c.want) {
			t.Errorf("readPicks(%q, %d) = %v %t, want %v", c.text, c.n, got, ok, c.want)
		}
	}
	for _, c := range []struct {
		text string
		n    int
	}{
		{"Lets go!", 1},           // no side named
		{"ja", 1},                 // a yes is no side
		{"1.0.31", 1},             // a version's text is never read
		{"nicht T", 1},            // a no
		{"T, mach ich später", 1}, // a put-off
		{"T oder S?", 1},          // a question
		{"T", 2},                  // too few
		{"T S", 1},                // too many
		{"1T S", 2},               // half numbered
		{"1T 1S", 2},              // a number twice
		{"3T 1S", 2},              // out of range
		{"1", 1},                  // a number alone
		{"ist das so", 1},         // words beside no letter
		{"T", 0},
		{"it's T", 2},   // the apostrophe's s is no side
		{"that’s T", 1}, // typographic, likewise
	} {
		if got, ok := readPicks(c.text, c.n); ok {
			t.Errorf("readPicks(%q, %d) = %v, want no answer", c.text, c.n, got)
		}
	}
}

// Only a go-ahead naming no side, or letters of the wrong count, is asked
// again; a command, a question or a no goes on as any reply.
func TestPickNudge(t *testing.T) {
	for text, want := range map[string]bool{
		"Lets go!": true, "ja": true, "ok bitte": true, "👍": false,
		"review": false, "continue": false, "explain the difference": false,
		"nein, mach ich selbst": false, "which one?": false, "": false,
	} {
		if got := pickNudge(text, 1); got != want {
			t.Errorf("pickNudge(%q, 1) = %t, want %t", text, got, want)
		}
	}
	if !pickNudge("T", 2) || !pickNudge("1T 2S", 3) {
		t.Error("letters of the wrong count are not asked again")
	}
}

// pickRepo: both branches bumped one dependency to their own version (a
// pick for the author), and both added a different repository at one spot
// (code's rule).
func pickRepo(t *testing.T) *conflictRepo {
	t.Helper()
	gradle := func(version, repo string) string {
		return "dependencies {\n    implementation 'ch.x:client:" + version + "'\n}\n// a\n// b\n// c\n// d\n// e\nrepositories {\n" + repo + "}\n"
	}
	return newConflictRepo(t,
		map[string]string{"build.gradle": gradle("1.0.27", "")},
		map[string]string{"build.gradle": gradle("1.0.31", "    mavenCentral()\n")},
		map[string]string{"build.gradle": gradle("1.0.29", "    google()\n")})
}

// askPick runs the request up to the question: the model leaves the
// version to the author. The question's comment id is returned.
func askPick(t *testing.T, h *harness) int {
	t.Helper()
	h.llm.conflict = inference.ConflictResult{Outcome: inference.OutcomeOK,
		Resolution: inference.ConflictResolution{Feasible: true, Lines: map[int][]string{}, Ask: []int{1}},
		Cost:       inference.CallCost{Answered: true, NanoUSD: ptri(1000)}}
	h.askConflict(t, "alice")
	if h.st.conflictChoice == nil {
		t.Fatalf("no question kept; replies %+v", h.bb.Replies)
	}
	return h.st.conflictChoice.CommentID
}

// answerPick replies text under the question, the event naming the PR's
// head the question was asked at.
func answerPick(h *harness, question int, author, text string) {
	answerPickAt(h, question, author, text, h.st.conflictChoice.SourceHead)
}

func answerPickAt(h *harness, question int, author, text, head string) {
	p := mentionPayload(text, author)
	p.Comment.ID = 78
	p.CommentParentID = question
	p.PullRequest.FromRef.LatestCommit = head
	h.r.HandleMention(context.Background(), p, "platform", &inlineScheduler{})
}

// Two versions of one line are the author's pick: noergler asks, keeps the
// merge, pushes nothing; "T" settles it with no second call, the rule's
// conflict resolved as before.
func TestConflictPickAskedThenApplied(t *testing.T) {
	repo := pickRepo(t)
	before := repo.run(repo.bare, "rev-parse", "feature/thing")
	h := conflictHarness(t, repo)

	question := askPick(t, h)

	if got := repo.run(repo.bare, "rev-parse", "feature/thing"); got != before {
		t.Fatal("pushed before the author picked")
	}
	if len(h.st.Fixes) != 0 {
		t.Errorf("fixes = %+v: the request ends with the answer", h.st.Fixes)
	}
	text := h.bb.Replies[len(h.bb.Replies)-1].Text
	for _, want := range []string{"`build.gradle:2`", "**T** from `master`", "1.0.31", "**S** from `feature/thing`", "1.0.29", "Reply **T** or **S** here."} {
		if !strings.Contains(text, want) {
			t.Errorf("question lacks %q:\n%s", want, text)
		}
	}
	if c := h.st.conflictChoice; len(c.Conflicts) != 2 || c.Conflicts[0].How != store.ChoiceAsk || c.Conflicts[1].How != store.ChoiceRule {
		t.Fatalf("kept %+v", c)
	}

	answerPick(h, question, "alice", "T")

	if len(h.llm.Conflicts) != 1 {
		t.Errorf("%d model calls, want the request's only", len(h.llm.Conflicts))
	}
	want := "dependencies {\n    implementation 'ch.x:client:1.0.31'\n}\n// a\n// b\n// c\n// d\n// e\nrepositories {\n    mavenCentral()\n    google()\n}"
	if got := repo.show("feature/thing:build.gradle"); got != want {
		t.Fatalf("build.gradle = %q", got)
	}
	if msg := repo.run(repo.bare, "log", "-1", "--format=%B", "feature/thing"); !strings.Contains(msg, "build.gradle:2 (author)") ||
		!strings.Contains(msg, "(rule)") {
		t.Errorf("commit:\n%s", msg)
	}
	reply := h.bb.Replies[len(h.bb.Replies)-1].Text
	if !strings.Contains(reply, "`build.gradle:2` (your pick)") || !strings.Contains(reply, "1.0.29") {
		t.Errorf("reply:\n%s", reply)
	}
	// The row and the commit name the request, the confirmation answers
	// the pick.
	if len(h.st.Fixes) != 1 || h.st.Fixes[0].Outcome != store.FixCommitted || h.st.Fixes[0].ConflictCommentID != 77 {
		t.Errorf("fixes = %+v", h.st.Fixes)
	}
	if last := h.bb.Replies[len(h.bb.Replies)-1]; last.Line != 78 {
		t.Errorf("confirmation under comment %d, want the pick 78", last.Line)
	}
	if h.st.conflictChoice != nil {
		t.Error("the question stayed open after its answer")
	}
	repo.run(repo.bare, "fsck", "--strict")
}

// A branch moved since the question: its merge is no longer the one asked
// about, so nothing is pushed and the author asks again.
// The event names the moved head (refused before any fetch), or still the
// old one (found by the fetch).
func TestConflictPickAfterTheBranchMoved(t *testing.T) {
	for _, eventSeesIt := range []bool{true, false} {
		t.Run(fmt.Sprintf("event names the move: %t", eventSeesIt), func(t *testing.T) {
			repo := pickRepo(t)
			h := conflictHarness(t, repo)
			question := askPick(t, h)
			old := h.st.conflictChoice.SourceHead
			repo.run(repo.work, "checkout", "-q", "feature/thing")
			repo.commit(map[string]string{"other.txt": "x\n"}, "more")
			repo.run(repo.work, "push", "-q", repo.bare, "feature/thing")
			moved := repo.run(repo.bare, "rev-parse", "feature/thing")
			head := old
			if eventSeesIt {
				head = moved
			}

			answerPickAt(h, question, "alice", "S", head)

			if got := repo.run(repo.bare, "rev-parse", "feature/thing"); got != moved {
				t.Fatal("a merge was pushed onto a moved branch")
			}
			if len(h.st.Fixes) != 1 || h.st.Fixes[0].Outcome != store.FixConflict ||
				!strings.Contains(h.bb.Replies[len(h.bb.Replies)-1].Text, "Please ask for the merge conflict again.") {
				t.Errorf("fixes = %+v, replies = %+v", h.st.Fixes, h.bb.Replies)
			}
			if h.st.conflictChoice != nil {
				t.Error("a stale question stayed open")
			}
		})
	}
}

// A failure on the way (Bitbucket's merge status) keeps the question and
// writes no row: the same answer, again, settles it with no new call, one
// row for the request.
func TestConflictPickFailureKeepsTheQuestion(t *testing.T) {
	repo := pickRepo(t)
	h := conflictHarness(t, repo)
	question := askPick(t, h)
	h.bb.mergeErr = errors.New("timeout")

	answerPick(h, question, "alice", "T")

	if h.st.conflictChoice == nil || len(h.st.Fixes) != 0 ||
		!strings.Contains(h.bb.Replies[len(h.bb.Replies)-1].Text, "failed") {
		t.Fatalf("choice %+v, fixes %+v, replies %+v", h.st.conflictChoice, h.st.Fixes, h.bb.Replies)
	}
	h.bb.mergeErr = nil

	answerPick(h, question, "alice", "T")

	if len(h.st.Fixes) != 1 || h.st.Fixes[0].Outcome != store.FixCommitted || len(h.llm.Conflicts) != 1 {
		t.Errorf("fixes = %+v, model calls %d", h.st.Fixes, len(h.llm.Conflicts))
	}
}

// Two spots asked, "T" alone: asked again for both letters, no call.
func TestConflictPickWrongCountAskedAgain(t *testing.T) {
	gradle := func(a, b string) string {
		return "x 'a:" + a + "'\n// 1\n// 2\n// 3\n// 4\n// 5\ny 'b:" + b + "'\n"
	}
	repo := newConflictRepo(t,
		map[string]string{"build.gradle": gradle("1", "1")},
		map[string]string{"build.gradle": gradle("3", "3")},
		map[string]string{"build.gradle": gradle("2", "2")})
	h := conflictHarness(t, repo)
	h.llm.conflict = inference.ConflictResult{Outcome: inference.OutcomeOK,
		Resolution: inference.ConflictResolution{Feasible: true, Lines: map[int][]string{}, Ask: []int{1, 2}}}
	h.askConflict(t, "alice")
	if h.st.conflictChoice == nil || h.st.conflictChoice.Asked() != 2 {
		t.Fatalf("choice %+v, replies %+v", h.st.conflictChoice, h.bb.Replies)
	}
	n := len(h.bb.Replies)

	answerPick(h, h.st.conflictChoice.CommentID, "alice", "T")

	if len(h.bb.Replies) != n+1 || h.bb.Replies[n].Text != "Reply here in order, e.g. `1T 2S`." || len(h.llm.Mentions) != 0 {
		t.Errorf("replies = %+v, mentions %d", h.bb.Replies[n:], len(h.llm.Mentions))
	}

	answerPick(h, h.st.conflictChoice.CommentID, "alice", "1S 2T")

	if got := repo.show("feature/thing:build.gradle"); got != strings.TrimSuffix(gradle("2", "3"), "\n") {
		t.Errorf("build.gradle = %q", got)
	}
}

// Under the question: the author's short reply naming no side is asked
// again for the letters, no call; anyone else's pick is told only the
// author may; a question goes to the Q&A.
func TestConflictPickOtherReplies(t *testing.T) {
	repo := pickRepo(t)
	t.Run("author, no side", func(t *testing.T) {
		h := conflictHarness(t, repo)
		question := askPick(t, h)
		n := len(h.bb.Replies)

		answerPick(h, question, "alice", "Lets go!")

		if len(h.bb.Replies) != n+1 || h.bb.Replies[n].Text != "Reply **T** or **S** here." {
			t.Errorf("replies = %+v", h.bb.Replies[n:])
		}
		if len(h.llm.Conflicts)+len(h.llm.Mentions) != 1 || h.st.conflictChoice == nil {
			t.Error("a reply naming no side called a model or closed the question")
		}
	})
	t.Run("not the author", func(t *testing.T) {
		h := conflictHarness(t, repo)
		question := askPick(t, h)
		n := len(h.bb.Replies)

		answerPick(h, question, "bob", "T")

		if len(h.bb.Replies) != n+1 || !strings.Contains(h.bb.Replies[n].Text, "only when the PR's author asks") {
			t.Errorf("replies = %+v", h.bb.Replies[n:])
		}
		if h.st.conflictChoice == nil {
			t.Error("someone else's pick closed the author's question")
		}
	})
	t.Run("a question", func(t *testing.T) {
		h := conflictHarness(t, repo)
		question := askPick(t, h)
		h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "1.0.31 is newer."}

		answerPick(h, question, "alice", "which one is newer?")

		if len(h.llm.Mentions) != 1 {
			t.Errorf("mentions = %d, want the Q&A", len(h.llm.Mentions))
		}
	})
}

// A model that leaves to the author a conflict no whole side settles is
// refused: nothing asked, nothing pushed.
func TestConflictPickNotChoosable(t *testing.T) {
	repo := newConflictRepo(t,
		map[string]string{"a.txt": "a\nb\n"},
		map[string]string{"a.txt": "a2\nb\n"},
		map[string]string{"a.txt": "a\nb2\n"})
	h := conflictHarness(t, repo)
	h.llm.conflict = inference.ConflictResult{Outcome: inference.OutcomeOK,
		Resolution: inference.ConflictResolution{Feasible: true, Lines: map[int][]string{}, Ask: []int{1}}}

	h.askConflict(t, "alice")

	if h.st.conflictChoice != nil || len(h.st.Fixes) != 1 || h.st.Fixes[0].Outcome != store.FixRejected {
		t.Errorf("choice %+v, fixes %+v", h.st.conflictChoice, h.st.Fixes)
	}
}
