package review

import (
	"context"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
)

func TestConflictConsent(t *testing.T) {
	for text, want := range map[string]bool{
		"ja": true, "Ja!": true, "ok bitte": true, "👍": true, "yes please": true, "Ja, gerne": true,
		"go ahead": true, "einverstanden": true, "Ja gerne": true, "ja klar": true, "ok sure": true, "yes yes": true,
		"ja mach das": true,
		"nein":        false, "ja, aber später": false, "ok, mach ich selbst": false, "no thanks": false,
		"ja, warum?": false, "Lass uns das versuchen": false, "ja, und schau dir auch den Test an": false,
		// An acknowledgement as often as a yes: the model's.
		"ok, danke": false, "ok thanks": false, "ok, I'll do it": false,
	} {
		if got := conflictConsent(text); got != want {
			t.Errorf("conflictConsent(%q) = %v, want %v", text, got, want)
		}
	}
}

func TestConflictVeto(t *testing.T) {
	for text, want := range map[string]bool{
		"Lass uns das versuchen": false, "ich bin einverstanden": false, "klingt gut, mach mal": false,
		"nicht jetzt": true, "mach ich selbst": true, "no thanks": true, "lieber nicht": true,
		"später vielleicht": true, "nein danke": true,
		"ok, I'll do it": true, "mach ich": true, "ich mach das": true, "übernehme ich": true,
		"kein Problem, leg los": false, "warum nicht": false, "mach halt": false, "no problem, go ahead": false,
		"can't wait, go": false, "ja, kein Stress": false, "yes, no rush": false,
	} {
		if got := conflictVeto(text); got != want {
			t.Errorf("conflictVeto(%q) = %v, want %v", text, got, want)
		}
	}
}

// offerReply is a plain reply to the open offer (comment 900).
func (h *harness) offerReply(t *testing.T, text, author string, parent int) {
	t.Helper()
	h.st.skipState = &store.SkipState{ID: 1}
	h.st.conflictOffer = 900
	p := mentionPayload(text, author)
	p.CommentParentID = parent
	h.r.HandleMention(context.Background(), p, "platform", &inlineScheduler{})
}

func conflictRepoByCode(t *testing.T) *conflictRepo {
	return newConflictRepo(t, map[string]string{"l": "a\n"}, map[string]string{"l": "a\nt\n"}, map[string]string{"l": "a\ns\n"})
}

// The author's "ja" under the open offer is the request: resolved and
// pushed, no Q&A call.
func TestConflictConsentByCode(t *testing.T) {
	repo := conflictRepoByCode(t)
	before := repo.run(repo.bare, "rev-parse", "feature/thing")
	h := conflictHarness(t, repo)

	h.offerReply(t, "ja", "alice", 900)

	if len(h.llm.Mentions) != 0 {
		t.Fatalf("Q&A calls = %d", len(h.llm.Mentions))
	}
	if got := repo.run(repo.bare, "rev-parse", "feature/thing"); got == before {
		t.Fatal("nothing pushed")
	}
}

// "ja" anywhere else is a question as before.
func TestConflictConsentOnlyUnderTheOpenOffer(t *testing.T) {
	repo := conflictRepoByCode(t)
	before := repo.run(repo.bare, "rev-parse", "feature/thing")
	h := conflictHarness(t, repo)

	h.offerReply(t, "ja", "alice", 901)

	if got := repo.run(repo.bare, "rev-parse", "feature/thing"); got != before {
		t.Fatal("a yes under another comment pushed")
	}
	if len(h.llm.Mentions) != 1 {
		t.Fatalf("Q&A calls = %d", len(h.llm.Mentions))
	}
	for _, tool := range h.llm.Mentions[0].Tools {
		if tool.Name == "resolve_merge_conflict" {
			t.Fatal("the tool was offered off the offer")
		}
	}
}

// Someone other than the author says yes: told, kindly, that only the
// author may ask; nothing pushed (user, 2026-11-15: a general rule).
func TestConflictConsentNotTheAuthor(t *testing.T) {
	repo := conflictRepoByCode(t)
	before := repo.run(repo.bare, "rev-parse", "feature/thing")
	h := conflictHarness(t, repo)

	h.offerReply(t, "ok", "bob", 900)

	if got := repo.run(repo.bare, "rev-parse", "feature/thing"); got != before {
		t.Fatal("pushed for someone other than the author")
	}
	if len(h.bb.Replies) != 1 || !strings.Contains(h.bb.Replies[0].Text, "only when the PR's author asks") {
		t.Fatalf("replies = %+v", h.bb.Replies)
	}
}

// Any other wording reaches the Q&A, which has the tool there; the model's
// yes starts the request, its own sentence is not posted.
func TestConflictConsentByTheModel(t *testing.T) {
	repo := conflictRepoByCode(t)
	before := repo.run(repo.bare, "rev-parse", "feature/thing")
	h := conflictHarness(t, repo)
	h.llm.onMention = func(_ context.Context, tools []string, run func(name, args string) string) string {
		if !slices.Contains(tools, "resolve_merge_conflict") {
			t.Errorf("tools = %v", tools)
		}
		if got := run("resolve_merge_conflict", "{}"); !strings.Contains(got, `"ok"`) {
			t.Errorf("tool = %s", got)
		}
		return "MODEL SENTENCE"
	}

	h.offerReply(t, "Lass uns das versuchen", "alice", 900)

	if got := repo.run(repo.bare, "rev-parse", "feature/thing"); got == before {
		t.Fatal("nothing pushed")
	}
	for _, r := range h.bb.Replies {
		if strings.Contains(r.Text, "MODEL SENTENCE") {
			t.Fatal("the model's sentence was posted")
		}
	}
}

// Without the tool (no room for it) the model may promise a resolution
// nobody starts: the reply says what starts it instead.
func TestConflictConsentToolsOff(t *testing.T) {
	h := conflictHarness(t, conflictRepoByCode(t))
	h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "Alles klar, ich löse ihn.", ToolsOff: "no room"}

	h.offerReply(t, "Lass uns das versuchen", "alice", 900)

	// The answer stays (it may answer a question); the hint follows it.
	if len(h.bb.Replies) != 1 || !strings.Contains(h.bb.Replies[0].Text, "ich löse ihn") ||
		!strings.Contains(h.bb.Replies[0].Text, "reply **yes** here") {
		t.Fatalf("replies = %+v", h.bb.Replies)
	}
}

// On a PR ignored since its summary was deleted a plain reply, a yes too,
// is not answered: a plain reply never brings a PR back (known limit: the
// command with a mention still works there).
func TestConflictConsentOnAnIgnoredPR(t *testing.T) {
	repo := conflictRepoByCode(t)
	before := repo.run(repo.bare, "rev-parse", "feature/thing")
	h := conflictHarness(t, repo)
	now := time.Now()
	h.st.skipState = &store.SkipState{ID: 1, IgnoredAt: &now, IgnoreReason: store.IgnoreSummaryDeleted}
	h.st.conflictOffer = 900
	h.st.botReplies = map[int]*store.BotReply{900: {Kind: store.BotReplyQuestion}}
	p := mentionPayload("ja", "alice")
	p.CommentParentID = 900

	handled, _ := h.r.HandleFindingReply(context.Background(), p, "platform", &inlineScheduler{})

	if !handled || repo.run(repo.bare, "rev-parse", "feature/thing") != before || len(h.bb.Replies) != 0 {
		t.Fatalf("handled %t, replies %+v", handled, h.bb.Replies)
	}
}

// The model reads a refusal as a yes: code's veto stands, nothing pushed.
func TestConflictConsentVetoedByCode(t *testing.T) {
	repo := conflictRepoByCode(t)
	before := repo.run(repo.bare, "rev-parse", "feature/thing")
	h := conflictHarness(t, repo)
	var tool string
	h.llm.onMention = func(_ context.Context, _ []string, run func(name, args string) string) string {
		tool = run("resolve_merge_conflict", "{}")
		return "Alles klar."
	}

	h.offerReply(t, "Danke, mach ich später selbst", "alice", 900)

	if !strings.Contains(tool, "rejected") {
		t.Fatalf("tool = %s", tool)
	}
	if got := repo.run(repo.bare, "rev-parse", "feature/thing"); got != before {
		t.Fatal("pushed despite the veto")
	}
	// The answer says what would start it, in case the veto misread a yes.
	if len(h.bb.Replies) != 1 || !strings.Contains(h.bb.Replies[0].Text, "Alles klar.") ||
		!strings.Contains(h.bb.Replies[0].Text, "reply **yes** here") {
		t.Fatalf("replies = %+v", h.bb.Replies)
	}
}
