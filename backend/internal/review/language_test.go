package review

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/store"
)

func TestLanguageKeywordIsExact(t *testing.T) {
	for text, want := range map[string]lang.Lang{
		"english": lang.EN, "English!": lang.EN, "  EN ": lang.EN, "auf Englisch": lang.EN,
		"deutsch": lang.DE, "Deutsch.": lang.DE, "de": lang.DE, "in German": lang.DE,
	} {
		if got, ok := languageKeyword(text); !ok || got != want {
			t.Errorf("languageKeyword(%q) = %q, %v, want %q", text, got, ok, want)
		}
	}
	// A question that names a language is a question.
	for _, text := range []string{"", "is this english?", "why is the doc in deutsch", "review", "end", "dem"} {
		if _, ok := languageKeyword(text); ok {
			t.Errorf("languageKeyword(%q) matched, want a question", text)
		}
	}
}

func TestMentionKeywordSetsTheCommentersLanguage(t *testing.T) {
	h := newHarness(t, nil)

	h.r.HandleMention(context.Background(), mentionPayload("@noergler English", "Bob"), "payments", nil)

	if got := h.st.languages["bob"]; got != lang.EN {
		t.Errorf("bob's language = %q, want en", got)
	}
	if len(h.llm.Mentions)+len(h.llm.Reviews) != 0 {
		t.Error("a language switch must not call the model")
	}
	if len(h.bb.Replies) != 1 || !strings.Contains(h.bb.Replies[0].Text, "in English") {
		t.Errorf("replies = %+v, want one English confirmation", h.bb.Replies)
	}

	h.r.HandleMention(context.Background(), mentionPayload("@noergler deutsch", "bob"), "payments", nil)
	if got := h.st.languages["bob"]; got != lang.DE {
		t.Errorf("bob's language = %q, want de", got)
	}
	if len(h.bb.Replies) != 2 || !strings.Contains(h.bb.Replies[1].Text, "auf Deutsch") {
		t.Errorf("replies = %+v, want a German confirmation", h.bb.Replies)
	}
}

// A switch is a setting, not engagement: an author who removed the summary
// keeps the PR ignored.
func TestMentionKeywordLeavesAnIgnoredPRIgnored(t *testing.T) {
	h := newHarness(t, nil)
	now := time.Now()
	h.st.skipState = &store.SkipState{IgnoredAt: &now}

	h.r.HandleMention(context.Background(), mentionPayload("@noergler english", "bob"), "payments", nil)

	if h.st.Reactived != 0 {
		t.Error("a language switch reactivated an ignored PR")
	}
}

func TestFindingReplyKeywordSetsTheLanguageAndStampsNothing(t *testing.T) {
	h, p := replyHarness(t)
	p.Comment.Text = "english"

	handled, _ := h.r.HandleFindingReply(context.Background(), p, "payments", nil)

	if !handled {
		t.Fatal("a language switch under a finding is the reply handler's")
	}
	if got := h.st.languages["alice"]; got != lang.EN {
		t.Errorf("alice's language = %q, want en", got)
	}
	if len(h.llm.Replies) != 0 || len(h.st.Replied) != 0 {
		t.Error("a language switch must not call the model or stamp the finding")
	}
}

// The team default is German; the author's own choice wins over it, and
// the asker's language decides a Q&A answer, not the PR author's.
func TestLanguageResolution(t *testing.T) {
	t.Run("German by default", func(t *testing.T) {
		h := newHarness(t, nil)
		h.r.cfg.Language = ""
		h.r.ReviewPullRequest(context.Background(), prPayload("pr:opened"), false)
		if len(h.bb.Posted) != 1 || !strings.Contains(h.bb.Posted[0].Text, "### Überblick") {
			t.Fatalf("summary not German:\n%+v", h.bb.Posted)
		}
	})
	t.Run("the author's own choice wins", func(t *testing.T) {
		h := newHarness(t, nil)
		h.r.cfg.Language = lang.DE
		h.st.languages = map[string]lang.Lang{"alice": lang.EN}
		h.r.ReviewPullRequest(context.Background(), prPayload("pr:opened"), false)
		if len(h.bb.Posted) != 1 || !strings.Contains(h.bb.Posted[0].Text, "### Overview") {
			t.Fatalf("summary not English:\n%+v", h.bb.Posted)
		}
	})
	t.Run("a DB fault falls back to the team default", func(t *testing.T) {
		h := newHarness(t, nil)
		h.r.cfg.Language = lang.EN
		h.st.failAll = true
		if got := h.r.languageFor(context.Background(), "alice"); got != lang.EN {
			t.Errorf("languageFor = %q, want the team's en", got)
		}
	})
	t.Run("a Q&A answer is the asker's", func(t *testing.T) {
		h := newHarness(t, nil)
		h.r.cfg.Language = lang.DE
		h.st.languages = map[string]lang.Lang{"bob": lang.EN}
		h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK}
		h.r.HandleMention(context.Background(), mentionPayload("@noergler why?", "bob"), "payments", nil)
		if len(h.bb.Replies) != 1 || h.bb.Replies[0].Text != inference.MentionEmptyReply(lang.EN) {
			t.Errorf("replies = %+v, want the English empty-answer reply", h.bb.Replies)
		}
	})
}

func TestGermanAnswersGetSwissSpelling(t *testing.T) {
	h := newHarness(t, nil)
	h.r.cfg.Language = lang.DE
	h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "Das ist grösser, weil `maß` gross ist: Maß."}

	h.r.HandleMention(context.Background(), mentionPayload("@noergler warum?", "carol"), "payments", nil)

	if len(h.bb.Replies) != 1 || h.bb.Replies[0].Text != "Das ist grösser, weil `maß` gross ist: Mass." {
		t.Errorf("replies = %+v", h.bb.Replies)
	}
}
