package review

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/render"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

func TestStopRequest(t *testing.T) {
	cases := []struct {
		text string
		want bool
	}{
		{"stop", true},
		{"Stop!", true},
		{"stopp bitte", true},
		{"stop monitoring", true},
		{"stop reviewing this PR please", true},
		{"Stop watching this PR.", true},
		{"Höre auf, diesen PR zu beobachten", true},
		{"hör auf diesen PR zu prüfen", true},
		{"Bitte diesen PR nicht mehr reviewen", true},
		{"don't review this PR anymore", true},
		{"Reviews stoppen", true},
		{", stop monitoring", true},
		// A question is for the Q&A.
		{"should I stop reviewing this?", false},
		{"kannst du aufhören, diesen PR zu prüfen?", false},
		// Negated.
		{"don't stop reviewing", false},
		{"keep monitoring, do not stop", false},
		{"bitte nicht aufhören zu prüfen", false},
		// About the code, not the bot.
		{"stop the goroutine leak in the handler", false},
		{"why does the loop stop early", false},
		{"stopped reviewing", false},
		{"review", false},
		{"", false},
		// Too long to be a command: a sentence for the model.
		{"stop reviewing is what the CLI does when the flag is set and the env var is missing too", false},
		// ASCII \b would split "stopü".
		{"stopü", false},
		{"das ist halt nicht reviewbar", false},
		// Statements about the code, not orders (review, high).
		{"this no longer checks nil here", false},
		{"warum prüft der Code den Token nicht mehr", false},
		{"Nicht mehr relevant", false},
		{"stop words here", false},
		// A bare "keep" does not negate.
		{"stop reviewing, keep the summary", true},
		{"stop monitoring but keep answering questions", true},
		{"keep reviewing", false},
		{"stop please", true},
		{"bitte stopp", true},
		// Negations in any apostrophe and with other verbs (review, medium).
		{"don’t stop reviewing", false},
		{"please don't pause reviews", false},
		{"Reviews bitte nicht stoppen", false},
		// Code and words far apart are no order.
		{"the review comment about stop() is wrong", false},
		{"the review comment about `stop` is wrong", false},
		{"review again pls, fixed the stop condition", false},
		{"checking quit early here is intended", false},
		{"Keine Reviews mehr bitte", true},
	}
	for _, c := range cases {
		if got := stopRequest(c.text); got != c.want {
			t.Errorf("stopRequest(%q) = %v, want %v", c.text, got, c.want)
		}
	}
}

func TestResumeRequest(t *testing.T) {
	cases := []struct {
		text string
		want bool
	}{
		{"resume", true},
		{"weiter", true},
		{"continue reviewing", true},
		{"resume monitoring this PR", true},
		{"bitte wieder prüfen", true},
		{"Reviews wieder aufnehmen", true},
		{"continue?", false},
		{"does the loop continue after an error", false},
		{"stop", false},
		{"nicht mehr weiter prüfen", false},
		{"weiterleitung", false},
		// A short phrase is a resume only when nothing but filler is
		// beside the verb.
		{"continue explaining", false},
		{"weiter so", false},
		{"wieder kaputt", false},
		{"bitte weiter", true},
		{"continue explaining the review finding on line 4", false},
	}
	for _, c := range cases {
		if got := resumeRequest(c.text); got != c.want {
			t.Errorf("resumeRequest(%q) = %v, want %v", c.text, got, c.want)
		}
	}
}

// The tool's guard needs a stop word and a target in the commenter's text,
// so a question about the code cannot be steered into a stop.
func TestStopIntent(t *testing.T) {
	cases := []struct {
		text string
		want bool
	}{
		{"could you please stop reviewing this one, the spike is throwaway?", true},
		{"kannst du aufhören, diesen PR zu prüfen?", true},
		{"no more reviews on this PR please", true},
		{"why does this loop stop early?", false},
		{"does this still work anymore?", false},
		{"please don't stop reviewing this", false},
	}
	for _, c := range cases {
		if got := stopIntent(c.text); got != c.want {
			t.Errorf("stopIntent(%q) = %v, want %v", c.text, got, c.want)
		}
	}
}

// On an active PR a resume phrase is a question, not a paid review.
func TestResumePhraseOnAnActivePRIsAQuestion(t *testing.T) {
	h := newHarness(t, nil)
	h.r.HandleMention(context.Background(), mentionPayload("@noergler continue", "alice"), "platform", &inlineScheduler{})
	if len(h.llm.Reviews) != 0 || len(h.llm.Mentions) != 1 {
		t.Errorf("reviews %d, mentions %d: want the Q&A", len(h.llm.Reviews), len(h.llm.Mentions))
	}
}

// A cost banner already on the summary stays on top; the next refused
// question replaces it instead of stacking a second one.
func TestStoppedBannerGoesBelowTheCostBanner(t *testing.T) {
	h := stoppedHarness(t)
	h.st.prCost = ptri(6_000_000_000)
	h.bb.setComment(55, render.CostLimitBanner(lang.EN, 6, 5, false)+"\n\n## PR review summary\n\nOverview.", 1)

	h.r.HandleMention(context.Background(), mentionPayload("@noergler stop", "bob"), "platform", &inlineScheduler{})
	body := h.bb.comments[55].Text
	if !strings.HasPrefix(body, render.CostBannerSentinel) || !strings.Contains(body, render.StoppedBannerSentinel) {
		t.Fatalf("summary body:\n%s", body)
	}
	// Two refused questions at the cap: still one cost banner. once=true
	// leaves it as it is, and an edit must not stack another.
	for range 2 {
		h.r.HandleMention(context.Background(), mentionPayload("@noergler what is this?", "alice"), "platform", &inlineScheduler{})
		h.r.costLimitNotice(context.Background(), lang.EN, "PROJ", "my-repo", 42, 1, 7, 5, false, false)
	}
	body = h.bb.comments[55].Text
	if n := strings.Count(body, render.CostBannerSentinel); n != 1 {
		t.Errorf("%d cost banners:\n%s", n, body)
	}
	if n := strings.Count(body, render.StoppedBannerSentinel); n != 1 {
		t.Errorf("%d stopped banners:\n%s", n, body)
	}
}

// A stopped PR whose summary was deleted without the webhook resumes as the
// deleted case: the review posts a fresh summary instead of being skipped.
func TestResumeAfterTheSummaryWasDeleted(t *testing.T) {
	h := stoppedHarness(t)
	h.r.HandleMention(context.Background(), mentionPayload("@noergler stop", "bob"), "platform", &inlineScheduler{})
	delete(h.bb.comments, 55)

	h.r.HandleMention(context.Background(), mentionPayload("@noergler review", "alice"), "platform", &inlineScheduler{})

	if h.st.Ignored != 1 || len(h.llm.Reviews) != 1 || len(h.bb.Posted) != 1 {
		t.Errorf("ignored %d, reviews %d, posted %d: want the deletion seen and a fresh summary",
			h.st.Ignored, len(h.llm.Reviews), len(h.bb.Posted))
	}
}

// stoppedHarness is a PR reviewed once: a row, a summary on Bitbucket.
func stoppedHarness(t *testing.T) *harness {
	t.Helper()
	h := newHarness(t, nil)
	summary := &store.SummaryComment{ID: 55, Version: 1}
	h.st.summary = summary
	h.st.skipState = &store.SkipState{ID: 1, Summary: summary}
	h.bb.setComment(55, "## PR review summary\n\nOverview.", 1)
	return h
}

// A stop is decided by code: the state, the banner on the summary, the
// confirmation. No model call, no attempt.
func TestMentionStopStopsTheReviews(t *testing.T) {
	h := stoppedHarness(t)

	h.r.HandleMention(context.Background(), mentionPayload("@noergler Höre auf, diesen PR zu beobachten", "bob"), "platform", &inlineScheduler{})

	if len(h.llm.Mentions)+len(h.llm.Reviews) != 0 {
		t.Error("a stop must not call the model")
	}
	if !h.st.skipState.Stopped() || h.st.skipState.StoppedBy != "bob" {
		t.Fatalf("skip state = %+v, want stopped by bob", h.st.skipState)
	}
	if len(h.st.Attempts) != 0 {
		t.Errorf("attempts = %+v, a stop records none", h.st.Attempts)
	}
	body := h.bb.comments[55].Text
	if !strings.HasPrefix(body, render.StoppedBannerSentinel) || !strings.Contains(body, "`bob`") ||
		!strings.Contains(body, "Overview.") {
		t.Errorf("summary body:\n%s", body)
	}
	if len(h.bb.Replies) != 1 || !strings.Contains(h.bb.Replies[0].Text, "no longer review this PR") {
		t.Errorf("replies = %+v", h.bb.Replies)
	}

	// A push on the stopped PR is skipped as pr_stopped, without reading the
	// summary back.
	h.bb.FetchedC = nil
	h.r.reviewInline(context.Background(), prPayload(webhook.EventFromRefUpdated), false)
	if len(h.llm.Reviews) != 0 {
		t.Error("a stopped PR was reviewed")
	}
	if len(h.st.Attempts) != 1 || h.st.Attempts[0].Reason != string(SkipStoppedPR) {
		t.Errorf("attempts = %+v, want one pr_stopped", h.st.Attempts)
	}
	if len(h.bb.FetchedC) != 0 {
		t.Errorf("fetched comments %v: a stopped PR's summary is alive by design", h.bb.FetchedC)
	}
}

// A stop costs nothing, so it works at the cost cap.
func TestMentionStopWorksAtTheCap(t *testing.T) {
	h := stoppedHarness(t)
	h.st.prCost = ptri(6_000_000_000)

	h.r.HandleMention(context.Background(), mentionPayload("@noergler stop reviewing", "bob"), "platform", &inlineScheduler{})

	if !h.st.skipState.Stopped() {
		t.Errorf("skip state = %+v, want stopped", h.st.skipState)
	}
}

// Stopping twice says so and leaves one banner.
func TestMentionStopTwice(t *testing.T) {
	h := stoppedHarness(t)
	for range 2 {
		h.r.HandleMention(context.Background(), mentionPayload("@noergler stop", "bob"), "platform", &inlineScheduler{})
	}
	if n := strings.Count(h.bb.comments[55].Text, render.StoppedBannerSentinel); n != 1 {
		t.Errorf("%d banners", n)
	}
	if len(h.bb.Replies) != 2 || !strings.Contains(h.bb.Replies[1].Text, "already stopped") {
		t.Errorf("replies = %+v", h.bb.Replies)
	}
}

// A PR never reviewed gets its row, so the stop has somewhere to live.
func TestMentionStopOnAPRNeverSeen(t *testing.T) {
	h := newHarness(t, nil)

	h.r.HandleMention(context.Background(), mentionPayload("@noergler stop monitoring", "bob"), "platform", &inlineScheduler{})

	if len(h.st.Upserts) != 1 || !h.st.Upserts[0].KeepPointer || !h.st.skipState.Stopped() {
		t.Errorf("upserts = %+v, state = %+v", h.st.Upserts, h.st.skipState)
	}
	if len(h.bb.Posted) != 0 {
		t.Error("no summary: the stop must not post one")
	}
}

// A question on a stopped PR is answered; the PR stays stopped.
func TestQuestionOnAStoppedPRKeepsItStopped(t *testing.T) {
	h := stoppedHarness(t)
	h.r.HandleMention(context.Background(), mentionPayload("@noergler stop", "bob"), "platform", &inlineScheduler{})
	h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "It locks the map."}

	h.r.HandleMention(context.Background(), mentionPayload("@noergler is the lock held?", "alice"), "platform", &inlineScheduler{})

	if len(h.llm.Mentions) != 1 {
		t.Fatalf("mention calls = %d, want the question answered", len(h.llm.Mentions))
	}
	if h.st.Reactived != 0 || !h.st.skipState.Stopped() {
		t.Errorf("reactivated %d, state %+v: a question must not resume", h.st.Reactived, h.st.skipState)
	}
}

// A review or resume request brings the reviews back: the banner goes, the
// summary is updated in place, not posted fresh.
func TestResumeReviewsAStoppedPR(t *testing.T) {
	for _, text := range []string{"@noergler review", "@noergler weiter beobachten"} {
		t.Run(text, func(t *testing.T) {
			h := stoppedHarness(t)
			h.r.HandleMention(context.Background(), mentionPayload("@noergler stop", "bob"), "platform", &inlineScheduler{})

			h.r.HandleMention(context.Background(), mentionPayload(text, "alice"), "platform", &inlineScheduler{})

			if h.st.Reactived != 1 || h.st.skipState.IgnoredAt != nil {
				t.Errorf("reactivated %d, state %+v", h.st.Reactived, h.st.skipState)
			}
			if len(h.llm.Reviews) != 1 {
				t.Fatalf("reviews = %d, want one", len(h.llm.Reviews))
			}
			if len(h.bb.Posted) != 0 {
				t.Errorf("posted %d fresh comments: a resumed PR keeps its summary", len(h.bb.Posted))
			}
			if strings.Contains(h.bb.comments[55].Text, render.StoppedBannerSentinel) {
				t.Errorf("banner still on the summary:\n%s", h.bb.comments[55].Text)
			}
		})
	}
}

// A stop on a PR ignored after its summary was deleted is stored: the next
// question must not bring it back against the stop. The dead summary is
// forgotten, so a later resume posts a fresh one.
func TestStopOnADeletedSummaryIsStored(t *testing.T) {
	h := newHarness(t, nil)
	now := time.Now()
	h.st.summary = &store.SummaryComment{ID: 55, Version: 1}
	h.st.skipState = &store.SkipState{ID: 1, IgnoredAt: &now, IgnoreReason: store.IgnoreSummaryDeleted, Summary: h.st.summary}

	h.r.HandleMention(context.Background(), mentionPayload("@noergler stop", "bob"), "platform", &inlineScheduler{})
	h.r.HandleMention(context.Background(), mentionPayload("@noergler what is this?", "alice"), "platform", &inlineScheduler{})

	if !h.st.skipState.Stopped() || h.st.skipState.Summary != nil || h.st.Reactived != 0 {
		t.Errorf("state = %+v, reactivated %d", h.st.skipState, h.st.Reactived)
	}
	if len(h.bb.Replies) == 0 || !strings.Contains(h.bb.Replies[0].Text, "no longer review this PR") {
		t.Errorf("replies = %+v", h.bb.Replies)
	}
}

// "@bot stop reviewing" under a finding is the stop, not a dispute.
func TestFindingThreadStopIsNotADispute(t *testing.T) {
	h, p := replyHarness(t)
	p.Comment.Text = "@noergler stop reviewing this PR"

	handled, handedOff := h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

	if !handled || handedOff {
		t.Errorf("handled %v, handed off %v", handled, handedOff)
	}
	if len(h.llm.Replies) != 0 || len(h.st.Replied) != 0 {
		t.Error("a stop was answered as a finding reply")
	}
	if !h.st.skipState.Stopped() || h.st.skipState.StoppedBy != "alice" {
		t.Errorf("state = %+v, want stopped by alice", h.st.skipState)
	}
}

// The stop is about reviews: the author's plain reply on a finding, and a
// plain reply under a bot answer, are still answered on a stopped PR.
func TestStoppedPRStillAnswersReplies(t *testing.T) {
	now := time.Now()
	stopped := &store.SkipState{ID: 1, IgnoredAt: &now, IgnoreReason: store.IgnoreStopped, StoppedBy: "bob"}

	h, p := replyHarness(t)
	h.st.skipState = stopped
	h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})
	if len(h.llm.Replies) != 1 {
		t.Errorf("finding reply calls = %d, want the reply answered", len(h.llm.Replies))
	}

	h = newHarness(t, nil)
	h.st.skipState = stopped
	h.st.botReplies = map[int]*store.BotReply{600: {CommentID: 600, Kind: store.BotReplyQuestion}}
	h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "Yes."}
	q := mentionPayload("does this also stop the Almanach send", "carol")
	q.CommentParentID = 600
	h.r.HandleFindingReply(context.Background(), q, "payments", &inlineScheduler{})
	if len(h.llm.Mentions) != 1 || !h.st.skipState.Stopped() {
		t.Errorf("mention calls = %d, state %+v: want answered, still stopped", len(h.llm.Mentions), h.st.skipState)
	}
}

// The tool stops only when the commenter's own words say stop, and the
// posting stage confirms it before the answer.
func TestStopReviewingTool(t *testing.T) {
	t.Run("needs the commenter's words", func(t *testing.T) {
		h := newHarness(t, nil)
		ask := testAsk("Dana", "what does this function do")
		out := toolByName(t, h, ask, "stop_reviewing")("{}")
		if ask.stop || !strings.Contains(out, "rejected") {
			t.Errorf("stop %v: %s", ask.stop, out)
		}
	})
	t.Run("stops and confirms", func(t *testing.T) {
		h := stoppedHarness(t)
		h.llm.onMention = func(_ context.Context, _ []string, run func(name, args string) string) string {
			run("stop_reviewing", "{}")
			return "Done."
		}
		h.r.HandleMention(context.Background(),
			mentionPayload("@noergler could you please stop reviewing this one, the spike is throwaway?", "bob"),
			"platform", &inlineScheduler{})

		if !h.st.skipState.Stopped() {
			t.Fatalf("state = %+v", h.st.skipState)
		}
		if len(h.bb.Replies) != 1 || !strings.Contains(h.bb.Replies[0].Text, "no longer review this PR") ||
			!strings.HasSuffix(h.bb.Replies[0].Text, "Done.") {
			t.Errorf("replies = %+v", h.bb.Replies)
		}
	})
}
