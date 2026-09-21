package review

import (
	"context"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/webhook"
)

// payloadBy builds the smallest payload that reaches the author gate.
func payloadBy(author string) *webhook.Payload {
	repo := &webhook.Repository{
		Slug:    "m2m-service",
		Project: webhook.Project{Key: "M2M_BETRIEBE"},
	}
	p := &webhook.Payload{EventKey: "pr:opened"}
	p.PullRequest.ID = 79
	p.PullRequest.Author.User.Name = author
	p.PullRequest.FromRef.Repository = repo
	p.PullRequest.ToRef.Repository = repo
	return p
}

// The ignore list wins over the allow list inside IsAutoReviewAuthor, so the
// caller gets a bare false and cannot say which list decided. Python reported
// every skip as an allow-list miss, which misstates why an ignored bot was
// skipped: prod logged `renovate_diecibaerg (not in auto-review authors)` for
// an author that was in ignore_authors.
//
// The gate returns before any HTTP or store call, so a bare Reviewer with no
// upstreams wired is enough to drive it.
func TestSkipReasonNamesTheListThatDecided(t *testing.T) {
	cases := []struct {
		name     string
		auto     []string
		ignore   []string
		author   string
		want     string
		unwanted string
	}{
		{
			name:     "an ignored author is reported as ignored",
			ignore:   []string{"renovate_diecibaerg"},
			author:   "renovate_diecibaerg",
			want:     "(ignored author)",
			unwanted: "not in auto-review authors",
		},
		{
			name:     "ignore still wins when an allow list exists",
			auto:     []string{"alice", "renovate_diecibaerg"},
			ignore:   []string{"renovate_diecibaerg"},
			author:   "renovate_diecibaerg",
			want:     "(ignored author)",
			unwanted: "not in auto-review authors",
		},
		{
			name:     "an author merely absent from the allow list keeps the old wording",
			auto:     []string{"alice"},
			author:   "bob",
			want:     "(not in auto-review authors)",
			unwanted: "ignored author",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r, buf := capturingReviewer(t)
			r.SetAuthorLists(tc.auto, tc.ignore)

			r.ReviewPullRequest(context.Background(), payloadBy(tc.author), false)

			out := buf.String()
			if !strings.Contains(out, tc.want) {
				t.Errorf("log %q does not contain %q", out, tc.want)
			}
			if strings.Contains(out, tc.unwanted) {
				t.Errorf("log %q should not contain %q", out, tc.unwanted)
			}
			// The author and the PR tag stay in the line either way.
			if !strings.Contains(out, tc.author) {
				t.Errorf("log %q does not name the author", out)
			}
			if !strings.Contains(out, "M2M_BETRIEBE/m2m-service#79") {
				t.Errorf("log %q does not carry the pr tag", out)
			}
		})
	}
}

// An @mention bypasses the gate entirely, so an ignored bot's PR can still be
// reviewed on request. Nothing about naming the ignore list may change that.
//
// Asserted on the decision rather than by driving ReviewPullRequest: with
// skipAuthorCheck the run continues past the gate into the store and
// Bitbucket, which a bare Reviewer has not got, and a test that swallowed the
// resulting panic would pass whether or not the gate had been consulted.
func TestIgnoredAuthorIsStillReviewableOnMention(t *testing.T) {
	r, _ := capturingReviewer(t)
	r.SetAuthorLists(nil, []string{"renovate_diecibaerg"})

	autoReview, ignored := r.autoReviewDecision("renovate_diecibaerg")
	if autoReview || !ignored {
		t.Fatalf("autoReviewDecision = (%v, %v), want (false, true)", autoReview, ignored)
	}
}

// The decision and its reason must come from ONE snapshot. Asking
// IsAutoReviewAuthor and then isIgnoredAuthor took two, so a settings write
// landing between them could report a reason the decision never used.
func TestDecisionAndReasonComeFromOneSnapshot(t *testing.T) {
	r, _ := capturingReviewer(t)
	r.SetAuthorLists(nil, []string{"bot"})

	// Two states that DISAGREE about bot, so a decision and a reason taken
	// from different snapshots are detectable:
	//   A: auto=[bot], ignore=[]   -> autoReview=true,  ignored=false
	//   B: auto=[],    ignore=[bot] -> autoReview=false, ignored=true
	// Any other pairing means the two reads straddled a write. The old code
	// (IsAutoReviewAuthor then isIgnoredAuthor) could return (false, false)
	// when B was replaced by A between the calls.
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 20000; i++ {
			if i%2 == 0 {
				r.SetAuthorLists([]string{"bot"}, nil)
			} else {
				r.SetAuthorLists(nil, []string{"bot"})
			}
		}
	}()

	for i := 0; i < 20000; i++ {
		autoReview, ignored := r.autoReviewDecision("bot")
		if autoReview == ignored {
			t.Fatalf("decision and reason disagree: autoReview=%v ignored=%v; "+
				"the two values came from different snapshots", autoReview, ignored)
		}
	}
	<-done
}
