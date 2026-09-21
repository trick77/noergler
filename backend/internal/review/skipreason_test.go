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
// an author that was in ignore_authors. Divergence from Python, on purpose.
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
func TestIgnoredAuthorIsNotSkippedOnMention(t *testing.T) {
	r, buf := capturingReviewer(t)
	r.SetAuthorLists(nil, []string{"renovate_diecibaerg"})

	// skipAuthorCheck = true. The review will fail later for want of
	// upstreams; all that matters is that it did not stop at the author gate.
	func() {
		defer func() { _ = recover() }()
		r.ReviewPullRequest(context.Background(), payloadBy("renovate_diecibaerg"), true)
	}()

	if out := buf.String(); strings.Contains(out, "Skipping") {
		t.Errorf("an @mention must bypass the author gate, got %q", out)
	}
}
