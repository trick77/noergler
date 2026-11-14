package review

import (
	"context"
	"errors"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/render"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// offerReview runs one review of the conflict repo's PR (feature/thing
// into master) with the given heads.
func (h *harness) offerReview(t *testing.T, sourceHead, targetHead string) {
	t.Helper()
	h.llm.review = okResultWith()
	// The source head's files, as the harness serves src1's.
	h.bb.files[sourceHead+":AGENTS.md"] = h.bb.files["src1:AGENTS.md"]
	h.bb.files[sourceHead+":a.go"] = h.bb.files["src1:a.go"]
	p := prPayload(webhook.EventFromRefUpdated)
	p.PullRequest.FromRef.LatestCommit, p.PullRequest.ToRef.LatestCommit = sourceHead, targetHead
	h.r.reviewInline(context.Background(), p, false)
}

// offers is the offer comments posted, the summary left out.
func (h *harness) offers() []string {
	var out []string
	for _, c := range h.bb.Posted {
		if strings.Contains(c.Text, "merge conflict with") {
			out = append(out, c.Text)
		}
	}
	return out
}

// A conflict code resolves is offered in a comment of its own that
// mentions the author: no model, no push, no fix row.
func TestConflictOfferByCode(t *testing.T) {
	repo := newConflictRepo(t, map[string]string{"l": "a\n"}, map[string]string{"l": "a\nt\n"}, map[string]string{"l": "a\ns\n"})
	before := repo.run(repo.bare, "rev-parse", "feature/thing")
	h := conflictHarness(t, repo)

	h.offerReview(t, "src1", "tgt1")

	offers := h.offers()
	if len(offers) != 1 {
		t.Fatalf("offers = %q", offers)
	}
	for _, want := range []string{"@alice ", "`master`", "without the model", "`@noergler resolve the merge conflict`"} {
		if !strings.Contains(offers[0], want) {
			t.Errorf("offer lacks %q:\n%s", want, offers[0])
		}
	}
	if h.st.conflictOffer == 0 {
		t.Error("the open offer was not stored")
	}
	if len(h.llm.Conflicts) != 0 || len(h.st.Fixes) != 0 {
		t.Errorf("model calls = %d, fixes = %+v", len(h.llm.Conflicts), h.st.Fixes)
	}
	if got := repo.run(repo.bare, "rev-parse", "feature/thing"); got != before {
		t.Fatal("an offer pushed")
	}
}

// A conflict the model must answer is offered as a try; the model is not
// asked.
func TestConflictOfferByModel(t *testing.T) {
	repo := newConflictRepo(t, map[string]string{"a.go": "x := 1\n"}, map[string]string{"a.go": "x := 2\n"}, map[string]string{"a.go": "x := 3\n"})
	h := conflictHarness(t, repo)

	h.offerReview(t, "src1", "tgt1")

	if offers := h.offers(); len(offers) != 1 || !strings.Contains(offers[0], "simple enough for me to try") {
		t.Fatalf("offers = %q", offers)
	}
	if len(h.llm.Conflicts) != 0 {
		t.Error("an offer asked the model")
	}
}

// Past merge's bounds nothing is offered: Bitbucket shows the conflict.
func TestConflictOfferNoneForALockfile(t *testing.T) {
	repo := newConflictRepo(t, map[string]string{"go.sum": "a\n"}, map[string]string{"go.sum": "b\n"}, map[string]string{"go.sum": "c\n"})
	h := conflictHarness(t, repo)
	h.offerReview(t, "src1", "tgt1")
	if offers := h.offers(); len(offers) != 0 || h.st.conflictOffer != 0 {
		t.Fatalf("offers = %q, open = %d", offers, h.st.conflictOffer)
	}
}

// A fork's PR is never offered, conflict or not.
func TestConflictOfferNoneForAFork(t *testing.T) {
	repo := newConflictRepo(t, map[string]string{"l": "a\n"}, map[string]string{"l": "a\nt\n"}, map[string]string{"l": "a\ns\n"})
	h := conflictHarness(t, repo)
	h.llm.review = okResultWith()
	p := prPayload(webhook.EventOpened)
	p.PullRequest.FromRef.Repository = &webhook.Repository{Slug: "my-repo", Project: webhook.Project{Key: "~ALICE"}}
	h.r.reviewInline(context.Background(), p, false)
	if offers := h.offers(); len(offers) != 0 {
		t.Fatalf("offers = %q", offers)
	}
}

// One open offer at a time: a later review with the conflict still there
// posts none, and on the same heads fetches nothing again.
func TestConflictOfferOnceWhileOpen(t *testing.T) {
	repo := newConflictRepo(t, map[string]string{"l": "a\n"}, map[string]string{"l": "a\nt\n"}, map[string]string{"l": "a\ns\n"})
	h := conflictHarness(t, repo)
	h.offerReview(t, "src1", "tgt1")
	h.offerReview(t, "src2", "tgt1")
	if offers := h.offers(); len(offers) != 1 {
		t.Fatalf("offers = %d, want 1 while it is open", len(offers))
	}

	// Same heads, no fetch: with the remote gone and the offer cleared, the
	// remembered check still offers.
	h.bb.remoteErr = errors.New("no fetch expected")
	h.st.conflictOffer = 0
	h.offerReview(t, "src2", "tgt1")
	if offers := h.offers(); len(offers) != 2 {
		t.Fatalf("offers = %d: the same heads were checked again", len(offers))
	}
}

// The conflict is gone: the open offer is resolved and cleared. A conflict
// later is offered anew.
func TestConflictOfferResolvedThenOfferedAgain(t *testing.T) {
	repo := newConflictRepo(t, map[string]string{"l": "a\n"}, map[string]string{"l": "a\nt\n"}, map[string]string{"l": "a\ns\n"})
	h := conflictHarness(t, repo)
	h.offerReview(t, "src1", "tgt1")
	first := h.st.conflictOffer

	h.bb.mergeStatus = bitbucket.MergeStatus{CanMerge: true}
	h.offerReview(t, "src2", "tgt1")
	if h.st.conflictOffer != 0 || !slices.Contains(h.bb.Calls, "resolve:"+strconv.Itoa(first)) {
		t.Fatalf("open = %d, calls = %v", h.st.conflictOffer, h.bb.Calls)
	}

	h.bb.mergeStatus = bitbucket.MergeStatus{Conflicted: true}
	h.offerReview(t, "src3", "tgt2")
	if offers := h.offers(); len(offers) != 2 || h.st.conflictOffer == 0 || h.st.conflictOffer == first {
		t.Fatalf("offers = %d, open = %d (first %d)", len(offers), h.st.conflictOffer, first)
	}
}

// An offer a human deleted is cleared; a failed resolve keeps it for the
// next review.
func TestConflictOfferResolveErrors(t *testing.T) {
	t.Run("deleted", func(t *testing.T) {
		h := newHarness(t, nil)
		h.st.conflictOffer = 555
		h.bb.resolveErr = &bitbucket.StatusError{Method: "GET", Path: "/comments", Status: 404}
		h.offerReview(t, "src1", "tgt1")
		if h.st.conflictOffer != 0 {
			t.Fatalf("open = %d, want cleared", h.st.conflictOffer)
		}
	})
	t.Run("failed", func(t *testing.T) {
		h := newHarness(t, nil)
		h.st.conflictOffer = 555
		h.bb.resolveErr = &bitbucket.StatusError{Method: "PUT", Path: "/comments", Status: 500}
		h.offerReview(t, "src1", "tgt1")
		if h.st.conflictOffer != 555 {
			t.Fatalf("open = %d, want kept", h.st.conflictOffer)
		}
	})
}

// The review that takes the PR to its cost cap checks nothing (user,
// 2026-11-14: stop there): no fetch, no offer, an open one withdrawn, since
// a model resolution would be refused at the cap.
func TestConflictOfferStopsAtTheCap(t *testing.T) {
	repo := newConflictRepo(t, map[string]string{"l": "a\n"}, map[string]string{"l": "a\nt\n"}, map[string]string{"l": "a\ns\n"})
	h := conflictHarness(t, repo)
	h.bb.remoteErr = errors.New("no fetch expected")
	h.st.prCost = ptri(4_950_000_000) // the run's $0.12 takes it past $5
	h.st.conflictOffer = 555
	h.offerReview(t, "src1", "tgt1")
	if len(h.st.Runs) != 1 {
		t.Fatalf("runs = %d: the review itself must run", len(h.st.Runs))
	}
	if offers := h.offers(); len(offers) != 0 || h.st.conflictOffer != 0 || !slices.Contains(h.bb.Calls, "resolve:555") {
		t.Fatalf("offers = %q, open = %d, calls = %v", offers, h.st.conflictOffer, h.bb.Calls)
	}
}

// A check that fails (merge status, fetch) or finds another one fetching
// is unknown: it leaves the open offer alone (read as none, it would be
// withdrawn and posted again next time) and posts none.
func TestConflictOfferUnknownKeepsTheOpenOffer(t *testing.T) {
	for name, tweak := range map[string]func(*testing.T, *harness){
		"merge status": func(_ *testing.T, h *harness) { h.bb.mergeErr = errors.New("bitbucket down") },
		"fetch":        func(_ *testing.T, h *harness) { h.bb.remoteErr = errors.New("no clone URL") },
		"busy": func(t *testing.T, _ *harness) {
			offerSlot <- struct{}{}
			t.Cleanup(func() { <-offerSlot })
		},
	} {
		t.Run(name, func(t *testing.T) {
			repo := newConflictRepo(t, map[string]string{"l": "a\n"}, map[string]string{"l": "a\nt\n"}, map[string]string{"l": "a\ns\n"})
			t.Run("open", func(t *testing.T) {
				h := conflictHarness(t, repo)
				tweak(t, h)
				h.st.conflictOffer = 555
				h.offerReview(t, "src1", "tgt1")
				if h.st.conflictOffer != 555 || slices.Contains(h.bb.Calls, "resolve:555") {
					t.Fatalf("open = %d, calls = %v", h.st.conflictOffer, h.bb.Calls)
				}
			})
			// None open: a conflict code resolves is not offered while the
			// check cannot run (the repo would offer it otherwise).
			t.Run("none open", func(t *testing.T) {
				h := conflictHarness(t, repo)
				tweak(t, h)
				h.offerReview(t, "src1", "tgt1")
				if offers := h.offers(); len(offers) != 0 {
					t.Fatalf("offers = %q", offers)
				}
			})
		})
	}
}

// The offer is a notice the bot posted: a plain reply under it reaches the
// bot.
func TestConflictOfferIsABotReply(t *testing.T) {
	repo := newConflictRepo(t, map[string]string{"l": "a\n"}, map[string]string{"l": "a\nt\n"}, map[string]string{"l": "a\ns\n"})
	h := conflictHarness(t, repo)
	h.offerReview(t, "src1", "tgt1")
	found := false
	for _, b := range h.st.BotReplies {
		if b.CommentID == h.st.conflictOffer && b.Kind == store.BotReplyQuestion {
			found = true
		}
	}
	if !found {
		t.Fatalf("offer %d not recorded as a bot reply: %+v", h.st.conflictOffer, h.st.BotReplies)
	}
}

// noergler's own push of the merge withdraws the open offer at once.
func TestConflictOfferWithdrawnWhenResolved(t *testing.T) {
	repo := newConflictRepo(t, map[string]string{"l": "a\n"}, map[string]string{"l": "a\nt\n"}, map[string]string{"l": "a\ns\n"})
	h := conflictHarness(t, repo)
	h.offerReview(t, "src1", "tgt1")
	open := h.st.conflictOffer
	h.askConflict(t, "alice")
	if len(h.st.Fixes) != 1 || h.st.Fixes[0].Outcome != store.FixCommitted {
		t.Fatalf("fixes = %+v", h.st.Fixes)
	}
	if h.st.conflictOffer != 0 || !slices.Contains(h.bb.Calls, "resolve:"+strconv.Itoa(open)) {
		t.Fatalf("open = %d, calls = %v", h.st.conflictOffer, h.bb.Calls)
	}
}

// The command an offer quotes is one the request recognises.
func TestConflictOfferCommandIsRecognised(t *testing.T) {
	for _, cmd := range []string{render.ConflictCommandDE, render.ConflictCommandEN} {
		if !conflictRequest(cmd) {
			t.Errorf("conflictRequest(%q) = false", cmd)
		}
	}
}
