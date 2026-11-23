package review

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/render"
	"github.com/trick77/noergler/internal/store"
)

// The staleness banner on a summary the other bot posted: the edit is
// refused (403), so the bannered summary is posted fresh and tracked.
func TestFailureNoticeRepostsAnotherAccountsSummary(t *testing.T) {
	h := newHarness(t, func(h *harness) {
		h.st.summary = &store.SummaryComment{ID: 55, Version: 3}
		h.bb.setComment(55, "the previous summary", 3)
		h.bb.updateErr = &bitbucket.StatusError{Status: 403}
	})

	h.r.failureNotice(context.Background(), "PROJ", "my-repo", 42, 7, "fresh", "stale banner", "timeout notice")

	if len(h.bb.Posted) != 1 || !strings.Contains(h.bb.Posted[0].Text, "the previous summary") {
		t.Fatalf("posted = %+v, want the bannered summary once", h.bb.Posted)
	}
	if n := len(h.st.Summaries); n == 0 || h.st.Summaries[n-1].ID == 55 {
		t.Errorf("the fresh comment is not tracked: %+v", h.st.Summaries)
	}
}

// The cost-limit banner on the other bot's summary goes up fresh too.
func TestCostLimitNoticeRepostsAnotherAccountsSummary(t *testing.T) {
	h := newHarness(t, func(h *harness) {
		h.st.summary = &store.SummaryComment{ID: 55, Version: 3}
		h.bb.setComment(55, "the previous summary", 3)
		h.bb.updateErr = &bitbucket.StatusError{Status: 403}
	})

	h.r.costLimitNotice(context.Background(), lang.Default, "PROJ", "my-repo", 42, 7, 6, 5, true, false)

	if len(h.bb.Posted) != 1 || !strings.Contains(h.bb.Posted[0].Text, "the previous summary") {
		t.Fatalf("posted = %+v, want the bannered summary once", h.bb.Posted)
	}
}

// The fresh "Review skipped" notice is wrapped; the banner put on an existing
// summary stays one row, which is what its stripper matches.
func TestFailureNoticeIsWrapped(t *testing.T) {
	for _, l := range []lang.Lang{lang.EN, lang.DE} {
		for _, o := range []inference.Outcome{inference.OutcomeTimedOut, inference.OutcomeUnparseable, inference.OutcomeTooLarge} {
			h := newHarness(t, func(*harness) {})
			h.r.failureNoticeFor(context.Background(), o, 5*time.Minute, l, "PROJ", "my-repo", 42, 7, "abcdef0123", "")
			if len(h.bb.Posted) != 1 {
				t.Fatalf("%s %s: posted %d comments, want 1", l, o, len(h.bb.Posted))
			}
			lines := strings.Split(h.bb.Posted[0].Text, "\n")
			if len(lines) < 2 {
				t.Errorf("%s %s: not wrapped: %q", l, o, h.bb.Posted[0].Text)
			}
			for i, line := range lines {
				if n := utf8.RuneCountInString(line); n > render.WrapWidth {
					t.Errorf("%s %s: line of %d columns: %q", l, o, n, line)
				}
				if i > 0 && strings.HasPrefix(line, "- ") {
					t.Errorf("%s %s: a wrapped line reads as a list item: %q", l, o, line)
				}
			}

			h = newHarness(t, func(h *harness) {
				h.st.summary = &store.SummaryComment{ID: 55, Version: 3}
				h.bb.setComment(55, "the previous summary", 3)
			})
			h.r.failureNoticeFor(context.Background(), o, 5*time.Minute, l, "PROJ", "my-repo", 42, 7, "abcdef0123", "0123456789")
			if len(h.bb.Updates) != 1 {
				t.Fatalf("%s %s: updated %d comments, want 1", l, o, len(h.bb.Updates))
			}
			if got := strings.Split(h.bb.Updates[0].Text, "\n"); len(got) != 4 || got[3] != "the previous summary" {
				t.Errorf("%s %s: the banner is not one row: %q", l, o, h.bb.Updates[0].Text)
			}
		}
	}
}

// A cost notice that is its own comment is wrapped, and the wrapped one is
// still read as blocked: the next blocked push leaves it alone.
func TestCostLimitNoticeFreshIsWrapped(t *testing.T) {
	for _, l := range []lang.Lang{lang.EN, lang.DE} {
		h := newHarness(t, func(*harness) {})
		h.r.costLimitNotice(context.Background(), l, "PROJ", "my-repo", 42, 7, 6, 5, true, false)
		if len(h.bb.Posted) != 1 {
			t.Fatalf("%s: posted %d comments, want 1", l, len(h.bb.Posted))
		}
		text := h.bb.Posted[0].Text
		for _, line := range strings.Split(text, "\n") {
			if n := utf8.RuneCountInString(line); n > render.WrapWidth {
				t.Errorf("%s: line of %d columns: %q", l, n, line)
			}
		}
		if !render.CostBannerSaysBlocked(text) {
			t.Errorf("%s: the wrapped banner does not read as blocked: %q", l, text)
		}
	}
}

// Any other refusal still gives up without a duplicate.
func TestFailureNoticeGivesUpOnOtherErrors(t *testing.T) {
	h := newHarness(t, func(h *harness) {
		h.st.summary = &store.SummaryComment{ID: 55, Version: 3}
		h.bb.setComment(55, "the previous summary", 3)
		h.bb.updateErr = &bitbucket.StatusError{Status: 500}
	})
	h.r.failureNotice(context.Background(), "PROJ", "my-repo", 42, 7, "fresh", "stale banner", "timeout notice")
	if len(h.bb.Posted) != 0 {
		t.Errorf("posted %d comments on a 500", len(h.bb.Posted))
	}
}

// A team that switched bots finds its summary written by the other account,
// which Bitbucket refuses to let the new one edit (403): post a fresh one.
func TestShouldRepost(t *testing.T) {
	for _, tc := range []struct {
		err  error
		want bool
	}{
		{bitbucket.ErrVersionConflict, true},
		{&bitbucket.StatusError{Status: 404}, true},
		{&bitbucket.StatusError{Status: 403}, true},
		{&bitbucket.StatusError{Status: 500}, false},
		{&bitbucket.StatusError{Status: 401}, false},
		{errors.New("connection reset"), false},
	} {
		if got := shouldRepost(tc.err); got != tc.want {
			t.Errorf("shouldRepost(%v) = %v, want %v", tc.err, got, tc.want)
		}
	}
}
