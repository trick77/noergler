package review

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/lang"
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
