package review

import (
	"context"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/webhook"
)

// The footer links the run's dashboard page, keyed by the attempt the run
// just recorded, and prints no link without a public URL: never one to a
// host that does not exist.
func TestFooterLinksTheRunPage(t *testing.T) {
	h := newHarness(t, nil)
	h.r.publicURL = "https://noergler.example.com"
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)
	if len(h.bb.Posted) == 0 {
		t.Fatal("no summary posted")
	}
	want := "https://noergler.example.com/runs/1)"
	if s := h.bb.Posted[0].Text; !strings.Contains(s, want) {
		t.Errorf("summary lacks the run link %q:\n%s", want, s)
	}

	h2 := newHarness(t, nil)
	h2.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)
	if s := h2.bb.Posted[0].Text; strings.Contains(s, "/runs/") {
		t.Errorf("no public URL, yet a run link:\n%s", s)
	}
}

// A file the size gate kept out is named in the footer. SkippedFiles was
// never filled before, so this line could not appear in production.
func TestFooterNamesTheOversizedFile(t *testing.T) {
	h := newHarness(t, nil)
	var props strings.Builder
	props.WriteString("diff --git a/messages.properties b/messages.properties\n--- a/messages.properties\n+++ b/messages.properties\n@@ -0,0 +1,60 @@\n")
	for i := 0; i < 60; i++ {
		props.WriteString("+key.x=value\n")
	}
	h.bb.prDiff = sampleDiff + props.String()
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)
	s := h.bb.Posted[0].Text
	if !strings.Contains(s, "Not reviewed (too large): `messages.properties` ⚠️") {
		t.Errorf("footer does not name the oversized file:\n%s", s)
	}
	// The count is the files that entered the review, the same set as the
	// +/- lines: the gated file never did, and is named above instead.
	if !strings.Contains(s, " · 1 file · +1 / -1 · ") {
		t.Errorf("footer count or lines include the gated file:\n%s", s)
	}
}

// The footer carries no token budget, prompt breakdown or key spend any
// more: those live on the run page (decided).
func TestFooterKeepsTheDetailOnTheRunPage(t *testing.T) {
	h := newHarness(t, nil)
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)
	s := h.bb.Posted[0].Text
	for _, gone := range []string{"Tokens used", "Input tokens", "key total", "↑", "cross-file"} {
		if strings.Contains(s, gone) {
			t.Errorf("footer still has %q:\n%s", gone, s)
		}
	}
}
