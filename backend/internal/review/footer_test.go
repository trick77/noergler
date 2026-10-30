package review

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/webhook"
)

// The footer links the PR's page by the PR row's link token, never by an
// id, and prints no link without a public URL or a readable token: never
// one to a host or a page that does not exist.
func TestFooterLinksThePRPage(t *testing.T) {
	h := newHarness(t, nil)
	h.r.publicURL = "https://noergler.example.com"
	h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)
	if len(h.bb.Posted) == 0 {
		t.Fatal("no summary posted")
	}
	want := "https://noergler.example.com/prs/" + h.st.prToken + ")"
	if s := h.bb.Posted[0].Text; !strings.Contains(s, want) {
		t.Errorf("summary lacks the PR link %q:\n%s", want, s)
	}

	h2 := newHarness(t, nil)
	h2.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)
	if s := h2.bb.Posted[0].Text; strings.Contains(s, "/prs/") {
		t.Errorf("no public URL, yet a PR link:\n%s", s)
	}

	// The page is the PR row's: a lost attempt row keeps the link.
	h4 := newHarness(t, nil)
	h4.r.publicURL = "https://noergler.example.com"
	h4.st.attemptErr = errors.New("db down")
	h4.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)
	if s := h4.bb.Posted[0].Text; !strings.Contains(s, "/prs/"+h4.st.prToken+")") {
		t.Errorf("attempt not written, and the PR link went with it:\n%s", s)
	}

	h3 := newHarness(t, nil)
	h3.r.publicURL = "https://noergler.example.com"
	h3.st.prToken = ""
	h3.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)
	if s := h3.bb.Posted[0].Text; strings.Contains(s, "/prs/") {
		t.Errorf("no token, yet a PR link:\n%s", s)
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
	h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)
	s := h.bb.Posted[0].Text
	if !strings.Contains(s, "Not reviewed (too large): `messages.properties` ⚠️") {
		t.Errorf("footer does not name the oversized file:\n%s", s)
	}
	// The gated file is one of the PR's files and its lines count, as the
	// merge-time rollup counts them; the count says it was not reviewed.
	if !strings.Contains(s, " · 2 files, 1 not reviewed · +61 / -1 · ") {
		t.Errorf("footer count disagrees with the not-reviewed line:\n%s", s)
	}
}

// The footer carries no token budget, prompt breakdown or key spend any
// more: those live on the run page (decided).
func TestFooterKeepsTheDetailOnTheRunPage(t *testing.T) {
	h := newHarness(t, nil)
	h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)
	s := h.bb.Posted[0].Text
	for _, gone := range []string{"Tokens used", "Input tokens", "key total", "↑", "cross-file"} {
		if strings.Contains(s, gone) {
			t.Errorf("footer still has %q:\n%s", gone, s)
		}
	}
}
