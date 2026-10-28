package review

import (
	"bytes"
	"context"
	"errors"
	"image"
	"image/color"
	"image/png"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/webhook"
)

const shotDir = "e2e/login.spec.ts-snapshots/"

// binary is one baseline's entry in a diff, as git writes it.
func binary(name string) string {
	p := shotDir + name
	return "diff --git a/" + p + " b/" + p + "\nindex 1111111..2222222 100644\nBinary files a/" + p + " and b/" + p + " differ\n"
}

// screenshotDiff is sampleDiff plus the baselines screenshotHarness's
// change set names, and a logo that is no baseline.
var screenshotDiff = sampleDiff + binary("form-chromium-linux.png") + binary("form-firefox-linux.png") +
	binary("new.png") + binary("gone.png") +
	"diff --git a/public/logo.png b/public/logo.png\nBinary files a/public/logo.png and b/public/logo.png differ\n"

func shotPNG(t *testing.T, dot bool) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 400, 300))
	for i := range img.Pix {
		img.Pix[i] = 255
	}
	if dot {
		img.Set(200, 150, color.RGBA{255, 0, 0, 255})
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// screenshotChangeSet is the PR's changes: the form changed in chromium
// and firefox, new.png added, gone.png deleted, a logo changed. The base
// commit is base1, not the target's tip tgt1.
func screenshotChangeSet() bitbucket.ChangeSet {
	return bitbucket.ChangeSet{FromHash: "base1", ToHash: "src1", Changes: []bitbucket.Change{
		{Path: shotDir + "form-chromium-linux.png", Type: "MODIFY", FromContentID: "c1", ContentID: "c2"},
		{Path: shotDir + "form-firefox-linux.png", Type: "MODIFY", FromContentID: "f1", ContentID: "f2"},
		{Path: shotDir + "new.png", Type: "ADD", ContentID: "n1"},
		{Path: shotDir + "gone.png", Type: "DELETE", FromContentID: "g1"},
		{Path: "public/logo.png", Type: "MODIFY", FromContentID: "l1", ContentID: "l2"},
	}}
}

// screenshotHarness serves the baselines at the PR's base (base1) and head
// (src1). The target's tip holds a third, newer form, which a comparison
// against it would wrongly report.
func screenshotHarness(t *testing.T, vision bool) *harness {
	return newHarness(t, func(h *harness) {
		h.bb.prDiff = screenshotDiff
		h.bb.changeSet = screenshotChangeSet()
		h.bb.blobs["base1:"+shotDir+"form-chromium-linux.png"] = shotPNG(t, false)
		h.bb.blobs["tgt1:"+shotDir+"form-chromium-linux.png"] = shotPNG(t, true)
		h.bb.blobs["src1:"+shotDir+"form-chromium-linux.png"] = shotPNG(t, true)
		h.bb.blobs["base1:"+shotDir+"form-firefox-linux.png"] = shotPNG(t, false)
		h.bb.blobs["src1:"+shotDir+"form-firefox-linux.png"] = shotPNG(t, true)
		h.bb.blobs["src1:"+shotDir+"new.png"] = shotPNG(t, false)
		h.llm.vision = vision
	})
}

// lastSummary is the summary as posted last, its whitespace collapsed: the
// assertions are about the words, not where the wrap fell.
func lastSummary(t *testing.T, h *harness) string {
	t.Helper()
	var text string
	switch {
	case len(h.bb.Updates) > 0:
		text = h.bb.Updates[len(h.bb.Updates)-1].Text
	case len(h.bb.Posted) > 0:
		text = h.bb.Posted[len(h.bb.Posted)-1].Text
	default:
		t.Fatal("no summary posted")
	}
	return strings.Join(strings.Fields(text), " ")
}

// The changed and the added baseline go to the model in the review call,
// the changed one against the PR's base; its lines land in the summary and
// the store; the deleted one is named, the firefox variant and the logo are
// not sent.
func TestScreenshotsAreDescribedInTheSummary(t *testing.T) {
	h := screenshotHarness(t, true)
	res := okResultWith()
	res.Review.Visual = []inference.VisualChange{
		{ID: "V1", Change: "A red dot appears mid-form; nothing in the PR explains it."},
		{ID: "V2", Change: "New empty page baseline."},
		{ID: "V9", Change: "an id never sent"},
	}
	h.llm.review = res

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.llm.Reviews) != 1 {
		t.Fatalf("reviews = %d", len(h.llm.Reviews))
	}
	req := h.llm.Reviews[0]
	if len(req.Images) != 2 || req.Images[0].Path != shotDir+"form-chromium-linux.png" || req.Images[1].Path != shotDir+"new.png" {
		t.Fatalf("images = %+v", req.Images)
	}
	if req.Images[0].Before == "" || !req.Images[0].Cropped || req.Images[1].Before != "" {
		t.Errorf("modified pair is before+after, cropped; added is after only: %+v", req.Images)
	}
	if req.PromptTokens <= 0 {
		t.Errorf("the images count toward the fit check: PromptTokens = %d", req.PromptTokens)
	}

	got := lastSummary(t, h)
	for _, want := range []string{
		"### Visual changes -",
		"`" + shotDir + "form-chromium-linux.png` — A red dot appears mid-form; nothing in the PR explains it. _(1 more variant)_",
		"`" + shotDir + "new.png` — New empty page baseline.",
		"`" + shotDir + "gone.png` — removed.",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("summary lacks %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "logo.png") || strings.Contains(got, "an id never sent") {
		t.Errorf("only baselines the run sent are listed:\n%s", got)
	}
	if len(h.st.VisualInserts) != 2 {
		t.Errorf("stored %d lines, want 2", len(h.st.VisualInserts))
	}
}

// The next run knows the same pairs by their blob ids: nothing is read or
// sent again, yet the section is still the whole PR's.
func TestScreenshotLinesAreReusedOnTheNextRun(t *testing.T) {
	h := screenshotHarness(t, true)
	res := okResultWith()
	res.Review.Visual = []inference.VisualChange{{ID: "V1", Change: "Dot added."}, {ID: "V2", Change: "New page."}}
	h.llm.review = res
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)
	reads := h.bb.byteReads

	h.llm.review = okResultWith()
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.llm.Reviews) != 2 || len(h.llm.Reviews[1].Images) != 0 {
		t.Fatalf("second run sent %d image pair(s), want none", len(h.llm.Reviews[1].Images))
	}
	if h.bb.byteReads != reads {
		t.Errorf("second run read %d screenshot version(s), want none: the blob ids are the key", h.bb.byteReads-reads)
	}
	got := lastSummary(t, h)
	if !strings.Contains(got, "— Dot added.") || !strings.Contains(got, "— New page.") {
		t.Errorf("stored lines missing from the second summary:\n%s", got)
	}
}

// A model without vision is sent no image and nothing is read for it; the
// summary says why.
func TestScreenshotsWithoutVisionAreNotCompared(t *testing.T) {
	h := screenshotHarness(t, false)
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if n := len(h.llm.Reviews[0].Images); n != 0 {
		t.Fatalf("sent %d image pair(s) to a model without vision", n)
	}
	if h.bb.byteReads != 0 {
		t.Errorf("read %d screenshot version(s) for a model that cannot see them", h.bb.byteReads)
	}
	got := lastSummary(t, h)
	if strings.Count(got, "not compared: the model takes no images") != 2 {
		t.Errorf("want both baselines marked not compared:\n%s", got)
	}
}

// A sent pair the model skipped says so, and nothing is stored for it.
func TestScreenshotSkippedByTheModelIsMarked(t *testing.T) {
	h := screenshotHarness(t, true)
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if got := lastSummary(t, h); strings.Count(got, "the model said nothing about it") != 2 {
		t.Errorf("want both sent baselines marked unanswered:\n%s", got)
	}
	if len(h.st.VisualInserts) != 0 {
		t.Errorf("stored %v for pairs the model never described", h.st.VisualInserts)
	}
}

// A PR without baselines has no section, sends no image and reads no
// changes.
func TestNoScreenshotsNoSection(t *testing.T) {
	h := newHarness(t, func(h *harness) {
		h.llm.vision = true
		h.bb.changeSetErr = errors.New("must not be called")
	})
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)
	if len(h.llm.Reviews[0].Images) != 0 || strings.Contains(lastSummary(t, h), "Visual changes") {
		t.Error("no baselines, no images and no section")
	}
}

// Changes without types or blob ids: the 404s decide added and deleted, and
// the bytes key the cache.
func TestScreenshotWithoutTypesIsClassifiedByFetch(t *testing.T) {
	h := newHarness(t, func(h *harness) {
		h.bb.prDiff = sampleDiff + binary("added.png") + binary("dropped.png")
		h.bb.changeSet = bitbucket.ChangeSet{FromHash: "base1", Changes: []bitbucket.Change{
			{Path: shotDir + "added.png"}, {Path: shotDir + "dropped.png"},
		}}
		h.bb.blobs["src1:"+shotDir+"added.png"] = shotPNG(t, true)
		h.bb.blobs["base1:"+shotDir+"dropped.png"] = shotPNG(t, false)
		h.llm.vision = true
	})
	res := okResultWith()
	res.Review.Visual = []inference.VisualChange{{ID: "V1", Change: "New baseline."}}
	h.llm.review = res

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	imgs := h.llm.Reviews[0].Images
	if len(imgs) != 1 || imgs[0].Path != shotDir+"added.png" || imgs[0].Before != "" {
		t.Fatalf("want the added baseline sent alone, after only: %+v", imgs)
	}
	got := lastSummary(t, h)
	for _, want := range []string{"`" + shotDir + "added.png` — New baseline.", "`" + shotDir + "dropped.png` — removed."} {
		if !strings.Contains(got, want) {
			t.Errorf("summary lacks %q:\n%s", want, got)
		}
	}
	if len(h.st.VisualInserts) != 1 || !strings.HasPrefix(h.st.VisualInserts[0].AfterID, "sha256:") {
		t.Errorf("a pair without blob ids is keyed by its bytes: %+v", h.st.VisualInserts)
	}
}

// Chromium only re-encoded, firefox really changed: the firefox variant is
// the one shown and sent, not lost behind its chromium twin.
func TestScreenshotChangeOnlyAVariantShowsIsKept(t *testing.T) {
	h := screenshotHarness(t, true)
	h.bb.blobs["src1:"+shotDir+"form-chromium-linux.png"] = shotPNG(t, false)
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	imgs := h.llm.Reviews[0].Images
	if len(imgs) == 0 || imgs[0].Path != shotDir+"form-firefox-linux.png" {
		t.Fatalf("want the firefox variant sent: %+v", imgs)
	}
	// The unchanged chromium pair is stored as such, never read again.
	if len(h.st.VisualInserts) != 1 || h.st.visual[h.st.VisualInserts[0]] != "" {
		t.Errorf("want the unchanged pair stored empty: %+v", h.st.VisualInserts)
	}
}

// A push that refreshes baselines and nothing else is still reviewed: the
// drift the section exists for.
func TestBaselineOnlyPushIsReviewed(t *testing.T) {
	h := screenshotHarness(t, true)
	h.bb.prDiff = binary("form-chromium-linux.png")
	res := okResultWith()
	res.Review.Visual = []inference.VisualChange{{ID: "V1", Change: "Dot added; no code explains it."}}
	h.llm.review = res

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.llm.Reviews) != 1 || len(h.llm.Reviews[0].Images) == 0 {
		t.Fatalf("a baseline-only diff must reach the model with its images")
	}
	if !strings.Contains(lastSummary(t, h), "Dot added; no code explains it.") {
		t.Errorf("summary lacks the line:\n%s", lastSummary(t, h))
	}
}

// The PR's changes did not fetch: the section says so instead of vanishing.
func TestScreenshotsUnreadSaySo(t *testing.T) {
	h := screenshotHarness(t, true)
	h.bb.changeSetErr = errors.New("503")
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if got := lastSummary(t, h); !strings.Contains(got, "### Visual changes _Not read this run: the PR's changes could not be fetched._") {
		t.Errorf("summary lacks the unread note:\n%s", got)
	}
}

// A baseline kept in Git LFS comes back as its pointer: named, not decoded.
func TestScreenshotInLFSIsNamed(t *testing.T) {
	h := screenshotHarness(t, true)
	h.bb.blobs["src1:"+shotDir+"new.png"] = []byte("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1234\n")
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if got := lastSummary(t, h); !strings.Contains(got, "`"+shotDir+"new.png` — _not compared: stored in Git LFS._") {
		t.Errorf("summary lacks the LFS line:\n%s", got)
	}
}

// No base commit in the PR's changes: a changed baseline is not compared,
// and the target's tip, which may hold another PR's baseline, is never read
// in its place.
func TestScreenshotWithoutBaseIsNotCompared(t *testing.T) {
	h := screenshotHarness(t, true)
	h.bb.changeSet.FromHash = ""
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	for _, img := range h.llm.Reviews[0].Images {
		if img.Before != "" {
			t.Errorf("sent a before side with no base: %s", img.Path)
		}
	}
	if got := lastSummary(t, h); !strings.Contains(got, "`"+shotDir+"form-chromium-linux.png` — _not compared: image could not be read._") {
		t.Errorf("want the changed baseline not compared:\n%s", got)
	}
}

// The PR moved on between the webhook and the changes read: the images come
// from the reviewed commit, and the blob ids, the newer head's, are not the
// key.
func TestScreenshotsComeFromTheReviewedCommit(t *testing.T) {
	h := screenshotHarness(t, true)
	h.bb.changeSet.ToHash = "newer"
	res := okResultWith()
	res.Review.Visual = []inference.VisualChange{{ID: "V1", Change: "Dot added."}}
	h.llm.review = res
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.llm.Reviews[0].Images) == 0 {
		t.Fatal("want the screenshots read at the reviewed commit")
	}
	if len(h.st.VisualInserts) == 0 || !strings.HasPrefix(h.st.VisualInserts[0].AfterID, "sha256:") {
		t.Errorf("want the pair keyed by its bytes: %+v", h.st.VisualInserts)
	}
}

// A push that only deletes a baseline still runs, so the summary says
// removed rather than keeping the old line.
func TestBaselineDeletionOnlyRefreshesTheSummary(t *testing.T) {
	h := screenshotHarness(t, true)
	h.bb.prDiff = binary("gone.png")
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.llm.Reviews) != 1 {
		t.Fatalf("reviews = %d, want the run to go on", len(h.llm.Reviews))
	}
	if !strings.Contains(lastSummary(t, h), "`"+shotDir+"gone.png` — removed.") {
		t.Errorf("summary lacks the removal:\n%s", lastSummary(t, h))
	}
}
