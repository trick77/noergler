package review

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/bitbucket"
)

// fileDiff builds a per-file diff big enough to be measured.
func fileDiff(path string, bodyBytes int) string {
	return fmt.Sprintf("diff --git a/%s b/%s\n--- a/%s\n+++ b/%s\n@@ -0,0 +1 @@\n+%s\n",
		path, path, path, path, strings.Repeat("x", bodyBytes))
}

// The point of the line: name every file and say whether it would have been
// reviewed. A fixture-heavy PR must not bury the one source file.
func TestTooLargeReportNamesEveryFileWithStatus(t *testing.T) {
	head := fileDiff("fixtures/huge.json", 5000) + fileDiff("diagrams/flow.puml", 800) +
		fileDiff("src/app.ts", 300)

	got := tooLargeReport([]byte(head), 1000, nil)

	for _, want := range []string{"fixtures/huge.json", "diagrams/flow.puml", "src/app.ts"} {
		if !strings.Contains(got, want) {
			t.Errorf("got %q, want %s named", got, want)
		}
	}
	// Last file in the head, so its size is a lower bound.
	if !strings.Contains(got, "src/app.ts  >=") || !strings.Contains(got, "REVIEWED") {
		t.Errorf("got %q, want src/app.ts marked REVIEWED", got)
	}
	// .json and .puml are both in the skip lists.
	if !strings.Contains(got, "FILTERED") || strings.Count(got, "FILTERED") != 2 {
		t.Errorf("got %q, want exactly the two skip-listed files FILTERED", got)
	}
	if !strings.Contains(got, "1 reviewable") || !strings.Contains(got, "2 filtered") {
		t.Errorf("got %q, want the totals split by status", got)
	}
}

// Reviewable files sort first: on a fixture-heavy PR the thing the operator is
// looking for must not sit below a screen of JSON. This inverts the previous
// largest-first ordering, deliberately.
func TestTooLargeReportPutsReviewableFilesFirst(t *testing.T) {
	head := fileDiff("fixtures/big.json", 9000) + fileDiff("src/app.go", 50)

	got := tooLargeReport([]byte(head), 1000, nil)

	if strings.Index(got, "src/app.go") > strings.Index(got, "fixtures/big.json") {
		t.Errorf("got %q, want the reviewable file listed before the fixture", got)
	}
}

// Within a status group, biggest first: that is what blew the cap.
func TestTooLargeReportRanksBySizeWithinAGroup(t *testing.T) {
	head := fileDiff("small.go", 10) + fileDiff("huge.go", 5000) + fileDiff("mid.go", 500)

	got := tooLargeReport([]byte(head), 1000, nil)

	huge, mid, small := strings.Index(got, "huge.go"), strings.Index(got, "mid.go"), strings.Index(got, "small.go")
	if huge > mid || mid > small {
		t.Errorf("got %q, want huge.go before mid.go before small.go", got)
	}
}

// The full list reaches past the cap. Files the head never saw are named with
// an unknown size rather than omitted, which is the whole reason for /changes.
func TestTooLargeReportCoversFilesPastTheCap(t *testing.T) {
	head := fileDiff("src/seen.go", 100)
	paths := []string{"src/seen.go", "src/past-the-cap.go", "fixtures/past.json"}

	got := tooLargeReport([]byte(head), 1000, paths)

	if !strings.Contains(got, "src/past-the-cap.go  ?  REVIEWED") {
		t.Errorf("got %q, want the unseen source file named with an unknown size", got)
	}
	if !strings.Contains(got, "fixtures/past.json  ?  FILTERED") {
		t.Errorf("got %q, want the unseen fixture filtered", got)
	}
	// seen.go appears in both sources and must not be listed twice.
	if n := strings.Count(got, "src/seen.go"); n != 1 {
		t.Errorf("got %q, listed src/seen.go %d times, want 1", got, n)
	}
	if !strings.Contains(got, "3 file(s) in the PR") {
		t.Errorf("got %q, want the count to describe the PR, not the head", got)
	}
}

// Without /changes the report must say it only covers the head, so the counts
// are not mistaken for the PR's.
func TestTooLargeReportScopesToTheHeadWithoutTheFullList(t *testing.T) {
	head := fileDiff("src/app.go", 100)

	got := tooLargeReport([]byte(head), 1000, nil)

	if !strings.Contains(got, "in the first") {
		t.Errorf("got %q, want the head-only scope stated", got)
	}
	if strings.Contains(got, "in the PR") {
		t.Errorf("got %q, must not claim to describe the whole PR", got)
	}
}

// The likeliest real cause: one committed blob so big it is the only file in
// the head, and it is the truncated last part. Dropping the tail would report
// nothing at all here, which is the case the log exists for.
func TestTooLargeReportKeepsTheTruncatedTail(t *testing.T) {
	head := fileDiff("dist/bundle.js", 9000)
	head = head[:len(head)-200] // cut mid-file, as the cap does

	got := tooLargeReport([]byte(head), 1000, nil)

	if !strings.Contains(got, ">=") {
		t.Errorf("got %q, want the tail marked as a lower bound", got)
	}
}

// Only the last part is a lower bound; a complete earlier file is exact.
func TestTooLargeReportMarksOnlyTheTailAsPartial(t *testing.T) {
	head := fileDiff("a.go", 100) + fileDiff("b.go", 50)

	got := tooLargeReport([]byte(head), 1000, nil)

	if !strings.Contains(got, "a.go  1") {
		t.Errorf("got %q, want a.go with an exact size", got)
	}
	if !strings.Contains(got, "b.go  >=") {
		t.Errorf("got %q, want b.go as a lower bound", got)
	}
}

// "Nothing here was worth reviewing" is the answer the operator most needs, so
// it is stated in words rather than left to be inferred from a zero.
func TestTooLargeReportSaysWhenNothingIsReviewable(t *testing.T) {
	head := fileDiff("fixtures/a.json", 100) + fileDiff("fixtures/b.json", 200)

	got := tooLargeReport([]byte(head), 1000, nil)

	if !strings.Contains(got, "no reviewable files") {
		t.Errorf("got %q, want the empty reviewable set stated outright", got)
	}
}

// Preamble before the first `diff --git ` is its own part with no path.
func TestTooLargeReportSkipsUnparseableParts(t *testing.T) {
	head := "some preamble\nwithout a header\n" + fileDiff("real.go", 100)

	got := tooLargeReport([]byte(head), 1000, nil)

	if !strings.Contains(got, "1 file(s)") {
		t.Errorf("got %q, want only the one parseable file counted", got)
	}
}

func TestTooLargeReportEmptyWhenNothingParses(t *testing.T) {
	if got := tooLargeReport(nil, 1000, nil); got != "" {
		t.Errorf("got %q, want empty for a nil head and no paths", got)
	}
	if got := tooLargeReport([]byte("no diff headers here"), 1000, nil); got != "" {
		t.Errorf("got %q, want empty when no path parses", got)
	}
}

// A Content-Length refusal reads no body: the head is empty, but /changes
// still answers, and that list is the only thing standing between the operator
// and an opaque skip.
func TestTooLargeReportWorksWithNoHeadAtAll(t *testing.T) {
	got := tooLargeReport(nil, 1000, []string{"src/app.go", "fixtures/x.json"})

	if !strings.Contains(got, "src/app.go  ?  REVIEWED") {
		t.Errorf("got %q, want the file list reported without a head", got)
	}
	if !strings.Contains(got, "2 file(s) in the PR") {
		t.Errorf("got %q, want both files counted", got)
	}
}

// The wiring, not just the helper: the too-large path must emit the error line
// and the report, each carrying the pr tag.
func TestLogDiffTooLargeEmitsBothLines(t *testing.T) {
	r, buf := capturingReviewer(t)
	head := []byte(fileDiff("src/app.go", 50) + fileDiff("fixtures/big.json", 4000))
	size := 12345
	err := &bitbucket.ContentTooLarge{
		What: "PROJ/repo#7 diff", Limit: 1000, Size: &size, Head: head,
	}

	r.logDiffTooLarge(context.Background(), "PROJ/repo#7", "PROJ", "repo", 7, err, err)

	out := buf.String()
	if !strings.Contains(out, "skipping review") {
		t.Errorf("log %q missing the skip line", out)
	}
	if !strings.Contains(out, "12345 bytes") {
		t.Errorf("log %q does not report the measured size", out)
	}
	if !strings.Contains(out, "fixtures/big.json") {
		t.Errorf("log %q does not name the offending file", out)
	}
	if !strings.Contains(out, "REVIEWED") || !strings.Contains(out, "FILTERED") {
		t.Errorf("log %q does not mark files by reviewable status", out)
	}
	if strings.Count(out, "PROJ/repo#7") < 2 {
		t.Errorf("log %q does not carry the pr tag on both lines", out)
	}
}

// /changes failing must not turn a failure path into a second failure: the
// head-only report still goes out, and the log says why it is partial.
func TestLogDiffTooLargeDegradesWhenChangesFails(t *testing.T) {
	r, buf := capturingReviewer(t)
	r.bitbucket.(*fakeBitbucket).changesErr = errors.New("boom")
	head := []byte(fileDiff("src/app.go", 50))
	size := 12345
	err := &bitbucket.ContentTooLarge{
		What: "PROJ/repo#7 diff", Limit: 1000, Size: &size, Head: head,
	}

	r.logDiffTooLarge(context.Background(), "PROJ/repo#7", "PROJ", "repo", 7, err, err)

	out := buf.String()
	if !strings.Contains(out, "full file list unavailable") {
		t.Errorf("log %q does not say the file list was unavailable", out)
	}
	if !strings.Contains(out, "src/app.go") {
		t.Errorf("log %q dropped the head-based report", out)
	}
	if strings.Contains(out, "in the PR") {
		t.Errorf("log %q claims PR scope with no file list", out)
	}
}
