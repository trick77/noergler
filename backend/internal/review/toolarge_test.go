package review

import (
	"context"
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

// The whole point of the line: name the biggest files, largest first.
func TestTooLargeOffendersRanksBySize(t *testing.T) {
	head := fileDiff("small.txt", 10) + fileDiff("huge.json", 5000) + fileDiff("mid.go", 500)

	got := tooLargeOffenders([]byte(head))
	if !strings.Contains(got, "3 file(s)") {
		t.Errorf("got %q, want a count of 3 files", got)
	}
	huge, mid := strings.Index(got, "huge.json"), strings.Index(got, "mid.go")
	small := strings.Index(got, "small.txt")
	if huge < 0 || mid < 0 || small < 0 {
		t.Fatalf("got %q, want all three paths named", got)
	}
	if !(huge < mid && mid < small) {
		t.Errorf("got %q, want huge.json before mid.go before small.txt", got)
	}
}

// The likeliest real cause: one committed blob so big it is the only file in
// the head, and it is the truncated last part. Dropping the tail would report
// nothing at all here, which is the case the log exists for.
func TestTooLargeOffendersKeepsTheTruncatedTail(t *testing.T) {
	head := fileDiff("dist/bundle.js", 9000)
	head = head[:len(head)-200] // cut mid-file, as the cap does

	got := tooLargeOffenders([]byte(head))
	if !strings.Contains(got, "dist/bundle.js>=") {
		t.Errorf("got %q, want dist/bundle.js marked as a lower bound", got)
	}
}

// Only the last part is a lower bound; a complete earlier file is exact.
func TestTooLargeOffendersMarksOnlyTheTailAsPartial(t *testing.T) {
	head := fileDiff("a.go", 100) + fileDiff("b.go", 50)

	got := tooLargeOffenders([]byte(head))
	if !strings.Contains(got, "a.go=") {
		t.Errorf("got %q, want a.go with an exact size", got)
	}
	if !strings.Contains(got, "b.go>=") {
		t.Errorf("got %q, want b.go as a lower bound", got)
	}
}

// Preamble before the first `diff --git ` is its own part with no path.
func TestTooLargeOffendersSkipsUnparseableParts(t *testing.T) {
	head := "some preamble\nwithout a header\n" + fileDiff("real.go", 100)

	got := tooLargeOffenders([]byte(head))
	if !strings.Contains(got, "1 file(s)") {
		t.Errorf("got %q, want only the one parseable file counted", got)
	}
}

func TestTooLargeOffendersCapsTheList(t *testing.T) {
	var head strings.Builder
	for i := 0; i < maxTooLargeOffenders+5; i++ {
		head.WriteString(fileDiff(fmt.Sprintf("f%02d.go", i), 100*(i+1)))
	}

	got := tooLargeOffenders([]byte(head.String()))
	if !strings.Contains(got, fmt.Sprintf("%d file(s)", maxTooLargeOffenders+5)) {
		t.Errorf("got %q, want the full count reported", got)
	}
	if n := strings.Count(got, ".go"); n != maxTooLargeOffenders {
		t.Errorf("named %d files, want %d", n, maxTooLargeOffenders)
	}
}

// Nothing to say means no line at all: the Content-Length branch carries no
// head, and the caller skips the log rather than printing an empty list.
func TestTooLargeOffendersEmptyWhenNothingParses(t *testing.T) {
	if got := tooLargeOffenders(nil); got != "" {
		t.Errorf("got %q, want empty for a nil head", got)
	}
	if got := tooLargeOffenders([]byte("no diff headers here")); got != "" {
		t.Errorf("got %q, want empty when no path parses", got)
	}
}

// The wiring, not just the helper: the too-large path must emit BOTH lines,
// the error and the offenders, each carrying the pr tag. Without this, the
// offender line could be deleted and only the helper's own tests would fail.
func TestLogDiffTooLargeEmitsBothLines(t *testing.T) {
	r, buf := capturingReviewer(t)
	head := []byte(fileDiff("src/app.go", 50) + fileDiff("fixtures/big.json", 4000))
	size := 12345
	err := &bitbucket.ContentTooLarge{
		What: "PROJ/repo#7 diff", Limit: 1000, Size: &size, Head: head,
	}

	r.logDiffTooLarge(context.Background(), "PROJ/repo#7", err, err)

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
	if strings.Count(out, "PROJ/repo#7") < 2 {
		t.Errorf("log %q does not carry the pr tag on both lines", out)
	}
	// Largest first: the fixture must precede the small source file.
	if strings.Index(out, "fixtures/big.json") > strings.Index(out, "src/app.go") {
		t.Errorf("log %q does not rank the biggest file first", out)
	}
}

// A Content-Length refusal reads no body, so there is nothing to name and the
// second line must not appear at all rather than printing an empty list.
func TestLogDiffTooLargeSkipsTheOffenderLineWithoutAHead(t *testing.T) {
	r, buf := capturingReviewer(t)
	size := 99999
	err := &bitbucket.ContentTooLarge{What: "PROJ/repo#7 diff", Limit: 1000, Size: &size}

	r.logDiffTooLarge(context.Background(), "PROJ/repo#7", err, err)

	out := buf.String()
	if !strings.Contains(out, "skipping review") {
		t.Errorf("log %q missing the skip line", out)
	}
	if strings.Contains(out, "largest:") {
		t.Errorf("log %q emitted an offender line with no head", out)
	}
}
