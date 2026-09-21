package diff

import (
	"reflect"
	"strings"
	"testing"
)

func TestSplitByFileIsLossless(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  int
	}{
		{"empty", "", 0},
		{"single file", "diff --git a/f.py b/f.py\n@@ -1,1 +1,1 @@\n-a\n+b\n", 1},
		{"two files", "diff --git a/a.py b/a.py\n-a\ndiff --git a/b.py b/b.py\n-b\n", 2},
		// Verified against Python: a preamble before the first `diff --git `
		// becomes its own part, it is not glued onto the first file.
		{"preamble is its own part", "warning: noise\ndiff --git a/a.py b/a.py\n-a\n", 2},
		{"no diff header at all", "just some text\nmore text\n", 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			parts := SplitByFile(tc.input)
			if len(parts) != tc.want {
				t.Errorf("got %d parts, want %d: %q", len(parts), tc.want, parts)
			}
			if joined := strings.Join(parts, ""); joined != tc.input {
				t.Errorf("not lossless:\n got %q\nwant %q", joined, tc.input)
			}
		})
	}
}

// Python's splitlines breaks on more than \n, so a form feed starts a new line.
func TestSplitByFileFormFeed(t *testing.T) {
	input := "diff --git a/a.py b/a.py\n-x\x0c-y\ndiff --git a/b.py b/b.py\n-z\n"
	parts := SplitByFile(input)
	if len(parts) != 2 {
		t.Fatalf("got %d parts, want 2: %q", len(parts), parts)
	}
	if joined := strings.Join(parts, ""); joined != input {
		t.Errorf("not lossless: %q", joined)
	}
}

func TestExtractPath(t *testing.T) {
	cases := []struct{ name, input, want string }{
		{"standard a/ b/", "diff --git a/src/main.go b/src/main.go\n", "src/main.go"},
		{"bitbucket src:// dst://", "diff --git src://old.go dst://new.go\n", "new.go"},
		{"plus plus plus fallback", "--- a/x.go\n+++ b/pkg/x.go\n", "pkg/x.go"},
		{"dst fallback", "+++ dst://pkg/y.go\n", "pkg/y.go"},
		{"crlf trimmed", "diff --git a/f.go b/f.go\r\n", "f.go"},
		{"unparseable", "some random text\n", ""},
		{"path with spaces", "diff --git a/my file.go b/my file.go\n", "my file.go"},
		{"case preserved", "diff --git a/Main.PY b/Main.PY\n", "Main.PY"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ExtractPath(tc.input); got != tc.want {
				t.Errorf("ExtractPath() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestIsDeleted(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  bool
	}{
		{"deleted mid-diff", "diff --git a/f.go b/f.go\n+++ /dev/null\n", true},
		{"deleted at start", "+++ /dev/null\n", true},
		{"not deleted", "diff --git a/f.go b/f.go\n+++ b/f.go\n", false},
		{"dev null on old side only", "--- /dev/null\n+++ b/f.go\n", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsDeleted(tc.input); got != tc.want {
				t.Errorf("IsDeleted() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestParseHunks(t *testing.T) {
	t.Run("absent counts mean 1", func(t *testing.T) {
		_, hunks := ParseHunks("@@ -5 +7 @@\n-a\n+b\n")
		if len(hunks) != 1 {
			t.Fatalf("got %d hunks, want 1", len(hunks))
		}
		h := hunks[0]
		if h.OldStart != 5 || h.OldCount != 1 || h.NewStart != 7 || h.NewCount != 1 {
			t.Errorf("got -%d,%d +%d,%d, want -5,1 +7,1", h.OldStart, h.OldCount, h.NewStart, h.NewCount)
		}
	})

	t.Run("trailing function context is discarded", func(t *testing.T) {
		_, hunks := ParseHunks("@@ -1,2 +1,3 @@ def foo():\n-a\n+b\n")
		if len(hunks) != 1 {
			t.Fatalf("got %d hunks, want 1", len(hunks))
		}
		if got := hunks[0].BodyLines; !reflect.DeepEqual(got, []string{"-a", "+b"}) {
			t.Errorf("BodyLines = %q", got)
		}
	})

	t.Run("header lines kept before first hunk", func(t *testing.T) {
		header, hunks := ParseHunks("diff --git a/f.go b/f.go\n--- a/f.go\n+++ b/f.go\n@@ -1,1 +1,1 @@\n-a\n+b\n")
		want := []string{"diff --git a/f.go b/f.go", "--- a/f.go", "+++ b/f.go"}
		if !reflect.DeepEqual(header, want) {
			t.Errorf("header = %q, want %q", header, want)
		}
		if len(hunks) != 1 {
			t.Errorf("got %d hunks, want 1", len(hunks))
		}
	})

	t.Run("no hunks", func(t *testing.T) {
		_, hunks := ParseHunks("diff --git a/f.go b/f.go\nsimilarity index 100%\n")
		if len(hunks) != 0 {
			t.Errorf("got %d hunks, want 0", len(hunks))
		}
	})

	// The trailing "" from splitting on "\n" is dropped, so no phantom blank
	// reaches the body.
	t.Run("trailing empty body line dropped", func(t *testing.T) {
		_, hunks := ParseHunks("@@ -1,1 +1,1 @@\n-a\n+b\n")
		if got := hunks[0].BodyLines; !reflect.DeepEqual(got, []string{"-a", "+b"}) {
			t.Errorf("BodyLines = %q, want no trailing blank", got)
		}
	})

	t.Run("only the last hunk carries the artifact", func(t *testing.T) {
		_, hunks := ParseHunks("@@ -1,1 +1,1 @@\n-a\n+b\n@@ -5,1 +5,1 @@\n-c\n+d\n")
		if len(hunks) != 2 {
			t.Fatalf("got %d hunks, want 2", len(hunks))
		}
		if got := hunks[0].BodyLines; !reflect.DeepEqual(got, []string{"-a", "+b"}) {
			t.Errorf("hunk 0 BodyLines = %q", got)
		}
		if got := hunks[1].BodyLines; !reflect.DeepEqual(got, []string{"-c", "+d"}) {
			t.Errorf("hunk 1 BodyLines = %q", got)
		}
	})
}

func TestHasContent(t *testing.T) {
	if (FileReviewData{Content: ""}).HasContent() {
		t.Error("empty content should report false, matching Python truthiness")
	}
	if !(FileReviewData{Content: "x"}).HasContent() {
		t.Error("non-empty content should report true")
	}
}
