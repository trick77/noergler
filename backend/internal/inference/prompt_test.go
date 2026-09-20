package inference

import (
	"strings"
	"testing"

	"github.com/trick77/noergler-go/internal/diff"
)

// Expectations generated from the running Python (format_file_entry).
func TestFormatFileEntryMatchesPython(t *testing.T) {
	const changes = "### Changes (diff: lines with `-` are REMOVED, lines with `+` are ADDED):\n```diff\n@@ -1 +1 @@\n+x\n```"

	cases := []struct {
		name string
		in   diff.FileReviewData
		want string
	}{
		{
			"content present",
			diff.FileReviewData{Path: "a.py", Diff: "@@ -1 +1 @@\n+x", Content: "line1\nline2", ContentFetched: true},
			"## File: a.py\n### Full file content (new version):\n```py\nline1\nline2\n```\n" + changes,
		},
		{
			// A fetched but empty file renders an empty code block, NOT the
			// omitted notice. This is why ContentFetched exists.
			"content empty string",
			diff.FileReviewData{Path: "a.py", Diff: "@@ -1 +1 @@\n+x", Content: "", ContentFetched: true},
			"## File: a.py\n### Full file content (new version):\n```py\n\n```\n" + changes,
		},
		{
			"content not fetched",
			diff.FileReviewData{Path: "a.py", Diff: "@@ -1 +1 @@\n+x"},
			"## File: a.py\n_(full file content omitted — review diff only)_\n" + changes,
		},
		{
			"deleted file",
			diff.FileReviewData{Path: "a.py", Diff: "+++ /dev/null\n-x"},
			"## File: a.py\n_(file deleted)_\n### Changes (diff: lines with `-` are REMOVED, lines with `+` are ADDED):\n```diff\n+++ /dev/null\n-x\n```",
		},
		{
			// No extension means an unlabelled fence.
			"no extension",
			diff.FileReviewData{Path: "Makefile", Diff: "@@ -1 +1 @@\n+x", Content: "all:", ContentFetched: true},
			"## File: Makefile\n### Full file content (new version):\n```\nall:\n```\n" + changes,
		},
		{
			// A dotfile's leading dot is not an extension.
			"dotfile",
			diff.FileReviewData{Path: ".env", Diff: "@@ -1 +1 @@\n+x", Content: "A=1", ContentFetched: true},
			"## File: .env\n### Full file content (new version):\n```\nA=1\n```\n" + changes,
		},
		{
			"multi dot takes the last extension",
			diff.FileReviewData{Path: "a.spec.ts", Diff: "@@ -1 +1 @@\n+x", Content: "q", ContentFetched: true},
			"## File: a.spec.ts\n### Full file content (new version):\n```ts\nq\n```\n" + changes,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := FormatFileEntry(tc.in); got != tc.want {
				t.Errorf("FormatFileEntry() =\n%q\nwant\n%q", got, tc.want)
			}
		})
	}
}

func TestRenderFileGroup(t *testing.T) {
	files := []diff.FileReviewData{
		{Path: "a.py", Diff: "d1"},
		{Path: "b.py", Diff: "d2"},
	}
	got := RenderFileGroup(files)
	if strings.Count(got, "## File:") != 2 {
		t.Errorf("want two entries:\n%s", got)
	}
	if !strings.Contains(got, "```\n\n## File: b.py") {
		t.Errorf("entries must be blank-line separated:\n%s", got)
	}
	if RenderFileGroup(nil) != "" {
		t.Error("no files renders empty")
	}
}

func TestRenderSupplementaryContext(t *testing.T) {
	t.Run("fixed section order", func(t *testing.T) {
		got := RenderSupplementaryContext([]string{"o.py"}, []string{"d.py"}, []string{"r.py"})
		iOther := strings.Index(got, "Other modified files")
		iRenamed := strings.Index(got, "Renamed files")
		iDeleted := strings.Index(got, "Deleted files")
		if iOther >= iRenamed || iRenamed >= iDeleted {
			t.Errorf("want other, renamed, deleted:\n%s", got)
		}
	})

	t.Run("empty groups are omitted", func(t *testing.T) {
		got := RenderSupplementaryContext([]string{"o.py"}, nil, nil)
		if strings.Contains(got, "Renamed") || strings.Contains(got, "Deleted") {
			t.Errorf("empty groups must not render:\n%s", got)
		}
	})

	t.Run("nothing renders empty", func(t *testing.T) {
		if RenderSupplementaryContext(nil, nil, nil) != "" {
			t.Error("want empty")
		}
	})
}

func TestRenderCumulativePRDiff(t *testing.T) {
	t.Run("blank renders empty", func(t *testing.T) {
		if RenderCumulativePRDiff("   \n  ") != "" {
			t.Error("whitespace-only must render empty")
		}
		if RenderCumulativePRDiff("") != "" {
			t.Error("empty must render empty")
		}
	})

	t.Run("wraps in the tag", func(t *testing.T) {
		got := RenderCumulativePRDiff("@@ -1 +1 @@")
		if !strings.Contains(got, "<cumulative_pr_diff>\n@@ -1 +1 @@\n</cumulative_pr_diff>") {
			t.Errorf("diff not wrapped:\n%s", got)
		}
	})
}

// Expectations generated from the running Python
// (render_previously_posted_findings).
func TestRenderPreviouslyPostedFindings(t *testing.T) {
	line := func(n int) *int { return &n }

	t.Run("renders each finding on one line", func(t *testing.T) {
		got := RenderPreviouslyPostedFindings([]PostedFinding{
			{FilePath: "a.py", LineNumber: line(12), Severity: "major", CommentText: "bad\nthing"},
		})
		if !strings.Contains(got, "- a.py:12 [major] bad thing") {
			t.Errorf("newlines must be flattened:\n%s", got)
		}
	})

	t.Run("defaults", func(t *testing.T) {
		got := RenderPreviouslyPostedFindings([]PostedFinding{
			{FilePath: "", LineNumber: line(3), CommentText: ""},
		})
		if !strings.Contains(got, "- <unknown>:3 [suggestion] ") {
			t.Errorf("want the unknown/suggestion defaults:\n%s", got)
		}
	})

	t.Run("no line number omits the colon", func(t *testing.T) {
		got := RenderPreviouslyPostedFindings([]PostedFinding{
			{FilePath: "b.py", Severity: "minor", CommentText: "x"},
		})
		if !strings.Contains(got, "- b.py [minor] x") {
			t.Errorf("want a bare path:\n%s", got)
		}
	})

	// Over 300 characters is truncated to 297 plus an ellipsis.
	t.Run("long text truncated", func(t *testing.T) {
		got := RenderPreviouslyPostedFindings([]PostedFinding{
			{FilePath: "b.py", CommentText: strings.Repeat("x", 350)},
		})
		if !strings.Contains(got, strings.Repeat("x", 297)+"...") {
			t.Error("want 297 characters plus an ellipsis")
		}
		if strings.Contains(got, strings.Repeat("x", 298)) {
			t.Error("truncated too late")
		}
	})

	t.Run("exactly at the cap is not truncated", func(t *testing.T) {
		got := RenderPreviouslyPostedFindings([]PostedFinding{
			{FilePath: "b.py", CommentText: strings.Repeat("x", 300)},
		})
		if strings.Contains(got, "...") {
			t.Error("300 characters is within the cap")
		}
	})

	// The cap counts characters, not bytes.
	t.Run("truncation counts runes", func(t *testing.T) {
		got := RenderPreviouslyPostedFindings([]PostedFinding{
			{FilePath: "b.py", CommentText: strings.Repeat("ü", 200)},
		})
		if strings.Contains(got, "...") {
			t.Error("200 runes is within the cap, even at 400 bytes")
		}
	})

	t.Run("none renders empty", func(t *testing.T) {
		if RenderPreviouslyPostedFindings(nil) != "" {
			t.Error("want empty")
		}
	})
}

// {files} is substituted before the other blocks so the cached prefix stays
// stable, and substitution is literal: file content carrying a JSON brace or
// another placeholder's name must not be reinterpreted.
func TestRenderReviewPromptSubstitution(t *testing.T) {
	tpl := "A{files}B{cumulative_pr_diff}C{previously_posted_findings}D{repo_instructions}E"
	got := RenderReviewPrompt(tpl, "F", "C1", "P1", "R1")
	if got != "AFBC1CP1DR1E" {
		t.Errorf("got %q", got)
	}

	// One pass, so replacement text is never rescanned: PR file content
	// spelling another placeholder's name stays that text rather than being
	// expanded into the real block. Python reaches the same result by
	// substituting {files} last. Attacker-controlled file content must not
	// rewrite another section of the prompt.
	t.Run("a placeholder inside file content is left alone", func(t *testing.T) {
		got := RenderReviewPrompt("{files}|{cumulative_pr_diff}", "{cumulative_pr_diff}", "REAL", "", "")
		if got != "{cumulative_pr_diff}|REAL" {
			t.Errorf("got %q, want the file's text left alone", got)
		}
	})

	t.Run("JSON braces survive", func(t *testing.T) {
		got := RenderReviewPrompt("{files}", `{"a": {"b": 1}}`, "", "", "")
		if got != `{"a": {"b": 1}}` {
			t.Errorf("got %q", got)
		}
	})
}
