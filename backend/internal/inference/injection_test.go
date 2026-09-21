package inference

import (
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/diff"
)

// PR file content is attacker-controlled. It must not be able to rewrite
// another section of the prompt by spelling a placeholder's name, which is
// what a multi-pass substitution allowed.
func TestHostileFileContentCannotRewritePromptSections(t *testing.T) {
	const template = "## Review\n{files}\n---\n{cumulative_pr_diff}\n---\n{previously_posted_findings}"

	hostile := diff.FileReviewData{
		Path:           "evil.py",
		Diff:           "@@ -1 +1 @@\n+x",
		Content:        "# {previously_posted_findings}\n# {cumulative_pr_diff}\n",
		ContentFetched: true,
	}

	files := RenderFileGroup([]diff.FileReviewData{hostile})
	posted := RenderPreviouslyPostedFindings([]PostedFinding{
		{FilePath: "real.py", Severity: "issue", CommentText: "a genuine earlier finding"},
	})
	cumulative := RenderCumulativePRDiff("@@ real cumulative diff @@")

	got := RenderReviewPrompt(template, files, cumulative, posted, "")

	// The real blocks appear exactly once each, where the template put them.
	if n := strings.Count(got, "a genuine earlier finding"); n != 1 {
		t.Errorf("the posted-findings block appears %d times, want 1", n)
	}
	if n := strings.Count(got, "@@ real cumulative diff @@"); n != 1 {
		t.Errorf("the cumulative diff appears %d times, want 1", n)
	}
	// The file's text survives as text.
	if !strings.Contains(got, "# {previously_posted_findings}") {
		t.Error("the hostile file's literal text was expanded instead of kept")
	}
}

// The same holds for a file whose content spells the files placeholder.
func TestSelfReferentialPlaceholderInFileContent(t *testing.T) {
	got := RenderReviewPrompt("{files}", "{files}", "", "", "")
	if got != "{files}" {
		t.Errorf("got %q, want the text left alone rather than re-expanded", got)
	}
}
