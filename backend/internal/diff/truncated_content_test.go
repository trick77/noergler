package diff

import "testing"

// Content is byte-capped at the socket (BITBUCKET_MAX_FILE_BYTES) while the
// diff is not, so a truncated large file plus a late hunk drives the dynamic
// scope search past the end of content. Indexing past the end would panic and
// take down the single queue worker, killing reviews for every team, not just
// this one.
func TestDynamicSearchSurvivesTruncatedContent(t *testing.T) {
	cases := []struct {
		name    string
		content string
		diff    string
	}{
		{"hunk far past EOF", numberedLines(10), "@@ -100,1 +100,1 @@\n-line 100\n+new\n"},
		{"hunk just past EOF", numberedLines(10), "@@ -12,1 +12,1 @@\n-line 12\n+new\n"},
		{"single-line content", "x", "@@ -50,1 +50,1 @@\n-line 50\n+new\n"},
		{
			"multiple hunks past EOF", numberedLines(5),
			"@@ -50,1 +50,1 @@\n-a\n+b\n@@ -60,1 +60,1 @@\n-c\n+d\n",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("panicked: %v", r)
				}
			}()
			ExpandContext(tc.diff, tc.content, "f.py", 5, 2, 10, true)
		})
	}
}

// The same shape with dynamic context off must also survive, since the context
// loops have always been guarded.
func TestStaticExpansionSurvivesTruncatedContent(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("panicked: %v", r)
		}
	}()
	ExpandContext("@@ -100,1 +100,1 @@\n-line 100\n+new\n", numberedLines(10), "f.py", 5, 2, 10, false)
}
