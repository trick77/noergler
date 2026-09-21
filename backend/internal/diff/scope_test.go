package diff

import "testing"

// Asserts exact 1-based line numbers; 0 means no enclosing scope was found.
func TestFindEnclosingScopeLine(t *testing.T) {
	cases := []struct {
		name     string
		lines    []string
		fromLine int
		maxLines int
		path     string
		want     int
	}{
		{
			"python def",
			[]string{"import os", "", "def my_function():", "    x = 1", "    y = 2", "    return x + y"},
			5, 8, "app/utils.py", 3,
		},
		{
			// Finds the nearest match, the inner def, not the class.
			"python class finds the inner def first",
			[]string{"import os", "", "class MyClass:", "    def method(self):", "        pass"},
			5, 8, "app/models.py", 4,
		},
		{
			"python async def",
			[]string{"", "async def handler():", "    await something()", "    return result"},
			4, 8, "app/main.py", 2,
		},
		{
			"java method",
			[]string{"package com.example;", "", "public class Foo {", "    private int bar() {", "        return 42;", "    }", "}"},
			5, 8, "src/Foo.java", 4,
		},
		{
			"typescript function",
			[]string{"import { x } from 'y';", "", "function doStuff() {", "  const a = 1;", "  return a;", "}"},
			5, 8, "src/utils.ts", 3,
		},
		{
			"no scope found",
			[]string{"x = 1", "y = 2", "z = 3"},
			3, 8, "script.py", 0,
		},
		{
			"skips non-code files",
			[]string{"key: value", "other: stuff", "changed: true"},
			3, 8, "config.yaml", 0,
		},
		{
			"skips docs",
			[]string{"# Title", "", "Some text"},
			3, 8, "README.md", 0,
		},
		{
			// maxLines=2 cannot reach the def on line 1.
			"max lines respected",
			[]string{"def far_away():", "    pass", "", "", "", "", "    x = 1"},
			7, 2, "app/foo.py", 0,
		},
		{
			"generic language with a brace",
			[]string{"func main() {", "    fmt.Println()", "    return", "}"},
			3, 8, "main.go", 1,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := findEnclosingScopeLine(tc.lines, tc.fromLine, tc.maxLines, tc.path); got != tc.want {
				t.Errorf("findEnclosingScopeLine() = %d, want %d", got, tc.want)
			}
		})
	}
}

// Asserts header and hunk counts plus exact start/count values.
func TestParseHunksCases(t *testing.T) {
	t.Run("single hunk", func(t *testing.T) {
		diff := "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n" +
			"@@ -10,3 +10,4 @@\n-old line\n+new line\n+added line"
		headers, hunks := ParseHunks(diff)
		if len(headers) != 3 {
			t.Errorf("got %d header lines, want 3", len(headers))
		}
		if len(hunks) != 1 {
			t.Fatalf("got %d hunks, want 1", len(hunks))
		}
		h := hunks[0]
		if h.OldStart != 10 || h.OldCount != 3 || h.NewStart != 10 || h.NewCount != 4 {
			t.Errorf("got -%d,%d +%d,%d, want -10,3 +10,4", h.OldStart, h.OldCount, h.NewStart, h.NewCount)
		}
	})

	t.Run("multiple hunks", func(t *testing.T) {
		diff := "diff --git a/file.py b/file.py\n@@ -1,2 +1,3 @@\n+added\n@@ -20,1 +21,1 @@\n-old\n+new"
		_, hunks := ParseHunks(diff)
		if len(hunks) != 2 {
			t.Fatalf("got %d hunks, want 2", len(hunks))
		}
		if hunks[0].NewStart != 1 {
			t.Errorf("hunk 0 NewStart = %d, want 1", hunks[0].NewStart)
		}
		if hunks[1].NewStart != 21 {
			t.Errorf("hunk 1 NewStart = %d, want 21", hunks[1].NewStart)
		}
	})

	t.Run("no hunks", func(t *testing.T) {
		headers, hunks := ParseHunks("diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py")
		if len(hunks) != 0 {
			t.Errorf("got %d hunks, want 0", len(hunks))
		}
		if len(headers) != 3 {
			t.Errorf("got %d header lines, want 3", len(headers))
		}
	})
}
