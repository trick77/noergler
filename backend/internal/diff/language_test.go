package diff

import (
	"reflect"
	"testing"
)

func TestDetectLanguage(t *testing.T) {
	cases := []struct{ path, want string }{
		{"main.py", "python"},
		{"types.pyi", "python"},
		{"Main.java", "jvm"},
		{"Main.kt", "jvm"},
		{"app.ts", "typescript"},
		{"app.tsx", "typescript"},
		{"app.mjs", "typescript"},
		{"page.html", "html"},
		{"style.scss", "css"},
		{"config.yaml", "config"},
		{"config.yml", "config"},
		{"README.md", "docs"},
		{"notes.txt", "docs"},
		{"Makefile", "other"},
		{"src/deep/main.py", "python"},
		// Build files resolve before the extension map.
		{"pom.xml", "build-config"},
		{"build.gradle", "build-config"},
		{"build.gradle.kts", "build-config"},
		{"package.json", "build-config"},
		{"tsconfig.json", "build-config"},
		{"tsconfig.app.json", "build-config"},
		// Case-sensitive, unlike IsTestFile.
		{"Main.PY", "other"},
		{"main.Py", "other"},
		// Build-file map is basename-exact, so a nested one still matches.
		{"app/pom.xml", "build-config"},
	}
	for _, tc := range cases {
		t.Run(tc.path, func(t *testing.T) {
			if got := DetectLanguage(tc.path); got != tc.want {
				t.Errorf("DetectLanguage(%q) = %q, want %q", tc.path, got, tc.want)
			}
		})
	}
}

func TestIsTestFile(t *testing.T) {
	cases := []struct {
		path string
		want bool
	}{
		{"foo_test.py", true},
		{"test_foo.py", true},
		{"FooTest.java", true},
		{"FooTests.java", true},
		{"app.spec.ts", true},
		{"app.test.tsx", true},
		{"src/tests/helper.go", true},
		{"src/test/helper.go", true},
		{"src/__tests__/helper.js", true},
		// Case-insensitive, unlike DetectLanguage.
		{"FOO_TEST.PY", true},
		{"src/TESTS/x.go", true},
		{"main.py", false},
		{"contest.py", false},
	}
	for _, tc := range cases {
		t.Run(tc.path, func(t *testing.T) {
			if got := IsTestFile(tc.path); got != tc.want {
				t.Errorf("IsTestFile(%q) = %v, want %v", tc.path, got, tc.want)
			}
		})
	}
}

func TestSortByLanguagePriority(t *testing.T) {
	in := []FileReviewData{
		{Path: "README.md"},
		{Path: "src/app.ts"},
		{Path: "src/main.py"},
		{Path: "tests/test_main.py"},
		{Path: "Main.java"},
		{Path: "style.css"},
		{Path: "page.html"},
	}
	got := SortByLanguagePriority(in)

	want := []string{
		// group 0, by language rank: python, jvm, typescript, html, css
		"src/main.py",
		"Main.java",
		"src/app.ts",
		"page.html",
		"style.css",
		// group 1: tests
		"tests/test_main.py",
		// group 2: docs
		"README.md",
	}
	var gotPaths []string
	for _, f := range got {
		gotPaths = append(gotPaths, f.Path)
	}
	if !reflect.DeepEqual(gotPaths, want) {
		t.Errorf("order =\n %q\nwant\n %q", gotPaths, want)
	}
}

// Group 2 is checked before the test group, so a deprioritized language inside a
// tests/ directory sorts as group 2, not group 1.
func TestDeprioritizedBeatsTestGroup(t *testing.T) {
	in := []FileReviewData{
		{Path: "tests/config.yaml"},
		{Path: "tests/test_main.py"},
	}
	got := SortByLanguagePriority(in)
	if got[0].Path != "tests/test_main.py" {
		t.Errorf("first = %q, want tests/test_main.py (config.yaml is group 2)", got[0].Path)
	}
}

// The sort key reads only the path, so re-reviews keep prefix-cache order.
func TestSortIsContentIndependent(t *testing.T) {
	a := []FileReviewData{
		{Path: "b.py", Diff: "x", Content: "one"},
		{Path: "a.py", Diff: "y", Content: "two"},
	}
	b := []FileReviewData{
		{Path: "b.py", Diff: "completely", Content: "different"},
		{Path: "a.py", Diff: "values", Content: "entirely"},
	}
	ga, gb := SortByLanguagePriority(a), SortByLanguagePriority(b)
	if ga[0].Path != gb[0].Path || ga[1].Path != gb[1].Path {
		t.Error("sort order depends on diff or content")
	}
}

func TestSortDoesNotMutateInput(t *testing.T) {
	in := []FileReviewData{{Path: "z.md"}, {Path: "a.py"}}
	_ = SortByLanguagePriority(in)
	if in[0].Path != "z.md" {
		t.Error("input slice was reordered")
	}
}
