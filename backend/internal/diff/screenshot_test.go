package diff

import (
	"slices"
	"testing"
)

func TestIsScreenshot(t *testing.T) {
	cases := []struct {
		path string
		want bool
	}{
		{"tests/login.spec.ts-snapshots/login-chromium-linux.png", true},
		{"src/__screenshots__/button.test.tsx/primary-1.png", true},
		{"src/__image_snapshots__/card-snap.jpg", true},
		{"Tests/Login.spec.ts-Snapshots/LOGIN.PNG", true},
		// An image elsewhere is a logo, not a baseline.
		{"public/logo.png", false},
		// A snapshot dir holding text snapshots.
		{"src/__screenshots__/notes.txt", false},
		// The file itself named like the dir does not count.
		{"img/foo-snapshots.png", false},
		{"tests/x.spec.ts-snapshots/a.svg", false},
	}
	for _, c := range cases {
		if got := IsScreenshot(c.path); got != c.want {
			t.Errorf("IsScreenshot(%q) = %v, want %v", c.path, got, c.want)
		}
	}
}

func TestHasScreenshot(t *testing.T) {
	const code = "diff --git a/src/app.ts b/src/app.ts\n@@ -1 +1 @@\n-a\n+b\n"
	const logo = "diff --git a/public/logo.png b/public/logo.png\nBinary files a/public/logo.png and b/public/logo.png differ\n"
	const shot = "diff --git src://t/a.spec.ts-snapshots/x.png dst://t/a.spec.ts-snapshots/x.png\n" +
		"Binary files src://t/a.spec.ts-snapshots/x.png and dst://t/a.spec.ts-snapshots/x.png differ\n"
	if HasScreenshot(code+logo) || HasScreenshot("") {
		t.Error("code and a logo are no screenshot")
	}
	if !HasScreenshot(code + shot) {
		t.Error("a baseline in Bitbucket's src:// dst:// form is one")
	}
}

// Variants group chromium first, then by path; a deleted variant is its own
// group beside a modified one.
func TestGroupVariantsPutsChromiumFirst(t *testing.T) {
	shots := []Screenshot{
		{Path: "s-snapshots/login-firefox-linux.png"},
		{Path: "s-snapshots/login-chromium-linux.png"},
		{Path: "s-snapshots/login-webkit-darwin.png"},
		{Path: "s-snapshots/header-firefox-linux.png"},
		{Path: "s-snapshots/header-webkit-linux.png", Change: ScreenshotDeleted},
		{Path: "s-snapshots/plain.png"},
	}
	var got [][]string
	for _, g := range GroupVariants(shots) {
		var paths []string
		for _, s := range g {
			paths = append(paths, s.Path)
		}
		got = append(got, paths)
	}
	want := [][]string{
		{"s-snapshots/login-chromium-linux.png", "s-snapshots/login-firefox-linux.png", "s-snapshots/login-webkit-darwin.png"},
		{"s-snapshots/header-firefox-linux.png"},
		{"s-snapshots/header-webkit-linux.png"},
		{"s-snapshots/plain.png"},
	}
	if !slices.EqualFunc(got, want, slices.Equal) {
		t.Errorf("groups = %v, want %v", got, want)
	}
}
