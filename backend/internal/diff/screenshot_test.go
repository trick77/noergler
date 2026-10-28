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

func TestScreenshotsReadsChangeKind(t *testing.T) {
	const d = "diff --git a/src/app.ts b/src/app.ts\n@@ -1 +1 @@\n-a\n+b\n" +
		"diff --git a/t/a.spec.ts-snapshots/mod.png b/t/a.spec.ts-snapshots/mod.png\n" +
		"index 1111111..2222222 100644\nBinary files a/t/a.spec.ts-snapshots/mod.png and b/t/a.spec.ts-snapshots/mod.png differ\n" +
		"diff --git a/t/a.spec.ts-snapshots/new.png b/t/a.spec.ts-snapshots/new.png\n" +
		"new file mode 100644\nindex 0000000..2222222\nBinary files /dev/null and b/t/a.spec.ts-snapshots/new.png differ\n" +
		"diff --git src://t/a.spec.ts-snapshots/gone.png dst://t/a.spec.ts-snapshots/gone.png\n" +
		"Binary files src://t/a.spec.ts-snapshots/gone.png and /dev/null differ\n" +
		"diff --git a/t/old-snapshots/x.png b/t/a.spec.ts-snapshots/moved.png\n" +
		"similarity index 100%\nrename from t/old-snapshots/x.png\nrename to t/a.spec.ts-snapshots/moved.png\n" +
		"diff --git a/public/logo.png b/public/logo.png\nBinary files a/public/logo.png and b/public/logo.png differ\n"

	got := Screenshots(d)
	want := []Screenshot{
		{Path: "t/a.spec.ts-snapshots/gone.png", OldPath: "t/a.spec.ts-snapshots/gone.png", Change: ScreenshotDeleted},
		{Path: "t/a.spec.ts-snapshots/mod.png", OldPath: "t/a.spec.ts-snapshots/mod.png", Change: ScreenshotModified},
		{Path: "t/a.spec.ts-snapshots/moved.png", OldPath: "t/old-snapshots/x.png", Change: ScreenshotModified},
		{Path: "t/a.spec.ts-snapshots/new.png", OldPath: "t/a.spec.ts-snapshots/new.png", Change: ScreenshotAdded},
	}
	if !slices.Equal(got, want) {
		t.Fatalf("Screenshots:\n got %+v\nwant %+v", got, want)
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
