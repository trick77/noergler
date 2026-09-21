package inference

import "testing"

// Expectations generated from Python Path(p).suffix.lstrip("."). Go's
// path.Ext disagrees on a dotfile, calling ".env" an extension of ".env",
// which would label the fence "env".
func TestFenceLanguageIsPinned(t *testing.T) {
	cases := []struct{ path, want string }{
		{".env", ""},
		{"a.py", "py"},
		{"Makefile", ""},
		{"a.spec.ts", "ts"},
		{".gitignore", ""},
		{"dir/.env", ""},
		// A trailing dot is a suffix of "." which lstrips to nothing.
		{"a.", ""},
		{"src/deep/main.go", "go"},
		{"", ""},
	}
	for _, tc := range cases {
		if got := fenceLanguage(tc.path); got != tc.want {
			t.Errorf("fenceLanguage(%q) = %q, want %q (Python)", tc.path, got, tc.want)
		}
	}
}
