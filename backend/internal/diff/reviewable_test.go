package diff

import (
	"strings"
	"testing"
)

func TestIsReviewable(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  bool
	}{
		{"plain source file", "diff --git a/src/main.go b/src/main.go\n@@ -1,1 +1,1 @@\n", true},
		{"png skipped by extension", "diff --git a/logo.png b/logo.png\n", false},
		// The head heuristic scans the whole first line, so the a/ side counts.
		{"old name png, new name py", "diff --git a/logo.png b/logo.py\n", false},
		{"quoted name", "diff --git \"a/my logo.png\" \"b/my logo.png\"\n", false},
		{"extension at end of line", "diff --git a/x.py b/x.png", false},
		{"binary marker", "diff --git a/f.bin2 b/f.bin2\nBinary files a/f.bin2 and b/f.bin2 differ\n", false},
		{"binary marker mixed case", "diff --git a/f.q b/f.q\nBINARY FILES a/f.q and b/f.q DIFFER\n", false},
		// Unparseable path is reviewable, not skipped.
		{"path does not parse", "@@ -1,1 +1,1 @@\n-a\n+b\n", true},
		{"go.sum skipped by name", "diff --git a/go.sum b/go.sum\n", false},
		{"pnpm lock skipped by name", "diff --git a/pnpm-lock.yaml b/pnpm-lock.yaml\n", false},
		{"gradlew skipped", "diff --git a/gradlew b/gradlew\n", false},
		{"protobuf suffix skipped", "diff --git a/api_pb2.py b/api_pb2.py\n", false},
		{"grpc protobuf suffix skipped", "diff --git a/api_pb2_grpc.py b/api_pb2_grpc.py\n", false},
		{"dotfile basename skipped", "diff --git a/.env b/.env\n", false},
		{"dotfile in normal directory skipped", "diff --git a/src/.hidden b/src/.hidden\n", false},
		{"dot directory skipped", "diff --git a/.github/workflows/ci.yaml b/.github/workflows/ci.yaml\n", false},
		{"node_modules skipped", "diff --git a/node_modules/x/i.js b/node_modules/x/i.js\n", false},
		{"target dir skipped", "diff --git a/target/out.go b/target/out.go\n", false},
		{"egg-info dir skipped", "diff --git a/pkg.egg-info/PKG b/pkg.egg-info/PKG\n", false},
		// Every .json is skipped here, even ones DetectLanguage calls build-config.
		{"package.json skipped", "diff --git a/package.json b/package.json\n", false},
		{"tsconfig.json skipped", "diff --git a/tsconfig.json b/tsconfig.json\n", false},
		{"src:// dst:// header", "diff --git src://main.go dst://main.go\n", true},
		{"src:// dst:// png skipped", "diff --git src://logo.png dst://logo.png\n", false},
		{"uppercase extension", "diff --git a/LOGO.PNG b/LOGO.PNG\n", false},
		{"deep path ok", "diff --git a/a/b/c/d.go b/a/b/c/d.go\n", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsReviewable(tc.input); got != tc.want {
				t.Errorf("IsReviewable() = %v, want %v", got, tc.want)
			}
		})
	}
}

// The binary-marker scan is bounded at 500 characters, not bytes, so a
// multi-byte prefix must not shift the cut.
func TestBinaryMarkerScanIsRuneBounded(t *testing.T) {
	// 400 multi-byte runes, then the marker: still inside the 500-rune window.
	prefix := "diff --git a/f.q b/f.q\n" + strings.Repeat("ü", 400) + "\n"
	if IsReviewable(prefix + "Binary files a/f.q and b/f.q differ\n") {
		t.Error("marker within 500 runes should be seen")
	}
	// Push the marker past 500 runes: no longer seen.
	far := "diff --git a/f.q b/f.q\n" + strings.Repeat("ü", 600) + "\n"
	if !IsReviewable(far + "Binary files a/f.q and b/f.q differ\n") {
		t.Error("marker beyond 500 runes should not be seen")
	}
}
