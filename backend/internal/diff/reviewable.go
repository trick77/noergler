package diff

import "strings"

// skipExtensions are extensions never worth reviewing.
var skipExtensions = map[string]bool{
	// Binary / media
	".png": true, ".jpg": true, ".jpeg": true, ".gif": true, ".bmp": true,
	".ico": true, ".svg": true, ".webp": true,
	".mp3": true, ".mp4": true, ".wav": true, ".ogg": true, ".avi": true,
	".mov": true, ".mkv": true,
	".ttf": true, ".otf": true, ".woff": true, ".woff2": true, ".eot": true,
	".pdf": true, ".doc": true, ".docx": true, ".xls": true, ".xlsx": true,
	".ppt": true, ".pptx": true,
	".zip": true, ".tar": true, ".gz": true, ".bz2": true, ".xz": true,
	".7z": true, ".rar": true, ".jar": true, ".war": true,
	".exe": true, ".dll": true, ".so": true, ".dylib": true, ".bin": true,
	".class": true, ".pyc": true, ".o": true, ".a": true,
	".wasm": true,
	".db":   true, ".sqlite": true, ".sqlite3": true,
	// Data / config that rarely benefits from code review
	".json": true, ".lock": true, ".min.js": true, ".min.css": true, ".csv": true,
	".map": true,
	// Diagram markup, not code. .drawio/.excalidraw are serialized editor state
	// rather than hand-written markup, so they are noise either way.
	".puml": true, ".plantuml": true, ".pu": true, ".iuml": true,
	".mmd": true, ".drawio": true, ".dio": true, ".excalidraw": true,
	// Generated output: regenerating these produces large diffs with no
	// reviewable intent behind them.
	".snap": true, ".ambr": true,
	".po": true, ".mo": true, ".xlf": true,
	// Build / config files
	".bat": true, ".cmd": true,
}

// sizeGatedExtensions are reviewed only while their diff stays small: the
// value is the most changed (+/-) lines a file's diff may carry. A hand edit to
// Spring config or a few i18n keys is worth a look; a regenerated or bulk
// translation bundle is not. Unmeasured: tune from the "skipped as oversized"
// log line.
var sizeGatedExtensions = map[string]int{".properties": 50}

// skipFiles are lockfiles whose names carry no extension we can filter on:
// .lock misses both go.sum and pnpm-lock.yaml, and .yaml/.sum are legitimate
// source extensions.
var skipFiles = map[string]bool{
	"gradlew": true, "mvnw": true, "go.sum": true, "pnpm-lock.yaml": true,
}

// skipFileSuffixes are generated sources sharing an extension with hand-written
// code, so only the filename suffix distinguishes them (protobuf output).
var skipFileSuffixes = []string{"_pb2.py", "_pb2_grpc.py"}

var skipDirs = map[string]bool{
	"target": true, "build": true, "node_modules": true, "dist": true,
	"__pycache__": true,
}

// skipDirSuffixes are directory names carrying a project-specific prefix,
// matched by suffix rather than exact name (e.g. myproject.egg-info).
var skipDirSuffixes = []string{".egg-info"}

// binaryMarkerScanRunes bounds the binary-marker scan. Counted in runes, not
// bytes, so a multi-byte diff header does not shift the cut.
const binaryMarkerScanRunes = 500

// IsReviewable decides whether a per-file diff is worth sending to the model,
// before its content is fetched.
//
// Order matters because only step 3 can return early with true:
//  1. an extension heuristic over the lowercased raw first line, which catches
//     quoted and octal-escaped names the path regex misses;
//  2. a binary marker in the first 500 characters;
//  3. the parsed path.
//
// A diff whose path does not parse is reviewable, not skipped, so running
// step 3 first would pass a binary diff or a skipped extension whose header
// the path regex could not parse.
func IsReviewable(fileDiff string) bool {
	head := strings.ToLower(firstLine(fileDiff))
	// The heuristic scans the whole first line, so it sees the a/ side too:
	// `diff --git a/logo.png b/logo.py` is skipped on the strength of the old name.
	for ext := range skipExtensions {
		if strings.Contains(head, ext+"\"") || strings.Contains(head, ext+" ") ||
			strings.HasSuffix(head, ext) {
			return false
		}
	}

	scan := strings.ToLower(firstRunes(fileDiff, binaryMarkerScanRunes))
	if strings.Contains(scan, "binary files") && strings.Contains(scan, "differ") {
		return false
	}

	path := ExtractPath(fileDiff)
	if path == "" {
		return true
	}
	if !PathIsReviewable(path) {
		return false
	}
	_, _, over := ExceedsSizeGate(fileDiff)
	return !over
}

// ExceedsSizeGate reports whether a file diff of a sizeGatedExtensions type
// changes more lines than its limit. A diff whose path does not parse, or of
// any other type, is never over.
func ExceedsSizeGate(fileDiff string) (changed, limit int, over bool) {
	path := strings.ToLower(ExtractPath(fileDiff))
	if path == "" {
		return 0, 0, false
	}
	basename := path[strings.LastIndexByte(path, '/')+1:]
	for ext, max := range sizeGatedExtensions {
		if strings.HasSuffix(basename, ext) {
			added, removed := CountChangedLines(fileDiff)
			return added + removed, max, added+removed > max
		}
	}
	return 0, 0, false
}

// CountChangedLines counts one file diff's added and removed lines, not its
// ---/+++ headers.
func CountChangedLines(fileDiff string) (added, removed int) {
	for _, line := range strings.Split(fileDiff, "\n") {
		switch {
		case strings.HasPrefix(line, "+++"), strings.HasPrefix(line, "---"):
		case strings.HasPrefix(line, "+"):
			added++
		case strings.HasPrefix(line, "-"):
			removed++
		}
	}
	return added, removed
}

// PathIsReviewable applies the name-based half of IsReviewable to a bare path.
//
// It exists for callers holding a path but no diff text: the too-large log
// path, which lists a PR's files from `/changes` after the cap tripped at the
// socket. The extension heuristic and the binary-marker scan in IsReviewable
// both need the diff header, so a path-only verdict is the weaker of the two
// and may call a binary file reviewable when its extension is not listed.
// Likewise a sizeGatedExtensions file reads reviewable here: there is no diff
// to count.
//
// Shares the skip lists with IsReviewable: one source of truth, so a list
// edit cannot make the two disagree.
func PathIsReviewable(path string) bool {
	if path == "" {
		return true
	}
	path = strings.ToLower(path)

	parts := strings.Split(path, "/")
	basename := parts[len(parts)-1]
	for ext := range skipExtensions {
		if strings.HasSuffix(basename, ext) {
			return false
		}
	}
	if skipFiles[basename] || hasAnySuffix(basename, skipFileSuffixes) {
		return false
	}
	if strings.HasPrefix(basename, ".") {
		return false
	}
	for _, p := range parts[:len(parts)-1] {
		if skipDirs[p] || strings.HasPrefix(p, ".") || hasAnySuffix(p, skipDirSuffixes) {
			return false
		}
	}
	return true
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

// firstRunes returns at most n runes of s, so a multi-byte diff does not
// shift the cut where a byte slice would.
func firstRunes(s string, n int) string {
	count := 0
	for i := range s {
		if count == n {
			return s[:i]
		}
		count++
	}
	return s
}

func hasAnySuffix(s string, suffixes []string) bool {
	for _, suf := range suffixes {
		if strings.HasSuffix(s, suf) {
			return true
		}
	}
	return false
}
