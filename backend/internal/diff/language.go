package diff

import (
	"regexp"
	"sort"
	"strings"
)

// extensionLanguageMap maps a file extension to a language group.
// Case-sensitive on purpose: Main.PY is "other", not "python".
// TestDetectLanguage pins it.
var extensionLanguageMap = map[string]string{
	// Python
	".py": "python", ".pyi": "python",
	// JVM
	".java": "jvm", ".kt": "jvm", ".kts": "jvm", ".groovy": "jvm",
	// TypeScript/Angular
	".ts": "typescript", ".tsx": "typescript", ".js": "typescript",
	".jsx": "typescript", ".mjs": "typescript",
	// HTML (Angular templates)
	".html": "html", ".htm": "html",
	// CSS/Styles
	".css": "css", ".scss": "css", ".less": "css", ".sass": "css",
	// Config
	".yaml": "config", ".yml": "config", ".toml": "config",
	".ini": "config", ".cfg": "config",
	// Docs
	".md": "docs", ".rst": "docs", ".txt": "docs",
}

// buildFileMap is checked before the extension map, so build.gradle.kts is
// build-config rather than jvm.
var buildFileMap = map[string]string{
	"pom.xml":          "build-config",
	"build.gradle":     "build-config",
	"build.gradle.kts": "build-config",
	"angular.json":     "build-config",
	"nx.json":          "build-config",
	"project.json":     "build-config",
	"package.json":     "build-config",
}

var languagePriority = []string{
	"python", "jvm", "typescript", "html", "css", "build-config", "config", "docs", "other",
}

var deprioritizedLanguages = map[string]bool{
	"build-config": true, "config": true, "docs": true, "other": true,
}

// testPatterns is case-insensitive and unanchored, unlike DetectLanguage.
var testPatterns = regexp.MustCompile(`(?i)(?:` +
	`_test\.py$|test_[^/]*\.py$` +
	`|Test\.java$|Tests\.java$` +
	`|\.spec\.ts$|\.test\.ts$` +
	`|\.spec\.tsx$|\.test\.tsx$` +
	`|\.spec\.js$|\.test\.js$` +
	`|/tests?/|/__tests__/` +
	`)`)

// DetectLanguage returns the language group for a path. Case-sensitive.
func DetectLanguage(path string) string {
	basename := path
	if i := strings.LastIndex(path, "/"); i >= 0 {
		basename = path[i+1:]
	}
	if lang, ok := buildFileMap[basename]; ok {
		return lang
	}
	if strings.HasPrefix(basename, "tsconfig") && strings.HasSuffix(basename, ".json") {
		return "build-config"
	}
	ext := ""
	if dot := strings.LastIndex(basename, "."); dot >= 0 {
		ext = basename[dot:]
	}
	if lang, ok := extensionLanguageMap[ext]; ok {
		return lang
	}
	return "other"
}

// IsTestFile reports whether a path looks like a test. Case-insensitive.
func IsTestFile(path string) bool { return testPatterns.MatchString(path) }

// SortByLanguagePriority orders files by (group, language rank, path).
//
// The key reads only the path, never the diff or content, so the same PR sorts
// identically across re-reviews and unchanged files stay in the prefix-cache
// window. Sorting is stable, and Go string byte order equals code-point
// order, so the ordering is deterministic for any input.
func SortByLanguagePriority(files []FileReviewData) []FileReviewData {
	rank := make(map[string]int, len(languagePriority))
	for i, lang := range languagePriority {
		rank[lang] = i
	}
	maxRank := len(languagePriority)

	// group: 2 is checked before the test group, so tests/config.yaml is
	// group 2, not group 1.
	group := func(path string) int {
		lang := DetectLanguage(path)
		switch {
		case deprioritizedLanguages[lang]:
			return 2
		case IsTestFile(path):
			return 1
		default:
			return 0
		}
	}

	out := make([]FileReviewData, len(files))
	copy(out, files)
	sort.SliceStable(out, func(i, j int) bool {
		gi, gj := group(out[i].Path), group(out[j].Path)
		if gi != gj {
			return gi < gj
		}
		ri, ok := rank[DetectLanguage(out[i].Path)]
		if !ok {
			ri = maxRank
		}
		rj, ok := rank[DetectLanguage(out[j].Path)]
		if !ok {
			rj = maxRank
		}
		if ri != rj {
			return ri < rj
		}
		return out[i].Path < out[j].Path
	})
	return out
}
