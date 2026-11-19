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

// testPatterns is case-insensitive, unlike DetectLanguage, except for the
// JVM suffixes: the capitalised Test, TestCase, IT (Failsafe, after a
// lowercase letter or digit, so PIT.java is no test) and Spec (Spock,
// Kotest) are the convention, and folding them made Latest.java and
// Contest.java tests. A Python test_ starts a path segment or a snake_case
// word (integration_test_helpers.py), never inside a word (latest_utils.py);
// test(s).py, conftest.py and *_tests.py are tests by name. A test
// directory starts a path segment, and so does a camelCase source set ending
// in Test (androidTest, commonTest, integrationTest) or testFixtures.
var testPatterns = regexp.MustCompile(`(?:` +
	`(?i:_tests?\.py$|(?:^|[/_])test_[^/]*\.py$|(?:^|/)(?:tests?|conftest)\.py$)` +
	`|(?:Tests?|TestCase|[a-z0-9]IT)\.(?:java|kt|groovy)$|Spec\.(?:groovy|kt)$` +
	`|(?i:\.(?:spec|test|e2e-spec|cy)\.(?:ts|tsx|js|jsx|mjs)$)` +
	`|(?i:(?:^|/)(?:tests?|__tests__|testFixtures)/)` +
	`|(?:^|/)src/[a-z][A-Za-z]*Test/` +
	`)`)

// mainSourceSet is Maven's and Gradle's production tree: a file under it is
// never a test, whatever its name (a pipeline DSL's PipelineSpec.groovy, a
// SpeedTest feature).
var mainSourceSet = regexp.MustCompile(`(?:^|/)src/main/`)

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

// IsTestFile reports whether a path looks like a test (testPatterns).
func IsTestFile(path string) bool {
	return !mainSourceSet.MatchString(path) && testPatterns.MatchString(path)
}

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
