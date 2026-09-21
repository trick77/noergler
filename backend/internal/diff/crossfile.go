package diff

import (
	"fmt"
	"regexp"
	"strings"
)

// symbolNamePatterns extract the symbol name from a scope definition line.
//
// Every \w is [\pL\pN_]: RE2's \w is ASCII, Python's is Unicode. All four use
// unanchored search, and the jvm one has no method alternative, so a plain
// `public void processOrder(...)` yields nothing. Intentional, pinned by a
// Python test.
var symbolNamePatterns = map[string]*regexp.Regexp{
	"python": regexp.MustCompile(`^\s*(?:async\s+)?(?:def|class)\s+([\pL\pN_]+)`),
	"jvm": regexp.MustCompile(
		`(?:(?:public|private|protected|static|final|abstract|override|default)\s+)*` +
			`(?:fun|class|interface|enum|record)\s+([\pL\pN_]+)`),
	"typescript": regexp.MustCompile(
		`(?:export\s+)?(?:default\s+)?(?:async\s+)?` +
			`(?:function|class|interface|enum|type|const)\s+([\pL\pN_]+)`),
	"other": regexp.MustCompile(`(?:def|func|fn|class)\s+([\pL\pN_]+)`),
}

const (
	// minSymbolLen avoids matching single-letter variables.
	minSymbolLen = 3
	// maxRefsPerSymbol caps references per target file, not per symbol, so one
	// symbol across ten files can carry fifty references.
	maxRefsPerSymbol = 5
	// maxRelationshipLines caps the rendered section. It counts header and
	// reference lines but not the blank separators.
	maxRelationshipLines = 30
	// refLineTextRunes caps a reference line's text. Python sliced by
	// characters, so this counts runes.
	refLineTextRunes = 120
)

// SymbolReference is one reference to a symbol in another file.
type SymbolReference struct {
	File       string
	LineNumber int
	LineText   string
	// FromDiff records that the reference was found in the file's diff rather
	// than its content, so the renderer can label it honestly.
	FromDiff bool
}

// CrossFileRelationship is a changed symbol and where it is referenced.
type CrossFileRelationship struct {
	Symbol     string
	DefinedIn  string
	References []SymbolReference
}

// extractChangedSymbols pulls symbol names from added scope lines in a diff.
//
// Cross-file extraction deliberately does not honour noDynamicLanguages, so a
// `+def foo` in a .md diff is still extracted.
func extractChangedSymbols(f FileReviewData) []string {
	lang := DetectLanguage(f.Path)
	scopePattern, ok := scopePatterns[lang]
	if !ok {
		scopePattern = scopePatterns["other"]
	}
	namePattern, ok := symbolNamePatterns[lang]
	if !ok {
		namePattern = symbolNamePatterns["other"]
	}

	var symbols []string
	seen := map[string]bool{}
	// Mirrors Python splitlines().
	for _, line := range splitLines(f.Diff) {
		if !strings.HasPrefix(line, "+") || strings.HasPrefix(line, "+++") {
			continue
		}
		codeLine := line[1:]
		if !scopePattern.MatchString(codeLine) {
			continue
		}
		m := namePattern.FindStringSubmatch(codeLine)
		if m == nil {
			continue
		}
		name := m[1]
		if len([]rune(name)) >= minSymbolLen && !seen[name] {
			symbols = append(symbols, name)
			seen[name] = true
		}
	}
	return symbols
}

// findReferences locates a symbol in a target file's content, or its diff when
// no content was fetched.
func findReferences(pattern *regexp.Regexp, target FileReviewData) []SymbolReference {
	// Python's `content or diff`: an empty content string falls through.
	text, fromDiff := target.Content, false
	if text == "" {
		text, fromDiff = target.Diff, true
	}

	var refs []SymbolReference
	for i, line := range splitLines(text) {
		if !pattern.MatchString(line) {
			continue
		}
		stripped := strings.TrimSpace(line)
		// Comment-only lines are skipped. The definition line itself is not,
		// despite a Python comment claiming otherwise.
		if strings.HasPrefix(stripped, "#") || strings.HasPrefix(stripped, "//") ||
			strings.HasPrefix(stripped, "*") {
			continue
		}
		refs = append(refs, SymbolReference{
			File:       target.Path,
			LineNumber: i + 1,
			LineText:   firstRunes(stripped, refLineTextRunes),
			FromDiff:   fromDiff,
		})
		if len(refs) >= maxRefsPerSymbol {
			break
		}
	}
	return refs
}

// BuildRelationships finds, for every symbol changed in one file, the other PR
// files that reference it.
func BuildRelationships(files []FileReviewData) []CrossFileRelationship {
	if len(files) < 2 {
		return nil
	}

	type fileSymbols struct {
		path    string
		symbols []string
	}
	var ordered []fileSymbols
	for _, f := range files {
		if syms := extractChangedSymbols(f); len(syms) > 0 {
			ordered = append(ordered, fileSymbols{path: f.Path, symbols: syms})
		}
	}
	if len(ordered) == 0 {
		return nil
	}

	// Python compiled the word-boundary regex per (symbol, target file) pair,
	// inside the inner loop. One per symbol is identical in behaviour and the
	// pair loop is quadratic.
	patterns := map[string]*regexp.Regexp{}
	patternFor := func(symbol string) *regexp.Regexp {
		if re, ok := patterns[symbol]; ok {
			return re
		}
		patterns[symbol] = symbolBoundaryRE(symbol)
		return patterns[symbol]
	}

	var relationships []CrossFileRelationship
	for _, src := range ordered {
		for _, symbol := range src.symbols {
			rel := CrossFileRelationship{Symbol: symbol, DefinedIn: src.path}
			re := patternFor(symbol)
			for _, target := range files {
				if target.Path == src.path {
					continue
				}
				rel.References = append(rel.References, findReferences(re, target)...)
			}
			if len(rel.References) > 0 {
				relationships = append(relationships, rel)
			}
		}
	}
	return relationships
}

// RenderRelationships renders the relationships as a prompt section.
//
// Output ends with a trailing newline when not truncated, and with the exact
// string "_(additional relationships truncated)_" and no trailing newline when
// it is.
func RenderRelationships(relationships []CrossFileRelationship) string {
	if len(relationships) == 0 {
		return ""
	}

	lines := []string{
		"## Cross-file relationships",
		"",
		"The following symbols were changed and are referenced in other files in this PR. " +
			"Pay special attention to whether callers/consumers are updated consistently.",
		"",
	}

	total := 0
	truncated := false
	for _, rel := range relationships {
		if total >= maxRelationshipLines {
			truncated = true
			break
		}
		lines = append(lines, fmt.Sprintf("**`%s`** (changed in `%s`) is referenced in:", rel.Symbol, rel.DefinedIn))
		total++
		for _, ref := range rel.References {
			if total >= maxRelationshipLines {
				truncated = true
				break
			}
			// A reference found in a diff is labelled as a diff line rather
			// than passed off as a file line.
			location := fmt.Sprintf("%s:%d", ref.File, ref.LineNumber)
			if ref.FromDiff {
				location = fmt.Sprintf("%s (diff line %d)", ref.File, ref.LineNumber)
			}
			lines = append(lines, fmt.Sprintf("- `%s` — `%s`", location, ref.LineText))
			total++
		}
		lines = append(lines, "")
	}

	if truncated {
		lines = append(lines, "_(additional relationships truncated)_")
	}
	return strings.Join(lines, "\n")
}

// symbolBoundaryRE matches symbol delimited by non-word runes.
//
// RE2's \b is ASCII-only, so `\bÖlservice\b` never matches `new Ölservice();`:
// RE2 sees no word character at the Ö and the assertion fails. Python's \b is
// Unicode-aware and does match, so the explicit boundaries restore parity
// rather than diverge from it. Same trap as \w, third occurrence.
func symbolBoundaryRE(symbol string) *regexp.Regexp {
	const notWord = `[^\pL\pN_]`
	return regexp.MustCompile(
		`(?:^|` + notWord + `)` + regexp.QuoteMeta(symbol) + `(?:$|` + notWord + `)`)
}

// splitLines mirrors Python str.splitlines (no terminators kept).
func splitLines(s string) []string {
	var out []string
	for _, l := range splitLinesKeepEnds(s) {
		out = append(out, strings.TrimRight(l, "\n\r\v\f\u001c\u001d\u001e\u0085  "))
	}
	return out
}
