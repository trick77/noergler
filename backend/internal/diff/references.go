package diff

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// A changed signature breaks callers the PR does not touch, and the review
// sees only the files it changes. One code search per review finds those
// callers: code picks the symbols and the lines, never the model.
//
// Bounded on purpose (user, 2026-11-20: never a burst of searches from one
// review): one request, at most maxSearchSymbols names in it, at most
// maxRefsPerSearchSymbol lines a name and maxSearchRefLines in all.
const (
	maxSearchSymbols       = 6
	maxRefsPerSearchSymbol = 5
	maxSearchRefLines      = 20
	// minSearchSymbolLen is longer than minSymbolLen: a short name ("Get",
	// "run") matches half a repository, and a search of it is noise.
	minSearchSymbolLen = 4
	// maxSearchQueryLen is the search's own limit on a query.
	maxSearchQueryLen = 250
	// maxSearchRefRunes is the longest line a reference may be.
	maxSearchRefRunes = 200
)

// searchableName is a name the search indexes whole: letters, digits, "_"
// and "." survive, every other punctuation character is dropped by the
// index, so a name holding one would be searched as two words.
var searchableName = regexp.MustCompile(`^[\pL\pN_.]+$`)

// definitionPatterns find the name an added line defines, methods included:
// a changed method is what an unchanged caller breaks on. Their own set, not
// symbolNamePatterns, which leaves methods out on purpose for the cross-file
// block (TestExtractChangedSymbols). Every \w spelled [\pL\pN_] (RE2's \w
// is ASCII). A JVM method needs a modifier, so a call statement
// (`order.process(x);`) is never read as one.
//
// JVM: Kotlin's `enum class`, `data class`, `fun interface` and an extension
// function's receiver (`fun String.toSlug(`) are stepped over, so the name
// is captured, never the keyword or the receiver type. TypeScript: a
// `const` counts at column 0 or after `export` only (an indented one is a
// local); a class method needs a modifier, as a JVM one does.
//
// Written with \s for reading; unicodeSpaced makes each one Unicode-aware.
var definitionPatterns = map[string]*regexp.Regexp{
	"go":     unicodeSpaced(`^\s*func\s+(?:\([^)]*\)\s*)?([\pL\pN_]+)\s*[\[(]`),
	"python": unicodeSpaced(`^\s*(?:async\s+)?(?:def|class)\s+([\pL\pN_]+)`),
	"jvm": unicodeSpaced(
		`^\s*(?:@[\pL\pN_.]+(?:\([^)]*\))?\s+)*` +
			`(?:(?:public|private|protected|internal|static|final|abstract|override|open|suspend|synchronized|default|inline|operator|infix|tailrec)\s+)*` +
			`(?:(?:enum|data|sealed|annotation|value|inner|fun)\s+)?` +
			`(?:(?:fun|class|interface|enum|record|object)\s+(?:<[^>]*>\s*)?(?:[\pL\pN_]+(?:<[^>]*>)?\??\.)?([\pL\pN_]+)` +
			`|[\pL\pN_<>\[\],.? ]+?\s+([\pL\pN_]+)\s*\()`),
	"typescript": unicodeSpaced(
		`^(?:export\s+(?:default\s+)?(?:async\s+)?(?:function\*?|class|interface|(?:const\s+)?enum|type|const|let)\s+([\pL\pN_]+)` +
			`|(?:async\s+)?(?:function\*?|class|interface|(?:const\s+)?enum|type|const)\s+([\pL\pN_]+))` +
			`|^\s+(?:(?:public|private|protected|static|async|readonly|override|abstract|get|set)\s+)+\*?([\pL\pN_]+)\s*(?:<[^>]*>)?\s*\(`),
}

// jvmKeywordLed is a JVM line led by a modifier: the pattern's method branch
// (no fun/class keyword) counts only then, so `order.process(x);` and
// `return build(x);` are no definitions. The keyword branch needs none.
var jvmKeywordLed = unicodeSpaced(`^\s*(?:@[\pL\pN_.]+(?:\([^)]*\))?\s+)*(?:(?:public|private|protected|internal|static|final|abstract|override|open|suspend|synchronized|default|inline|operator|infix|tailrec)\s+)+`)

// unicodeSpaced compiles a pattern with each \s widened to
// [\s\p{Zs}]: RE2's \s is ASCII, a no-break space would end a match. No
// pattern here holds \s inside a bracket class, where the swap would nest.
func unicodeSpaced(p string) *regexp.Regexp {
	return regexp.MustCompile(strings.ReplaceAll(p, `\s`, `[\s\p{Zs}]`))
}

// reservedNames are words a pattern must never yield as a name
// (`export default class extends Base` names no class).
var reservedNames = map[string]bool{
	"extends": true, "implements": true,
	"class": true, "interface": true, "enum": true, "record": true, "object": true, "fun": true,
	"function": true, "const": true, "type": true, "return": true, "new": true,
}

// commonNames are names so many types define (interface methods, entry
// points, conventions) that their hits would fill the one search and push
// out the callers that matter.
var commonNames = map[string]bool{
	"main": true, "init": true, "String": true, "Error": true, "Close": true, "Read": true, "Write": true,
	"Len": true, "Less": true, "Swap": true, "ServeHTTP": true, "MarshalJSON": true, "UnmarshalJSON": true,
	"__init__": true, "__str__": true, "__repr__": true, "__eq__": true, "__hash__": true,
	"toString": true, "equals": true, "hashCode": true, "compareTo": true, "constructor": true,
	"render": true, "ngOnInit": true, "ngOnDestroy": true,
}

// definedName is the name one diff line defines, "" for none.
func definedName(p *regexp.Regexp, lang, code string) string {
	m := p.FindStringSubmatch(code)
	if m == nil {
		return ""
	}
	for i, name := range m[1:] {
		if name == "" {
			continue
		}
		// The JVM's method branch (the second group) needs a modifier.
		if lang == "jvm" && i == 1 && !jvmKeywordLed.MatchString(code) {
			return ""
		}
		if reservedNames[name] {
			return ""
		}
		return name
	}
	return ""
}

// definedNames are the names the diff's added and removed lines define,
// each in order.
func definedNames(f FileReviewData) (added, removed []string) {
	lang := DetectLanguage(f.Path)
	if strings.HasSuffix(f.Path, ".go") {
		lang = "go"
	}
	p, ok := definitionPatterns[lang]
	if !ok {
		return nil, nil
	}
	for _, line := range splitLines(f.Diff) {
		switch {
		case strings.HasPrefix(line, "+++"), strings.HasPrefix(line, "---"):
		case strings.HasPrefix(line, "+"):
			if n := definedName(p, lang, line[1:]); n != "" {
				added = append(added, n)
			}
		case strings.HasPrefix(line, "-"):
			if n := definedName(p, lang, line[1:]); n != "" {
				removed = append(removed, n)
			}
		}
	}
	return added, removed
}

// SearchSymbols are the names a search looks for, ranked by how likely an
// unchanged caller breaks: a definition the diff both removes and adds (a
// changed signature) first, then one it only removes (renamed or deleted:
// the callers of the old name break). A name the diff only adds is never
// searched: the default branch has no caller of it yet, so every hit would
// be another symbol of the same name, shown as one the PR changes.
// Searchable, long enough, not a name every type defines; capped.
func SearchSymbols(files []FileReviewData) []string {
	addedSet, removedSet := map[string]bool{}, map[string]bool{}
	var order []string
	seen := map[string]bool{}
	for _, f := range files {
		added, removed := definedNames(f)
		for _, s := range removed {
			removedSet[s] = true
			if !seen[s] {
				seen[s] = true
				order = append(order, s)
			}
		}
		for _, s := range added {
			addedSet[s] = true
			if !seen[s] {
				seen[s] = true
				order = append(order, s)
			}
		}
	}
	rank := func(s string) int {
		switch {
		case addedSet[s] && removedSet[s]:
			return 0
		case removedSet[s]:
			return 1
		}
		return 2
	}
	var out []string
	for r := 0; r <= 1; r++ {
		for _, s := range order {
			if rank(s) != r || len([]rune(s)) < minSearchSymbolLen || !searchableName.MatchString(s) || commonNames[s] {
				continue
			}
			out = append(out, s)
			if len(out) == maxSearchSymbols {
				return out
			}
		}
	}
	return out
}

// SearchQuery is one query for the symbols in one repository, within the
// search's length limit: names that would take it over are left out, last
// first. used are the names it carries; none when not even one fits.
func SearchQuery(symbols []string, project, repo string) (query string, used []string) {
	scope := fmt.Sprintf(" project:%s repo:%s", project, repo)
	for n := len(symbols); n > 0; n-- {
		terms := strings.Join(symbols[:n], " OR ")
		if n > 1 {
			terms = "(" + terms + ")"
		}
		if q := terms + scope; len(q) <= maxSearchQueryLen {
			return q, symbols[:n]
		}
	}
	return "", nil
}

// SearchHit is one file a search found, with the lines it returned.
type SearchHit struct {
	File  string
	Lines []SearchLine
}

// SearchLine is one returned line, as plain text.
type SearchLine struct {
	Line int
	Text string
}

// UnchangedRef is one line of a file the PR does not change that names a
// symbol the PR changes.
type UnchangedRef struct {
	Symbol string `json:"symbol"`
	File   string `json:"file"`
	Line   int    `json:"line"`
	Text   string `json:"text"`
}

// MatchReferences keeps the returned lines that name a symbol as a whole
// word, case as written (the search ignores case; a reference does not), in
// a file outside skip (the PR's own files: the search reads the default
// branch, where they are still the old version). A line that starts as a
// comment is no reference, nor is a line in docs or config. Capped per
// symbol and in all; ordered by symbol, production before test files, then
// path and line, so the same PR renders the same block.
func MatchReferences(symbols []string, hits []SearchHit, skip map[string]bool) []UnchangedRef {
	// One entry per symbol: its pattern and the lines found for it.
	type perSymbol struct {
		name string
		re   *regexp.Regexp
		refs []UnchangedRef
	}
	bySymbol := make([]*perSymbol, len(symbols))
	for i, s := range symbols {
		bySymbol[i] = &perSymbol{name: s, re: symbolBoundaryRE(s)}
	}
	seen := map[string]bool{}
	for _, h := range hits {
		if skip[h.File] || !callerLanguage(h.File) {
			continue
		}
		for _, l := range h.Lines {
			stripped := strings.TrimSpace(l.Text)
			// A line longer than the cap is left out, never cut: the model
			// quotes what it sees, and a cut line quotes as no line at all.
			if isCommentLine(stripped) || len([]rune(stripped)) > maxSearchRefRunes {
				continue
			}
			for _, ps := range bySymbol {
				if !ps.re.MatchString(l.Text) {
					continue
				}
				key := fmt.Sprintf("%s|%s|%d", ps.name, h.File, l.Line)
				if !seen[key] {
					seen[key] = true
					ps.refs = append(ps.refs, UnchangedRef{Symbol: ps.name, File: h.File, Line: l.Line,
						Text: stripped})
				}
			}
		}
	}
	var out []UnchangedRef
	for _, ps := range bySymbol {
		refs := ps.refs
		// Production callers before tests: a test calling the old form is
		// worth less than code that ships, and the cap would cut by path.
		sort.Slice(refs, func(a, b int) bool {
			if ta, tb := testCaller(refs[a].File), testCaller(refs[b].File); ta != tb {
				return tb
			}
			if refs[a].File != refs[b].File {
				return refs[a].File < refs[b].File
			}
			return refs[a].Line < refs[b].Line
		})
		for _, r := range refs[:min(len(refs), maxRefsPerSearchSymbol)] {
			if len(out) == maxSearchRefLines {
				return out
			}
			out = append(out, r)
		}
	}
	return out
}

// RenderUnchangedReferences is the prompt block; "" for none.
func RenderUnchangedReferences(refs []UnchangedRef) string {
	if len(refs) == 0 {
		return ""
	}
	lines := []string{
		"## References in unchanged files",
		"",
		"Lines from files this PR does not change, as they stand on the repository's default branch, " +
			"that name a symbol the PR changes. Check that the PR's change still fits them. They are context: " +
			"quote one as evidence beside a line of the PR's own diff, never anchor a finding on it. " +
			"Only these lines of those files are shown; do not guess at the rest.",
		"",
	}
	symbol := ""
	for _, r := range refs {
		if r.Symbol != symbol {
			if symbol != "" {
				lines = append(lines, "")
			}
			symbol = r.Symbol
			lines = append(lines, fmt.Sprintf("**`%s`** is used in:", r.Symbol))
		}
		lines = append(lines, fmt.Sprintf("- `%s:%d` — %s", r.File, r.Line, codeSpan(r.Text)))
	}
	return strings.Join(lines, "\n")
}

// callerLanguage reports whether a file can hold a call: docs, config and
// build files only mention a name, and would take a symbol's few slots.
func callerLanguage(path string) bool {
	switch DetectLanguage(path) {
	case "docs", "config", "build-config":
		return false
	}
	return true
}

// testCaller reports whether a caller sits in a test: IsTestFile, plus Go's
// _test.go, which its patterns leave to the language.
func testCaller(path string) bool {
	return IsTestFile(path) || strings.HasSuffix(path, "_test.go")
}

// isCommentLine reports whether a trimmed line opens as a comment: `#`,
// `//`, `/*`, a docstring quote, or a block comment's `*` continuation
// (a lone `*` or one followed by a space or `/`; `*out = f(x)` is code).
func isCommentLine(s string) bool {
	switch {
	case s == "", s == "*", strings.HasPrefix(s, "#"), strings.HasPrefix(s, "//"), strings.HasPrefix(s, "/*"),
		strings.HasPrefix(s, `"""`), strings.HasPrefix(s, "'''"), strings.HasPrefix(s, "* "), strings.HasPrefix(s, "*/"):
		return true
	}
	return false
}

// codeSpan wraps code in a backtick run longer than any inside it, padded
// with a space when it starts or ends with a backtick (CommonMark), so a
// backtick in the line never closes the span early.
func codeSpan(s string) string {
	longest, run := 0, 0
	for _, r := range s {
		if r == '`' {
			run++
			longest = max(longest, run)
			continue
		}
		run = 0
	}
	fence := strings.Repeat("`", longest+1)
	if strings.HasPrefix(s, "`") || strings.HasSuffix(s, "`") {
		return fence + " " + s + " " + fence
	}
	return fence + s + fence
}

// ReferenceLines are the referenced lines' texts, for the anchor index:
// quotable as evidence, never an anchor.
func ReferenceLines(refs []UnchangedRef) []string {
	out := make([]string, len(refs))
	for i, r := range refs {
		out[i] = r.Text
	}
	return out
}
