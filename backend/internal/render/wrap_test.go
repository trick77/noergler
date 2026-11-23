package render

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"unicode/utf8"
)

// TestWrapProseGoldenCorpus is the oracle for the whole wrap path.
//
// testdata/wrap_golden.json holds 31 inputs with their expected output,
// compared byte for byte: the wrapper is hand-written, so it is pinned
// against a fixed corpus rather than against a reading of the code. Five of
// the cases exist only to catch the Unicode-class traps in the two
// structural regexes.
func TestWrapProseGoldenCorpus(t *testing.T) {
	blob, err := os.ReadFile("testdata/wrap_golden.json")
	if err != nil {
		t.Fatalf("read golden corpus: %v", err)
	}
	var cases []struct {
		In  string `json:"in"`
		Out string `json:"out"`
	}
	if err := json.Unmarshal(blob, &cases); err != nil {
		t.Fatalf("decode golden corpus: %v", err)
	}
	if len(cases) < 36 {
		t.Fatalf("golden corpus has %d cases, expected at least 36", len(cases))
	}
	for i, c := range cases {
		got := WrapProse(c.In)
		if got != c.Out {
			t.Errorf("case %d\n in:   %q\n got:  %q\n want: %q", i, c.In, got, c.Out)
		}
	}
}

// A non-breaking space is not in the wrapper's whitespace set, so it never
// becomes a break point. strings.Fields would split on it.
func TestNBSPIsNotABreakPoint(t *testing.T) {
	in := "Text with a non breaking space run that is long enough to need wrapping across the configured column width limit here."
	got := WrapProse(in)
	if !strings.Contains(got, "non breaking space") {
		t.Errorf("NBSP run was split: %q", got)
	}
}

// An NBSP-indented quote or heading is structural and passes through
// unwrapped. RE2's \s is ASCII and would miss it, hence [\s\p{Zs}].
func TestNBSPIndentedStructureIsNotWrapped(t *testing.T) {
	for _, in := range []string{
		" > Quote indented with a non-breaking space that must never wrap no matter how long the line runs in total.",
		" # Heading indented with a non-breaking space that must never wrap no matter how long the line runs here.",
	} {
		if got := WrapProse(in); got != in {
			t.Errorf("structural line was wrapped\n in:  %q\n got: %q", in, got)
		}
	}
}

// An Arabic-Indic marker is a numbered list item and wraps at the narrower
// list width. RE2's \d is ASCII and would miss it, hence \p{Nd}.
func TestArabicIndicMarkerIsAListItem(t *testing.T) {
	in := "١. Numbered item with an Arabic-Indic marker and enough words that the wrapping kicks in at the list width, which is the prose width."
	got := WrapProse(in)
	if !strings.Contains(got, "\n") {
		t.Fatalf("expected a wrap at the list width, got %q", got)
	}
	// The hang indent is three columns, matching the marker width.
	lines := strings.Split(got, "\n")
	if !strings.HasPrefix(lines[1], "   ") {
		t.Errorf("continuation line lacks the hang indent: %q", lines[1])
	}
}

// Tabs expand to 8-column stops before wrapping, so the output is space
// padded and carries no tabs at all.
func TestTabsExpandToEightColumnStops(t *testing.T) {
	got := WrapProse("a\tb\tc")
	if strings.ContainsRune(got, '\t') {
		t.Errorf("tabs survived expansion: %q", got)
	}
	if got != "a       b       c" {
		t.Errorf("tab expansion = %q, want %q", got, "a       b       c")
	}
}

// _wrap_chunks calls chunk.strip(), which is Unicode, so a chunk that is only
// non-breaking spaces is blank and gets dropped -- even though the SPLITTER
// never treats NBSP as a separator. The two rules differ on purpose.
func TestNBSPChunkCountsAsBlank(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{" ", nil},
		{"hello  ", []string{"hello "}},
		{"hello  ", []string{"hello"}},
		{" ", nil},
	}
	for _, c := range cases {
		got := textwrapWrap(c.in, 10, "", "")
		if len(got) != len(c.want) {
			t.Errorf("textwrapWrap(%q) = %q, want %q", c.in, got, c.want)
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("textwrapWrap(%q) = %q, want %q", c.in, got, c.want)
				break
			}
		}
	}
}

// Tab expansion resets the column on \n and \r only; \v and \f are ordinary
// width-1 characters.
func TestExpandTabsOnlyResetsOnNewlineAndCarriageReturn(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"a\vb\tc", "a\vb     c"},
		{"a\fb\tc", "a\fb     c"},
		{"a\nb\tc", "a\nb       c"},
		{"a\rb\tc", "a\rb       c"},
	}
	for _, c := range cases {
		if got := expandTabs(c.in); got != c.want {
			t.Errorf("expandTabs(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestShortStringIsUnchanged(t *testing.T) {
	if got := WrapProse("Looks good."); got != "Looks good." {
		t.Errorf("short string changed: %q", got)
	}
	if got := WrapProse(""); got != "" {
		t.Errorf("empty string changed: %q", got)
	}
}

func TestIdempotent(t *testing.T) {
	in := "The repository layer now batches writes, which materially reduces round trips to PostgreSQL under sustained review load on large PRs and beyond."
	once := WrapProse(in)
	if twice := WrapProse(once); twice != once {
		t.Errorf("not idempotent\n once:  %q\n twice: %q", once, twice)
	}
}

// A break never makes structure out of prose: whatever the column the word
// lands on, no line the wrap added starts with a Markdown block marker, before
// or after NoEmDash, and no text is lost.
func TestWrapNeverStartsALineWithABlockMarker(t *testing.T) {
	// The literal marker is the oracle, never startsBlock itself.
	words := func(s string) []string { return strings.FieldsFunc(s, isWrapWhitespace) }
	for _, marker := range []string{
		"-", "*", "+", "•", "—", "1.", "28.", "3)", "#", "##", ">", ">=5", "---", "——", "=", "==", "***", "___",
		"1.\u00a0Januar", "-\u00a0foo", "#\u00a0x", "```go fmt```", "```", "~~~",
	} {
		first := words(marker)[0]
		for _, prefix := range []string{"", "- ", "1. "} {
			for n := 60; n <= 2*WrapWidth; n++ {
				text := strings.Repeat("word ", n/5) + strings.Repeat("x", n%5+1)
				in := prefix + text + " " + marker + " rest " + marker + " " + marker + " of the sentence — it goes on for a while " + marker
				got := WrapProse(in)
				if strings.Join(words(got), " ") != in {
					t.Fatalf("%q after %d columns: text changed\n got: %q", marker, n, got)
				}
				for _, line := range strings.Split(got, "\n")[1:] {
					if words(line)[0] == first {
						t.Fatalf("%q after %d columns: line starts with it: %q", marker, n, line)
					}
				}
				if marker != "```" && strings.Contains(NoEmDash(got), emDash) { // two ``` are a code span
					t.Fatalf("%q after %d columns: an em dash is left: %q", marker, n, NoEmDash(got))
				}
				for _, line := range strings.Split(got, "\n") {
					if w := utf8.RuneCountInString(line); w > WrapWidth {
						t.Fatalf("%q after %d columns: line of %d columns: %q", marker, n, w, line)
					}
				}
			}
		}
	}
}

// The markers that only count when a line holds one of them: two pipes are a
// table row and pass through, two fences a code span.
func TestWrapNeverStartsALineWithALoneMarker(t *testing.T) {
	for _, marker := range []string{"|", "```"} {
		for n := 60; n <= 2*WrapWidth; n++ {
			got := WrapProse(strings.Repeat("word ", n/5) + strings.Repeat("x", n%5+1) + " " + marker + " the rest of the sentence that goes on for a while")
			if strings.Contains(got, "\n"+marker) {
				t.Fatalf("%q after %d columns: %q", marker, n, got)
			}
		}
	}
}

// With no other word to move down, the marker stays on the line and the line
// overflows.
func TestWrapMarkerAfterAnOverlongWordOverflows(t *testing.T) {
	url := "https://bitbucket.example.com/" + strings.Repeat("x", WrapWidth)
	got := WrapProse(url + " — - the rest of the sentence")
	if want := url + " — -\nthe rest of the sentence"; got != want {
		t.Errorf("\n got:  %q\n want: %q", got, want)
	}
}

func TestLongTokenNeverSplit(t *testing.T) {
	url := "https://bitbucket.example.com/projects/FOO/repos/bar/pull-requests/42"
	got := WrapProse("See the discussion at " + url + " for the full rationale here and more words to push it over the cap.")
	if !strings.Contains(got, url) {
		t.Errorf("URL was chopped: %q", got)
	}
}

func TestCodeSpansPreserved(t *testing.T) {
	got := WrapProse("The handler now calls `do thing` before returning, which keeps the ordering stable across retries and avoids a subtle race condition.")
	if !strings.Contains(got, "`do thing`") {
		t.Errorf("code span with a space was split: %q", got)
	}

	got = WrapProse("Compare `alpha` and `beta` carefully here because the ordering of those two calls genuinely matters for correctness in this path.")
	if !strings.Contains(got, "`alpha`") || !strings.Contains(got, "`beta`") {
		t.Errorf("adjacent spans not both restored: %q", got)
	}
}

// The mask is padded to the span's real width so the wrapper counts the
// span's true column width; an unpadded placeholder would let lines overflow.
func TestCodeSpanCountsTowardWidth(t *testing.T) {
	span := "`ServiceLeistungsfallDatenschutz`"
	got := WrapProse("Removes the personengruppe Fremdfall privacy workaround from " + span +
		", including the injected feature-flag usage entirely from the service.")
	for _, line := range strings.Split(got, "\n") {
		if n := utf8.RuneCountInString(line); n > WrapWidth {
			t.Errorf("line of %d runes exceeds the %d cap: %q", n, WrapWidth, line)
		}
	}
}

// Fenced blocks pass through whole, however long the lines inside run.
func TestFencedBlockUntouched(t *testing.T) {
	in := "```\nsome code that is very long and must not be wrapped at all even though it exceeds the width limit easily\n```"
	if got := WrapProse(in); got != in {
		t.Errorf("fence contents changed\n in:  %q\n got: %q", in, got)
	}
}

// \x00 and \x01 in the input would collide with the mask placeholders and are
// stripped before wrapping.
func TestPlaceholderCharsStrippedFromInput(t *testing.T) {
	got := WrapProse("A line with a \x00 null and a \x01 filler embedded in prose.")
	if strings.ContainsAny(got, "\x00\x01") {
		t.Errorf("placeholder chars survived: %q", got)
	}
}

// A table row is structure, like a heading: a reply's settings table broken
// across lines is no table.
func TestTableRowIsNotWrapped(t *testing.T) {
	row := "| Setting | Value that is long enough to cross the prose width if anything here were to wrap it, which it must not |"
	if got := WrapProse(row); got != row {
		t.Errorf("table row was wrapped: %q", got)
	}
}

// Lists wrap at the prose width, not earlier than the paragraphs beside them.
func TestListWrapsAtTheProseWidth(t *testing.T) {
	item := "- " + strings.Repeat("word ", 21) + "end"
	if got := WrapProse(item); strings.Contains(got, "\n") {
		t.Errorf("a %d-rune item broke before the prose width: %q", len(item), got)
	}
}

// Code a reply carries outside ``` fences is left whole: a ~~~ fence, an
// indented block and a pipe table without its leading pipe.
func TestCodeOutsideBacktickFencesIsNotWrapped(t *testing.T) {
	long := strings.Repeat("token ", 25)
	for _, in := range []string{
		"~~~go\n" + long + "\n~~~",
		"    " + long,
		"\t" + long,
		"Setting | Value | " + long,
	} {
		if got := WrapProse(in); got != in {
			t.Errorf("code was wrapped\n in:  %q\n got: %q", in, got)
		}
	}
}
