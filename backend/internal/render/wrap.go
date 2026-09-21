package render

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// WrapWidth is the target column for plain prose.
//
// Bitbucket renders LLM free-text as one paragraph the full browser width.
// We hard-wrap so no line runs full width.
const WrapWidth = 110

// ListWrapWidth is the narrower target for list items: a marker plus a hang
// indent would otherwise let them render wider than the surrounding text.
const ListWrapWidth = 90

// HardBreak is the token joining wrapped pieces. Bitbucket Data Center's
// CommonMark renderer treats a single newline inside a paragraph as a visible
// break, so it is just "\n". Changing it to a multi-character token would
// also require changing the re-split in WrapProse.
const HardBreak = "\n"

// neverWrap matches fence-adjacent structure we pass through untouched:
// block quotes and ATX headings.
//
// The indent class must be Unicode-aware: a line indented with a
// non-breaking space is still a heading or quote. RE2's \s is ASCII and
// would miss it, so the class is spelled [\s\p{Zs}]. Without it an
// NBSP-indented " > quote" gets wrapped and its block structure is
// mangled (TestNBSPIndentedStructureIsNotWrapped).
var neverWrap = regexp.MustCompile(`^[\s\p{Zs}]*(?:>[\s\p{Zs}]|#{1,6}[\s\p{Zs}])`)

// listItem captures (marker, content) so the content wraps while the marker
// is preserved and continuation lines hang-indent under the text.
//
// Same Unicode-class treatment as neverWrap, plus \p{Nd} for the digits:
// RE2's \d is ASCII, so "١. item" would read as plain prose instead of a
// numbered item (TestArabicIndicMarkerIsAListItem). The marker set must stay
// in sync with what the model emits; an unrecognized marker falls through to
// the wider prose path.
var listItem = regexp.MustCompile(`^([\s\p{Zs}]*(?:[-*+\x{2022}]|\p{Nd}+[.)])[\s\p{Zs}]+)(.*)$`)

// codeSpan matches inline code. Spans are masked before wrapping so one with
// internal spaces is never split across a break, then restored verbatim.
var codeSpan = regexp.MustCompile("`[^`]+`")

// placeholder matches a mask: an index between NULs, padded with \x01 to the
// span's real width so the wrapper counts the span's true column width.
var placeholder = regexp.MustCompile("\x00(\\d+)\x00\x01*")

// WrapProse hard-wraps plain prose lines at width and passes structural lines
// through.
//
// Headings, block quotes and fenced code blocks are left intact; list items
// wrap at listWidth with their marker preserved. Inline code spans and long
// unbreakable tokens (URLs, path/to/file) are never split, so the width is a
// target and not a guarantee.
//
// Idempotent, with one bounded exception: a list-item continuation line that
// is a single over-width token loses its hang indent on a second pass.
// Cosmetic, and every caller wraps fresh model output once.
func WrapProse(text string) string {
	return WrapProseWidth(text, WrapWidth, ListWrapWidth)
}

// WrapProseWidth is WrapProse with explicit widths, for tests.
func WrapProseWidth(text string, width, listWidth int) string {
	if text == "" {
		return text
	}

	// \x00 and \x01 never render and would collide with the mask
	// placeholders, so they are stripped from the input first.
	text = strings.NewReplacer("\x00", "", "\x01", "").Replace(text)

	var out []string
	inFence := false
	for _, line := range strings.Split(text, "\n") {
		// Unicode strip: a non-breaking space counts as indentation here,
		// which is why unicode.IsSpace is right where the regexes above
		// needed an explicit class.
		stripped := strings.TrimLeftFunc(line, unicode.IsSpace)
		if strings.HasPrefix(stripped, "```") {
			inFence = !inFence
			out = append(out, line)
			continue
		}
		if inFence || stripped == "" || neverWrap.MatchString(line) {
			out = append(out, line)
			continue
		}
		if m := listItem.FindStringSubmatch(line); m != nil {
			out = append(out, wrapLine(m[2], listWidth, m[1]))
			continue
		}
		out = append(out, wrapLine(line, width, ""))
	}
	return strings.Join(out, "\n")
}

// wrapLine wraps one line, masking inline code spans first.
//
// prefix is a list marker; when given it goes on the first piece and
// continuation lines are indented by the same width.
func wrapLine(line string, width int, prefix string) string {
	var spans []string
	masked := codeSpan.ReplaceAllStringFunc(line, func(span string) string {
		idx := len(spans)
		spans = append(spans, span)
		// No spaces, so the wrapper keeps it as one unbreakable token, and
		// adjacent spans stay separately addressable on restore. Padded to
		// the span's real width in RUNES, because widths are counted in
		// characters and not bytes.
		base := fmt.Sprintf("\x00%d\x00", idx)
		pad := utf8.RuneCountInString(span) - utf8.RuneCountInString(base)
		if pad < 0 {
			pad = 0
		}
		return base + strings.Repeat("\x01", pad)
	})

	indent := strings.Repeat(" ", utf8.RuneCountInString(prefix))
	pieces := textwrapWrap(masked, width, prefix, indent)
	if len(pieces) == 0 {
		return prefix + line
	}
	wrapped := strings.Join(pieces, HardBreak)
	if len(spans) > 0 {
		wrapped = placeholder.ReplaceAllStringFunc(wrapped, func(m string) string {
			sub := placeholder.FindStringSubmatch(m)
			i, err := strconv.Atoi(sub[1])
			if err != nil || i >= len(spans) {
				return m
			}
			return spans[i]
		})
	}
	return wrapped
}
