// Package render turns review results into the markdown noergler posts on a
// Bitbucket pull request: the summary comment, the inline comment bodies, the
// skip notices and the two sentinel-marked banners.
//
// Everything here is a pure function of its arguments. The review pipeline
// owns the I/O; this package owns the text.
package render

import (
	"fmt"
	"math"
	"strings"

	"github.com/trick77/noergler/internal/lang"
)

// Fmt renders a count with an apostrophe as the thousands separator.
//
// Swiss orthography: the apostrophe, not the comma, groups thousands.
func Fmt(n int) string {
	s := fmt.Sprintf("%d", n)
	neg := strings.HasPrefix(s, "-")
	if neg {
		s = s[1:]
	}
	var b strings.Builder
	for i, r := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			b.WriteByte('\'')
		}
		b.WriteRune(r)
	}
	if neg {
		return "-" + b.String()
	}
	return b.String()
}

// pct is the percentage sites in the summary, rounded half-to-even
// (TestPctUsesBankersRounding).
func pct(part, whole int) int {
	if whole == 0 {
		return 0
	}
	return int(math.RoundToEven(float64(part) / float64(whole) * 100))
}

// PluralIn renders the count with the singular or plural word of l.
func PluralIn(l lang.Lang, n int, deOne, deMany, enOne, enMany string) string {
	if n == 1 {
		return fmt.Sprintf("%d %s", n, l.Pick(deOne, enOne))
	}
	return fmt.Sprintf("%d %s", n, l.Pick(deMany, enMany))
}

// Plural renders "1 finding" / "2 findings".
func Plural(n int, word string) string {
	if n == 1 {
		return fmt.Sprintf("%d %s", n, word)
	}
	return fmt.Sprintf("%d %ss", n, word)
}
