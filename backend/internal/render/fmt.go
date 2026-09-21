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

// FmtK renders a token count compactly: "628k", "1.5M", "2M".
//
// Rounding is half-to-even: 2500 renders "2k", not "3k". math.RoundToEven
// does that; math.Round would not (TestFmtKUsesBankersRounding).
func FmtK(n int) string {
	if n >= 1_000_000 {
		s := fmt.Sprintf("%.1fM", float64(n)/1_000_000)
		return strings.Replace(s, ".0M", "M", 1)
	}
	return fmt.Sprintf("%dk", int(math.RoundToEven(float64(n)/1000)))
}

// pct is the percentage sites in the summary, on the same half-to-even
// rounding rule as FmtK (TestPctUsesBankersRounding).
func pct(part, whole int) int {
	if whole == 0 {
		return 0
	}
	return int(math.RoundToEven(float64(part) / float64(whole) * 100))
}

// Plural renders "1 finding" / "2 findings".
func Plural(n int, word string) string {
	if n == 1 {
		return fmt.Sprintf("%d %s", n, word)
	}
	return fmt.Sprintf("%d %ss", n, word)
}
