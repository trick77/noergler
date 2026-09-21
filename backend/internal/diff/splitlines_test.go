package diff

import (
	"reflect"
	"testing"
)

// Empirically: it breaks on \n, \r, \r\n, \v, \f, \x1c, \x1d, \x1e, \x85,
// U+2028 and U+2029. A bug here shifts every line number, so the boundary set
// is pinned rather than assumed.
func TestSplitLinesKeepEndsIsPinned(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"a\nb\n", []string{"a\n", "b\n"}},
		{"a\nb", []string{"a\n", "b"}},
		{"", nil},
		{"\n", []string{"\n"}},
		{"a\r\nb\r\n", []string{"a\r\n", "b\r\n"}},
		{"a\rb", []string{"a\r", "b"}},
		{"a\vb", []string{"a\v", "b"}},
		{"a\fb", []string{"a\f", "b"}},
		{"a\x1cb\x1dc\x1ed", []string{"a\x1c", "b\x1d", "c\x1e", "d"}},
		{"a\u0085b", []string{"a\u0085", "b"}},
		{"a b c", []string{"a ", "b ", "c"}},
		{"a\n\nb", []string{"a\n", "\n", "b"}},
		{"\n\n", []string{"\n", "\n"}},
		{"abc", []string{"abc"}},
		{"a\r\n\r\nb", []string{"a\r\n", "\r\n", "b"}},
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			if got := splitLinesKeepEnds(tc.in); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("splitLinesKeepEnds(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// splitLines uses the same boundary set but drops the terminators.
func TestSplitLinesIsPinned(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"a\nb\n", []string{"a", "b"}},
		{"a\nb", []string{"a", "b"}},
		{"", nil},
		{"\n", []string{""}},
		{"a\r\nb\r\n", []string{"a", "b"}},
		{"a\rb", []string{"a", "b"}},
		{"a\vb", []string{"a", "b"}},
		{"a\fb", []string{"a", "b"}},
		{"a\x1cb\x1dc\x1ed", []string{"a", "b", "c", "d"}},
		{"a\u0085b", []string{"a", "b"}},
		{"a b c", []string{"a", "b", "c"}},
		{"a\n\nb", []string{"a", "", "b"}},
		{"\n\n", []string{"", ""}},
		{"abc", []string{"abc"}},
		{"a\r\n\r\nb", []string{"a", "", "b"}},
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			if got := splitLines(tc.in); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("splitLines(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}
