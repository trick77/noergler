package diff

import (
	"reflect"
	"testing"
)

// Expectations generated from CPython's str.splitlines. A bug here shifts every
// line number, so the boundary set is pinned rather than assumed.
func TestSplitLinesKeepEndsMatchesPython(t *testing.T) {
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

// splitLines drops the terminators, matching Python's plain splitlines.
func TestSplitLinesMatchesPython(t *testing.T) {
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
