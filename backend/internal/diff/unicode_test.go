package diff

import "testing"

// RE2's \w is ASCII-only, so every pattern here spells it [\pL\pN_] and a
// non-ASCII identifier still matches: the whole of "caféHandler" is extracted,
// not just "caf".
func TestSymbolExtractionHandlesNonASCII(t *testing.T) {
	got := extractChangedSymbols(FileReviewData{
		Path: "a.ts",
		Diff: "+const caféHandler = (x) => x\n",
	})
	if len(got) != 1 || got[0] != "caféHandler" {
		t.Errorf("got %q, want [caféHandler]", got)
	}
}

func TestSymbolNamePatternsAreUnicodeAware(t *testing.T) {
	cases := []struct{ lang, line, want string }{
		{"python", "def prüfen(x):", "prüfen"},
		{"typescript", "export function laufenÜber(a) {", "laufenÜber"},
		{"jvm", "public class Ölservice {", "Ölservice"},
		{"other", "func größe() {", "größe"},
	}
	for _, tc := range cases {
		t.Run(tc.lang, func(t *testing.T) {
			m := symbolNamePatterns[tc.lang].FindStringSubmatch(tc.line)
			if m == nil {
				t.Fatalf("%s pattern did not match %q", tc.lang, tc.line)
			}
			if m[1] != tc.want {
				t.Errorf("captured %q, want %q", m[1], tc.want)
			}
		})
	}
}
