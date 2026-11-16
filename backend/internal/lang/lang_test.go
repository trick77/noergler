package lang

import "testing"

func TestParse(t *testing.T) {
	for in, want := range map[string]Lang{"de": DE, " EN ": EN, "De": DE} {
		if got, ok := Parse(in); !ok || got != want {
			t.Errorf("Parse(%q) = %q, %v", in, got, ok)
		}
	}
	for _, in := range []string{"", "fr", "deutsch", "en-US"} {
		if _, ok := Parse(in); ok {
			t.Errorf("Parse(%q) accepted", in)
		}
	}
}

// Anything but English is German: an unset language never writes English
// by accident.
func TestPickAndOr(t *testing.T) {
	if EN.Pick("de", "en") != "en" || DE.Pick("de", "en") != "de" || Lang("").Pick("de", "en") != "de" {
		t.Error("Pick")
	}
	if Lang("").Or(EN) != EN || DE.Or(EN) != DE {
		t.Error("Or")
	}
}

// WithoutCode blanks the code SwissSS leaves alone and keeps the prose.
func TestWithoutCode(t *testing.T) {
	for in, want := range map[string]string{
		"a `b` c":                "a   c",
		"a ``x `y` z`` c":        "a   c",
		"a `open":                "a  ",
		"a\n```\n@Test\n```\nb":  "a\n\n\n\nb",
		"a\n~~~\n@Test\n~~~\nb":  "a\n\n\n\nb",
		"a\n    @Test\n\tx\nb":   "a\n\n\nb",
		"no code, Strasse und ß": "no code, Strasse und ß",
	} {
		if got := WithoutCode(in); got != want {
			t.Errorf("WithoutCode(%q) = %q, want %q", in, got, want)
		}
	}
}
