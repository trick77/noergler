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
