package render

import "testing"

// Expected values are written out case by case, not derived from the code
// under test.
func TestFmt(t *testing.T) {
	cases := []struct {
		in   int
		want string
	}{
		{0, "0"},
		{1, "1"},
		{12, "12"},
		{123, "123"},
		{1000, "1'000"},
		{1234, "1'234"},
		{12345, "12'345"},
		{999999, "999'999"},
		{1000000, "1'000'000"},
		{7000, "7'000"},
		{628000, "628'000"},
		{-1234, "-1'234"},
		{1234567, "1'234'567"},
	}
	for _, c := range cases {
		if got := Fmt(c.in); got != c.want {
			t.Errorf("Fmt(%d) = %q, want %q", c.in, got, c.want)
		}
	}
}

// Rounding is half-to-even, so 500 -> 0k, 1500 -> 2k, 2500 -> 2k and
// 3500 -> 4k. math.Round would give 1k, 2k, 3k, 4k and differ on two of
// the four.
func TestFmtKUsesBankersRounding(t *testing.T) {
	cases := []struct {
		in   int
		want string
	}{
		{0, "0k"},
		{499, "0k"},
		{500, "0k"},
		{501, "1k"},
		{1000, "1k"},
		{1500, "2k"},
		{2500, "2k"},
		{3500, "4k"},
		{7000, "7k"},
		{628000, "628k"},
		{999499, "999k"},
		{999500, "1000k"},
		{1000000, "1M"},
		{1050000, "1.1M"},
		{1500000, "1.5M"},
		{2000000, "2M"},
	}
	for _, c := range cases {
		if got := FmtK(c.in); got != c.want {
			t.Errorf("FmtK(%d) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestPlural(t *testing.T) {
	if got := Plural(1, "finding"); got != "1 finding" {
		t.Errorf("Plural(1) = %q", got)
	}
	if got := Plural(2, "finding"); got != "2 findings" {
		t.Errorf("Plural(2) = %q", got)
	}
	if got := Plural(0, "finding"); got != "0 findings" {
		t.Errorf("Plural(0) = %q", got)
	}
}

// The summary's two percentage sites use the same rule as FmtK.
func TestPctUsesBankersRounding(t *testing.T) {
	if got := pct(5, 1000); got != 0 {
		t.Errorf("pct(5,1000) = %d, want 0 (0.5 rounds to even)", got)
	}
	if got := pct(15, 1000); got != 2 {
		t.Errorf("pct(15,1000) = %d, want 2 (1.5 rounds to even)", got)
	}
	if got := pct(25, 1000); got != 2 {
		t.Errorf("pct(25,1000) = %d, want 2 (2.5 rounds to even)", got)
	}
	if got := pct(1, 0); got != 0 {
		t.Errorf("pct with zero whole = %d, want 0", got)
	}
}
