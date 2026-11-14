package render

import (
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/lang"
)

func TestMention(t *testing.T) {
	for in, want := range map[string]string{
		"alice":      "@alice",
		"j.doe-2_x":  "@j.doe-2_x",
		"john doe":   `@"john doe"`,
		"a@corp.com": `@"a@corp.com"`,
	} {
		if got := Mention(in); got != want {
			t.Errorf("Mention(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestConflictOffer(t *testing.T) {
	for _, c := range []struct {
		l      lang.Lang
		byCode bool
		want   []string
	}{
		{lang.DE, true, []string{"@alice Dieser PR hat einen Merge-Konflikt mit `master`.", "ohne Modell", "`@noergler behebe den Merge-Konflikt`"}},
		{lang.DE, false, []string{"versuchen", "sage ich dir, warum"}},
		{lang.EN, true, []string{"@alice This PR has a merge conflict with `master`.", "without the model", "`@noergler resolve the merge conflict`"}},
		{lang.EN, false, []string{"simple enough for me to try", "I'll tell you why"}},
	} {
		got := ConflictOffer(c.l, c.byCode, "alice", "noergler", "master")
		for _, w := range c.want {
			if !strings.Contains(got, w) {
				t.Errorf("%v byCode=%t lacks %q:\n%s", c.l, c.byCode, w, got)
			}
		}
		if strings.Contains(got, "ß") || strings.Contains(got, "—") {
			t.Errorf("ß or em dash in %q", got)
		}
	}
}
