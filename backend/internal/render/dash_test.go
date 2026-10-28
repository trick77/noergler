package render

import "testing"

func TestNoEmDash(t *testing.T) {
	cases := []struct{ name, in, want string }{
		{"prose", "**Approve ✅** — fine — really.", "**Approve ✅** - fine - really."},
		{"nothing to do", "plain - text", "plain - text"},
		{"inline code kept", "- `a—b.png` — removed", "- `a—b.png` - removed"},
		{"double backtick span kept", "see ``src/a`b—c.go`` — here", "see ``src/a`b—c.go`` - here"},
		{"unmatched backtick is literal", "x — `y — z", "x - `y - z"},
		{"fence kept", "Fix:\n```suggestion\nlabel := \"a — b\"\n```\nDone — ok.",
			"Fix:\n```suggestion\nlabel := \"a — b\"\n```\nDone - ok."},
		{"tilde fence kept", "~~~\n—\n~~~\n—", "~~~\n—\n~~~\n-"},
		{"backtick fence inside a tilde fence is code", "~~~\n```\n—\n~~~\n—", "~~~\n```\n—\n~~~\n-"},
	}
	for _, c := range cases {
		if got := NoEmDash(c.in); got != c.want {
			t.Errorf("%s: NoEmDash(%q) = %q, want %q", c.name, c.in, got, c.want)
		}
	}
}
