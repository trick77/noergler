package inference

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"
)

func TestExtractBlockingRules(t *testing.T) {
	md := strings.Join([]string{
		"# Rules",
		"[blocking] No SQL in controllers.",
		"- [BLOCKING] Every handler checks the tenant.",
		"  * [Blocking]   Money is nano-USD,\tnever float.\r",
		"- [blocking] No SQL in controllers.", // repeat
		"- [blocking]",                        // empty
		"- not [blocking] mid-line",
		"1. [blocking] Numbered items count.",
		"+ [blocking] So do plus items.",
		"```md",
		"- [blocking] an example in a fence",
		"```",
		"~~~",
		"[blocking] another example",
		"~~~",
		" - [blocking] Unicode indentation counts.",
		"- [blocking] A wrapped rule goes",
		"  on over indented lines,",
		"\tall of them.",
		"  - but a nested item ends it",
		"- [blocking] So does",
		"a line at the margin.",
		"- [blocking] And a",
		"",
		"  blank line.",
		"- [blocking]   No SQL",
		"  in controllers.", // re-wrapped repeat of B1
	}, "\n")
	rules, over := ExtractBlockingRules(md)
	want := []BlockingRule{
		{ID: "B1", Text: "No SQL in controllers."},
		{ID: "B2", Text: "Every handler checks the tenant."},
		{ID: "B3", Text: "Money is nano-USD, never float."},
		{ID: "B4", Text: "Numbered items count."},
		{ID: "B5", Text: "So do plus items."},
		{ID: "B6", Text: "Unicode indentation counts."},
		{ID: "B7", Text: "A wrapped rule goes on over indented lines, all of them."},
		{ID: "B8", Text: "So does"},
		{ID: "B9", Text: "And a"},
	}
	if fmt.Sprint(rules) != fmt.Sprint(want) || len(over) != 0 {
		t.Errorf("rules = %q, over = %q\nwant %q", rules, over, want)
	}
}

func TestExtractBlockingRulesCaps(t *testing.T) {
	var b strings.Builder
	for i := range MaxBlockingRules + 2 {
		fmt.Fprintf(&b, "- [blocking] rule %d\n", i)
	}
	fmt.Fprintf(&b, "- [blocking] %s\n", strings.Repeat("x", 400))
	rules, over := ExtractBlockingRules(b.String())
	if len(rules) != MaxBlockingRules || len(over) != 3 || rules[9].ID != "B10" || over[0] != "rule 10" {
		t.Errorf("len = %d, over = %q, last = %+v", len(rules), over, rules[len(rules)-1])
	}
	long, _ := ExtractBlockingRules("[blocking] " + strings.Repeat("y", 400))
	if n := len([]rune(long[0].Text)); n != maxBlockingRuleRunes || !strings.HasSuffix(long[0].Text, "…") {
		t.Errorf("long rule = %d runes: %q", n, long[0].Text)
	}
}

func TestRenderBlockingRules(t *testing.T) {
	if got := RenderBlockingRules(nil); got != "" {
		t.Errorf("no rules must render nothing, got %q", got)
	}
	got := RenderBlockingRules([]BlockingRule{{ID: "B1", Text: "No SQL."}, {ID: "B2", Text: "Tenant."}})
	if !strings.HasPrefix(got, "### Blocking rules\n") || !strings.HasSuffix(got, "- B1: No SQL.\n- B2: Tenant.") {
		t.Errorf("block = %q", got)
	}
	// A nested file's rule names the files it covers.
	got = RenderBlockingRules([]BlockingRule{{ID: "B1", Text: "No SQL."}, {ID: "B2", Text: "Cents.", Dir: "services/pay/"}})
	if !strings.HasSuffix(got, "- B1: No SQL.\n- B2 (only files under `services/pay/`): Cents.") {
		t.Errorf("nested block = %q", got)
	}
}

// Several files: root first, one numbering, one cap; a text repeats only
// within one directory, and a nested repeat of a root rule takes no slot.
func TestExtractBlockingRulesFrom(t *testing.T) {
	var root strings.Builder
	for i := range 8 {
		fmt.Fprintf(&root, "- [blocking] root %d\n", i)
	}
	rules, over := ExtractBlockingRulesFrom([]RuleSource{
		{Text: root.String() + "- [blocking] shared\n"},
		{Dir: "a/", Text: "- [blocking] shared\n- [blocking] a only\n- [blocking] a only\n- [blocking] sibling\n"},
		{Dir: "b/", Text: "- [blocking] sibling\n- [blocking] b two\n"},
	})
	if len(rules) != MaxBlockingRules {
		t.Fatalf("rules = %d, want the shared cap %d", len(rules), MaxBlockingRules)
	}
	if r := rules[8]; r.ID != "B9" || r.Text != "shared" || r.Dir != "" {
		t.Errorf("B9 = %+v, want the root's shared", r)
	}
	if r := rules[9]; r.ID != "B10" || r.Text != "a only" || r.Dir != "a/" {
		t.Errorf("B10 = %+v, want a/'s own rule: its repeat of the root's takes no slot", r)
	}
	if want := []string{"sibling", "sibling", "b two"}; !slices.Equal(over, want) {
		t.Errorf("over = %q, want %q (a sibling's same text is a rule of its own)", over, want)
	}
}

func TestCheckBlocking(t *testing.T) {
	rules := []BlockingRule{{ID: "B1", Text: "No SQL."}, {ID: "B2", Text: "Tenant."}, {ID: "B3", Text: "Cents.", Dir: "pay/"}}
	conf := func(n int) *int { return &n }
	id := func(s string) *string { return &s }
	for name, c := range map[string]struct {
		f    ReviewFinding
		want BlockingOutcome
		ok   bool
		rule string
	}{
		"accepted":       {ReviewFinding{Severity: "issue", Confidence: conf(90), BlockingRule: id("B2")}, BlockingTask, true, "Tenant."},
		"id any case":    {ReviewFinding{Severity: "issue", Confidence: conf(99), BlockingRule: id(" b1 ")}, BlockingTask, true, "No SQL."},
		"unknown id":     {ReviewFinding{Severity: "issue", Confidence: conf(99), BlockingRule: id("B4")}, BlockingUnknownRule, false, ""},
		"in scope":       {ReviewFinding{File: "pay/a.go", Severity: "issue", Confidence: conf(99), BlockingRule: id("B3")}, BlockingTask, true, "Cents."},
		"deep in scope":  {ReviewFinding{File: "pay/x/y/a.go", Severity: "issue", Confidence: conf(99), BlockingRule: id("B3")}, BlockingTask, true, "Cents."},
		"out of scope":   {ReviewFinding{File: "web/a.ts", Severity: "issue", Confidence: conf(99), BlockingRule: id("B3")}, BlockingUnknownRule, false, "Cents."},
		"prefix only":    {ReviewFinding{File: "payments/a.go", Severity: "issue", Confidence: conf(99), BlockingRule: id("B3")}, BlockingUnknownRule, false, "Cents."},
		"suggestion":     {ReviewFinding{Severity: "suggestion", Confidence: conf(99), BlockingRule: id("B1")}, BlockingSeverity, false, "No SQL."},
		"under 90":       {ReviewFinding{Severity: "issue", Confidence: conf(89), BlockingRule: id("B1")}, BlockingConfidence, false, "No SQL."},
		"no confidence":  {ReviewFinding{Severity: "issue", BlockingRule: id("B1")}, BlockingConfidence, false, "No SQL."},
		"rule, no rules": {ReviewFinding{Severity: "issue", Confidence: conf(99), BlockingRule: id("B1")}, BlockingUnknownRule, false, ""},
	} {
		t.Run(name, func(t *testing.T) {
			rs := rules
			if name == "rule, no rules" {
				rs = nil
			}
			rule, outcome, ok := CheckBlocking(c.f, rs)
			text := ""
			if rule != nil {
				text = rule.Text
			}
			if outcome != c.want || ok != c.ok || text != c.rule {
				t.Errorf("= %q %v %q, want %q %v %q", outcome, ok, text, c.want, c.ok, c.rule)
			}
		})
	}
}

// A category is kept only when it is one of the enum; anything else reads
// as none and the finding stands.
func TestParseCategory(t *testing.T) {
	for raw, want := range map[string]string{
		`"security"`: "security",
		`"secret"`:   "secret",
		`"style"`:    "",
		`null`:       "",
		`3`:          "",
	} {
		item := `{"file": "a.go", "line": 1, "severity": "issue", "comment": "c", "category": ` + raw + `}`
		f, ok := parseFinding(json.RawMessage(item))
		if !ok {
			t.Errorf("%s: the finding must stand", raw)
			continue
		}
		got := ""
		if f.Category != nil {
			got = *f.Category
		}
		if got != want {
			t.Errorf("%s: category = %q, want %q", raw, got, want)
		}
	}
}

// A blocking_rule that is no string is no claim; the finding stands.
func TestParseBlockingRule(t *testing.T) {
	for raw, want := range map[string]string{
		`"B1"`:   "B1",
		`null`:   "",
		`""`:     "",
		`3`:      "",
		`["B1"]`: "",
	} {
		item := `{"file": "a.go", "line": 1, "severity": "issue", "comment": "c", "blocking_rule": ` + raw + `}`
		f, ok := parseFinding(json.RawMessage(item))
		if !ok {
			t.Errorf("%s: the finding must stand", raw)
			continue
		}
		got := ""
		if f.BlockingRule != nil {
			got = *f.BlockingRule
		}
		if got != want || f.Marked() != (want != "") {
			t.Errorf("%s: rule = %q, marked = %v", raw, got, f.Marked())
		}
	}
}
