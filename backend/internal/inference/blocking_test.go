package inference

import (
	"encoding/json"
	"fmt"
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
		{"B1", "No SQL in controllers."},
		{"B2", "Every handler checks the tenant."},
		{"B3", "Money is nano-USD, never float."},
		{"B4", "Numbered items count."},
		{"B5", "So do plus items."},
		{"B6", "Unicode indentation counts."},
		{"B7", "A wrapped rule goes on over indented lines, all of them."},
		{"B8", "So does"},
		{"B9", "And a"},
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
	got := RenderBlockingRules([]BlockingRule{{"B1", "No SQL."}, {"B2", "Tenant."}})
	if !strings.HasPrefix(got, "### Blocking rules\n") || !strings.HasSuffix(got, "- B1: No SQL.\n- B2: Tenant.") {
		t.Errorf("block = %q", got)
	}
}

func TestCheckBlocking(t *testing.T) {
	rules := []BlockingRule{{"B1", "No SQL."}, {"B2", "Tenant."}}
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
		"unknown id":     {ReviewFinding{Severity: "issue", Confidence: conf(99), BlockingRule: id("B3")}, BlockingUnknownRule, false, ""},
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
