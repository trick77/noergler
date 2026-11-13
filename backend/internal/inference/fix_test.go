package inference

import (
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/lang"
)

func TestParseFindingFix(t *testing.T) {
	for name, c := range map[string]struct {
		in   string
		ok   bool
		want FindingFix
	}{
		"feasible": {`{"feasible":true,"reason":"","old":"a","new":"b","commit_subject":" fix a "}`, true,
			FindingFix{Feasible: true, Old: "a", New: "b", CommitSubject: "fix a"}},
		"not feasible": {`{"feasible":false,"reason":" needs the generator ","old":"","new":"","commit_subject":""}`, true,
			FindingFix{Reason: "needs the generator"}},
		// A feasible fix with nothing to replace is no fix.
		"feasible without old": {`{"feasible":true,"reason":"","old":"","new":"b","commit_subject":"x"}`, true,
			FindingFix{New: "b", CommitSubject: "x"}},
		"fenced": {"```json\n{\"feasible\":false,\"reason\":\"r\",\"old\":\"\",\"new\":\"\",\"commit_subject\":\"\"}\n```", true,
			FindingFix{Reason: "r"}},
		"not json": {"sure, here is the fix", false, FindingFix{}},
	} {
		t.Run(name, func(t *testing.T) {
			got, ok := ParseFindingFix(c.in)
			if ok != c.ok || got != c.want {
				t.Errorf("got %+v, %v; want %+v, %v", got, ok, c.want, c.ok)
			}
		})
	}
}

func TestValidateFix(t *testing.T) {
	const file = "package a\n\nfunc a() {\n\treturn\n}\n\nfunc b() {\n\treturn\n}\n"
	fix := func(old, new string) FindingFix { return FindingFix{Feasible: true, Old: old, New: new} }
	for name, c := range map[string]struct {
		content  string
		from, to int
		fix      FindingFix
		want     string
		reason   string
	}{
		"spliced": {file, 1, 9, fix("func a() {\n\treturn\n}", "func a() {\n\tpanic(1)\n}"),
			"package a\n\nfunc a() {\n\tpanic(1)\n}\n\nfunc b() {\n\treturn\n}\n", ""},
		"ambiguous":   {file, 1, 9, fix("\treturn\n", "\tpanic(1)\n"), "", "old_ambiguous"},
		"not found":   {file, 1, 9, fix("func c() {", "func d() {"), "", "old_not_found"},
		"no change":   {file, 1, 9, fix("func a() {", "func a() {"), "", "no_change"},
		"empty old":   {file, 1, 9, fix("", "x"), "", "empty_old"},
		"outside":     {file, 1, 4, fix("func b() {", "func bb() {"), "", "outside_excerpt"},
		"binary":      {"a\x00b", 1, 1, fix("a", "c"), "", "binary"},
		"too large":   {file, 1, 9, fix("func a() {", strings.Repeat("x\n", FixMaxLines+1)), "", "too_large"},
		"secret":      {file, 1, 9, fix("func a() {", "func a() {\n\tpassword := \"hunter2hunter2\""), "", "secret"},
		"crlf kept":   {"a\r\nb\r\nc\r\n", 1, 3, fix("b\n", "x\ny\n"), "a\r\nx\r\ny\r\nc\r\n", ""},
		"bom kept":    {"\ufeffa\nb\n", 1, 2, fix("b", "c"), "\ufeffa\nc\n", ""},
		"no final nl": {"a\nb", 1, 2, fix("b", "c"), "a\nc", ""},
	} {
		t.Run(name, func(t *testing.T) {
			out, reason := ValidateFix([]byte(c.content), c.from, c.to, c.fix)
			if reason != c.reason || string(out) != c.want {
				t.Errorf("got %q, %q; want %q, %q", out, reason, c.want, c.reason)
			}
		})
	}
}

func TestCheckCommitSubject(t *testing.T) {
	for in, want := range map[string]string{
		"Fix loop redirection.":           "fix loop redirection",
		"`behebe Schleifenumleitung`":     "behebe Schleifenumleitung",
		"  Überprüfe die Eingabe ":        "überprüfe die Eingabe",
		"two\nlines":                      "",
		"":                                "",
		strings.Repeat("x", 73):           "",
		strings.Repeat("ä", 72):           strings.Repeat("ä", 72),
	} {
		got, ok := CheckCommitSubject(in)
		if got != want || ok != (want != "") {
			t.Errorf("CheckCommitSubject(%q) = %q, %v; want %q", in, got, ok, want)
		}
	}
}

func TestRenderFindingFixPrompt(t *testing.T) {
	tmpl := "{finding}|{file_path}|{code}|{start_line}|{anchor_line}|{reply}|{commit_language}|{repo_instructions}"
	for l, style := range map[lang.Lang]string{lang.EN: "fix loop redirection", lang.DE: "behebe Schleifenumleitung"} {
		got := RenderFindingFixPrompt(FindingFixPromptRequest{
			Template: tmpl, Finding: "F", FilePath: "a.go", Code: "x := \"```\"", StartLine: 3, AnchorLine: 7,
			Reply: "flick das {code}", Language: l, RepoInstructions: "R",
		})
		for _, want := range []string{"F|a.go|````\nx := \"```\"\n````|3|7|flick das {code}|", style, "|R"} {
			if !strings.Contains(got, want) {
				t.Errorf("%v: prompt lacks %q:\n%s", l, want, got)
			}
		}
	}
}

// A fix request is its own intent; the parser keeps it.
func TestParseFindingReplyKeepsAFixRequest(t *testing.T) {
	got, ok := ParseFindingReply(`{"intent":"requests_fix","reply":"On it.","agents_md_line":null,"matches_signal":null}`)
	if !ok || got.Intent != IntentRequestsFix {
		t.Errorf("got %+v, %v", got, ok)
	}
}
