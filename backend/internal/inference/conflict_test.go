package inference

import (
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/prompts"
)

func TestRenderConflictPrompt(t *testing.T) {
	p := RenderConflictPrompt(ConflictPromptRequest{
		Template: prompts.MergeConflict, TargetBranch: "master", SourceBranch: "feature/x",
		Hunks: []ConflictHunk{{
			Path: "a.go", Before: []string{"func f() {"}, After: []string{"}"},
			Base: []string{"\treturn 1"}, Target: []string{"\treturn 2"}, Source: []string{"\treturn 3", "\t// ```"},
		}},
		Language: lang.DE,
	})
	for _, want := range []string{"### Conflict 1 in `a.go`", "B1: \treturn 1", "T1: \treturn 2", "S2: \t// ```",
		"Target branch `master` (T)", "PR branch `feature/x` (S)", "func f() {"} {
		if !strings.Contains(p, want) {
			t.Errorf("prompt lacks %q", want)
		}
	}
	// A code line holding a fence does not close the block it sits in.
	if !strings.Contains(p, "````") {
		t.Error("the fence was not widened past the code's own")
	}
	for _, ph := range []string{PlaceholderConflicts, PlaceholderTargetBranch, PlaceholderSourceBranch, PlaceholderOutputLanguage} {
		if strings.Contains(p, ph) {
			t.Errorf("%s left in the prompt", ph)
		}
	}
	if strings.Index(p, "## Untrusted input") > strings.Index(p, "### Conflict 1") {
		t.Error("the code is above the untrusted marker")
	}
}

func TestParseConflictResolution(t *testing.T) {
	r, ok := ParseConflictResolution(`{"feasible":true,"reason":"","resolutions":[{"conflict":1,"lines":["T1","S1"]},{"conflict":2,"lines":[]}]}`)
	if !ok || !r.Feasible || len(r.Lines) != 2 || strings.Join(r.Lines[1], ",") != "T1,S1" {
		t.Fatalf("got %+v, %t", r, ok)
	}
	if _, ok := ParseConflictResolution(`{"feasible":true,"reason":"","resolutions":[{"conflict":1,"lines":["T1"]},{"conflict":1,"lines":["S1"]}]}`); ok {
		t.Error("a conflict answered twice was accepted")
	}
	if _, ok := ParseConflictResolution(`not json`); ok {
		t.Error("garbage was accepted")
	}
	r, ok = ParseConflictResolution("```json\n{\"feasible\":false,\"reason\":\" needs a new line \",\"resolutions\":[]}\n```")
	if !ok || r.Feasible || r.Reason != "needs a new line" {
		t.Fatalf("got %+v, %t", r, ok)
	}
}
