package evals

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/merge"
)

func TestLoadConflictCorpus(t *testing.T) {
	cases, err := LoadConflictCorpus()
	if err != nil {
		t.Fatal(err)
	}
	if len(cases) == 0 {
		t.Fatal("empty conflict corpus")
	}
	controls := 0
	for _, c := range cases {
		if c.Control {
			controls++
		}
		req := c.Request("", func(s string) int { return len(s) / 4 })
		if !strings.Contains(req.Prompt, "### Conflict 1 in") || strings.Contains(req.Prompt, "{conflicts}") {
			t.Errorf("%s: prompt not rendered", c.Name)
		}
	}
	// A corpus without controls cannot show the number that counts.
	if controls == 0 {
		t.Error("no control in the conflict corpus")
	}
}

// A hand-written conflict must be the one merge.Merge3 shows production,
// edges trimmed and all, or the eval asks something production never
// would; and a trimmed edge, which no diff3 text shows, must reach Trivial.
func TestLoadIsWhatMergeShows(t *testing.T) {
	c := ConflictCase{Name: "x", Origin: OriginInvented, Source: "s", Why: "w", TargetBranch: "m", SourceBranch: "f", Control: true,
		Files: []ConflictFile{{Path: "f.txt", Merged: "a\n<<<<<<< source\ns\nshared\n||||||| base\nb\n=======\nt\nshared\n>>>>>>> target\n"}}}
	if _, err := LoadConflictCase(c); err == nil || !strings.Contains(err.Error(), "other conflicts") {
		t.Errorf("an untrimmed shared edge loaded: %v", err)
	}
	// Both added one block: Merge3 settled its "}", and the rest is no
	// longer code's to resolve, though the text alone looks trivial.
	c.Files[0].Merged = "a\n<<<<<<< source\ng() {\n||||||| base\n=======\nf() {\n>>>>>>> target\n}\n"
	if _, err := LoadConflictCase(c); err == nil || !strings.Contains(err.Error(), "nothing for the model") {
		t.Fatalf("untrimmed, code's: %v", err)
	}
	c.Files[0].Trimmed = []int{1}
	if l, err := LoadConflictCase(c); err != nil || len(l.asked) != 1 {
		t.Fatalf("trimmed: asked %d, err %v", len(l.asked), err)
	}
	c.Files[0].Trimmed = []int{2}
	if _, err := LoadConflictCase(c); err == nil {
		t.Error("a trimmed conflict past the file's loaded")
	}
}

const twoSided = "a\n<<<<<<< source\ns1\nshared\ns3\n||||||| base\nb1\n=======\nt1\nshared\nt3\n>>>>>>> target\nz\n"

func loadTwoSided(t *testing.T, control bool, accept ...ConflictAccept) LoadedConflictCase {
	t.Helper()
	l, err := LoadConflictCase(ConflictCase{Name: "x", Origin: OriginInvented, Source: "test", Why: "test",
		TargetBranch: "main", SourceBranch: "f", Control: control,
		Files: []ConflictFile{{Path: "f.txt", Merged: twoSided}}, Accept: accept})
	if err != nil {
		t.Fatal(err)
	}
	return l
}

func TestParseDiff3(t *testing.T) {
	chunks, err := ParseDiff3(twoSided)
	if err != nil {
		t.Fatal(err)
	}
	cs := merge.Conflicts(chunks)
	if len(chunks) != 3 || len(cs) != 1 || len(cs[0].Source) != 3 || len(cs[0].Base) != 1 || len(cs[0].Target) != 3 {
		t.Fatalf("chunks = %+v", chunks)
	}
	for _, bad := range []string{
		"<<<<<<< source\nx\n=======\ny\n>>>>>>> target\n", // no base section
		"<<<<<<< HEAD\nx\n||||||| base\n=======\ny\n>>>>>>> target\n",
		"<<<<<<< source\nx\n||||||| base\n=======\ny\n",
	} {
		if _, err := ParseDiff3(bad); err == nil {
			t.Errorf("ParseDiff3(%q) passed", bad)
		}
	}
}

func TestLoadConflictCaseRejects(t *testing.T) {
	base := ConflictCase{Name: "x", Origin: OriginInvented, Source: "s", Why: "w", TargetBranch: "m", SourceBranch: "f",
		Files: []ConflictFile{{Path: "f.txt", Merged: twoSided}}}
	for name, mut := range map[string]func(*ConflictCase){
		"origin":    func(c *ConflictCase) { c.Origin = "other" },
		"no accept": func(*ConflictCase) {},
		"same path": func(c *ConflictCase) {
			c.Control = true
			c.Files = append(c.Files, c.Files[0])
		},
		"excluded": func(c *ConflictCase) { c.Control = true; c.Files[0].Path = "web/dist/index.html" },
		"control+acc": func(c *ConflictCase) {
			c.Control = true
			c.Accept = []ConflictAccept{{File: "f.txt", Conflict: 1, Texts: []string{"t1\n"}}}
		},
		"inexpressible": func(c *ConflictCase) {
			c.Accept = []ConflictAccept{{File: "f.txt", Conflict: 1, Texts: []string{"new\n"}}}
		},
		"base line": func(c *ConflictCase) {
			c.Accept = []ConflictAccept{{File: "f.txt", Conflict: 1, Texts: []string{"b1\n"}}}
		},
		"unknown": func(c *ConflictCase) {
			c.Accept = []ConflictAccept{{File: "f.txt", Conflict: 2, Texts: []string{"t1\n"}}}
		},
		"trivial only": func(c *ConflictCase) {
			c.Control = true
			c.Files[0].Merged = "<<<<<<< source\ns\n||||||| base\n=======\nt\n>>>>>>> target\n"
		},
	} {
		c := base
		c.Files = append([]ConflictFile(nil), base.Files...)
		mut(&c)
		if _, err := LoadConflictCase(c); err == nil {
			t.Errorf("%s: loaded", name)
		}
	}
}

func TestExpressible(t *testing.T) {
	c := &merge.Conflict{Target: []string{"a\n", "b\n"}, Source: []string{"x\n", "b\n"}}
	for _, tc := range []struct {
		want []string
		ok   bool
	}{
		{[]string{"a", "x", "b"}, true},
		{[]string{"x", "a", "b"}, true},
		{[]string{"b", "a"}, true},       // S2 then T1
		{[]string{"b", "b", "a"}, false}, // both b used up, a would break target's order
		{[]string{"x", "x"}, false},
		{[]string{"a", "b", "b"}, true},
		{[]string{"a", "b", "b", "b"}, false},
		{nil, false},
	} {
		if got := Expressible(c, tc.want); got != tc.ok {
			t.Errorf("Expressible(%v) = %v", tc.want, got)
		}
	}
}

func answer(feasible bool, lines map[int][]string) inference.ConflictResult {
	return inference.ConflictResult{Resolution: inference.ConflictResolution{Feasible: feasible, Lines: lines}}
}

func TestScoreConflict(t *testing.T) {
	solve := loadTwoSided(t, false, ConflictAccept{File: "f.txt", Conflict: 1, Texts: []string{"t1\ns1\nshared\n"}})
	anyOrder := loadTwoSided(t, false, ConflictAccept{File: "f.txt", Conflict: 1, Texts: []string{"t1\ns1\nshared\n"}, AnyOrder: true})
	control := loadTwoSided(t, true)
	for _, tc := range []struct {
		name string
		c    LoadedConflictCase
		res  inference.ConflictResult
		want string
	}{
		{"same text other ids", solve, answer(true, map[int][]string{1: {"T1", "S1", "T2"}}), VerdictCorrect},
		{"dropped a side", solve, answer(true, map[int][]string{1: {"S1", "S2"}}), VerdictWrong},
		{"any order", anyOrder, answer(true, map[int][]string{1: {"S1", "T1", "S2"}}), VerdictCorrect},
		{"strict order", solve, answer(true, map[int][]string{1: {"S1", "T1", "S2"}}), VerdictWrong},
		{"declined", solve, answer(false, nil), VerdictDeclined},
		{"base ref", solve, answer(true, map[int][]string{1: {"B1"}}), VerdictRejected},
		{"order broken", solve, answer(true, map[int][]string{1: {"T2", "T1"}}), VerdictRejected},
		{"unanswered", solve, answer(true, map[int][]string{}), VerdictRejected},
		{"control answered", control, answer(true, map[int][]string{1: {"T1", "T2"}}), VerdictWrong},
		{"control declined", control, answer(false, nil), VerdictDeclined},
		{"control rejected", control, answer(true, map[int][]string{1: {"X"}}), VerdictRejected},
		{"unparseable", solve, inference.ConflictResult{Outcome: inference.OutcomeUnparseable}, VerdictRejected},
		{"timed out", solve, inference.ConflictResult{Outcome: inference.OutcomeTimedOut}, VerdictIncomplete},
		{"error", solve, inference.ConflictResult{Err: errors.New("503")}, VerdictIncomplete},
	} {
		if got := ScoreConflict(tc.c, tc.res).Verdict; got != tc.want {
			t.Errorf("%s: %s, want %s", tc.name, got, tc.want)
		}
	}
}

// An answer that would put a line without a newline inside the file is
// one production drops at assembly, never pushes.
func TestScoreConflictAssembles(t *testing.T) {
	c := &merge.Conflict{Base: []string{"b\n"}, Target: []string{"t\n"}, Source: []string{"s"}}
	l := LoadedConflictCase{
		ConflictCase: ConflictCase{Files: []ConflictFile{{Path: "f"}}},
		asked:        []asked{{file: "f", k: 0, conflict: c, accept: ConflictAccept{Texts: []string{"s\nt\n"}}}},
		chunks:       map[string][]merge.Chunk{"f": {{Conflict: c}}},
	}
	if v := ScoreConflict(l, answer(true, map[int][]string{1: {"S1", "T1"}})); v.Verdict != VerdictRejected {
		t.Errorf("unterminated line inside: %+v", v)
	}
	if v := ScoreConflict(l, answer(true, map[int][]string{1: {"T1", "S1"}})); v.Verdict != VerdictWrong {
		t.Errorf("unterminated line last: %+v", v)
	}
}

// stubResolver answers from a script, failing the first fails calls.
type stubResolver struct {
	mu    sync.Mutex
	fails int
	calls int
}

func (s *stubResolver) ResolveConflict(context.Context, inference.ConflictRequest) inference.ConflictResult {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls++
	if s.calls <= s.fails {
		return inference.ConflictResult{Outcome: inference.OutcomeTimedOut}
	}
	return answer(false, nil)
}

func TestRunConflictsRedrawsAndReports(t *testing.T) {
	c := loadTwoSided(t, true)
	stub := &stubResolver{fails: 1}
	run := RunConflicts(context.Background(), stub, []LoadedConflictCase{c}, "", func(string) int { return 1 }, 1, 1, nil)
	a := run.Cases[0].Answers[0]
	if a.Verdict != VerdictDeclined || a.Redraws != 1 || stub.calls != 2 {
		t.Fatalf("answer %+v after %d calls", a, stub.calls)
	}
	if run.ErrWrong() != nil || run.ErrIncomplete() != nil {
		t.Fatal("a declined control is no error")
	}
	var b strings.Builder
	run.Report(&b)
	if !strings.Contains(b.String(), "invented/control") {
		t.Errorf("report:\n%s", b.String())
	}
	stub = &stubResolver{fails: 10}
	run = RunConflicts(context.Background(), stub, []LoadedConflictCase{c}, "{conflicts}", func(string) int { return 1 }, 2, 2, nil)
	if run.ErrIncomplete() == nil || stub.calls != 2*(maxRedraws+1) {
		t.Fatalf("incomplete not reported after %d calls", stub.calls)
	}
}
