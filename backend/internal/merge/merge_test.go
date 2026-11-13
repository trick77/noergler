package merge

import (
	"errors"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func ls(s string) []string { return Lines([]byte(s)) }

func TestLinesRoundTrip(t *testing.T) {
	for _, s := range []string{"", "a", "a\n", "a\r\nb", "a\n\nb\n"} {
		if got := strings.Join(Lines([]byte(s)), ""); got != s {
			t.Errorf("Lines(%q) joins to %q", s, got)
		}
	}
}

func TestDiffMatchesShortestEdit(t *testing.T) {
	cases := []struct{ a, b string }{
		{"a\nb\nc\n", "a\nc\n"},
		{"a\nb\nc\n", "a\nb\nx\nc\n"},
		{"", "a\n"},
		{"a\n", ""},
		{"a\nb\nc\nd\n", "x\nb\ny\nd\nz\n"},
	}
	for _, c := range cases {
		a, b := ls(c.a), ls(c.b)
		hs, ok := diff(a, b, MaxEdits)
		if !ok {
			t.Fatalf("diff(%q,%q) gave up", c.a, c.b)
		}
		// Applying the hunks to a gives b.
		var out []string
		pos := 0
		for _, h := range hs {
			out = append(out, a[pos:h.b0]...)
			out = append(out, b[h.s0:h.s1]...)
			pos = h.b1
		}
		out = append(out, a[pos:]...)
		if strings.Join(out, "") != c.b {
			t.Errorf("hunks of %q -> %q rebuild %q", c.a, c.b, strings.Join(out, ""))
		}
	}
}

func TestDiffGivesUpPastMaxEdits(t *testing.T) {
	var a, b []string
	for i := range 50 {
		a = append(a, fmt.Sprintf("a%d\n", i))
		b = append(b, fmt.Sprintf("b%d\n", i))
	}
	if _, ok := diff(a, b, 10); ok {
		t.Fatal("100 edits passed a bound of 10")
	}
}

func TestMerge3(t *testing.T) {
	cases := []struct {
		name                 string
		base, target, source string
		conflicts            int
		trivial              bool
		clean                string // the result when conflicts == 0
	}{
		{name: "one side", base: "a\nb\nc\n", target: "a\nB\nc\n", source: "a\nb\nc\n", clean: "a\nB\nc\n"},
		{name: "apart", base: "a\nb\nc\nd\ne\n", target: "A\nb\nc\nd\ne\n", source: "a\nb\nc\nd\nE\n", clean: "A\nb\nc\nd\nE\n"},
		{name: "same change", base: "a\nb\n", target: "a\nX\n", source: "a\nX\n", clean: "a\nX\n"},
		{name: "both insert", base: "a\nz\n", target: "a\nt1\nt2\nz\n", source: "a\ns1\nz\n", conflicts: 1, trivial: true},
		{name: "both insert at end", base: "a\n", target: "a\nt\n", source: "a\ns\n", conflicts: 1, trivial: true},
		// A line both add at the edge is settled once, as in git: the rest
		// is trivial.
		{name: "both insert, shared edge", base: "a\nz\n", target: "a\nimport x\nimport y\nz\n", source: "a\nimport x\nimport w\nz\n", conflicts: 1, trivial: true},
		// Both add the same line in another order: target then source
		// would hold it twice.
		{name: "both insert, shared inside", base: "a\nz\n", target: "a\nimport x\nimport y\nz\n", source: "a\nimport y\nimport x\nz\n", conflicts: 1},
		// The same key set twice, values apart: the model's, never both.
		{name: "both insert, same key", base: "{\nz\n", target: "{\n  \"lodash\": \"^4.17.20\",\nz\n", source: "{\n  \"lodash\": \"^4.17.21\",\nz\n", conflicts: 1},
		{name: "both insert, same const", base: "a\nz\n", target: "a\nconst X = 1\nz\n", source: "a\nconst X = 2\nz\n", conflicts: 1},
		{name: "both insert, other keys", base: "{\nz\n", target: "{\n  \"a\": 1,\nz\n", source: "{\n  \"b\": 2,\nz\n", conflicts: 1, trivial: true},
		{name: "both modify", base: "a\nb\nc\n", target: "a\nB1\nc\n", source: "a\nB2\nc\n", conflicts: 1},
		{name: "adjacent", base: "a\nb\nc\n", target: "A\nb\nc\n", source: "a\nB\nc\n", conflicts: 1},
		{name: "delete vs modify", base: "a\nb\nc\n", target: "a\nc\n", source: "a\nB\nc\n", conflicts: 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			chunks, err := Merge3(ls(c.base), ls(c.target), ls(c.source))
			if err != nil {
				t.Fatal(err)
			}
			cs := Conflicts(chunks)
			if len(cs) != c.conflicts {
				t.Fatalf("conflicts = %d, want %d: %+v", len(cs), c.conflicts, chunks)
			}
			if c.conflicts == 0 {
				got, err := Assemble(chunks, nil)
				if err != nil || string(got) != c.clean {
					t.Fatalf("clean = %q, %v; want %q", got, err, c.clean)
				}
				return
			}
			if cs[0].Trivial() != c.trivial {
				t.Errorf("Trivial = %t, want %t: %+v", cs[0].Trivial(), c.trivial, cs[0])
			}
		})
	}
}

func TestTrivialResolutionKeepsBothSides(t *testing.T) {
	chunks, err := Merge3(ls("a\nz\n"), ls("a\nt1\nt2\nz\n"), ls("a\ns1\nz\n"))
	if err != nil {
		t.Fatal(err)
	}
	c := Conflicts(chunks)[0]
	got, err := Assemble(chunks, [][]Ref{TrivialResolution(c)})
	if err != nil {
		t.Fatal(err)
	}
	if want := "a\nt1\nt2\ns1\nz\n"; string(got) != want {
		t.Fatalf("got %q, want %q", got, want)
	}
	if LineOf(chunks, [][]Ref{TrivialResolution(c)}, 0) != 2 {
		t.Errorf("LineOf = %d, want 2", LineOf(chunks, nil, 0))
	}
}

func TestCRLFKeptByteForByte(t *testing.T) {
	chunks, err := Merge3(ls("a\r\nz\r\n"), ls("a\r\nt\r\nz\r\n"), ls("a\r\ns\r\nz\r\n"))
	if err != nil {
		t.Fatal(err)
	}
	got, err := Assemble(chunks, [][]Ref{TrivialResolution(Conflicts(chunks)[0])})
	if err != nil || string(got) != "a\r\nt\r\ns\r\nz\r\n" {
		t.Fatalf("got %q, %v", got, err)
	}
}

func TestAssembleRefusesAJoinedLine(t *testing.T) {
	// Both append to a file without a final newline: target's last line
	// would run into source's.
	chunks, err := Merge3(ls("a\n"), ls("a\nt"), ls("a\ns"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Assemble(chunks, [][]Ref{TrivialResolution(Conflicts(chunks)[0])}); !errors.Is(err, ErrNoEOL) {
		t.Fatalf("err = %v, want ErrNoEOL", err)
	}
}

func TestValidateResolution(t *testing.T) {
	c := &Conflict{Base: ls("b\n"), Target: ls("t1\nt2\n"), Source: ls("s1\n")}
	cases := []struct {
		refs string
		want string
	}{
		{"T1 T2 S1", ""},
		{"S1 T1", ""},
		{"T1", ""},
		{"", "empty"},
		{"T3", "unknown_ref"},
		{"S2", "unknown_ref"},
		{"T1 T1", "duplicate_ref"},
		{"T2 T1", "order"},
	}
	for _, tc := range cases {
		var refs []Ref
		for f := range strings.FieldsSeq(tc.refs) {
			r, ok := ParseRef(f)
			if !ok {
				t.Fatalf("ParseRef(%q)", f)
			}
			refs = append(refs, r)
		}
		if got := ValidateResolution(c, refs); got != tc.want {
			t.Errorf("%q: %q, want %q", tc.refs, got, tc.want)
		}
	}
	if ValidateResolution(&Conflict{Base: ls("b\n")}, nil) != "" {
		t.Error("both sides deleted: an empty resolution is the only one")
	}
}

func TestParseRefRefusesBase(t *testing.T) {
	for _, s := range []string{"B1", "T0", "S", "X1", "T-1", "t1"} {
		if _, ok := ParseRef(s); ok {
			t.Errorf("ParseRef(%q) accepted", s)
		}
	}
}

// A line a side repeats on purpose, one copy dropped, is a loss.
func TestDroppedRepeatedLine(t *testing.T) {
	c := &Conflict{Target: ls("x\n"), Source: ls("flush()\nflush()\n")}
	_, src := Dropped(c, []Ref{{N: 1}})
	if !slices.Equal(src, []string{"flush()"}) {
		t.Fatalf("dropped source = %q, want one flush()", src)
	}
	// Both sides' same line, kept once: no loss.
	c = &Conflict{Target: ls("a\n"), Source: ls("a\n")}
	if tgt, src := Dropped(c, []Ref{{Target: true, N: 1}}); len(tgt)+len(src) != 0 {
		t.Fatalf("dropped = %q / %q", tgt, src)
	}
}

func TestPatienceMatchesRebuild(t *testing.T) {
	cases := []struct{ a, b string }{
		{"a\n}\nb\n}\n", "a\n}\nx\nb\n}\n"},
		{"}\n}\n}\n", "}\n"},
		{"", "a\n"},
		{"a\nb\nc\n", "c\nb\na\n"},
	}
	for _, c := range cases {
		a, b := ls(c.a), ls(c.b)
		hs, ok := diffWith(patience, a, b, MaxEdits)
		if !ok {
			t.Fatalf("patience(%q,%q) gave up", c.a, c.b)
		}
		var out []string
		pos := 0
		for _, h := range hs {
			out = append(out, a[pos:h.b0]...)
			out = append(out, b[h.s0:h.s1]...)
			pos = h.b1
		}
		out = append(out, a[pos:]...)
		if strings.Join(out, "") != c.b {
			t.Errorf("patience hunks of %q -> %q rebuild %q", c.a, c.b, strings.Join(out, ""))
		}
	}
}

func TestDropped(t *testing.T) {
	c := &Conflict{Target: ls("x := f(a)\nlog()\n"), Source: ls("x := g(a)\nlog()\n")}
	refs := []Ref{{N: 1}, {Target: true, N: 2}}
	tgt, src := Dropped(c, refs)
	// S2 "log()" is not chosen but its text is kept from T2.
	if !slices.Equal(tgt, []string{"x := f(a)"}) || len(src) != 0 {
		t.Fatalf("dropped = %q / %q", tgt, src)
	}
}

func TestExcluded(t *testing.T) {
	cases := []struct {
		path string
		body string
		want string
	}{
		{"go.sum", "x", "lockfile"},
		{"ui/package-lock.json", "x", "lockfile"},
		{"x/foo.lock", "x", "lockfile"},
		{"web/app.min.js", "x", "generated"},
		{"gen.go", "// Code generated by x. DO NOT EDIT.\n", "generated"},
		{"a.bin", "a\x00b", "binary"},
		{"latin1.txt", "caf\xe9", "binary"},
		{"main.go", "package main\n", ""},
	}
	for _, c := range cases {
		if got := Excluded(c.path, []byte(c.body)); got != c.want {
			t.Errorf("Excluded(%q) = %q, want %q", c.path, got, c.want)
		}
	}
}

// TestAgreesWithGit runs random edits of a realistic file through Merge3
// and `git merge-file`, with git's Myers and histogram diffs both (a
// server's merge uses histogram): wherever Merge3 merges at all (it may
// refuse as ErrAmbiguous), it agrees with each on whether the merge is
// clean, and a clean merge comes out byte for byte the same. The edits
// repeat existing lines ("}", blank) too, where alignments differ.
// Skipped without git.
func TestAgreesWithGit(t *testing.T) {
	gitBin, err := exec.LookPath("git")
	if err != nil {
		t.Skip("no git binary")
	}
	src, err := os.ReadFile("merge.go")
	if err != nil {
		t.Fatal(err)
	}
	baseLines := Lines(src)[:120]
	rng := rand.New(rand.NewSource(1))
	dir := t.TempDir()
	disagree, ambiguous := 0, 0
	const trials = 600
	for trial := range trials {
		target := edit(rng, baseLines, trial, "t")
		source := edit(rng, baseLines, trial, "s")
		files := map[string][]string{"base": baseLines, "target": target, "source": source}
		for name, l := range files {
			if err := os.WriteFile(filepath.Join(dir, name), []byte(strings.Join(l, "")), 0o600); err != nil {
				t.Fatal(err)
			}
		}
		chunks, mErr := Merge3(baseLines, target, source)
		if errors.Is(mErr, ErrAmbiguous) {
			ambiguous++
			continue
		}
		if mErr != nil {
			t.Fatal(mErr)
		}
		ours := len(Conflicts(chunks)) == 0
		for _, algo := range []string{"myers", "histogram"} {
			cmd := exec.Command(gitBin, "merge-file", "-p", "--diff-algorithm="+algo,
				filepath.Join(dir, "target"), filepath.Join(dir, "base"), filepath.Join(dir, "source"))
			out, err := cmd.Output()
			var exit *exec.ExitError
			gitClean := err == nil
			if err != nil && !errors.As(err, &exit) {
				t.Fatal(err)
			}
			if ours != gitClean {
				disagree++
				t.Logf("trial %d: git %s clean=%t, Merge3 clean=%t", trial, algo, gitClean, ours)
				continue
			}
			if ours {
				got, err := Assemble(chunks, nil)
				if err != nil || string(got) != string(out) {
					t.Errorf("trial %d: clean merges differ from git %s", trial, algo)
				}
			}
		}
	}
	if disagree > 0 {
		t.Errorf("%d of %d comparisons disagree with git", disagree, 2*(trials-ambiguous))
	}
	// Refusing everything would agree trivially.
	if ambiguous > trials/4 {
		t.Errorf("%d of %d trials refused as ambiguous", ambiguous, trials)
	}
	t.Logf("%d of %d trials ambiguous", ambiguous, trials)
}

// edit makes one to three random line edits: insert a new line or a copy
// of an existing one, delete, or change.
func edit(rng *rand.Rand, base []string, trial int, tag string) []string {
	out := append([]string(nil), base...)
	for e := range 1 + rng.Intn(3) {
		at := rng.Intn(len(out) + 1)
		line := fmt.Sprintf("\t// %s edit %d.%d\n", tag, trial, e)
		if rng.Intn(4) == 0 {
			line = base[rng.Intn(len(base))]
		}
		switch rng.Intn(3) {
		case 0:
			out = slices.Insert(out, at, line)
		case 1:
			if at < len(out) {
				out = slices.Delete(out, at, at+1)
			}
		default:
			if at < len(out) {
				out[at] = line
			}
		}
	}
	return out
}
