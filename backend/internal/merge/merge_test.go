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

func TestSnapshot(t *testing.T) {
	for _, tc := range []struct {
		name string
		c    Conflict
		want bool
	}{
		{"source", Conflict{Target: ls("v 1.0\n"), Source: ls("v 1.1-SNAPSHOT\n")}, true},
		{"target", Conflict{Target: ls("v 1.1-SNAPSHOT\n"), Source: ls("v 1.0\n")}, true},
		{"lowercase", Conflict{Source: ls("<version>1.4.0-snapshot</version>\n")}, true},
		{"spring", Conflict{Source: ls("v 1.0.0.BUILD-SNAPSHOT\n")}, true},
		{"timestamped", Conflict{Source: ls("<version>2.1-20251110.081500-7</version>\n")}, true},
		{"base only", Conflict{Base: ls("v 1.0-SNAPSHOT\n"), Target: ls("v 1.0\n"), Source: ls("v 1.0.1\n")}, false},
		{"edges only", Conflict{Head: ls("v 1.0-SNAPSHOT\n"), Tail: ls("w 2-SNAPSHOT\n"), Target: ls("a\n"), Source: ls("b\n")}, false},
		{"identifier", Conflict{Target: ls("case SNAPSHOT:\n"), Source: ls("takeSnapshot()\n")}, false},
		// The loose rule: a repository name or Gradle logic refuses too.
		{"repository name", Conflict{Target: ls("<id>nexus-snapshots</id>\n"), Source: ls("b\n")}, true},
		{"gradle logic", Conflict{Target: ls("a\n"), Source: ls(`if (v.endsWith("-SNAPSHOT")) {` + "\n")}, true},
	} {
		if got := tc.c.Snapshot(); got != tc.want {
			t.Errorf("%s: Snapshot() = %v, want %v", tc.name, got, tc.want)
		}
	}
}

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
		// A line both add at the edge is settled once, as in git; the rest
		// is what is left of a block both added: the model's.
		{name: "both insert, shared edge", base: "a\nz\n", target: "a\nimport x\nimport y\nz\n", source: "a\nimport x\nimport w\nz\n", conflicts: 1},
		// git/git 8f0d663e: the same function added on both sides, its body
		// settled, headers left a space (or a tab) apart.
		{name: "both insert, same block", base: "a\nz\n", target: "a\nf () {\n\tbody\n}\nz\n",
			source: "a\n# doc\nf() {\n\tbody\n}\nz\n", conflicts: 1},
		{name: "both insert, same block, tab", base: "a\nz\n", target: "a\nf\t() {\n\tbody\n}\nz\n",
			source: "a\nf() {\n\tbody\n}\nz\n", conflicts: 1},
		// Keys apart only in inner whitespace are one key.
		{name: "both insert, same key, inner space", base: "a\nz\n", target: "a\nconst  X = 1\nz\n", source: "a\nconst X = 2\nz\n", conflicts: 1},
		// Both add the same line in another order: target then source
		// would hold it twice.
		{name: "both insert, shared inside", base: "a\nz\n", target: "a\nimport x\nimport y\nz\n", source: "a\nimport y\nimport x\nz\n", conflicts: 1},
		// The same key set twice, values apart: the model's, never both.
		{name: "both insert, same key", base: "{\nz\n", target: "{\n  \"lodash\": \"^4.17.20\",\nz\n", source: "{\n  \"lodash\": \"^4.17.21\",\nz\n", conflicts: 1},
		{name: "both insert, same const", base: "a\nz\n", target: "a\nconst X = 1\nz\n", source: "a\nconst X = 2\nz\n", conflicts: 1},
		// git/git 8f0d663e: the same function added on both sides, headers
		// one space apart; code would define it twice.
		{name: "both insert, same line but a space", base: "a\n}\n", target: "a\nf () {\n}\n",
			source: "a\n# doc\nf() {\n}\n", conflicts: 1},
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

func TestValidateResolutionUndoesChange(t *testing.T) {
	cases := []struct {
		name         string
		base, tg, sc string
		refs         string
		want         string
	}{
		// Target deleted "gone", source left it and added a line.
		{"deletion followed", "keep\ngone\n", "keep\n", "keep\ngone\nnew\n", "T1 S3", ""},
		{"deletion undone", "keep\ngone\n", "keep\n", "keep\ngone\nnew\n", "S1 S2 S3", "undoes_change"},
		{"kept line twice", "keep\ngone\n", "keep\n", "keep\ngone\nnew\n", "T1 S1 S3", "undoes_change"},
		// Target changed v, source left it and added w.
		{"change followed", "v=1\n", "v=2\n", "v=1\nw=1\n", "T1 S2", ""},
		{"change undone", "v=1\n", "v=2\n", "v=1\nw=1\n", "S1 S2", "undoes_change"},
		// A line both kept, kept once, from either side.
		{"both kept", "a\nb\n", "a\nb\nt\n", "a\nb\ns\n", "S1 S2 T3 S3", ""},
		// Target only re-indented: in YAML or Python that is the change.
		{"re-indent followed", "debug: true\n", "  debug: true\n", "debug: true\nlevel: 2\n", "T1 S2", ""},
		{"re-indent undone", "debug: true\n", "  debug: true\n", "debug: true\nlevel: 2\n", "S1 S2", "undoes_change"},
		// A line end is no change of the line.
		{"line end only", "a\r\n", "a\n", "a\r\nb\r\n", "S1 S2", ""},
		// A blank line target deleted comes back.
		{"blank undone", "x\n\ny\n", "x\ny\n", "x\n\ny\nz\n", "S1 S2 S3 S4", "undoes_change"},
		// No base: nothing to undo.
		{"no base", "", "t\n", "s\n", "S1 T1", ""},
	}
	for _, tc := range cases {
		c := &Conflict{Base: ls(tc.base), Target: ls(tc.tg), Source: ls(tc.sc)}
		var refs []Ref
		for f := range strings.FieldsSeq(tc.refs) {
			r, ok := ParseRef(f)
			if !ok {
				t.Fatalf("%s: ParseRef(%q)", tc.name, f)
			}
			refs = append(refs, r)
		}
		if got := ValidateResolution(c, refs); got != tc.want {
			t.Errorf("%s: %q, want %q", tc.name, got, tc.want)
		}
	}
	// Trimmed edges count: Merge3 settles the shared trailing "b", Base
	// keeps the whole region. Target added a second "b", so a
	// resolution keeping one beside the settled one undoes nothing.
	chunks, err := Merge3(ls("b\nc\n"), ls("}\nb\nb\n"), ls("c\n}\nb\n"))
	if err != nil {
		t.Fatal(err)
	}
	cs := Conflicts(chunks)
	if len(cs) != 1 || !cs[0].Trimmed || len(cs[0].Tail) != 1 {
		t.Fatalf("want one conflict with a trimmed tail: %+v", cs)
	}
	if got := ValidateResolution(cs[0], []Ref{{Target: true, N: 1}, {Target: true, N: 2}, {N: 2}}); got != "" {
		t.Errorf("trimmed: %q, want it to stand", got)
	}
	// ReadAnswer passes the reason on: no push.
	c := &Conflict{Base: ls("v=1\n"), Target: ls("v=2\n"), Source: ls("v=1\nw=1\n")}
	if _, why := ReadAnswer([]*Conflict{c}, map[int][]string{1: {"S1", "S2"}}); why != "conflict 1: undoes_change" {
		t.Errorf("ReadAnswer: %q", why)
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
	// The same line a space apart, kept once: no loss either.
	c = &Conflict{Target: ls("f () {\n"), Source: ls("# doc\nf() {\n")}
	if tgt, src := Dropped(c, []Ref{{Target: true, N: 1}, {N: 1}}); len(tgt)+len(src) != 0 {
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
		hs, err := diffWith(patience, false, a, b, MaxEdits)
		if err != nil {
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
		{"backend/web/dist/index.html", "x", "generated"},
		{"dist/app.js", "x", "generated"},
		{"src/distance.go", "x", ""},
		{"src/cmd/dist/build.go", "x", ""},
		{"pkg/dist/__init__.py", "x", ""},
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

// A merge that changes once its changes slide as git slides them depends
// on the alignment: refused. Target's extra blank, slid below the existing
// blank as git puts it, clears source's change of "}"; as Myers found it,
// right after "}", it conflicts. git merges cleanly; master's Merge3
// conflicted (found by TestAgreesWithGit once merge.go's first lines
// changed): a conflict git does not have, now never handed on.
func TestSlideChangesTheMergeIsRefused(t *testing.T) {
	// Target's first change keeps the common head from settling the rest.
	base := ls("a\nk\n}\n\n// T\nb\n")
	target := ls("a2\nk\n}\n\n\n// T\nb\n")
	source := ls("a\nk\nx\n\n// T\nb\n")
	if _, err := Merge3(base, target, source); !errors.Is(err, ErrAmbiguous) {
		t.Fatalf("err = %v, want ErrAmbiguous", err)
	}
	if _, clean := gitMerge(t, "histogram", base, target, source); !clean {
		t.Fatal("git conflicts: the case no longer shows what it pins")
	}
}

// Merges the slide settles cleanly where git's histogram merge conflicts
// (code review, 2026-11-15): never clean, refused or a conflict.
func TestNeverCleanWhereHistogramConflicts(t *testing.T) {
	cases := [][3]string{{
		"\n\n}\nu3\n\n\n}\n\n\n}\n\t}\n}\n\t}\nx\nx\n\t}\n\nx\n}\n}\n",
		"\n\n}\nu3\n\n}\n\n\n}\n\t}\n}\n\t}\nx\nt1\n\t}\n\nt2\nx\n}\n}\n",
		"\n\n}\n\n\n\n}\n\n\ns0\n\t}\n}\n\t}\nx\nx\n\t}\n\nx\n}\n}\n",
	}}
	for i, c := range cases {
		base, target, source := ls(c[0]), ls(c[1]), ls(c[2])
		chunks, err := Merge3(base, target, source)
		if err != nil || len(Conflicts(chunks)) > 0 {
			continue
		}
		got, _ := Assemble(chunks, nil)
		if out, clean := gitMerge(t, "histogram", base, target, source); !clean || out != string(got) {
			t.Errorf("case %d: Merge3 clean, git histogram clean=%t", i, clean)
		}
	}
}

// gitMerge is `git merge-file -p` of the three with git's algo, skipping
// the test without git or one too old for --diff-algorithm. An exit of 128
// or more is git failing, not a conflict count.
func gitMerge(t *testing.T, algo string, base, target, source []string) (out string, clean bool) {
	t.Helper()
	gitBin, err := exec.LookPath("git")
	if err != nil {
		t.Skip("no git binary")
	}
	dir := t.TempDir()
	for name, l := range map[string][]string{"base": base, "target": target, "source": source} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(strings.Join(l, "")), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	b, err := exec.Command(gitBin, "merge-file", "-p", "--diff-algorithm="+algo,
		filepath.Join(dir, "target"), filepath.Join(dir, "base"), filepath.Join(dir, "source")).Output()
	var exit *exec.ExitError
	switch {
	case err == nil:
		return string(b), true
	case !errors.As(err, &exit):
		t.Fatal(err)
	case exit.ExitCode() >= 128:
		t.Skipf("git merge-file failed (exit %d): %s", exit.ExitCode(), exit.Stderr)
	}
	return string(b), false
}

// marks is a changed-line slice from a pattern: 'x' changed, '.' not.
func marks(p string) []bool {
	out := make([]bool, len(p))
	for i, c := range p {
		out[i] = c == 'x'
	}
	return out
}

func pattern(c []bool) string {
	var b strings.Builder
	for _, x := range c {
		b.WriteByte(map[bool]byte{true: 'x', false: '.'}[x])
	}
	return b.String()
}

// compact as git's xdl_change_compact: a group slides to the bottom of
// its run, back to where it lined up with the other side's change, and
// joins a group it slides into.
func TestCompact(t *testing.T) {
	cases := []struct {
		name          string
		rec           []int
		changed, othr string
		want          string
	}{
		// "a x x b" -> "a x b": the deleted x goes to the bottom.
		{"down", []int{0, 1, 1, 2}, ".x..", "...", "..x."},
		// "a x x b" -> "a Y x b", found as: Y inserted, the second x
		// deleted. The deletion slides up to face Y: one replace.
		{"back to the other side", []int{0, 1, 1, 2}, "..x.", ".x..", ".x.."},
		// "a x x y b" -> "a x b", found as x(1) and y deleted: x(1) slides
		// into y's group, one group of two.
		{"joins", []int{0, 1, 1, 3, 2}, ".x.x.", "...", "..xx."},
		// Out of step (the other side's groups do not pair): refused.
		{"out of step", []int{0, 1, 1, 2}, "..x.", "", ""},
	}
	for _, c := range cases {
		changed := marks(c.changed)
		ok := compact(c.rec, changed, marks(c.othr))
		if c.want == "" {
			if ok {
				t.Errorf("%s: compacted out of step to %s", c.name, pattern(changed))
			}
			continue
		}
		if !ok || pattern(changed) != c.want {
			t.Errorf("%s: %s (ok %t), want %s", c.name, pattern(changed), ok, c.want)
		}
	}
	// Through diffWith, the side's pass: an inserted blank beside a blank
	// goes to the bottom; the first line's change keeps the common head
	// out of it.
	hs, err := diffWith(myers, true, ls("a\nk\n\n\nb\n"), ls("a2\nk\n\n\n\nb\n"), 100)
	if err != nil || len(hs) != 2 || hs[1] != (hunk{b0: 4, b1: 4, s0: 4, s1: 5}) {
		t.Fatalf("blank: hunks = %+v, %v", hs, err)
	}
}

// TestAgreesWithGit runs random edits of a realistic file through Merge3
// and `git merge-file`, with git's Myers and histogram diffs both (a
// server's merge uses histogram): wherever Merge3 merges at all (it may
// refuse as ErrAmbiguous), it agrees with each on whether the merge is
// clean, and a clean merge comes out byte for byte the same. The edits
// repeat existing lines ("}", blank) too, where alignments differ.
//
// The bases are fixed files in testdata, not merge.go itself: an edit of
// merge.go's head once changed what this test checked, and only then did
// it show a disagreement it had always had. agree_base_2 is that head.
// Skipped without git.
func TestAgreesWithGit(t *testing.T) {
	for _, name := range []string{"agree_base_1.txt", "agree_base_2.txt"} {
		src, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		baseLines := Lines(src)
		rng := rand.New(rand.NewSource(1))
		disagree, ambiguous := 0, 0
		const trials = 600
		for trial := range trials {
			target := edit(rng, baseLines, trial, "t")
			source := edit(rng, baseLines, trial, "s")
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
				out, gitClean := gitMerge(t, algo, baseLines, target, source)
				if ours != gitClean {
					disagree++
					t.Logf("%s trial %d: git %s clean=%t, Merge3 clean=%t", name, trial, algo, gitClean, ours)
					continue
				}
				if ours {
					got, err := Assemble(chunks, nil)
					if err != nil || string(got) != out {
						t.Errorf("%s trial %d: clean merges differ from git %s", name, trial, algo)
					}
				}
			}
		}
		if disagree > 0 {
			t.Errorf("%s: %d of %d comparisons disagree with git", name, disagree, 2*(trials-ambiguous))
		}
		// Refusing everything would agree trivially.
		if ambiguous > trials/4 {
			t.Errorf("%s: %d of %d trials refused as ambiguous", name, ambiguous, trials)
		}
		t.Logf("%s: %d of %d trials ambiguous", name, ambiguous, trials)
	}
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
