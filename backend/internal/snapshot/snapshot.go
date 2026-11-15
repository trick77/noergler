// Package snapshot spots an unreleased Maven or Gradle build: a version
// with the -SNAPSHOT suffix, or a resolved snapshot's timestamp. Blunt
// rules, never a model: what they decide blocks a merge.
package snapshot

import (
	"fmt"
	"path"
	"regexp"
	"strings"
	"unicode"

	"github.com/trick77/noergler/internal/diff"
)

// suffixRE is -SNAPSHOT on a version, not running on into a word, `_` or
// `/`. In capitals, the way versions write it, after a word character
// (`1.0.0-beta-SNAPSHOT`), `}`, `>` (`<changelist>-SNAPSHOT`) or `=`
// (`suffix=-SNAPSHOT`); in any case (Maven compares it ignoring case) only
// after a digit, `}` or BUILD. Gradle logic about the suffix has a quote,
// paren or blank before it (`endsWith("-SNAPSHOT")`, a comment), a
// repository's name is lower case after a word (`libs-snapshot`,
// `maven-snapshots`): neither is a version. A `-` after it is a qualifier
// (`1.0-SNAPSHOT-jdk17`). RE2 classes are ASCII, so the classes are \pL\pN
// (INVARIANTS: adapters).
var suffixRE = regexp.MustCompile(`(?:[\pL\pN}>=]-SNAPSHOT|(?i:(?:\p{Nd}|\}|build)-snapshot))(?:$|[^\pL\pN_/])`)

// stampRE is a resolved snapshot (1.0-20251115.093012-3): an unreleased
// build without the word.
var stampRE = regexp.MustCompile(`-\p{Nd}{8}\.\p{Nd}{6}-\p{Nd}+`)

// Line reports a line naming an unreleased build as a version: the strict
// rule a merge-blocking task hangs on.
func Line(l string) bool {
	return suffixRE.MatchString(l) || stampRE.MatchString(l)
}

// Mentions reports a line that mentions an unreleased build at all:
// `-snapshot` in any case anywhere, or a resolved snapshot's timestamp.
// The loose rule for refusing a merge-conflict resolution, where a stray
// match costs an offer, never a merge (merge.Conflict.Snapshot).
func Mentions(l string) bool {
	return strings.Contains(strings.ToLower(l), "-snapshot") || stampRE.MatchString(l)
}

// BuildFile reports a Maven or Gradle build file: `pom.xml`,
// `.mvn/maven.config` (`-Drevision=1.0-SNAPSHOT`), any Gradle script
// (`*.gradle`, `*.gradle.kts`: build, settings, `dependencies.gradle`),
// `gradle.properties` and a Version Catalog (`*.versions.toml`). Nothing
// else uses SNAPSHOT (user, 2026-11-15): npm, Go, Python and Cargo name a
// pre-release otherwise. Kotlin in buildSrc is code, left out.
func BuildFile(p string) bool {
	base := path.Base(p)
	switch base {
	case "pom.xml", "gradle.properties":
		return true
	case "maven.config":
		return path.Base(path.Dir(p)) == ".mvn"
	}
	return strings.HasSuffix(base, ".gradle") || strings.HasSuffix(base, ".gradle.kts") ||
		strings.HasSuffix(base, ".versions.toml")
}

// Hit is one added line naming an unreleased build.
type Hit struct {
	Path string
	// Line is the line's number in the new file.
	Line int
	Text string
	// Seq counts the file's hits with the same text, from 1: two
	// dependencies on one `<version>2.0-SNAPSHOT</version>` line each are
	// two hits.
	Seq int
}

// Key is what a hit is remembered by: its path, its text without any
// whitespace and Seq, blanks between (the text has none). No line number:
// a line the PR moves is the same hit.
func (h Hit) Key() string {
	return fmt.Sprintf("%s %s %d", h.Path, noSpace(h.Text), h.Seq)
}

func noSpace(s string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsSpace(r) {
			return -1
		}
		return r
	}, s)
}

// Added is the PR delta's hits: the lines a diff adds to a build file,
// never a context line, a removed one or any other file. A change that
// replaces a line naming an unreleased build with another (1.0-SNAPSHOT to
// 1.1-SNAPSHOT) brings none in (user, 2026-11-15): per change block, a run
// of removed and added lines, as many added hits are exempt as the block
// removes; only the rest count. A hit moved to another block counts there,
// the safe way.
func Added(rawDiff string) []Hit {
	var hits []Hit
	for _, fd := range diff.SplitByFile(rawDiff) {
		if p := diff.ExtractPath(fd); p != "" && BuildFile(p) {
			hits = append(hits, AddedIn(p, fd)...)
		}
	}
	return hits
}

// FileHits is the hits of a per-file diff response for path: only its part
// for path counts, whatever else the response carries; a response with one
// part, or none with a header, is that file's.
func FileHits(p, raw string) []Hit {
	var parts []string
	for _, part := range diff.SplitByFile(raw) {
		if strings.HasPrefix(part, "diff --git ") {
			parts = append(parts, part)
		}
	}
	switch len(parts) {
	case 0:
		return AddedIn(p, raw)
	case 1:
		return AddedIn(p, parts[0])
	}
	for _, part := range parts {
		if diff.ExtractPath(part) == p {
			return AddedIn(p, part)
		}
	}
	return nil
}

// AddedIn is Added for one file's diff, its path given rather than read from
// the header.
func AddedIn(p, fileDiff string) []Hit {
	if diff.IsDeleted(fileDiff) {
		return nil
	}
	var hits []Hit
	_, hunks := diff.ParseHunks(fileDiff)
	for _, h := range hunks {
		n, removed := h.NewStart, 0
		var added []Hit
		flush := func() {
			if len(added) > removed {
				hits = append(hits, added[removed:]...)
			}
			removed, added = 0, nil
		}
		for _, l := range h.BodyLines {
			l = strings.TrimSuffix(l, "\r")
			switch {
			case strings.HasPrefix(l, `\`):
				// "\ No newline at end of file" sits inside a block, never
				// ends one.
			case strings.HasPrefix(l, "-"):
				if Line(l[1:]) {
					removed++
				}
			case strings.HasPrefix(l, "+"):
				if Line(l[1:]) {
					added = append(added, Hit{Path: p, Line: n, Text: l[1:]})
				}
				n++
			default:
				flush()
				n++
			}
		}
		flush()
	}
	seq := map[string]int{}
	for i := range hits {
		t := noSpace(hits[i].Text)
		seq[t]++
		hits[i].Seq = seq[t]
	}
	return hits
}
