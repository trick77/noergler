//go:build ignore

// hack/strip-comment-lines.go <module-dir> < in.xml > out.xml
//
// Removes <line> entries for source lines that carry no code, so the patch
// coverage gate stops counting comments as untested.
//
// Why this exists: Go's own coverprofile records BLOCKS -- "lines A..B contain
// N statements" -- and says nothing about which lines inside the block are
// statements. gocover-cobertura expands each block into one <line> element per
// line in that range, so every comment and blank line inside a function body
// arrives in the report as hits="0". diff-cover then counts them against the
// diff, and a heavily commented change fails the gate on its prose.
//
// That is measuring the wrong thing twice over: a comment cannot be executed,
// so it can be neither covered nor uncovered, and no test anyone writes will
// ever turn it green. The only way to raise the number is to delete
// explanation, which is the opposite of what the gate is for.
//
// Exactness matters here, so this uses go/scanner rather than pattern-matching
// for "//": a line is CODE if the scanner emits at least one non-COMMENT token
// on it. That keeps `x := "http://example.com"` (a string that contains //) and
// `doThing() // why` (code with a trailing comment) counted as code, while
// dropping own-line comments, block-comment interiors and blank lines. Both are
// cases a regex gets wrong in opposite directions.
//
// Lines the file does not have, or files that cannot be parsed, are left
// exactly as they were: this may only ever remove lines it can prove carry no
// code. A parse failure means the file is being changed to something this tool
// does not understand, and silently dropping its lines could hide real
// uncovered code.
package main

import (
	"bufio"
	"fmt"
	"go/scanner"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
)

var (
	classRe = regexp.MustCompile(`<class[^>]*filename="([^"]*)"`)
	lineRe  = regexp.MustCompile(`^\s*<line number="(\d+)"`)
)

func main() {
	moduleDir := "."
	if len(os.Args) > 1 {
		moduleDir = os.Args[1]
	}

	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 0, 1024*1024), 64*1024*1024)
	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()

	// codeLines for the class currently being read; nil means "keep every
	// line", which is the safe default for a file that could not be parsed.
	var codeLines map[int]bool
	cache := map[string]map[int]bool{}
	dropped, kept := 0, 0

	for in.Scan() {
		line := in.Text()

		if m := classRe.FindStringSubmatch(line); m != nil {
			path := filepath.Join(moduleDir, m[1])
			if c, ok := cache[path]; ok {
				codeLines = c
			} else {
				codeLines = codeLinesOf(path)
				cache[path] = codeLines
			}
		}

		if codeLines != nil {
			if m := lineRe.FindStringSubmatch(line); m != nil {
				n, err := strconv.Atoi(m[1])
				if err == nil && !codeLines[n] {
					dropped++
					continue
				}
				kept++
			}
		}
		fmt.Fprintln(out, line)
	}
	if err := in.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "strip-comment-lines: read: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr,
		"strip-comment-lines: kept %d code lines, dropped %d comment/blank lines\n",
		kept, dropped)
}

// codeLinesOf returns the set of 1-based lines in path that carry at least one
// non-comment token. It returns nil when the file cannot be read or scanned, so
// the caller keeps that file's report untouched.
func codeLinesOf(path string) map[int]bool {
	src, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "strip-comment-lines: %s: %v (keeping all lines)\n", path, err)
		return nil
	}

	fset := token.NewFileSet()
	file := fset.AddFile(path, fset.Base(), len(src))

	var s scanner.Scanner
	bad := false
	errh := func(token.Position, string) { bad = true }
	// ScanComments so comments arrive as COMMENT tokens to be skipped, rather
	// than being silently dropped by the scanner and leaving no way to tell a
	// comment line from a blank one. (Both are removed, but only because
	// neither ends up in the set -- not by guessing.)
	s.Init(file, src, errh, scanner.ScanComments)

	lines := map[int]bool{}
	for {
		pos, tok, _ := s.Scan()
		if tok == token.EOF {
			break
		}
		if tok == token.COMMENT {
			continue
		}
		// SEMICOLON is included deliberately: an implicit one is positioned at
		// the newline ending a statement, which is a line that really does
		// carry code. An explicit one obviously does too.
		lines[file.Position(pos).Line] = true
	}
	if bad {
		fmt.Fprintf(os.Stderr, "strip-comment-lines: %s: scan errors (keeping all lines)\n", path)
		return nil
	}
	return lines
}
