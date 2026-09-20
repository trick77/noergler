#!/usr/bin/env bash
# hack/strip-comment-lines.test.sh — self-test for strip-comment-lines.go
#
# The cases that matter are the two a regex gets wrong in opposite directions:
# a string literal containing "//" (code that LOOKS like a comment) and a
# trailing comment on a real statement (a comment that shares a line with code).
# Dropping the first would hide genuinely uncovered code; keeping the second
# would defeat the whole point.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0

check() { # check <label> <expected> <actual>
  if [ "$2" = "$3" ]; then
    echo "  ok   $1"
  else
    echo "  FAIL $1 (expected '$2', got '$3')"; fail=1
  fi
}

mkdir -p "$TMP/src"
cat > "$TMP/src/sample.go" <<'GO'
package sample

// A leading comment line.
func F() string {
	// An own-line comment.
	url := "http://example.com" // trailing comment

	/*
	   A block comment.
	*/
	return url
}
GO

# Line numbers in sample.go:
#  1 package         code
#  2 (blank)
#  3 // leading      comment
#  4 func F()        code
#  5 // own-line     comment
#  6 url := ...      code WITH a trailing comment
#  7 (blank)
#  8 /*              comment
#  9    A block...   comment
# 10 */              comment
# 11 return url      code
# 12 }               code
report() { # report <line-numbers...>
  {
    echo '<coverage><packages><package><classes>'
    echo "<class filename=\"src/sample.go\"><lines>"
    for n in "$@"; do printf '<line number="%d" hits="0"></line>\n' "$n"; done
    echo '</lines></class></classes></package></packages></coverage>'
  }
}

kept() { # kept <line-numbers...> -> the line numbers that survived, comma-joined
  report "$@" | go run hack/strip-comment-lines.go "$TMP" 2>/dev/null |
    sed -nE 's/.*<line number="([0-9]+)".*/\1/p' | paste -sd, -
}

echo "== strip-comment-lines =="

check "keeps a plain statement" "1,4,11,12" "$(kept 1 4 11 12)"
check "drops own-line comments" "" "$(kept 3 5)"
check "drops blank lines" "" "$(kept 2 7)"
check "drops a block comment, interior included" "" "$(kept 8 9 10)"

# The two that a "starts with //" rule gets wrong.
check "keeps code carrying a trailing comment" "6" "$(kept 6)"
cat > "$TMP/src/urls.go" <<'GO'
package sample

func G() string {
	return "http://example.com"
}
GO
check "keeps a string that contains //" "4" "$(
  report 4 | sed 's|src/sample.go|src/urls.go|' |
    go run hack/strip-comment-lines.go "$TMP" 2>/dev/null |
    sed -nE 's/.*<line number="([0-9]+)".*/\1/p' | paste -sd, -
)"

# Conservative on anything it cannot read: this may only ever remove lines it
# can PROVE carry no code, or it would hide real uncovered code.
#
# The trigger is a LEXICAL error, not a syntax error — go/scanner tokenizes,
# so `func H( {` scans perfectly well and only the parser would object. An
# unterminated string is the real thing.
cat > "$TMP/src/broken.go" <<'GO'
package sample

var s = "unterminated
GO
check "keeps every line of an unscannable file" "1,2,3" "$(
  report 1 2 3 | sed 's|src/sample.go|src/broken.go|' |
    go run hack/strip-comment-lines.go "$TMP" 2>/dev/null |
    sed -nE 's/.*<line number="([0-9]+)".*/\1/p' | paste -sd, -
)"

# A syntax error is NOT a reason to bail: the file still tokenizes, so which
# lines carry code is still known exactly.
cat > "$TMP/src/unparsed.go" <<'GO'
package sample

// A comment above a function that does not compile.
func H( {
GO
check "still strips comments from a file that only fails to PARSE" "1,4" "$(
  report 1 3 4 | sed 's|src/sample.go|src/unparsed.go|' |
    go run hack/strip-comment-lines.go "$TMP" 2>/dev/null |
    sed -nE 's/.*<line number="([0-9]+)".*/\1/p' | paste -sd, -
)"
check "keeps every line of a missing file" "1,2" "$(
  report 1 2 | sed 's|src/sample.go|src/gone.go|' |
    go run hack/strip-comment-lines.go "$TMP" 2>/dev/null |
    sed -nE 's/.*<line number="([0-9]+)".*/\1/p' | paste -sd, -
)"

if [ "$fail" -eq 0 ]; then
  echo "all strip-comment-lines tests passed"
else
  echo "strip-comment-lines tests FAILED"
fi
exit "$fail"
