#!/usr/bin/env bash
# hack/coverage-gate.test.sh — self-test for coverage-gate.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0

check() { # check <label> <expected-exit> <actual-exit>
  if [ "$2" = "$3" ]; then
    echo "  ok   $1"
  else
    echo "  FAIL $1 (expected exit $2, got $3)"; fail=1
  fi
}

# 80 of 100 lines covered outside cmd/ -> 80.0%.
# cmd/ contributes 100 lines, all uncovered, and must be ignored entirely:
# including it would report 80/200 = 40.0%, a materially different number.
gen_lines() { # gen_lines <count> <hits>
  local i
  for ((i = 1; i <= $1; i++)); do
    printf '<line number="%d" hits="%d"></line>' "$i" "$2"
  done
}

cat > "$TMP/backend.xml" <<XML
<?xml version="1.0"?>
<coverage>
  <packages>
    <package name="github.com/trick77/noergler/internal/review">
      <classes>
        <class name="review" filename="internal/review/review.go">
          <lines>$(gen_lines 80 1)$(gen_lines 20 0)</lines>
        </class>
      </classes>
    </package>
    <package name="github.com/trick77/noergler/cmd/noergler">
      <classes>
        <class name="main" filename="cmd/noergler/main.go">
          <lines>$(gen_lines 100 0)</lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
XML

cat > "$TMP/backend-malformed.xml" <<'XML'
<coverage><packages><
XML

printf 'backend=79.0\nui=50.0\n' > "$TMP/floors-under"
printf 'backend=81.0\nui=50.0\n' > "$TMP/floors-over"

COVERAGE_FLOORS="$TMP/floors-under" COVERAGE_FILE="$TMP/backend.xml" \
  ./hack/coverage-gate.sh backend >/dev/null 2>&1
check "passes when above floor" 0 $?

COVERAGE_FLOORS="$TMP/floors-over" COVERAGE_FILE="$TMP/backend.xml" \
  ./hack/coverage-gate.sh backend >/dev/null 2>&1
check "fails when below floor" 1 $?

out=$(COVERAGE_FLOORS="$TMP/floors-under" COVERAGE_FILE="$TMP/backend.xml" \
  ./hack/coverage-gate.sh backend 2>&1)
case "$out" in
  *80.0*) echo "  ok   excludes cmd/ (reports 80.0%, not 40.0%)" ;;
  *)      echo "  FAIL excludes cmd/ - got: $out"; fail=1 ;;
esac

COVERAGE_FLOORS="$TMP/floors-under" COVERAGE_FILE="$TMP/backend-malformed.xml" \
  ./hack/coverage-gate.sh backend >/dev/null 2>&1
check "rejects malformed backend XML" 2 $?

./hack/coverage-gate.sh bogus >/dev/null 2>&1
check "rejects unknown side" 2 $?

printf 'backend=79.0\nui=abc\n' > "$TMP/floors-nonnumeric"
COVERAGE_FLOORS="$TMP/floors-nonnumeric" COVERAGE_FILE="$TMP/backend.xml" \
  ./hack/coverage-gate.sh ui >/dev/null 2>&1
check "rejects non-numeric floor" 2 $?

# --- proves LINES, not statements, are measured ---
# 50 of 100 lines covered (all in one "statement" as far as any statement-based
# counter would see it) -> 50.0%. A gate that silently still measured
# statements would report something else (e.g. 100% for one covered
# statement), so this fixture only passes if the line count drives the result.
cat > "$TMP/backend-lines.xml" <<XML
<?xml version="1.0"?>
<coverage>
  <packages>
    <package name="github.com/trick77/peeq/internal/lines">
      <classes>
        <class name="lines" filename="internal/lines/lines.go">
          <lines>$(gen_lines 50 1)$(gen_lines 50 0)</lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
XML

out=$(COVERAGE_FLOORS="$TMP/floors-under" COVERAGE_FILE="$TMP/backend-lines.xml" \
  ./hack/coverage-gate.sh backend 2>&1)
case "$out" in
  *50.0*) echo "  ok   measures lines, not statements (reports 50.0%)" ;;
  *)      echo "  FAIL measures lines, not statements — got: $out"; fail=1 ;;
esac

# --- ui branch ---

cat > "$TMP/ui-summary-good.json" <<'JSON'
{"total": {"lines": {"pct": 90.0}}}
JSON

cat > "$TMP/ui-summary-malformed.json" <<'JSON'
{not valid json
JSON

cat > "$TMP/ui-summary-missing-field.json" <<'JSON'
{"total": {"lines": {}}}
JSON

printf 'backend=79.0\nui=50.0\n' > "$TMP/floors-ui-under"

COVERAGE_FLOORS="$TMP/floors-ui-under" COVERAGE_FILE="$TMP/ui-summary-good.json" \
  ./hack/coverage-gate.sh ui >/dev/null 2>&1
check "ui passes when above floor" 0 $?

COVERAGE_FLOORS="$TMP/floors-ui-under" COVERAGE_FILE="$TMP/ui-summary-malformed.json" \
  ./hack/coverage-gate.sh ui >/dev/null 2>&1
check "ui rejects malformed JSON" 2 $?

COVERAGE_FLOORS="$TMP/floors-ui-under" COVERAGE_FILE="$TMP/ui-summary-missing-field.json" \
  ./hack/coverage-gate.sh ui >/dev/null 2>&1
check "ui rejects missing total.lines.pct" 2 $?

[ "$fail" = 0 ] && echo "coverage-gate: all checks passed"
exit "$fail"
