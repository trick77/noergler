#!/usr/bin/env bash
# hack/patch-coverage.sh [base-ref]
#
# Patch coverage: the lines this branch adds or changes must be at least
# PATCH_MIN% covered. Complements hack/coverage-gate.sh, which enforces the
# absolute project floor — the two answer different questions:
#
#   coverage-gate.sh   "is the codebase as a whole tested enough?"   (75% floor)
#   patch-coverage.sh  "is the code I just wrote tested?"            (75% patch)
#
# The floor alone lets a large well-tested codebase absorb untested new code
# without ever going red. The patch gate alone lets legacy debt sit forever.
# Both, together, pin the gain without punishing anyone for debt they inherited.
#
# The two numbers were deliberately different once — patch at 80 against a 75
# floor, holding new code to a higher bar than inherited debt. They are now the
# same. The remaining reason to run both is the QUESTION each asks, not the
# number: a branch can clear the project floor comfortably while the handful of
# lines it actually changed go untested, and only the patch gate sees that.
#
# Ported from loom, which has run this since its early days.
#
# Coverage reports must already exist — CI produces them before calling this.
set -euo pipefail

BASE_REF="${1:-origin/master}"
PATCH_MIN="${PATCH_MIN:-75}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail=0
checked=0
summary="${GITHUB_STEP_SUMMARY:-/dev/null}"
mkdir -p coverage

# diff-cover needs the base commit present; CI checkouts are shallow by default.
if ! git rev-parse --verify --quiet "$BASE_REF" >/dev/null; then
  echo "patch-coverage: base ref '$BASE_REF' not found." >&2
  echo "  In CI, actions/checkout needs fetch-depth: 0." >&2
  exit 2
fi

# Escape hatch, deliberately narrow and deliberately loud.
#
# Some changes cannot meet a patch bar no matter how good they are: a wholesale
# prettier or gofmt reformat, a mechanical rename, a vendored import. None of
# them add logic, but diff-cover counts every touched line, so what gets
# measured is the pre-existing coverage of code the change did not write. The
# question the gate asks is simply the wrong one for those.
#
# Opt out with [skip patch-coverage] in a commit message on the branch.
#
# This skips ONLY patch coverage. hack/coverage-gate.sh still enforces the
# absolute floor in the same CI job, so overall coverage can never silently
# fall — the worst this can do is let already-untested lines stay untested.
if git log --format='%B' "$(git merge-base "$BASE_REF" HEAD)"..HEAD 2>/dev/null |
  grep -qF '[skip patch-coverage]'; then
  echo "::warning::patch-coverage SKIPPED — a commit on this branch carries [skip patch-coverage]."
  echo "patch-coverage: SKIPPED by [skip patch-coverage] in a commit message." >&2
  echo "  The absolute floor (hack/coverage-gate.sh) still applies and is" >&2
  echo "  enforced separately, so total coverage cannot fall unnoticed." >&2
  exit 0
fi

# diff-cover prints "No lines with coverage information" and exits 0 when none of
# the report's paths match the diff. That is legitimate for a docs-only PR, but it
# is also exactly what a broken path mapping looks like — and that bug has shipped
# before (lcov SF: paths vs git paths). So treat the message as a failure only
# when that stack's sources actually changed.
assert_matched() {
  local report="$1" label="$2" coverage="$3" strip="$4" base changed
  shift 4
  base="$(git merge-base "$BASE_REF" HEAD)"
  # Test files are excluded from coverage reports by design, so a diff touching
  # only tests legitimately matches no coverage data — don't flag that.
  #
  # Vitest setup files are the same category: they are the `setupFiles` entry,
  # excluded from coverage, and therefore never present in the report. A commit
  # touching only the setup file would otherwise trip the absent-from-report
  # check below. Match all the conventions in use across this repo family rather
  # than one repo's filename — peeq and music use src/test-setup.ts, loom uses
  # vitest.setup.ts, lens uses src/test/setup.ts, lens-console uses
  # test-setup.ts. Type-only declarations are likewise never instrumented.
  #
  # Keep the filtered list rather than testing with `grep -qv`: under ugrep (a
  # common macOS `grep`) the -q/-v combination returns 1 even when non-matching
  # lines exist, which would silently disable this guard locally while it still
  # works under GNU grep. The names are needed below anyway.
  changed="$(git diff --name-only "$base"...HEAD -- "$@" |
    grep -vE '(_test\.go|\.test\.tsx?|\.d\.ts|(^|/)(test-setup|vitest\.setup|setup)\.ts)$' || true)"

  [[ -n "$changed" ]] || return 0
  grep -q 'No lines with coverage information' "$report" || return 0

  # Changed source lines can legitimately carry no coverage data of their own:
  # a Go string in a package-level data table, a type-only TS change, a struct
  # tag. None of those are instrumented, so diff-cover reports nothing — which
  # looks exactly like a broken path mapping. Tell the two apart by asking
  # whether the changed files appear in the coverage report at all. Present
  # means the mapping works and the diff simply touched no executable line;
  # absent is the real bug.
  local file
  while IFS= read -r file; do
    [[ -n "$file" ]] || continue
    if grep -qF "${file#"$strip"}" "$coverage"; then
      echo "patch-coverage: $label diff touched no executable lines; n/a."
      return 0
    fi
  done <<<"$changed"

  # Nothing changed is in the report. That is the broken-mapping symptom, but
  # it is ALSO what a whole-module type-only change looks like: a .ts file
  # holding only `export type` emits no runtime code, so it is absent from the
  # report entirely rather than present with zero hits. The loop above cannot
  # tell them apart, and only got away with it while such a file happened to be
  # accompanied by an instrumented one — any one match short-circuits it.
  #
  # So prove the mapping independently of the diff: if the report's OWN paths
  # resolve to files that exist here, the mapping is fine and this diff simply
  # touched nothing instrumented. A genuinely broken mapping still fails,
  # because then none of its paths would resolve.
  local reported
  reported="$(sed -nE 's/^SF:(.*)$/\1/p; s/.*filename="([^"]*)".*/\1/p' "$coverage" | head -100)"
  while IFS= read -r file; do
    [[ -n "$file" ]] || continue
    if [[ -e "${strip}${file}" ]]; then
      echo "patch-coverage: $label diff touched no instrumented lines; n/a."
      echo "  (report paths resolve, so the mapping is sound — the change is" \
        "type-only or otherwise not executable.)"
      return 0
    fi
  done <<<"$reported"

  echo "patch-coverage: FAIL — $label sources changed but are absent from the" >&2
  echo "  coverage report. This usually means the report's paths do not" >&2
  echo "  match git's paths." >&2
  fail=1
}

# --- backend ------------------------------------------------------------------
# CI already converts the coverprofile to Cobertura for hack/coverage-gate.sh;
# reuse that artifact rather than regenerating it.
if [[ -f coverage/backend.xml ]]; then
  checked=1
  echo "== backend patch coverage (>= ${PATCH_MIN}%) =="
  # Where the Go module lives, relative to the repo root. Repos with a UI keep
  # the module under backend/; Go-only repos have go.mod at the root. Detect it
  # rather than hardcoding, so this script stays byte-identical everywhere —
  # forking it per layout is exactly the drift this file exists to avoid.
  if [[ -f backend/go.mod ]]; then
    MODULE_DIR="backend"
    MODULE_PREFIX="backend/"
    MODULE_GLOB="backend/*.go"
  elif [[ -f go.mod ]]; then
    MODULE_DIR="."
    MODULE_PREFIX=""
    MODULE_GLOB="*.go"
  else
    echo "patch-coverage: no go.mod at backend/ or the repo root." >&2
    exit 2
  fi

  # gocover-cobertura writes an ABSOLUTE path into <sources>, e.g.
  # /home/runner/work/peeq/peeq/backend. diff-cover joins that with each
  # class's module-relative filename, producing an absolute path that never
  # matches git's repo-relative paths — so every file silently misses and the
  # gate passes vacuously at "No lines with coverage information".
  #
  # --src-roots does NOT override this; the embedded <sources> wins. Rewriting
  # the element to a repo-relative path is what actually makes the match work.
  # Verified: with the absolute path diff-cover reports 0 matched lines; with
  # the rewrite it correctly reports the changed lines and their coverage.
  sed "s|<source>.*</source>|<source>${MODULE_DIR}</source>|" \
    coverage/backend.xml > coverage/backend-rooted.xml

  # Comments are not code, and must never be counted.
  #
  # Go's coverprofile records BLOCKS — "lines A..B hold N statements" — and
  # never says WHICH lines in the block are statements. gocover-cobertura
  # expands each block into one <line> per line of that range, so every comment
  # and blank line inside a function body reaches the report as hits="0".
  # diff-cover then holds them against the diff, and a well-explained change
  # fails the gate on its prose — with no test anyone could write that would
  # turn those lines green. The only way to raise that number is to delete the
  # explanation, which inverts what the gate is for.
  #
  # strip-comment-lines.go drops a <line> only where go/scanner proves the line
  # carries no non-comment token, so a trailing `// why` and a string holding
  # "http://x" both stay counted. It is deliberately conservative: a file it
  # cannot read or parse keeps every line it had.
  go run hack/strip-comment-lines.go "$MODULE_DIR" \
    < coverage/backend-rooted.xml > coverage/backend-code-only.xml

  diff-cover coverage/backend-code-only.xml \
    --compare-branch "$BASE_REF" \
    --fail-under "$PATCH_MIN" \
    --format "markdown:coverage/backend-patch.md" || fail=1
  cat coverage/backend-patch.md >> "$summary" 2>/dev/null || true
  assert_matched coverage/backend-patch.md backend coverage/backend-code-only.xml \
    "$MODULE_PREFIX" "$MODULE_GLOB"
fi

# --- ui -----------------------------------------------------------------------
if [[ -f coverage/ui/lcov.info ]]; then
  checked=1
  echo "== ui patch coverage (>= ${PATCH_MIN}%) =="
  # vitest writes SF: paths relative to ui/. --src-roots does NOT rewrite these
  # for lcov (it only does so for Cobertura's <sources>), and a mismatch makes
  # diff-cover report "no lines with coverage information" and pass vacuously.
  # Rewrite them to repo-root-relative so they match git's paths.
  sed 's|^SF:|SF:ui/|' coverage/ui/lcov.info > coverage/ui-lcov-rooted.info

  diff-cover coverage/ui-lcov-rooted.info \
    --compare-branch "$BASE_REF" \
    --fail-under "$PATCH_MIN" \
    --format "markdown:coverage/ui-patch.md" || fail=1
  cat coverage/ui-patch.md >> "$summary" 2>/dev/null || true
  assert_matched coverage/ui-patch.md ui coverage/ui-lcov-rooted.info "" "ui/src/*.ts" "ui/src/*.tsx"
fi

# A gate that checked nothing must not report success: if neither report was
# produced (reporter moved, output dir changed), fail loudly instead of green.
if [[ "$checked" -eq 0 ]]; then
  echo "patch-coverage: no coverage reports found under coverage/." >&2
  echo "  Run the tests first — CI does this before invoking the gate." >&2
  exit 2
fi

exit "$fail"
