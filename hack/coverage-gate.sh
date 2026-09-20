#!/usr/bin/env bash
# hack/coverage-gate.sh <backend|ui>
#
# Fails when line coverage falls below the hard floor in hack/coverage-floors.
#
# Backend coverage comes from a Cobertura XML conversion of the coverprofile,
# not from `go tool cover -func`: that prints only per-function statement
# percentages, exposes no line metric, and gives no way to exclude a package.
# Anything under cmd/ (main() wiring) is deliberately not counted. Matching the
# whole cmd/ tree rather than one binary's name keeps this script identical
# across every repo in the family, including those with several binaries.
set -euo pipefail

# Force a C/POSIX numeric locale so awk always uses '.' as the decimal
# separator, regardless of the environment's locale settings.
export LC_ALL=C
export LC_NUMERIC=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLOORS="${COVERAGE_FLOORS:-$ROOT/hack/coverage-floors}"
SIDE="${1:-}"

die() { echo "coverage-gate: $*" >&2; exit 2; }

case "$SIDE" in
  backend|ui) ;;
  *) die "usage: $0 <backend|ui>" ;;
esac

is_number() { awk -v v="$1" 'BEGIN { exit !(v ~ /^[0-9]+(\.[0-9]+)?$/) }'; }

[ -f "$FLOORS" ] || die "floors file not found: $FLOORS"
FLOOR="$(awk -F= -v k="$SIDE" '$1==k {print $2; exit}' "$FLOORS")"
[ -n "$FLOOR" ] || die "no floor for '$SIDE' in $FLOORS"
is_number "$FLOOR" || die "floor for '$SIDE' is not numeric: '$FLOOR' in $FLOORS"

if [ "$SIDE" = backend ]; then
  FILE="${COVERAGE_FILE:-$ROOT/coverage/backend.xml}"
  [ -f "$FILE" ] || die "no Cobertura XML at $FILE — run the tests first"
  set +e
  PCT="$(python3 -c '
import sys
import xml.etree.ElementTree as ET

path = sys.argv[1]
try:
    root = ET.parse(path).getroot()
except ET.ParseError:
    sys.exit(3)

tot = cov = 0
for cls in root.iter("class"):
    fn = cls.get("filename", "")
    if fn.startswith("cmd/") or "/cmd/" in fn:
        continue
    for line in cls.iter("line"):
        tot += 1
        if int(line.get("hits", "0")) > 0:
            cov += 1

if tot == 0:
    print("0.0")
else:
    print("%.1f" % (100 * cov / tot))
' "$FILE" 2>/dev/null)"
  PY_RC=$?
  set -e
  [ "$PY_RC" -eq 0 ] || die "malformed Cobertura XML at $FILE"
else
  FILE="${COVERAGE_FILE:-$ROOT/coverage/ui/coverage-summary.json}"
  [ -f "$FILE" ] || die "no coverage summary at $FILE — run the UI tests first"
  # The node snippet catches both invalid JSON and a missing total.lines.pct
  # field, exiting 3 instead of letting a raw stack trace reach the caller.
  # Any non-zero exit here is a parse/config error.
  set +e
  PCT="$(node -e '
    try {
      const s = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"));
      const pct = s && s.total && s.total.lines && s.total.lines.pct;
      if (typeof pct !== "number" || Number.isNaN(pct)) {
        throw new Error("missing total.lines.pct");
      }
      process.stdout.write(pct.toFixed(1));
    } catch (e) {
      process.exit(3);
    }
  ' "$FILE" 2>/dev/null)"
  NODE_RC=$?
  set -e
  [ "$NODE_RC" -eq 0 ] || die "malformed coverage summary at $FILE"
fi
is_number "$PCT" || die "computed coverage percentage is not numeric: '$PCT'"

# Compare with a 0.05 grace so float formatting alone can never fail a build.
if awk -v p="$PCT" -v f="$FLOOR" 'BEGIN { exit !(p < f - 0.05) }'; then
  echo "coverage-gate: $SIDE FAILED — ${PCT}% of lines, floor is ${FLOOR}%" >&2
  echo "  Add tests to reach the floor in $FLOORS." >&2
  exit 1
fi

echo "coverage-gate: $SIDE ok — ${PCT}% of lines (floor ${FLOOR}%)"
