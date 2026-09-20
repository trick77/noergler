#!/usr/bin/env bash
# hack/parity.sh: run the same webhook through the Go service and the Python
# original and diff what each one produced.
#
# What this proves that the golden corpora do not: the corpora drive the
# renderers with curated arguments, so they pin formatting. They say nothing
# about what the pipeline FEEDS the renderers from a fetched diff, nor about the
# integration (HMAC, ownership, the queue, posting order, the riptide claim).
# That is what a replay against both services compares.
#
# Each side gets its own fakes process, its own port and its own database, so
# neither can see the other's comments, claims or rows. The canned review comes
# from one file, so both receive identical bytes.
#
# Usage: hack/parity.sh
# Env: PYTHON_REPO (/Users/jan/localgit/noergler), PGHOST/PGPORT, KEEP=1 to
#      keep the output directory.
set -euo pipefail
cd "$(dirname "$0")/.."

python_repo=${PYTHON_REPO:-/Users/jan/localgit/noergler}
pghost=${PGHOST:-localhost}
pgport=${PGPORT:-5432}
go_dsn=${GO_DSN:-postgres://noergler:changeme@$pghost:$pgport/noergler?sslmode=disable}
py_dsn=${PY_DSN:-postgres://noergler:changeme@$pghost:$pgport/noergler_py}

# The gateway alias both sides must put on the wire. Go maps its profile id onto
# it via LLMWIRE_LITELLM_MODELS; Python sends OPENAI_MODEL verbatim, so it is
# handed the alias directly. Getting this wrong makes every diff noise.
alias_model=ai-gateway-gpt-5.5
profile_model=gpt-5.5
effort=high

out=${OUT:-$(mktemp -d)}
mkdir -p "$out/go" "$out/py"

go_fakes_port=18190
py_fakes_port=18191
go_port=18180
py_port=18181

pids=()
cleanup() {
  for p in "${pids[@]:-}"; do kill -TERM "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT

# A slug and project key of their own, so a row left by an earlier run cannot
# make either reviewer skip the PR as already reviewed. The shared Postgres is
# the normal case here; this claims its own names instead of deleting rows
# other worktrees may be using.
run=$$
team="parity$run"
project="PAR$run"
secret=s

echo "== building"
go build -o "$out/noergler" ./cmd/noergler
go build -o "$out/fakes" ./hack/fakes

echo "== starting fakes (go :$go_fakes_port, py :$py_fakes_port)"
"$out/fakes" -addr ":$go_fakes_port" -record "$out/go" -review hack/testdata/review.json \
  > "$out/go-fakes.log" 2>&1 &
pids+=($!)
"$out/fakes" -addr ":$py_fakes_port" -record "$out/py" -review hack/testdata/review.json \
  > "$out/py-fakes.log" 2>&1 &
pids+=($!)
# Both must answer. A fake that lost the port bind dies at once, and without
# this check the run would look like the other implementation posting nothing:
# the first attempt at this diffed a live service against a dead port.
for port in "$go_fakes_port" "$py_fakes_port"; do
  up=""
  for _ in $(seq 50); do
    if curl -sf -o /dev/null "localhost:$port/rest/api/2/myself"; then
      up=1
      break
    fi
    sleep 0.1
  done
  if [ -z "$up" ]; then
    echo "!! fakes on :$port never answered"
    cat "$out/go-fakes.log" "$out/py-fakes.log"
    exit 1
  fi
done

# One teams.yaml shape serves both: the keys used here mean the same thing in
# each implementation. riptide is on, because the rollup payload is one of the
# outputs being compared and smoke.sh never exercised it.
write_teams() {
  cat > "$1" <<EOF
teams:
  - slug: $team
    webhook_secret_env: TEAM_PARITY_WEBHOOK_SECRET
    projects: [{key: $project}]
    inference: {api_key_env: TEAM_PARITY_OPENAI_API_KEY, reasoning_effort: $effort}
    riptide: {url: "http://localhost:$2", token_env: TEAM_PARITY_RIPTIDE_TOKEN}
EOF
}
write_teams "$out/teams-go.yaml" "$go_fakes_port"
write_teams "$out/teams-py.yaml" "$py_fakes_port"

payload="$out/webhook.json"
sed "s/\"PROJ\"/\"$project\"/g" internal/webhook/testdata/sample_webhook.json > "$payload"

# The dispatch reads eventKey from the BODY; the X-Event-Key header only gates
# the ping. A header-only change leaves a pr:merged replay running the review
# path, which is what the first run of this script did.
merged="$out/webhook-merged.json"
sed 's/"eventKey": "pr:opened"/"eventKey": "pr:merged"/' "$payload" > "$merged"
grep -q '"eventKey": "pr:merged"' "$merged" || {
  echo "!! could not rewrite eventKey; the fixture's shape changed"
  exit 1
}

echo "== migrating"
# Python needs a database of its own: its schema is accumulators where Go's is
# runs, so they cannot share one. Creating it is not destructive; alembic's
# error if it is missing does not say what is wrong.
docker exec "${PG_CONTAINER:-phase2-postgres-1}" psql -U noergler -d postgres \
  -tAc "select 1 from pg_database where datname='noergler_py'" 2>/dev/null | grep -q 1 ||
  docker exec "${PG_CONTAINER:-phase2-postgres-1}" psql -U noergler -d postgres \
    -c "CREATE DATABASE noergler_py" > /dev/null

DATABASE_URL="$go_dsn" "$out/noergler" migrate > "$out/go-migrate.log" 2>&1
(cd "$python_repo" && DATABASE_URL="$py_dsn" .venv/bin/alembic upgrade head) > "$out/py-migrate.log" 2>&1

echo "== starting go serve :$go_port"
env -i PATH="$PATH" \
  BITBUCKET_URL="http://localhost:$go_fakes_port" BITBUCKET_TOKEN=t BITBUCKET_USERNAME=bot \
  LLMWIRE_LITELLM_BASE_URL="http://localhost:$go_fakes_port" \
  LLMWIRE_LITELLM_MODELS="$profile_model=$alias_model" \
  OPENAI_MODEL="$profile_model" OPENAI_REASONING_EFFORT="$effort" \
  JIRA_URL="http://localhost:$go_fakes_port" JIRA_TOKEN=t \
  DATABASE_URL="$go_dsn" TEAMS_CONFIG="$out/teams-go.yaml" \
  TEAM_PARITY_WEBHOOK_SECRET="$secret" TEAM_PARITY_OPENAI_API_KEY=k \
  TEAM_PARITY_RIPTIDE_TOKEN=rt \
  SERVER_PORT="$go_port" NOERGLER_VERSION=parity \
  NOERGLER_PUBLIC_URL="http://localhost:$go_port" \
  "$out/noergler" serve > "$out/go-serve.log" 2>&1 &
pids+=($!)

echo "== starting python uvicorn :$py_port"
# Python takes the gateway ALIAS as OPENAI_MODEL: it sends that string on the
# wire, where Go sends the profile id and lets llmwire map it.
# TIKTOKEN_CACHE_DIR is pinned because the default lives in a temp directory
# the OS prunes, and a cold tiktoken wants the network at boot.
env -i PATH="$PATH" HOME="$HOME" \
  BITBUCKET_URL="http://localhost:$py_fakes_port" BITBUCKET_TOKEN=t BITBUCKET_USERNAME=bot \
  OPENAI_BASE_URL="http://localhost:$py_fakes_port" \
  OPENAI_MODEL="$alias_model" OPENAI_REASONING_EFFORT="$effort" \
  JIRA_URL="http://localhost:$py_fakes_port" JIRA_TOKEN=t \
  DATABASE_URL="$py_dsn" TEAMS_CONFIG="$out/teams-py.yaml" \
  TEAM_PARITY_WEBHOOK_SECRET="$secret" TEAM_PARITY_OPENAI_API_KEY=k \
  TEAM_PARITY_RIPTIDE_TOKEN=rt \
  NOERGLER_VERSION=parity NOERGLER_PUBLIC_URL="http://localhost:$py_port" \
  TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$HOME/.cache/tiktoken}" \
  "$python_repo/.venv/bin/uvicorn" app.main:app \
  --app-dir "$python_repo" --host 127.0.0.1 --port "$py_port" \
  > "$out/py-serve.log" 2>&1 &
pids+=($!)

# Both need their startup checks to finish before the route exists.
for port in "$go_port" "$py_port"; do
  ok=""
  for _ in $(seq 100); do
    if curl -sf -o /dev/null "localhost:$port/health"; then ok=1; break; fi
    sleep 0.2
  done
  if [ -z "$ok" ]; then
    echo "!! :$port never became healthy"
    tail -30 "$out/go-serve.log" "$out/py-serve.log"
    exit 1
  fi
done

echo "== replaying pr:opened to both"
PORT="$go_port" TEAM="$team" SECRET="$secret" hack/replay.sh "$payload" pr:opened
echo
PORT="$py_port" TEAM="$team" SECRET="$secret" hack/replay.sh "$payload" pr:opened
echo

# The webhook returns before the worker runs: wait for the output, not the
# response. One inline comment per finding in hack/testdata/review.json, plus
# the summary last, so the fixture's two findings mean three comments. Waiting
# for only two let the run continue with the summary still in flight, and
# cleanup would then kill both services before it landed.
# A bare `[ a ] && [ b ] && break` as the last command in the body would make
# errexit kill the whole run on the first not-yet iteration, taking both
# services down mid-review. Keep the test inside an if.
want_comments=3
count() { find "$1" -name "$2" 2>/dev/null | wc -l | tr -d ' '; }
for _ in $(seq 60); do
  if [ "$(count "$out/go" 'comment-*.json')" -ge "$want_comments" ] &&
    [ "$(count "$out/py" 'comment-*.json')" -ge "$want_comments" ]; then
    break
  fi
  sleep 0.5
done

echo "== replaying pr:merged to both (riptide rollup)"
PORT="$go_port" TEAM="$team" SECRET="$secret" hack/replay.sh "$merged" pr:merged
echo
PORT="$py_port" TEAM="$team" SECRET="$secret" hack/replay.sh "$merged" pr:merged
echo
for _ in $(seq 40); do
  if [ -f "$out/go/rollup-1.json" ] && [ -f "$out/py/rollup-1.json" ]; then
    break
  fi
  sleep 0.5
done

cleanup
trap - EXIT

# Normalize what is legitimately per-run before diffing: ids, timestamps,
# durations, and the elapsed seconds the summary footnote prints.
norm() {
  ALIAS_MODEL="$alias_model" PROFILE_MODEL="$profile_model" python3 - "$1" <<'PY'
import json, os, re, sys

from decimal import Decimal

# total_cost_usd is a decimal STRING on both sides and the same amount, but
# Python's comes off a NUMERIC(_,6) column through str(), so it keeps the
# column's trailing zeros ("0.012300") where Go trims them ("0.0123").
# Python's own test asserts "0.0123", so Go matches the intended contract and
# this is a storage artifact, not a divergence: compare the amounts.
NUMERIC_STRINGS = {"total_cost_usd"}

# Adjacent-hunk merge, the divergence AGENTS.md pins: the overlap comes off
# hunk 1's invented after-context here, where Python trimmed hunk 2's real body
# and so dropped removal lines. The visible trace is a blank CONTEXT line inside
# a ```diff fence that Python keeps and Go does not.
#
# Only blank lines inside those fences are folded, and only there: a blank line
# in prose still counts, and a "+"/"-" line never does, so a dropped removal
# would still fail the diff.
def fold_blank_diff_lines(s):
    out, in_diff = [], False
    for line in s.split("\n"):
        if line.startswith("```diff"):
            in_diff = True
        elif in_diff and line.startswith("```"):
            in_diff = False
        if in_diff and line == "":
            continue
        out.append(line)
    return "\n".join(out)

def scrub(o):
    if isinstance(o, dict):
        out = {}
        for k, v in sorted(o.items()):
            if k in {
                "id", "call_id", "created", "timestamp", "created_at", "updated_at",
                "started_at", "finished_at", "duration_ms", "elapsed_ms",
                "total_elapsed_ms", "first_reviewed_at", "last_reviewed_at",
                "emitted_at", "first_review_at", "closed_at",
            }:
                out[k] = "<scrubbed>"
            elif k in NUMERIC_STRINGS and isinstance(v, str):
                # Strip trailing zeros only AFTER a decimal point: a bare
                # rstrip("0") turns "10" into "1" and would quietly compare two
                # different amounts as equal.
                s = f"{Decimal(v):f}"
                out[k] = s.rstrip("0").rstrip(".") if "." in s else s
            else:
                out[k] = scrub(v)
        return out
    if isinstance(o, list):
        return [scrub(v) for v in o]
    if isinstance(o, str):
        # The displayed model string: Go shows the llmwire profile id, Python
        # the gateway alias it puts on the wire. Deliberate: the alias is the
        # operator's private naming and does not belong in a PR comment. Both
        # carry the reasoning effort. Fold the alias onto the profile id so a
        # real difference here would still show.
        s = o.replace(os.environ["ALIAS_MODEL"], os.environ["PROFILE_MODEL"])
        s = fold_blank_diff_lines(s)
        # The footnote's file-content figure differs by one token on a diff that
        # has consecutive blank context lines, and it is not the blank lines
        # themselves: tiktoken-go and tiktoken disagree on " \n \n". tiktoken
        # merges that run into one token (id 56319), tiktoken-go emits two, so
        # Go counts one extra per run. Verified on the library, not our wrapper,
        # which is a pass-through. Over-counting shrinks the usable budget, so
        # it errs safe; the figure is advisory in a footnote. Compare the label
        # rather than the digits here, and only here.
        s = re.sub(r"~[\d']+ file content", "~<n> file content", s)
        # Wall-clock figures inside rendered markdown.
        s = re.sub(r"⏱️ [\d.]+s", "⏱️ <t>", s)
        s = re.sub(r"\b\d{4}-\d{2}-\d{2}T[\d:.]+(Z|[+-]\d{2}:?\d{2})", "<ts>", s)
        s = re.sub(r"in [\d.]+s", "in <t>", s)
        return s
    return o

raw = open(sys.argv[1], "rb").read()
try:
    print(json.dumps(scrub(json.loads(raw)), indent=2, ensure_ascii=False, sort_keys=True))
except json.JSONDecodeError:
    print(raw.decode("utf-8", "replace"))
PY
}

# The normalizer folds the gateway alias onto the profile id in every string,
# which would also hide the one case that must never happen: Go putting the
# PROFILE id on the wire, because llmwire would then route it to
# api.openai.com instead of the gateway. Assert the raw wire name first.
status=0
for side in go py; do
  f="$out/$side/completion-1.json"
  if [ ! -f "$f" ]; then
    # Skipping this silently would let a run that never reached the gateway
    # look like one that passed the check.
    echo "!! $side made no completion request; nothing reached the gateway"
    status=1
    continue
  fi
  if ! grep -q "\"model\":[[:space:]]*\"$alias_model\"" "$f"; then
    echo "!! $side sent the wrong model on the wire (want $alias_model):"
    python3 -c "import json,sys; print('   model =', json.load(open(sys.argv[1])).get('model'))" "$f"
    status=1
  fi
done

# The blank-context fold above is only safe if no real diff line differs. Check
# the prompt's "+"/"-" lines directly: a dropped removal line, which is the bug
# the merge divergence exists to avoid, would show up here.
if [ -f "$out/go/completion-1.json" ] && [ -f "$out/py/completion-1.json" ]; then
  difflines() {
    python3 -c "
import json, sys
c = json.load(open(sys.argv[1]))['messages'][1]['content']
for line in c.split('\n'):
    t = line.strip()
    if (t.startswith('+') or t.startswith('-')) and not t.startswith(('+++', '---')):
        print(line)
" "$1"
  }
  difflines "$out/go/completion-1.json" > "$out/go-difflines.txt"
  difflines "$out/py/completion-1.json" > "$out/py-difflines.txt"
  if diff -u "$out/go-difflines.txt" "$out/py-difflines.txt" > "$out/difflines.diff"; then
    echo "--- prompt +/- diff lines: identical"
  else
    echo "--- prompt +/- diff lines: DIFFER (a real diff line changed, not just context)"
    sed -n '1,40p' "$out/difflines.diff"
    status=1
  fi
fi

report() {
  local kind=$1
  local gf="$out/go/$kind" pf="$out/py/$kind"
  if [ ! -f "$gf" ] && [ ! -f "$pf" ]; then
    # Not a pass. If neither side produced this, nothing was compared, and a
    # run where nothing happened at all (both webhooks rejected, both teams
    # disabled, no schema) would otherwise print PARITY OK and exit 0.
    echo "--- $kind: MISSING on BOTH sides, nothing was compared"
    status=1
    return
  fi
  if [ ! -f "$gf" ] || [ ! -f "$pf" ]; then
    echo "--- $kind: MISSING on one side (go=$([ -f "$gf" ] && echo yes || echo no) py=$([ -f "$pf" ] && echo yes || echo no))"
    status=1
    return
  fi
  norm "$gf" > "$out/go-$kind.norm"
  norm "$pf" > "$out/py-$kind.norm"
  if diff -u "$out/go-$kind.norm" "$out/py-$kind.norm" > "$out/$kind.diff"; then
    echo "--- $kind: identical"
  else
    echo "--- $kind: DIFFERS"
    sed -n '1,60p' "$out/$kind.diff"
    status=1
  fi
}

echo
echo "===== parity report ====="
# The comparison is on PARSED json, not raw bytes: Go and Python order object
# keys differently (Go emits a message's content before its role), which no HTTP
# API can observe. Raw sizes therefore differ by a few bytes even when every
# value matches.
echo "go  comments: $(count "$out/go" 'comment-*.json')   py comments: $(count "$out/py" 'comment-*.json')"

# Both sides must have produced the expected number of comments. Without this a
# run where neither service did anything has nothing to diff and would read as
# agreement.
for side in go py; do
  n=$(count "$out/$side" 'comment-*.json')
  if [ "$n" -ne "$want_comments" ]; then
    echo "!! $side posted $n comments, want $want_comments"
    status=1
  fi
done
# One replay of pr:opened (a review) and one of pr:merged (a rollup).
#
# Inline comments are posted one per finding and the summary goes last, so with
# the two-finding fixture comment-3 IS the summary. It has to be in this list:
# the footnote is the one place the model label is rendered, so leaving it out
# means the run cannot see the output this phase changed.
#
# No comment-update: the merged event emits the rollup and does not re-review,
# so nothing edits the summary on this path.
for k in completion-1.json comment-1.json comment-2.json comment-3.json \
  rollup-1.json; do
  report "$k"
done

echo
echo "--- rows (not comparable field by field: Python's schema is accumulators, Go's is runs)"
psql_go() { docker exec "${PG_CONTAINER:-phase2-postgres-1}" psql -U noergler -d noergler -tAc "$1" 2>/dev/null || echo "?"; }
psql_py() { docker exec "${PG_CONTAINER:-phase2-postgres-1}" psql -U noergler -d noergler_py -tAc "$1" 2>/dev/null || echo "?"; }
echo "go  review_runs=$(psql_go "select count(*) from review_runs r join pull_requests p on p.id=r.pull_request_id where p.project_key='$project'")" \
     "findings=$(psql_go "select count(*) from findings f join pull_requests p on p.id=f.pull_request_id where p.project_key='$project'")"
echo "py  pr_reviews=$(psql_py "select count(*) from pr_reviews where project_key='$project'")" \
     "findings=$(psql_py "select count(*) from review_findings f join pr_reviews p on p.id=f.pr_review_id where p.project_key='$project'")"

echo
if [ "$status" -eq 0 ]; then
  echo "PARITY OK"
else
  echo "PARITY DIFFERENCES: see $out/*.diff"
fi
echo "output: $out"
[ -n "${KEEP:-}" ] || echo "(set KEEP=1 to stop this being a mktemp dir you lose track of)"
exit "$status"
