#!/usr/bin/env bash
# hack/smoke.sh: build, migrate, start serve with a two-team config (one bad
# block), hit the probes, the teams API and a signed webhook replay, stop,
# print the log lines.
# Needs a Postgres: `docker compose up -d postgres` or DATABASE_URL.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
pid=""
fakes_pid=""
# The server dies with the script, even when the reader of our output goes
# away first: a stale listener on the port would answer the next run.
cleanup() {
  # Under errexit a failed kill (server already gone) would end the trap
  # before the rm and turn a green run into exit 1.
  if [ -n "$pid" ]; then kill -TERM "$pid" 2>/dev/null || true; fi
  if [ -n "$fakes_pid" ]; then kill -TERM "$fakes_pid" 2>/dev/null || true; fi
  rm -r "$tmp" || true
}
trap cleanup EXIT

go build -o "$tmp/noergler" ./cmd/noergler
go build -o "$tmp/fakes" ./hack/fakes

# serve now checks Bitbucket and Jira at startup, so both have to answer.
fakes_port=${FAKES_PORT:-18099}
"$tmp/fakes" -addr ":$fakes_port" > "$tmp/fakes.log" 2>&1 &
fakes_pid=$!
for _ in $(seq 50); do
  curl -sf -o /dev/null "localhost:$fakes_port/rest/api/2/myself" && break
  sleep 0.1
done
# A slug and a project key of their own per run. The script replays one fixed
# PR, and a row left by an earlier run would make the reviewer skip it as
# already reviewed. A shared Postgres is the normal case here, so the run
# claims its own names rather than deleting anyone else's rows.
run=$$
team="platform-$run"
project="PROJ$run"
cat > "$tmp/teams.yaml" <<EOF
teams:
  - slug: $team
    webhook_secret_env: TEAM_SMOKE_WEBHOOK_SECRET
    projects: [{key: $project}]
    inference: {api_key_env: TEAM_SMOKE_OPENAI_API_KEY}
  - slug: payments-$run
    webhook_secret_env: TEAM_BROKEN_WEBHOOK_SECRET
    inference: {api_key_env: TEAM_BROKEN_OPENAI_API_KEY, base_url: nope}
EOF
# The fixture names PROJ; rewrite it to this run's key.
payload="$tmp/webhook.json"
sed "s/\"PROJ\"/\"$project\"/g" internal/webhook/testdata/sample_webhook.json > "$payload"

port=${PORT:-18080}
dsn=${DATABASE_URL:-postgres://noergler:changeme@localhost:5432/noergler?sslmode=disable}
DATABASE_URL="$dsn" "$tmp/noergler" migrate > "$tmp/migrate.log" 2>&1 || { cat "$tmp/migrate.log"; exit 1; }
env -i PATH="$PATH" \
  BITBUCKET_URL="http://localhost:$fakes_port" BITBUCKET_TOKEN=t BITBUCKET_USERNAME=bot \
  LLMWIRE_LITELLM_BASE_URL="http://localhost:$fakes_port" LLMWIRE_LITELLM_MODELS=gpt-5.5=ai-gateway-gpt-5.5 \
  OPENAI_MODEL=gpt-5.5 JIRA_URL="http://localhost:$fakes_port" JIRA_TOKEN=t \
  DATABASE_URL="$dsn" TEAMS_CONFIG="$tmp/teams.yaml" \
  TEAM_SMOKE_WEBHOOK_SECRET=s TEAM_SMOKE_OPENAI_API_KEY=k \
  SERVER_PORT="$port" NOERGLER_VERSION=smoke NOERGLER_PUBLIC_URL="http://localhost:$port" \
  "$tmp/noergler" serve > "$tmp/serve.log" 2>&1 &
pid=$!
sleep 1

# RSS once the teams are up and the tokenizer is warm. The container sets
# GOMEMLIMIT=1500MiB against a 2Gi pod, so this is the number to watch: it is
# the floor a deployment starts from, before any PR is reviewed.
echo "RSS after warm-up: $(ps -o rss= -p "$pid" | tr -d ' ') KiB"

echo "GET /health: $(curl -s "localhost:$port/health")"
echo "GET /ready: HTTP $(curl -s -o /dev/null -w '%{http_code}' "localhost:$port/ready")"
echo "GET /nope: $(curl -s -H 'X-Request-Id: req-1' "localhost:$port/nope")"
echo "GET /teams/$team: $(curl -s -H 'Authorization: Bearer s' "localhost:$port/teams/$team")"

# A signed replay of the sample delivery: the only end-to-end proof that the
# route, the HMAC, the ownership check and the queue are wired together.
echo "POST /webhook/$team (signed): $(PORT="$port" TEAM="$team" SECRET=s hack/replay.sh "$payload" pr:opened)"
echo "POST /webhook/$team (bad signature): HTTP $(curl -s -o /dev/null -w '%{http_code}' -X POST \
  -H 'X-Hub-Signature: sha256=deadbeef' --data-binary "@$payload" "localhost:$port/webhook/$team")"

# The worker needs a moment to pick the job up before SIGTERM drains it.
sleep 2
kill -TERM "$pid"
wait "$pid" || true
echo "--- log"
grep -E "version|Database|Bitbucket:|Jira:|team_disabled|team_ready|teams_ready|http_request|stopped|listening|DISABLED|queue\[" "$tmp/serve.log"
echo "--- review posted"
grep -c "comment posted" "$tmp/fakes.log" || true
echo "--- endpoints hack/fakes does not serve (want none)"
grep "unhandled:" "$tmp/fakes.log" | sed 's/.*unhandled: //' | sort -u || true
