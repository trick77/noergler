#!/usr/bin/env bash
# hack/smoke.sh: build, migrate, start serve with a two-team config (one bad
# block), hit the probes and an unknown route, stop, print the log lines.
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
cat > "$tmp/teams.yaml" <<'EOF'
teams:
  - slug: platform
    webhook_secret_env: TEAM_PLATFORM_WEBHOOK_SECRET
    projects: [{key: PLAT}]
    inference: {api_key_env: TEAM_PLATFORM_OPENAI_API_KEY}
  - slug: payments
    webhook_secret_env: TEAM_PAYMENTS_WEBHOOK_SECRET
    inference: {api_key_env: TEAM_PAYMENTS_OPENAI_API_KEY, base_url: nope}
EOF

port=${PORT:-18080}
dsn=${DATABASE_URL:-postgres://noergler:changeme@localhost:5432/noergler?sslmode=disable}
DATABASE_URL="$dsn" "$tmp/noergler" migrate > "$tmp/migrate.log" 2>&1 || { cat "$tmp/migrate.log"; exit 1; }
env -i PATH="$PATH" \
  BITBUCKET_URL="http://localhost:$fakes_port" BITBUCKET_TOKEN=t BITBUCKET_USERNAME=bot \
  LLMWIRE_LITELLM_BASE_URL=https://llm.example.com/v1 LLMWIRE_LITELLM_MODELS=gpt-5.5=ai-gateway-gpt-5.5 \
  OPENAI_MODEL=gpt-5.5 JIRA_URL="http://localhost:$fakes_port" JIRA_TOKEN=t \
  DATABASE_URL="$dsn" TEAMS_CONFIG="$tmp/teams.yaml" \
  TEAM_PLATFORM_WEBHOOK_SECRET=s TEAM_PLATFORM_OPENAI_API_KEY=k \
  SERVER_PORT="$port" NOERGLER_VERSION=smoke \
  "$tmp/noergler" serve > "$tmp/serve.log" 2>&1 &
pid=$!
sleep 1

echo "GET /health: $(curl -s "localhost:$port/health")"
echo "GET /ready: HTTP $(curl -s -o /dev/null -w '%{http_code}' "localhost:$port/ready")"
echo "GET /nope: $(curl -s -H 'X-Request-Id: req-1' "localhost:$port/nope")"
kill -TERM "$pid"
wait "$pid" || true
echo "--- log"
grep -E 'version|Database|Bitbucket|Jira|team_disabled|teams_ready|http_request|stopped|listening|DISABLED' "$tmp/serve.log"
