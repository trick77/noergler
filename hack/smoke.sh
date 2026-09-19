#!/usr/bin/env bash
# hack/smoke.sh: build, start serve with a two-team config (one bad block),
# hit the probes and an unknown route, stop, print the interesting log lines.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

go build -o "$tmp/noergler" ./cmd/noergler
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
env -i PATH="$PATH" \
  BITBUCKET_URL=https://bb.example.com BITBUCKET_TOKEN=t BITBUCKET_USERNAME=bot \
  LLMWIRE_LITELLM_BASE_URL=https://llm.example.com/v1 LLMWIRE_LITELLM_MODELS=gpt-5.5=ai-gateway-gpt-5.5 \
  OPENAI_MODEL=gpt-5.5 JIRA_URL=https://jira.example.com JIRA_TOKEN=t \
  DATABASE_URL=postgres://x TEAMS_CONFIG="$tmp/teams.yaml" \
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
grep -E 'version|team_disabled|teams_ready|http_request|stopped|listening|DISABLED' "$tmp/serve.log"
