#!/usr/bin/env bash
# scripts/smoke.sh: build, migrate, start serve with a two-team config (one bad
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

go build -C backend -o "$tmp/noergler" ./cmd/noergler
# By file, not by package: scripts/ is outside the module now, so `./scripts/fakes`
# finds no go.mod on a fresh clone. It is a single stdlib-only file, which go
# builds without a module.
go build -o "$tmp/fakes" scripts/fakes/main.go

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
    projects: [{key: $project}, {key: ${project}NOREAD}]
    inference: {api_key_env: TEAM_SMOKE_OPENAI_API_KEY}
  - slug: payments-$run
    webhook_secret_env: TEAM_BROKEN_WEBHOOK_SECRET
    inference: {api_key_env: TEAM_BROKEN_OPENAI_API_KEY, base_url: nope}
EOF
# The fixture names PROJ; rewrite it to this run's key.
payload="$tmp/webhook.json"
sed "s/\"PROJ\"/\"$project\"/g" backend/internal/webhook/testdata/sample_webhook.json > "$payload"

port=${PORT:-18080}
dsn=${DATABASE_URL:-postgres://noergler:changeme@localhost:5432/noergler?sslmode=disable}
DATABASE_URL="$dsn" "$tmp/noergler" migrate > "$tmp/migrate.log" 2>&1 || { cat "$tmp/migrate.log"; exit 1; }
# One fixed key, never a fresh one per run: the DB is usually shared, and a
# credential sealed under another key aborts the next boot against it. The
# run also deletes every credential it pushes. A DB holding a local
# instance's credentials needs that instance's key here.
secrets_key=${NOERGLER_SECRETS_KEY:-5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a}
env -i PATH="$PATH" \
  BITBUCKET_URL="http://localhost:$fakes_port" BITBUCKET_TOKEN=t BITBUCKET_USERNAME=bot \
  LLMWIRE_LITELLM_BASE_URL="http://localhost:$fakes_port" LLMWIRE_LITELLM_MODELS=gpt-5.5=ai-gateway-gpt-5.5 \
  OPENAI_MODEL=gpt-5.5 JIRA_URL="http://localhost:$fakes_port" JIRA_TOKEN=t \
  DATABASE_URL="$dsn" TEAMS_CONFIG="$tmp/teams.yaml" NOERGLER_SECRETS_KEY="$secrets_key" \
  TEAM_SMOKE_WEBHOOK_SECRET=s TEAM_SMOKE_OPENAI_API_KEY=k \
  SERVER_PORT="$port" NOERGLER_PUBLIC_URL="http://localhost:$port" \
  "$tmp/noergler" serve > "$tmp/serve.log" 2>&1 &
pid=$!
sleep 1

# RSS once the teams are up and the tokenizer is warm. The container sets
# GOMEMLIMIT=1500MiB against a 2Gi pod, so this is the number to watch: it is
# the floor a deployment starts from, before any PR is reviewed.
echo "RSS after warm-up: $(ps -o rss= -p "$pid" | tr -d ' ') KiB"

echo "GET /healthz: $(curl -s "localhost:$port/healthz")"
echo "GET /readyz: HTTP $(curl -s -o /dev/null -w '%{http_code}' "localhost:$port/readyz")"
echo "GET /nope: $(curl -s -H 'X-Request-Id: req-1' "localhost:$port/nope")"
echo "GET /teams/$team: $(curl -s -H 'Authorization: Bearer s' "localhost:$port/teams/$team")"

# A signed replay of the sample delivery: the only end-to-end proof that the
# route, the HMAC, the ownership check and the queue are wired together.
echo "POST /webhook/$team (signed): $(PORT="$port" TEAM="$team" SECRET=s scripts/replay.sh "$payload" pr:opened)"
# A Q&A mention: the fake gateway asks for pr_status once, then answers with
# what the tool returned, so the log shows the staged tool loop end to end.
mention="$tmp/mention.json"
sed -e 's/"eventKey": "pr:opened"/"eventKey": "pr:comment:added"/' -e '$d' "$payload" > "$mention"
printf '%s\n' ',  "comment": {"id": 900, "text": "@bot why was this PR reviewed?", "author": {"name": "jan.username", "slug": "jan.username", "displayName": "Jan"}}' '}' >> "$mention"
sleep 1
echo "POST /webhook/$team (mention): $(PORT="$port" TEAM="$team" SECRET=s scripts/replay.sh "$mention" pr:comment:added)"
# Draft -> ready: the handoff to humans. The fake serves the review's inline
# comments with no threadResolved, so they count as open and the summary
# gets the handoff banner (log: "N of M noergler threads open"; the JSON log
# escapes the arrow before it as >, so grep for the words).
ready="$tmp/ready.json"
sed -e 's/"eventKey": "pr:opened"/"eventKey": "pr:modified"/' -e '$d' "$payload" > "$ready"
printf '%s\n' ',  "previousDraft": true' '}' >> "$ready"
sleep 1
echo "POST /webhook/$team (draft -> ready): $(PORT="$port" TEAM="$team" SECRET=s scripts/replay.sh "$ready" pr:modified)"
echo "POST /webhook/$team (bad signature): HTTP $(curl -s -o /dev/null -w '%{http_code}' -X POST \
  -H 'X-Hub-Signature: sha256=deadbeef' --data-binary "@$payload" "localhost:$port/webhook/$team")"

# Onboarding, end to end. This is the surface that writes webhooks into a
# customer's Bitbucket with a human's admin token, and until now nothing but
# unit tests covered it. The fake keeps the hooks it is given, so the status
# call below reports on what the onboard call actually wrote rather than on a
# canned answer.
#
# Two headers on every call: the team's own webhook secret authenticates the
# caller, and X-Bitbucket-Token carries the admin token the writes go out on.
#
# The token goes in bare: the Bitbucket client prepends "Bearer " itself, so
# sending one here would reach the fake as "Bearer Bearer …" and be refused.
#
# The body goes through a file rather than an argument. Passed inline, curl
# received one --data argument per top-level key with the outer braces gone
# ('"action":"onboard"' and '"projects":[...]' as separate arguments), so the
# server decoded a bare string and answered 422. A file body sidesteps the
# quoting entirely and is readable besides.
onboard() {
  printf '%s' "$2" > "$tmp/onboard-body.json"
  curl -s -X POST \
    -H "Authorization: Bearer s" \
    -H "X-Bitbucket-Token: $1" \
    -H 'Content-Type: application/json' \
    --data-binary "@$tmp/onboard-body.json" "localhost:$port/onboard/$team"
}
echo "--- onboarding"
echo "onboard: $(onboard admin-token '{"action":"onboard","projects":[{"key":"'"$project"'"}]}')"
# status takes no projects: onboard.go rejects that combination with a 400.
echo "status: $(onboard admin-token '{"action":"status"}')"
# grant-bot against the project the bot cannot read. Onboarding grants only
# when the bot's own read fails, so this is the one target that reaches
# GrantUserPermission; on the readable project it would correctly do nothing.
echo "grant-bot: $(onboard admin-token '{"action":"grant-bot","projects":[{"key":"'"${project}NOREAD"'"}]}')"
echo "remove (dry run): $(onboard admin-token '{"action":"remove","projects":[{"key":"'"$project"'"}],"dry_run":true}')"
# The negative case. A non-admin token must get nowhere: onboarding proves
# admin rights by listing webhooks, so the listing refuses it and the status
# call cannot come back healthy.
echo "status (non-admin token): $(onboard not-admin '{"action":"status"}')"
# And no token at all is a 401 from the route itself, before Bitbucket is
# touched.
echo "status (no token): HTTP $(curl -s -o /dev/null -w '%{http_code}' -X POST \
  -H 'Authorization: Bearer s' -H 'Content-Type: application/json' \
  --data '{"action":"status"}' "localhost:$port/onboard/$team")"

# The team's own bot and Jira token. A revoked token is refused and changes
# nothing; the team bot takes over the @mention, the shared one stops
# answering for this team, and DELETE hands it back.
creds() {
  printf '%s' "$2" > "$tmp/creds-body.json"
  curl -s -X "$1" -H "Authorization: Bearer s" -H 'Content-Type: application/json' \
    --data-binary "@$tmp/creds-body.json" "localhost:$port/teams/$team/credentials${3:-}"
}
echo "--- own accounts"
echo "push (revoked): $(creds PUT '{"bitbucket":{"username":"team-bot","token":"revoked-token"}}')"
echo "push (wrong username): $(creds PUT '{"bitbucket":{"username":"someone","token":"team-bot-token"}}')"
echo "push: $(creds PUT '{"bitbucket":{"username":"team-bot","token":"team-bot-token"},"jira":{"token":"jira-team-token"}}')"
sed -e 's/@bot why/@team-bot why/' -e 's/"id": 900/"id": 901/' "$mention" > "$tmp/mention-own.json"
echo "POST /webhook/$team (mention @team-bot): $(PORT="$port" TEAM="$team" SECRET=s scripts/replay.sh "$tmp/mention-own.json" pr:comment:added)"
sed -e 's/"id": 900/"id": 902/' "$mention" > "$tmp/mention-shared.json"
echo "POST /webhook/$team (mention @bot, want ignored): $(PORT="$port" TEAM="$team" SECRET=s scripts/replay.sh "$tmp/mention-shared.json" pr:comment:added)"
echo "GET /api/dashboard/teams: $(curl -s "localhost:$port/api/dashboard/teams" | grep -o '"bitbucket_account":{[^}]*}' | head -1)"
echo "delete bitbucket: $(creds DELETE '' /bitbucket)"
echo "delete bitbucket again (want 404): $(creds DELETE '' /bitbucket)"
echo "delete jira: $(creds DELETE '' /jira | grep -o '"jira_account":{[^}]*}')"

# The worker needs a moment to pick the job up before SIGTERM drains it.
sleep 2
kill -TERM "$pid"
wait "$pid" || true
echo "--- log"
grep -E "version|Database|Bitbucket:|Jira:|team_disabled|team_ready|teams_ready|http_request|stopped|listening|DISABLED|queue\[|priced by the gateway|HTTP totals|cost headers|Review of |context_window |context window |PR description |calls, tools|A reply on|noergler threads open|team credential|credentials push|comment mentions" "$tmp/serve.log"
echo "--- review posted"
grep -c "comment posted" "$tmp/fakes.log" || true
echo "--- onboarding writes seen by bitbucket (want a create, a grant and a refusal)"
grep "onboarding:" "$tmp/fakes.log" || true
echo "--- endpoints scripts/fakes does not serve (want none)"
grep "unhandled:" "$tmp/fakes.log" | sed 's/.*unhandled: //' | sort -u || true
