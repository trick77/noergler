<p>
  <img src="logo.png" alt="noergler" width="360">
</p>

[![CI](https://github.com/trick77/noergler-go/actions/workflows/ci.yaml/badge.svg)](https://github.com/trick77/noergler-go/actions/workflows/ci.yaml) ![Go 1.26](https://img.shields.io/badge/go-1.26-blue)

Code review agent for typical private cloud corporate environments. The name is
German for "Nörgler" (grumbler/complainer).

Built for the realities of enterprise setups: self-hosted Bitbucket Server,
on-prem Jira, an OpenAI-compatible LLM endpoint (for example a LiteLLM proxy),
and corporate CA certificates. Receives PR webhooks, sends diffs to the LLM,
and posts findings back as inline comments plus a summary comment on the PR.

Go port of [trick77/noergler](https://github.com/trick77/noergler) (Python).
Its output is byte-for-byte identical to the original's for the same input; see
`hack/parity.sh`.

![noergler inline review comment](review.png)

## Features

- **Multi-team.** One instance serves many teams. Each team has its own webhook
  secret, its own gateway key, its own claimed repositories and its own review
  settings. One team's misconfiguration disables that team only.
- **Incremental reviews.** A push is reviewed against the last reviewed commit,
  not the whole PR again.
- **Context beyond the diff.** Hunks are widened with surrounding code, the
  enclosing function or class is pulled in, and symbols shared between changed
  files are named.
- **Jira acceptance criteria.** Finds the ticket from the branch name or PR
  title and checks the change against criteria that are verifiable from code.
- **Repo-specific rules.** An `AGENTS.md` in the repository is passed to the
  model as project guidelines.
- **Ask it things.** Mention the bot in a PR comment to get an answer in the
  same thread.
- **Cost control.** Per-PR cost cap, per-run cost recorded, optional FinOps
  forwarding to riptide. Unpriced runs never block a review.
- **Team self-service.** Teams claim projects and repositories, create their own
  webhooks and manage their author lists over HTTP, proving admin rights with
  their own Bitbucket token.

## How it works

A webhook arrives, the team is authenticated from the path and the signature,
the job is queued, one worker reviews it, and the findings are posted. The full
pipeline is in [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## Interacting with noergler

Mention the bot in a PR comment (`@<BITBUCKET_USERNAME>`) to ask about the PR or
to request a review that the author gate or the cost cap would otherwise skip.
The reply lands in the same comment thread.

To make noergler ignore a PR entirely, delete its summary comment. To skip a
branch, put the opt-out keyword (`noergloff` by default) in the branch name.

## Summary comment

One summary comment per PR, edited in place on later runs, with eight sections:
Overview, Strengths, Issues / Suggestions, Security / Performance, Test
Coverage, Ticket or Requirement Compliance, Recommendation, and a footnote
carrying the scope, token and cost figures.

## Quick start

```bash
cp .env.example .env          # fill in the required values
cp teams.example.yaml teams.yaml
docker compose up -d
```

Compose brings up PostgreSQL, runs the migrations as a separate service, and
then starts noergler on port 8080. The migration service is the same image with
a different command, which is how it runs in a deployment too.

Without compose:

```bash
docker build -t noergler -f backend/Containerfile .
docker run --rm --env-file .env noergler migrate
docker run -p 8080:8080 --env-file .env \
  -e TEAMS_CONFIG=/app/teams.yaml \
  -v ./teams.yaml:/app/teams.yaml:ro \
  -v ./prompts:/app/prompts:ro \
  noergler
```

From source:

```bash
go build -C backend -o "$PWD/noergler" ./cmd/noergler
./noergler migrate
./noergler serve
```

Build into the repo root and run from there: `REVIEW_PROMPT_TEMPLATE` and
`TEAMS_CONFIG` default to `prompts/review.txt` and `teams.yaml`, both relative
to the working directory, and both live at the root beside `backend/`. Running
from inside `backend/` disables every team with a missing prompt template.

`serve` is the default subcommand. `migrate` never runs from `serve`: it is the
init container, and nothing creates the schema at runtime.

## Configuration

Two layers, environment for the instance and `teams.yaml` for each team. The
full reference is [CONFIGURATION.md](CONFIGURATION.md); `.env.example` and
`teams.example.yaml` are commented working templates.

The setting most often wrong is the model mapping. `OPENAI_MODEL` and a team's
`inference.model` are llmwire **profile ids**; the gateway knows **aliases**;
`LLMWIRE_LITELLM_MODELS` maps between them:

```
LLMWIRE_LITELLM_BASE_URL=https://litellm.company.com/v1
LLMWIRE_LITELLM_MODELS=gpt-5.5=ai-gateway-gpt-5.5
OPENAI_MODEL=gpt-5.5
```

A profile missing from that map disables the team on purpose: llmwire would
otherwise send the request to `api.openai.com`. The context window is read per
team from the gateway and must be at least 1M tokens.

## Webhook setup

Each team points one webhook at `POST /webhook/<slug>`, secured with its own
secret. Teams can do this themselves:

```bash
# what the team has: claims and review author lists
curl -H "Authorization: Bearer $SECRET" $HOST/teams/$TEAM

# status of every claim: ownership, bot access, webhook state, stray hooks. No writes.
curl -X POST -H "Authorization: Bearer $SECRET" -H "X-Bitbucket-Token: $ADMIN_TOKEN" \
  -H 'Content-Type: application/json' -d '{"action": "status"}' \
  $HOST/onboard/$TEAM
```

`Authorization` is the team's webhook secret. `X-Bitbucket-Token` is the
caller's **own** Bitbucket token with project admin, used to prove the caller
may write to each target; it is never logged and never stored. Ready-made
requests for the IntelliJ HTTP client are in [`http/`](http/).

`NOERGLER_PUBLIC_URL` must be set for onboarding, since it is the URL written
into the Bitbucket webhook.

## Customization

The review prompt is `prompts/review.txt`, the Q&A prompt
`prompts/mention.txt`, both mounted read-only and swappable with
`REVIEW_PROMPT_TEMPLATE` and `REVIEW_MENTION_PROMPT_TEMPLATE`.

Placeholder order in the review template matters for the gateway's prefix
cache: `{files}` must come before `{cumulative_pr_diff}` and
`{previously_posted_findings}`.

Per-repository rules go in the repository's own `AGENTS.md`. Reviews are skipped
for repositories without one unless `REVIEW_REQUIRE_AGENTS_MD=false`.

## Running tests

The Go module lives in `backend/`; `hack/` and the docs stay at the repo root.

```bash
cd backend
gofmt -l .                 # must print nothing
go vet ./...
go test -race ./...
```

Store tests need a database and are skipped without one:

```bash
docker compose up -d postgres
cd backend
NOERGLER_TEST_DSN=postgres://noergler:changeme@localhost:5432/noergler?sslmode=disable \
  go test -race ./internal/store/...
```

End to end against fake Bitbucket, Jira, gateway and riptide, from the repo root:

```bash
./hack/smoke.sh      # boots serve, replays a signed webhook, reports what was posted
./hack/parity.sh     # the same replay through this and the Python implementation, diffed
```

Coverage floor and per-PR patch coverage, the same gate the other repos use:

```bash
cd backend
go test -race -covermode=atomic -coverpkg=./... -coverprofile=../coverage/backend.out ./...
go run github.com/boumenot/gocover-cobertura@v1.5.0 < ../coverage/backend.out > ../coverage/backend.xml
cd ..
./hack/coverage-gate.sh backend
```

## Health check

`GET /health` always answers 200 with the enabled and disabled teams.
`GET /ready` answers 200 once at least one team is enabled, 503 otherwise.
Neither is logged.

`teams_ready enabled=[...] disabled=[...]` is logged once at startup, alongside
`team_ready` and `team_disabled team=<slug> reason=...` per team. Those are the
lines to alert on.

## Deployment notes

The image is `ghcr.io/trick77/noergler-go`, built and pushed on merge to master.
The init container runs `["/noergler","migrate"]`.

`GOMEMLIMIT=1500MiB` is set in the image against a 2 Gi pod. RSS after startup
and tokenizer warm-up is around 40 MiB; `hack/smoke.sh` prints it.

`SSL_CERT_FILE` works for corporate CA bundles without any setting of ours:
Go's `crypto/x509` reads it when building the system pool.

Migrating from the Python service is covered in [CUTOVER.md](CUTOVER.md).

## Licence

See [LICENSE](LICENSE).
