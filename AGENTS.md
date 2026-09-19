# AGENTS.md

Go port of trick77/noergler (Python). Bitbucket Server PR auto-review bridge,
inference through `github.com/trick77/llmwire` against a LiteLLM gateway.
The Python code is the spec where this repo is silent; its docs drifted, its
code did not.

## Commands

`gofmt -l .` (must print nothing), `go vet ./...`, `go build ./...`,
`go test -race ./...`. Store tests need `NOERGLER_TEST_DSN` (skipped without).
`./hack/smoke.sh` boots `serve` with a two-team config and hits the probes.
Go 1.26, `net/http` + `ServeMux` patterns, `pgx`, `yaml.v3`,
`tiktoken-go/tokenizer`. No web framework, no ORM, no logging library.

## Layout

`cmd/noergler` (`serve` default, `migrate`), `internal/config` (env + teams.yaml),
`internal/logging` (slog JSON handler), `internal/httpapi`, `internal/store`,
`internal/bitbucket`, `internal/jira`, `internal/riptide`, `internal/tokens`,
`internal/diff`, `internal/inference`, `internal/render`, `internal/review`,
`internal/queue`, `internal/teams`, `internal/onboarding`, `hack/`, `prompts/`.

## Invariants carried over from Python

- **Team identity = webhook path + that team's HMAC secret + ownership
  check.** Never trust `project.key` from the payload alone. Store the
  authenticated slug.
- **One team's fault disables that team only.** Never let a per-team error
  abort startup; never let a shared-layer error (DB, Bitbucket, Jira,
  unusable `teams.yaml`) disable just one team.
- Every line about a team carries `team=<slug>`: bind with
  `logging.WithTeam(ctx, slug)` at each boundary (webhook route, queue worker,
  per-team startup) and log with `*Context`.
- Secrets never in `teams.yaml`; `*_env` fields name env vars. Strict decode:
  an unknown key disables the team. Gateway host/models and the prompt
  templates are instance-only.
- **Single review worker**, per-PR supersede, FIFO jobs. `pr:deleted` and
  `pr:comment:deleted` run on the queue too (Python ran them concurrently;
  divergence). One prompt set resident at a time.
- Team self-service authenticates with the team's webhook secret (`Bearer`,
  constant-time compare). `/onboard` writes run on the caller's
  `X-Bitbucket-Token`, never logged or stored, proven per target by admin
  rights. Claims are written only after that proof; uniqueness is the DB's.
- Prompt placeholder order in `prompts/review.txt`: `{files}` BEFORE
  `{cumulative_pr_diff}` and `{previously_posted_findings}` (prefix cache).
  File order passed to the LLM is content-independent (group, language,
  path). Substitute with `strings.ReplaceAll`, never `text/template` (the
  files contain JSON braces).
- **Cost fails open.** Unpriced run = NULL cost, review proceeds; the per-PR
  cap skips only subsequent auto-runs. Cost is `BIGINT` nano-USD in the DB,
  USD only at the edges. Key spend is a gauge: shown, never summed.
- Riptide: best-effort, never fails a webhook; one rollup per PR, claimed
  before the POST, never retried; unknown cost = omit `total_cost_usd`;
  `reviewer_handle` + `reviewer_account_kind: "bot"` always.
- The disagree/feedback mechanic was removed deliberately. Do not reintroduce.

## llmwire rules

- Models are llmwire profile ids (`gpt-5.5`); the gateway alias is the
  operator's (`LLMWIRE_LITELLM_MODELS`). A profile not listed there disables
  the team (FromEnv would route it to api.openai.com).
- Per-team key via `Config.Lookup`: answer `LLMWIRE_LITELLM_API_KEY` with the
  team's `TEAM_<SLUG>_OPENAI_API_KEY`, delegate the rest to `os.LookupEnv`.
  `Config.APIKey` stays empty.
- `Chat` only, never streaming: a LiteLLM stream carries no cost header.
- Never retry. llmwire never retries; neither do we.
- `Usage.Cost.Provenance == Reported` is the only priced case.
- Context window: gateway `ListModels` `max_input_tokens` for the alias,
  `OPENAI_CONTEXT_WINDOW` overrides; `>= 1_000_000` required.

## Deliberate divergences from Python (pinned by tests)

Acceptance-criteria prefix needs a word boundary; Jira fetched once per
review; deleted/comment-deleted on the queue; dead code not ported
(`fetch_pr_comments`, `_estimate_review_effort`, `get_existing_finding_keys`,
`team_for`, `uncached_prompt`); no `/docs`; `SERVER_HOST`/`SERVER_PORT`
honoured; cross-file refs label diff lines as diff lines; riptide
`final_files_changed` counts reviewable files; declined PRs start fresh on
reopen.

## Logging

JSON, `timestamp` first, `msg`, `log_level` (debug/info/warning/error),
`service`, `env`. Splunk-reserved keys renamed `splunk_<key>`. Access line
`http_request` with `request_id`, `method`, `path`, `status_code`,
`duration_ms`; probes silent. Startup lines `team_disabled team=… reason=…`,
`team_ready`, `teams_ready enabled=[…] disabled=[…]` are alerted on.

## Memory

`GOMEMLIMIT=1500MiB` in the Containerfile against a 2 Gi pod. Bitbucket bodies
byte-capped at the socket (`BITBUCKET_MAX_DIFF_BYTES` 10Mi,
`BITBUCKET_MAX_FILE_BYTES` 1Mi), 4 file fetches in flight, tokenizer vocab
compiled in and warmed at boot.

## Git

Default branch `master`; branch + PR, never push to master. `.yaml` never
`.yml`, `Containerfile` never `Dockerfile`. A merge to master auto-tags and
pushes `ghcr.io/trick77/noergler-go`. `docs/plans/` is gitignored.
