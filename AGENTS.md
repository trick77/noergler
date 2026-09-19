# AGENTS.md

Go port of trick77/noergler (Python). Bitbucket Server PR auto-review bridge,
inference through `github.com/trick77/llmwire` against a LiteLLM gateway.
The Python code is the spec where this repo is silent; its docs drifted, its
code did not.

## Commands

`gofmt -l .` (must print nothing), `go vet ./...`, `go build ./...`,
`go test -race ./...`. Store tests need `NOERGLER_TEST_DSN` (skipped without):
`docker compose up -d postgres`, then
`NOERGLER_TEST_DSN=postgres://noergler:changeme@localhost:5432/noergler?sslmode=disable`.
Each test migrates its own schema and drops it.
`./hack/smoke.sh` boots `serve` against `hack/fakes` (Bitbucket, Jira and
riptide probes) and hits the endpoints. `serve` is the default subcommand,
`migrate` is the init container and never runs from `serve`.
Go 1.26, `net/http` + `ServeMux` patterns, `pgx`, `yaml.v3`,
`tiktoken-go/tokenizer`. No web framework, no ORM, no logging library.

## Invariants carried over from Python

- **Team identity = webhook path + that team's HMAC secret + ownership check.**
  Never trust payload `project.key` alone. Store the authenticated slug.
- **One team's fault disables that team only.** A per-team error never aborts
  startup; a shared-layer error (DB, Bitbucket, Jira, unusable `teams.yaml`)
  never disables one team alone.
- Every team line carries `team=<slug>`: `logging.WithTeam(ctx, slug)` at each
  boundary (webhook route, queue worker, per-team startup), log with `*Context`.
- No secrets in `teams.yaml`; `*_env` names env vars. Strict decode: unknown key
  disables the team. Gateway host/models and prompt templates are instance-only.
- **Single review worker**, per-PR supersede, FIFO jobs. `pr:deleted` and
  `pr:comment:deleted` on the queue too (Python ran them concurrently;
  divergence). One prompt set resident at a time.
- Team self-service authenticates with the team's webhook secret (`Bearer`,
  constant-time). `/onboard` writes use the caller's `X-Bitbucket-Token`, never
  logged or stored, proven per target by admin rights. Claims written only after
  that proof; uniqueness is the DB's.
- `prompts/review.txt` placeholder order: `{files}` BEFORE
  `{cumulative_pr_diff}` and `{previously_posted_findings}` (prefix cache). File
  order to the LLM is content-independent (group, language, path). Substitute
  with `strings.ReplaceAll`, never `text/template` (files contain JSON braces).
- **Cost fails open.** Unpriced run = NULL cost, review proceeds; the per-PR cap
  skips only later auto-runs. `BIGINT` nano-USD in the DB, USD at the edges.
  Key spend is a gauge: shown, never summed.
- Riptide: best-effort, never fails a webhook; one rollup per PR, claimed before
  the POST, never retried; unknown cost omits `total_cost_usd`;
  `reviewer_handle` + `reviewer_account_kind: "bot"` always.
- The disagree/feedback mechanic was removed deliberately. Do not reintroduce.

## Store

Embedded SQL in `internal/store/migrations/`, filename order, one transaction
per file, advisory lock `0x6E6F6572`, `schema_migrations` table. Never edit an
applied file; add the next number. Schema is runs, not accumulators: totals are
aggregates over `review_runs`; `PRCost` is NULL until a run is priced.
`ClaimRollup` stamps `riptide_emitted_at` in the same statement that reads
the snapshot. Every store call in the review path goes through a warn-and-
fallback wrapper: a DB fault never fails a review.

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

AC prefix needs a word boundary (`AC` no longer eats `Actual…`, `Req` no longer
`Request…`; `AK3` still matches); Jira fetched once per review;
deleted/comment-deleted on the queue; dead code not ported
(`fetch_pr_comments`, `_estimate_review_effort`, `get_existing_finding_keys`,
`team_for`, `uncached_prompt`); no `/docs`; `SERVER_HOST`/`SERVER_PORT`
honoured; cross-file refs label diff lines as diff lines; riptide
`final_files_changed` counts reviewable files; declined PRs start fresh on
reopen; `raw/{path}` URL-escaped (Python broke on a space or `#`).

## Adapters

- No interfaces here. Phase 6 defines them consumer-side.
- Never add `Client.Timeout` to Bitbucket: httpx's is per-operation, Go's spans
  the body read and cuts a 10 MiB diff. Caller's ctx bounds the total.
- Never follow redirects: a 3xx would replay the bearer token at the new host.
- `getTextCapped`: non-2xx beats the cap, else a big error page reports as an
  oversized diff. Then Content-Length, then the read. Strict `>`.
- Unreadable Jira ticket is not an error: the key came off a branch name, may be
  noise. Dial timeout and bad JSON do fail.
- `fields=` verbatim; `url.Values.Encode` escapes the commas.
- **RE2 has no lookaround.** Emphasis: one pass, boundary chars written back.
  `**bold**` stays `*bold*`; `ü*fett*` untouched (Python `\w` is Unicode).
  Diff against Python before touching these or the AC prefix pattern.
- Jira `imageRE` eats any `!…!` span on a line, so `Done! Ship it!` → `Done`.
  Python did the same; kept for parity. Fix needs a parity decision first.

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
