# AGENTS.md

Go port of trick77/noergler (Python). Bitbucket Server PR auto-review bridge,
inference through `github.com/trick77/llmwire` against a LiteLLM gateway.
**The Python code is the spec. Its docs drifted, its code did not.**

## Commands

`gofmt -l .` must print nothing. `go vet ./...`, `go test -race ./...`.
Store tests skip without `NOERGLER_TEST_DSN`; `docker compose up -d postgres`,
then DSN `postgres://noergler:changeme@localhost:5432/noergler?sslmode=disable`.
`./hack/smoke.sh` boots `serve` against `hack/fakes`.
No web framework, no ORM, no logging library. Do not add one.

## Team isolation

- **Team identity = webhook path + that team's HMAC secret + ownership check.**
  Never trust payload `project.key` alone. Store the authenticated slug.
- **One team's fault disables that team only.** A per-team error never aborts
  startup. A shared-layer error (DB, Bitbucket, Jira, unusable `teams.yaml`)
  never disables one team alone.
- Every team line carries `team=<slug>`: bind at each boundary (webhook route,
  queue worker, per-team startup).
- No secrets in `teams.yaml`; `*_env` names env vars. Unknown key disables the
  team. Gateway host/models and prompt templates are instance-only.
- Team self-service authenticates with the team's webhook secret, compared
  constant-time. `/onboard` writes use the caller's `X-Bitbucket-Token`, never
  logged or stored, proven per target by admin rights. Claims written only after
  that proof; uniqueness is the DB's.

## Money and telemetry

- **Cost fails open.** Unpriced run = NULL cost, review proceeds. The per-PR cap
  skips only later auto-runs. `BIGINT` nano-USD in the DB, USD at the edges.
- Key spend is a gauge: shown, never summed.
- `Usage.Cost.Provenance == Reported` is the only priced case.
- Riptide never fails a webhook. One rollup per PR, claimed before the POST,
  never retried. Unknown cost omits `total_cost_usd`; `reviewer_handle` and
  `reviewer_account_kind: "bot"` travel together. Cost is a decimal string,
  never a float, and never an exponent (`1E-9` breaks strict parsers).
- A DB fault never fails a review: the review path wraps store calls in
  warn-and-fallback.

## Prompt

- `prompts/review.txt` order: `{files}` BEFORE `{cumulative_pr_diff}` and
  `{previously_posted_findings}`. Prefix cache depends on it.
- File order to the LLM is content-independent (group, language, path).
- `strings.ReplaceAll`, never `text/template`: the files contain JSON braces.
- One prompt set resident at a time. **Single review worker**, per-PR supersede.

## llmwire

- Models are llmwire profile ids (`gpt-5.5`). The gateway alias is the
  operator's (`LLMWIRE_LITELLM_MODELS`); a profile missing there disables the
  team, else FromEnv routes it to api.openai.com.
- Per-team key via `Config.Lookup` answering with `TEAM_<SLUG>_OPENAI_API_KEY`;
  `Config.APIKey` stays empty.
- `Chat` only, never streaming: a LiteLLM stream carries no cost header.
- Never retry.
- Context window comes from the gateway's `ListModels` `max_input_tokens` for the
  alias, `OPENAI_CONTEXT_WINDOW` overrides, `>= 1_000_000` required.

## Adapters

- No interfaces here. Phase 6 defines them consumer-side.
- **Never add `Client.Timeout` to Bitbucket.** httpx's is per-operation, Go's
  spans the body read and cuts a 10 MiB diff. Caller's ctx bounds the total.
- **Never follow redirects.** A 3xx would replay the bearer token at the new
  host. Non-2xx, not `>= 400`: a redirect must not read as success.
- `getTextCapped`: non-2xx beats the cap, else a big error page reports as an
  oversized diff. Then Content-Length, then the read. Strict `>`.
- Unreadable Jira ticket is not an error: the key came off a branch name, may be
  noise. Dial timeout and bad JSON do fail.
- `fields=` verbatim; `url.Values.Encode` escapes the commas.
- **RE2 has no lookaround.** Emphasis: one pass, boundary chars written back.
  `**bold**` stays `*bold*`; `ü*fett*` untouched (Python `\w` is Unicode).
  Diff against Python before touching these or the AC prefix pattern.
- Jira `imageRE` eats any `!…!` span on a line: `Done! Ship it!` → `Done`.
  Python did the same, kept for parity. A fix needs a parity decision.

## Store

- Never edit an applied migration; add the next number.
- Schema is runs, not accumulators: totals aggregate over `review_runs`.
- `ClaimRollup` stamps `riptide_emitted_at` in the statement that reads the
  snapshot, so a crash cannot emit twice.
- `serve` never migrates. `migrate` is the init container.

## Ops

- `team_disabled`, `team_ready`, `teams_ready` are alerted on. Do not reword.
- Splunk-reserved keys renamed `splunk_<key>`, `timestamp` first (Splunk's auto
  timestamp guesses wrong otherwise).
- Bodies byte-capped at the socket, 4 file fetches in flight, tokenizer vocab
  compiled in and warmed at boot. The pod has 2 Gi.
- o200k_base vocab costs 6.7 MiB resident, not the 110 MiB the Python encoder
  needed. `GOMEMLIMIT=1500MiB` is generous, not tight.

## Diff engine

- Two header counting rules, both required, disagreeing on `\ No newline`: the
  per-hunk one counts "non-empty and not `+`/`-`", the merged one "starts with
  `-`/` `" and "`+`/` `". Do not unify.
- Bugs ported as is: `\ No newline` counts as a real line; `before_count` is
  unclamped, so a stale `content` yields header counts exceeding the body; a
  diff with no surviving hunks is filed under deleted, mislabelling a mode
  change.
- Every `\w` from a Python pattern is `[\pL\pN_]`, and `\b` is spelled out as
  `(?:^|[^\pL\pN_])`: RE2's `\w` and `\b` are ASCII, Python's are Unicode.
  `\bÖlservice\b` matches nothing in RE2. Third trap of this shape after the
  Jira regexes. Check every ported pattern for both.
- Line splitters differ per Python call site. `parse_hunks`/`expand_context`/
  `remove_deletion_only_hunks` split on `\n`; `split_by_file` and the symbol
  finders use `splitlines()` (also `\v`, `\f`, `\x1c`-`\x1e`, `\x85`, U+2028/9).
- Sort key reads only the path, never diff or content: prefix-cache order.

## Divergences from Python (pinned by tests)

AC prefix needs a word boundary (`AC` no longer eats `Actual…`, `Req` no longer
`Request…`; `AK3` still matches); Jira fetched once per review;
deleted/comment-deleted on the queue, not concurrent; dead code not ported
(`fetch_pr_comments`, `_estimate_review_effort`, `get_existing_finding_keys`,
`team_for`, `uncached_prompt`); no `/docs`; `SERVER_HOST`/`SERVER_PORT`
honoured; cross-file refs label diff lines as diff lines; riptide
`final_files_changed` counts reviewable files; declined PRs start fresh on
reopen; `raw/{path}` URL-escaped (Python broke on a space or `#`); adjacent
hunks merge without losing diff lines (Python trimmed hunk 2's body by the
whole overlap and dropped its removals); no phantom blank line in expanded
bodies (Python's `split("\n")` artifact survived mid-body); the dynamic scope
search skips lines past the end of truncated content instead of indexing past
it (Python raised `IndexError`; a panic would take the queue worker down).

The disagree/feedback mechanic was removed deliberately. Do not reintroduce.

## Git

Never push to master; branch + PR. `.yaml` never `.yml`, `Containerfile` never
`Dockerfile`. A merge to master auto-tags and pushes the image.
`docs/plans/` is gitignored.
