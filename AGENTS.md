# AGENTS.md

Bitbucket Server PR auto-review bridge, inference through
`github.com/trick77/llmwire` against a LiteLLM gateway. Ported from the
Python implementation now in `archive/`.
**That Python code is the spec for anything ported. Its docs drifted, its
code did not.**

## Commands

**The module is in `backend/`; `hack/`, docs and `prompts/` are at the root.**
Go commands run from `backend/`, scripts from the root.
`gofmt -l .` must print nothing. `go vet ./...`, `go test -race ./...`.
Store tests skip without `NOERGLER_TEST_DSN`; `docker compose up -d postgres`,
then DSN `postgres://noergler:changeme@localhost:5432/noergler?sslmode=disable`.
`./hack/smoke.sh` boots `serve` against `hack/fakes`; `./hack/parity.sh` diffs
this against the archived Python service in `archive/` (needs a venv there).
Coverage floor 75% (`hack/coverage-floors`),
gate `./hack/coverage-gate.sh backend` over `coverage/backend.xml`; `cmd/` is
excluded and `hack/` is outside the module.
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
- **Editing `prompts/review.txt` means re-running the evals**:
  `EVAL_BASE_URL=... EVAL_API_KEY=... go run ./cmd/evals` from `backend/`.
  Seeded-bug corpus in `internal/evals/corpus/`; exit 1 = a bug went
  unreported. The unit tests pin assembly only, never review quality.
- Evals default to **`mimo-v2.5-pro`, effort `high`** and stay there: a
  weaker model or less thinking scores worse on the same prompt, so a mixed
  history cannot be compared and a regression reads as a model change.
  `EVAL_MODEL` is an llmwire profile id, not the endpoint's own name; the
  gateway alias defaults to that id, which is what a plain
  OpenAI-compatible host serves (`EVAL_ALIAS` for one that renames).
- **Every eval run is committed** to `internal/evals/results/` with a row in
  its README, worse scores included: an uncommitted number cannot be
  compared with the next one, and a regression nobody recorded is invisible.
- The mimo endpoint lists no `max_input_tokens`, so a run needs
  `-context-window 1000000`; without it Startup fails and nothing is scored.
- Eval scoring must stay blunt (file + line window + keyword). A judge model
  would make a moved number two non-deterministic things to explain.
  The corpus keeps a case with NO expected findings: without it, "caught
  every seeded bug" is satisfied by reporting everything.

## llmwire

- Models are llmwire profile ids (`gpt-5.5`). The gateway alias is the
  operator's (`LLMWIRE_LITELLM_MODELS`); a profile missing there disables the
  team, else FromEnv routes it to api.openai.com.
- Per-team key via `Config.Lookup` answering with `TEAM_<SLUG>_OPENAI_API_KEY`;
  `Config.APIKey` stays empty. An empty team key falls through to the env; both
  empty and `FromEnv` returns `MissingEnvError`, `New` fails, team disabled.
  A no-auth gateway is unsupported: Python substituted a `no-auth` placeholder.
- `Chat` only, never streaming: a LiteLLM stream carries no cost header.
- Never retry.
- Context window comes from the gateway's `ListModels` `max_input_tokens` for the
  alias, `OPENAI_CONTEXT_WINDOW` overrides, `>= 1_000_000` required.
  `ListModels` warnings are kept: a present-but-unusable limit reads as a nil
  limit, so dropping them reports a garbage value as a missing field.
- No local `reasoning_effort` enum. Python's hardcoded
  {minimal, low, medium, high} was wrong both ways for the configured model;
  llmwire validates the level against the profile and the gateway's 400 covers
  the rest (`mapPingError`).

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
(`fetch_pr_comments`, `get_existing_finding_keys`, `team_for`,
`uncached_prompt` have no caller; `_estimate_review_effort` has three, but
only to set `ReviewResult.review_effort`, which nothing reads and which is
never serialized: computed, never observed); no `/docs`; `SERVER_HOST`/`SERVER_PORT`
honoured; cross-file refs label diff lines as diff lines; riptide
`final_files_changed` counts reviewable files; declined PRs start fresh on
reopen; `raw/{path}` URL-escaped (Python broke on a space or `#`); adjacent
hunks merge without losing diff lines (Python trimmed hunk 2's body by the
whole overlap and dropped its removals); no phantom blank line in expanded
bodies (Python's `split("\n")` artifact survived mid-body); the dynamic scope
search skips lines past the end of truncated content instead of indexing past
it (Python raised `IndexError`; a panic would take the queue worker down);
`findings_posted` counts comments actually posted (Python stored the attempted
count); an author skipped by `ignore_authors` is logged as `(ignored author)`,
not Python's blanket `(not in auto-review authors)` (the ignore check lives
inside `IsAutoReviewAuthor`, so the caller re-asks `isIgnoredAuthor`; the
precedence is unchanged, only the reported reason); the config dump renders
`context_window = 0` as `from gateway` and `inference/startup.go` logs the
resolved window per team (Python logged it from `llm_client.py`; the dump runs
before any team starts, so a bare 0 was the only window ever printed); cost log
lines carry 3 decimals, not 9 (`$0.000560` reads `$0.001`; the DB stays BIGINT
nano-USD and the riptide edge keeps its exact decimal string); the summary
headings drop Python's slash form and are sentence case: `Issues and
suggestions`, `Security and performance`, `Test coverage`, `Requirement
compliance` (`summary_golden.json` carries the rename and is byte-for-byte
otherwise; the prompt is unaffected, it names JSON keys like
`security_performance`); the parser's
six diagnostics are RETURNED as `ParsedReview.Diagnostics`
and emitted by `Client.Review`, so they bind `team=`/`pr_tag` from the call ctx
(`ParseReview` stays pure); a skipped item is logged as raw JSON where Python
logged a dict `repr`, so `{'requirement': 'r'}` reads `{"requirement":"r"}`;
placeholders substituted in ONE pass, so a `{files}` inside
`repo_instructions` or `ticket_context` stays literal (Python's sequential
order protects only file content); an empty `comment.text` is accepted and
answers 200 `comment without mention` (Pydantic's required str is satisfied by
`""`, probed against the venv); a malformed webhook body is 400, not Python's
accidental 500 from an unguarded `request.json()`; inbound bodies are capped
at 1 MiB (413 over it) where Python read them unbounded, because the webhook
body is read BEFORE the HMAC and a team slug is not a secret; `/onboard` and
`PUT /teams/{slug}/settings` decode the body AFTER authenticating, so an
unauthenticated caller cannot probe the schema (FastAPI validates first and
would 422 ahead of the 401; nothing pins that order).

NOT ours and not fixable here: `tiktoken-go` counts `" \n \n"` as two tokens
where tiktoken merges the run into one (id 56319), so a diff with consecutive
blank context lines counts one token high per run (~0.3% on a small PR). Errs
safe (a smaller usable budget) and only shows in the summary footnote's
file-content figure, which `hack/parity.sh` therefore compares by label.

## HTTP surface

Webhook check order is `main.py:389` step for step and is NOT what a Go author
would write: path-bound team (never the payload's `project.key`) -> ping before
the body is read -> raw body -> test-connection shortcut -> HMAC -> `pr:`
prefix off the RAW json -> validate -> repo from toRef then fromRef ->
**ownership** -> exclude_repos (review-starting events only) -> dispatch.
HMAC compares the hex STRINGS constant-time: uppercase hex must fail.
The `pr:` prefix precedes validation, so a non-PR event is ignored, not
refused; never call `Decode` before that check.
**The mention gate is the route's**: `HandleMention` does not check that a
comment names the bot, so without it every comment is an inference call.
Trigger is the instance `BITBUCKET_USERNAME`, case-insensitive substring, no
word boundary. Response bodies are structs, not maps: `encoding/json` sorts map
keys, and `pr:deleted`/`pr:comment:deleted` carry no `queue` key although Go
queues them.
404/503 for a slug come BEFORE the 401 on every route.
`teams.Runtime` is copy-on-write (`atomic.Pointer`): take ONE snapshot per
request, or a concurrent settings write lands between the ownership check and
the exclude check. `ApplySettings` must also mirror the two author lists onto
the live Reviewer, which copies `config.Review` by value; `exclude_repos` is
not mirrored. `teams.Reconcile` runs BEFORE any Reviewer is built, for the same
reason. The onboarding orchestrators never mutate the team they are given.

## Review pipeline

`OutcomeError` posts NO notice and writes NO row: Python re-raises a
non-overflow API error into the outer except, which only logs. Only
`timed_out`, `unparseable` and `too_large` post one. Each of those preserves
the PRIOR commit; only success and the three stable-state skips (opt-out
branch, AGENTS.md missing/oversized) advance the pointer.

PR cost total read only when THIS run is priced (Python nests it under
`run_cost_usd is not None`); an unconditional read shows a total Python never
showed and can trip the banner.

Queue worker recovers per job: Python cannot panic this way, and without it one
bad PR kills the only worker for every team. `Submit` never blocks (unbounded,
like `put_nowait`) or a backlog becomes a webhook timeout. Depth excludes the
in-flight item, as `qsize()` does.

RE2 vs Python regex, probed both directions against the venv: RE2's `\b`, `\d`,
`\s` are ASCII, Python's are Unicode. `(?:^|[^\pL\pN_])` for `\b`, `\p{Nd}` for
`\d`, `[\s\p{Zs}]` for `\s`. Bites the Jira key, the security keywords,
`_extract_question` and both `markdown_format` structural regexes. `textwrap`
likewise: ASCII-only whitespace (NBSP never breaks), tabs expand to 8-column
stops first. Pin with a golden corpus from the venv, never by reading Python.

The disagree/feedback mechanic was removed deliberately. Do not reintroduce.

## Git

Never push to master; branch + PR. `.yaml` never `.yml`, `Containerfile` never
`Dockerfile`. A merge to master auto-tags and pushes the image.
`docs/plans/` is gitignored.
