# AGENTS.md

Bitbucket Server PR auto-review bridge, inference through
`github.com/trick77/llmwire` against a LiteLLM gateway.

## Commands

**The module is in `backend/`; `hack/`, docs and `prompts/` are at the root.**
Go commands run from `backend/`, scripts from the root.
`gofmt -l .` must print nothing. `go vet ./...`, `go test -race ./...`.
Store tests skip without `NOERGLER_TEST_DSN`; `docker compose up -d postgres`,
then DSN `postgres://noergler:changeme@localhost:5432/noergler?sslmode=disable`.
`./hack/smoke.sh` boots `serve` against `hack/fakes`.
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
  never a float, and never an exponent (`1E-9` breaks strict parsers).

## Prompt

- `prompts/review.txt` order: `{files}` BEFORE `{cumulative_pr_diff}` and
  `{previously_posted_findings}`. Prefix cache depends on it.
- File order to the LLM is content-independent (group, language, path).
- `strings.ReplaceAll`, never `text/template`: the files contain JSON braces.
- **Single worker for every Bitbucket call**, per-PR supersede. Only the
  gateway call leaves it, onto a bounded pool (`REVIEW_INFERENCE_CONCURRENCY`
  6 global, `..._PER_TEAM` 2 nested inside it). Prepare and post keep the
  worker: serialization was for Bitbucket, never for inference.
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
  A no-auth gateway is unsupported.
- `Chat` only, never streaming: a LiteLLM stream carries no cost header
  (in-band chunk cost is opt-in gateway-side and unread here).
- Never retry.
- Context window comes from the gateway's `ListModels` `max_input_tokens` for the
  alias, `OPENAI_CONTEXT_WINDOW` overrides, `>= 1_000_000` required.
  `ListModels` warnings are kept: a present-but-unusable limit reads as a nil
  limit, so dropping them reports a garbage value as a missing field.

## Adapters

- **Never add `Client.Timeout` to Bitbucket.** Go's spans the body read and
  cuts a 10 MiB diff. The caller's ctx bounds the total instead.
- **Never follow redirects.** A 3xx would replay the bearer token at the new
  host. Non-2xx, not `>= 400`: a redirect must not read as success.
- `getTextCapped`: non-2xx beats the cap, else a big error page reports as an
  oversized diff. Then Content-Length, then the read. Strict `>`.
- Unreadable Jira ticket is not an error: the key came off a branch name, may be
  noise. Dial timeout and bad JSON do fail.
- `fields=` verbatim; `url.Values.Encode` escapes the commas.
- **RE2 has no lookaround.** Emphasis: one pass, boundary chars written back.
  `**bold**` stays `*bold*`; `ü*fett*` untouched (the word class is Unicode).
  These and the AC prefix pattern are pinned by a golden corpus.
- Jira `imageRE` eats any `!…!` span on a line: `Done! Ship it!` → `Done`.
  Known quirk, pinned by tests; changing it is a decision.

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
- 4 file fetches in flight, tokenizer vocab compiled in and warmed at boot.
  The pod has 2 Gi. Resident prompts are bounded too: the worker stops
  preparing at pool cap plus `stagedSlack`, so a 50-PR burst cannot hold 50
  assembled prompts.
- **The PR diff is UNCAPPED by default** (`BITBUCKET_MAX_DIFF_BYTES=0`); the
  inbound webhook body (1 MiB) and one file body (1 MiB) stay capped. A diff
  cap refuses the whole PR, and a diff's bytes are mostly files `IsReviewable`
  discards, so it measures what never becomes resident: a 19.4 MB PR with
  5 KB of reviewable source was skipped. `GOMEMLIMIT` bounds the process.
  Setting a diff cap is opt-in, for a pod too small to rely on that.
- o200k_base vocab costs 6.7 MiB resident. `GOMEMLIMIT=1500MiB` is generous,
  not tight.

## Diff engine

- Two header counting rules, both required, disagreeing on `\ No newline`: the
  per-hunk one counts "non-empty and not `+`/`-`", the merged one "starts with
  `-`/` `" and "`+`/` `". Do not unify.
- Known quirks, pinned by tests: `\ No newline` counts as a real line;
  `before_count` is unclamped, so a stale `content` yields header counts
  exceeding the body; a diff with no surviving hunks is filed under deleted,
  mislabelling a mode change.
- Every `\w` in these patterns is `[\pL\pN_]`, and `\b` is spelled out as
  `(?:^|[^\pL\pN_])`: RE2's `\w` and `\b` are ASCII, but the text is Unicode.
  `\bÖlservice\b` matches nothing in RE2. Third trap of this shape after the
  Jira regexes. Check every pattern for both.
- Line splitters differ per call site. `parse_hunks`/`expand_context`/
  `remove_deletion_only_hunks` split on `\n`; `split_by_file` and the symbol
  finders use `splitlines()` (also `\v`, `\f`, `\x1c`-`\x1e`, `\x85`, U+2028/9).
- Sort key reads only the path, never diff or content: prefix-cache order.

## Pinned behavior (tests enforce this)

Each of these is deliberate and pinned by a test. Changing one is a decision,
not a cleanup.

AC prefix needs a word boundary (`AC` does not eat `Actual…`, `Req` does not
eat `Request…`; `AK3` still matches); Jira fetched once per review;
deleted/comment-deleted on the queue, not concurrent; no `/docs`;
`SERVER_HOST`/`SERVER_PORT` honoured; cross-file refs label diff lines as diff
lines; riptide `final_files_changed` counts reviewable files; declined PRs
start fresh on reopen; `raw/{path}` URL-escaped (raw interpolation breaks on a
space or `#`); adjacent hunks merge without losing diff lines (trimming hunk
2's body by the whole overlap drops its removals); no phantom blank line in
expanded bodies (the `split("\n")` artifact must not survive mid-body); the
dynamic scope search skips lines past the end of truncated content instead of
indexing past it (indexing past it panics and takes the queue worker down);
`findings_posted` counts comments actually posted, not attempted; an author
skipped by `ignore_authors` is logged as `(ignored author)`, never a blanket
`(not in auto-review authors)` (the ignore check lives inside
`IsAutoReviewAuthor`, so the caller re-asks `isIgnoredAuthor`; the precedence
is unchanged, only the reported reason); the config dump renders
`context_window = 0` as `from gateway` and `inference/startup.go` logs the
resolved window per team (the dump runs before any team starts, so a bare 0
would be the only window ever printed); cost log lines carry 3 decimals, not 9
(`$0.000560` reads `$0.001`; the DB stays BIGINT nano-USD and the riptide edge
keeps its exact decimal string); the summary headings are sentence case with no
slash form: `Issues and suggestions`, `Security and performance`, `Test
coverage`, `Requirement compliance` (`summary_golden.json` carries the names;
the prompt is unaffected, it names JSON keys like `security_performance`); the
parser's six diagnostics are RETURNED as `ParsedReview.Diagnostics` and emitted
by `Client.Review`, so they bind `team=`/`pr_tag` from the call ctx
(`ParseReview` stays pure); a skipped item is logged as raw JSON, so
`{"requirement":"r"}`; placeholders substituted in ONE pass, so a `{files}`
inside `repo_instructions` or `ticket_context` stays literal (substituting
sequentially would protect only file content); an empty `comment.text` is
accepted and answers 200 `comment without mention`; a malformed webhook body is
400, never 500; inbound bodies are capped at 1 MiB (413 over it), because the
webhook body is read BEFORE the HMAC and a team slug is not a secret;
`/onboard` and `PUT /teams/{slug}/settings` decode the body AFTER
authenticating, so an unauthenticated caller cannot probe the schema (validate
first and a 422 would land ahead of the 401). An over-cap body is drained
and counted to 10x the cap AND a 10s timeout, so `ContentTooLarge.Size`
reports what was sent;
bytes alone are no bound, a stalled body never reaches the ceiling and would
hold the single review worker, and the review path has no ctx deadline; a
drain that stops early sets `Truncated` and stays `ContentTooLarge`, never a
generic error (that path is `OutcomeError`: no notice, no row); the too-large
review path (reachable ONLY when a diff cap is set - it is 0/unlimited by
default) names EVERY file, one log line each, `REVIEWED` or `FILTERED`,
reviewable first then size descending, tail as a lower bound. `Head` stops at
the cap, so the list comes from `FetchPRChanges` (`/changes`, paths only);
those carry `?` for size. `/changes` failing OR parsing to nothing is an error
that logs `full file list unavailable` and falls back to `Head` - an empty list
with no error would report head-only scope as if it had never been called.
`diff.PathIsReviewable` shares `IsReviewable`'s four skip lists; a test pins
that they agree.

NOT ours and not fixable here: `tiktoken-go` counts `" \n \n"` as two tokens
where the reference tokenizer merges the run into one (id 56319), so a diff
with consecutive blank context lines counts one token high per run (~0.3% on a
small PR). Errs safe (a smaller usable budget) and only shows in the summary
footnote's file-content figure.

## HTTP surface

The webhook check order is fixed and is NOT what a Go author would write, so
do not "tidy" it: path-bound team (never the payload's `project.key`) -> ping
before the body is read -> raw body -> test-connection shortcut -> HMAC ->
`pr:` prefix off the RAW json -> validate -> repo from toRef then fromRef ->
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

Three stages: `prepare` (guards, Bitbucket, Jira, prompt) -> `infer` (the
gateway call, the only stage off the worker) -> `post` (outcomes, comments,
rows, summary). `ReviewPullRequest` composes all three inline and is what
`HandleMention` and the tests use; `ReviewPullRequestStaged` is the queued
path.

- **A keyword mention is a review**, so it takes the staged path with
  `skipAuthorCheck` true (the person asking is the authorization). Running it
  inline would hold the worker for the whole gateway call and put it outside
  the pool, so in-flight inference could reach cap+1, and it could run
  alongside a staged review of the same PR. A Q&A mention is not a review and
  stays inline: one call, one reply.
- **Staged reviews report `handedOff`.** True keeps the PR's queue hold
  alive past the worker turn; a prepare exit returns false having logged its
  own HTTP totals. Get this wrong and either a second run of the same PR
  races the first on the pointer, the summary and the comments, or the PR is
  held forever.
- **The HTTP totals are NOT deferred in the entry function.** With the
  gateway call staged, a defer there fires before it and before posting:
  `inference=0` and no post-stage Bitbucket calls. Logged once, at the end
  of `post`.
- **`post` runs on the review's ctx, not the worker's job ctx.** The job ctx
  carries `team=` but no `pr_tag` and no httpstats scope. The queue passes
  the review's through structurally; do not "simplify" it to the parameter.
- The plan crossing the stages holds counts and names, never file bodies,
  the raw diff or anything but the one prompt string. `render.SummaryInput`
  has no content-bearing field, and `post`'s only use of the files is
  `len(files)`.
- `Queue.Stop` is a **fixpoint**: a review spawns inference which submits a
  post stage back onto the worker. It settles on "nothing runnable", not "no
  key held" (holds are released by inference goroutines, which `infWG` has
  already waited for, so a key still held is one nothing will release). While
  draining it starts no queued review, only the tails of reviews already in
  flight: shutdown is one round of work, not the whole backlog.
- Acquire the **team semaphore before the global one**, release in reverse.
  The other order holds a global slot while queuing for a team slot, which
  deadlocks the nested pair.

`OutcomeError` posts NO notice and writes NO row: a non-overflow API error is
only logged. Only
`timed_out`, `unparseable` and `too_large` post one. Each of those preserves
the PRIOR commit; only success and the three stable-state skips (opt-out
branch, AGENTS.md missing/oversized) advance the pointer.

PR cost total read only when THIS run is priced; an unconditional read shows a
total for an unpriced run and can trip the banner.

Queue worker recovers per job: without it one bad PR kills the only worker for
every team. `Submit` never blocks (unbounded,
like `put_nowait`) or a backlog becomes a webhook timeout. Depth excludes the
in-flight item, as `qsize()` does.

RE2's `\b`, `\d`, `\s` are ASCII while the text is Unicode, in both
directions. `(?:^|[^\pL\pN_])` for `\b`, `\p{Nd}` for `\d`,
`[\s\p{Zs}]` for `\s`. Bites the Jira key, the security keywords,
`_extract_question` and both `markdown_format` structural regexes. `textwrap`
likewise: ASCII-only whitespace (NBSP never breaks), tabs expand to 8-column
stops first. Pinned by a golden corpus.
