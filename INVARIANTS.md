# INVARIANTS.md

Rules the code alone does not tell you are deliberate. Read the section for
the area you touch; `AGENTS.md` holds what applies everywhere, `RATIONALE.md`
why. Behaviour pinned by a test is not repeated here: the test is the rule.

## Teams

- Store the authenticated slug, never one read from the payload.
- Self-service auth = the team's webhook secret, constant-time. `/onboard`
  writes use the caller's `X-Bitbucket-Token` (never logged or stored), proven
  per target by admin rights; claims written only after; uniqueness is the
  DB's.
- Gateway host/models and prompt templates are instance-only; an unknown
  `teams.yaml` key disables the team.

## Money

- Per-PR cap skips only later auto-runs; a mention overrides it.
- Billed call with no run row: cost on the attempt. `prCostSQL` and `spendCTE`
  both sum runs + those attempts: one figure everywhere.
- Read the PR total only when this run is priced.
- `null` = unpriced, never rendered `$0.000`. Only
  `Usage.Cost.Provenance == Reported` is priced. Reply and Q&A cost is logged, not recorded (decided).

## Prompt and findings

- File order to the model depends on path only, never content.
- Cumulative diff: `IsReviewable` parts, push-touched files first, whole files
  or none; a partial block never says "entire PR".
- Diff lines carry new-file numbers; the model never counts lines.
- `ValidateFindings`, same call in `post` and evals: removed, full-file,
  other-file and cumulative-diff lines are evidence, never anchors, but one
  diff line of the finding's own file is required. A comment line may be
  quoted without its marker; every marker but `//` keeps its space. Drops
  lower the verdict, never raise it.
- Fixed claims (`ValidateFixed`): incremental runs only; one quoted line added
  or removed, none moved; reply, then resolve only if the reply landed; the
  row is stamped regardless. Limits are in the doc comment and decided.
- Finding replies: PR author only, once, never resolve; a non-author `@bot`
  reply goes to the Q&A, a review-keyword one to the mention path (a review);
  nothing posted = nothing stamped; template optional at its default path
  only.
- `.properties` size gate is NOT in `IsReviewable` (the rollup must count the
  file).

## Evals

- Exit 1 = a seeded bug missed OR any finding on a clean control; two
  consecutive reds are the signal, one is noise; the exit code stays strict.
- No model or level default in code (`corpus/series.yaml`); one pair per
  series.
- Scoring stays blunt (file + line window + keyword); the corpus keeps a case
  with no expected findings.

## llmwire

- Capabilities checked at startup against the profile, never probed. Tests
  assert intent, never a vendor's wire spelling, level name or rate.
- Per-team key via `Config.Lookup`; `Config.APIKey` stays empty. No-auth
  gateways unsupported.
- Context window: the profile never overrides the gateway's limit. Keep
  `ListModels` warnings.

## Adapters

- Success = 2xx, not `< 400` (a redirect must not read as success).
- `getTextCapped`: status before the cap, then Content-Length, then the read.
- An unreadable Jira ticket is not an error; dial timeout and bad JSON are.
- RE2: `\w \b \d \s` are ASCII. Use `[\pL\pN_]`, `(?:^|[^\pL\pN_])`,
  `\p{Nd}`, `[\s\p{Zs}]`, and check both directions. No lookaround: write
  boundary chars back.

## Webhook and HTTP

- Check order, never tidied: path team -> ping before the body -> raw body ->
  test-connection -> HMAC (hex strings, constant-time, uppercase fails) ->
  `pr:` prefix off the RAW json (never `Decode` first) -> validate -> repo
  from toRef then fromRef -> ownership -> exclude_repos (review-starting
  events only) -> dispatch.
- The mention gate is the route's; `HandleMention` does not check.
- 404/503 for a slug before 401. `/onboard` and the settings PUT decode the
  body AFTER auth.
- Response bodies are structs, never maps (`encoding/json` sorts map keys).
- One `teams.Runtime` snapshot per request. `UpdateSettings` mirrors both
  author lists onto the live Reviewer (it copies `config.Review` by value),
  not `exclude_repos`; `Reconcile` runs before any Reviewer is built; the
  onboarding orchestrators never mutate their team.
- Self-service handlers and `api/openapi.yaml` change together. Nullable =
  `type: [x, "null"]`, never `oneOf` with null.
- Dashboard: unauthenticated by design (the ingress is the boundary).
  `ui/src/routing.ts` and the `web/embed.go` allowlist are one list. The
  chart series ramp is CVD-validated: never re-picked by eye.

## Review pipeline

- `Options.Owns` re-asked at the start of `prepare` and `post`.
- A keyword mention is a staged review; a Q&A mention stays inline.
- HTTP totals logged once at the end of `post`, never deferred in the entry.
  `post` runs on the review's ctx, never the job's.
- The plan across stages carries no file bodies and no raw diff.
- Team semaphore before the global one, released in reverse.
- `Queue.Stop` settles on "nothing runnable", not "no key held".
- `OutcomeError` posts no notice and writes no row.

## Store

- `review_runs` = successes only; failures and skips are `review_attempts`,
  written fail-open. A mention writes no attempt.
- `ClaimRollup` stamps `riptide_emitted_at` in the statement that reads the
  snapshot. Never hang the decline-stamp clear on the reopen's review.

## Ops

- The PR diff is uncapped by default; the webhook and file bodies stay capped.
- Splunk-reserved keys renamed `splunk_<key>`, `timestamp` first.

## Diff engine

- The two header counting rules disagree on `\ No newline`: both required,
  never unify.
- Line splitting differs per call site on purpose (`\n` vs `splitlines()`).
