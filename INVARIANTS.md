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
- Gateway host/models are instance-only; an unknown `teams.yaml` key
  disables the team.

## Money

- Per-PR cap stops everything: reviews, replies, Q&A, mention or not
  (decided). One notice: the summary's cost banner.
- Billed call with no run row: cost on the attempt. Reply and Q&A calls: an
  `llm_calls` row, no attempt. `prCostSQL` and `spendCTE` both sum runs +
  those attempts + `attempt_id IS NULL` calls: one figure everywhere. A
  review's call row is detail; never sum it.
- Read the PR total only when this run is priced.
- `null` = unpriced, never rendered `$0.000`. Only
  `Usage.Cost.Provenance == Reported` is priced. Reply and Q&A cost is recorded and counted (decided).
  A tool loop logs and records each call, logs one total (`SumCosts`):
  unpriced if any call is, the key spend its last reading. The total is
  never stored; the rows sum to it.

## Prompt and findings

- `pre_existing` (bugs in unchanged code): never posted, never moves the
  verdict, never scored. `ValidatePreExisting`: evidence in the shown file,
  none on a line the PR (all pushes, on an incremental review) added or
  removed, no in-diff rule; prose with a secret-like token dropped (code is
  fine); at most 5. Run page only.
- A secret literal the diff adds (code or config file, not moved) is a
  finding without a failing input; default-looking values (`admin123`,
  `changeme`) count, runtime lookups, empty or templated values and files
  only for tests, examples or local runs do not. Prose (summary fields
  included) names the field, never the value; `evidence` keeps it verbatim.
  `HasSecret` matches a literal given to a secret-named key, so the run page
  withholds that evidence.
- File order to the model depends on path only, never content.
- Cumulative diff: `IsReviewable` parts, push-touched files first, whole files
  or none; a partial block never says "entire PR".
- Diff lines carry new-file numbers; the model never counts lines.
- `ValidateFindings`, same call in `post` and evals: removed, full-file,
  other-file and cumulative-diff lines are evidence, never anchors, but one
  diff line of the finding's own file is required. A comment line may be
  quoted without its marker; every marker but `//` keeps its space. A quote
  wrapped in `"…"` is unwrapped as the last fallback, never with a marker put
  back on. One matcher for findings, fixed and pre-existing claims. Drops
  lower the verdict, never raise it.
- Fixed claims (`ValidateFixed`): incremental runs only; one quoted line added
  or removed, none moved; reply, then resolve only if the reply landed; the
  row is stamped regardless. Limits are in the doc comment and decided.
- Finding replies: PR author only, once, never resolve; only a reply directly
  under the finding's comment is found (Bitbucket sends the immediate
  parent; a reply nested under another reply is not, known limit); a non-author `@bot`
  reply goes to the Q&A, a review-keyword one to the mention path (a review);
  nothing posted = nothing stamped.
- Every reply the bot posts is stored (`bot_replies`, by comment id, with
  its PR). A plain reply directly under one is a Q&A from anyone, like a
  reply under the summary (user, 2026-10-22), matched from the DB, never by
  fetching the parent; the finding-thread answer included (decided). Never
  on an ignored PR: a plain reply never reactivates, only an `@mention`.
- Q&A on a reply: the immediate parent goes into the prompt as untrusted
  data, below the diff (prefix cache), capped; never the stored summary;
  a failed fetch answers without it. Another person's parent: no name, the
  asker is not the one answered (user, 2026-10-20), and `guardName` cuts a
  leading address to the parent's author; the asker's own parent keeps the
  name. A code block the parent leaves open is closed with its own fence.
- Author's first name: every question (Q&A, `question` replies) always;
  `acknowledges`/`disputes` may; `other` never (prompt asks, `guardName`
  cuts a leading one on `other`); reviews never; extracted by code
  from the operator pattern, single-case parts title-cased; never stored or
  logged.
- Language: a review is the PR author's, a Q&A the asker's; resolved user ->
  team -> `de`, failing open to the team. Set for the commenter alone, by an
  exact keyword or by `set_my_language` when their own text names the
  language as a full word (never "en"/"de" there); a switch never stamps a
  finding or reactivates an ignored PR. Prompts, enums,
  keys and logs stay English; `{output_language}` above "## Untrusted
  input"; `SwissSS` on model prose only, never code or `suggestion`. Evals
  pin `en`. Every banner marker exists per language.
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
- Incremental diff = PR `/diff?sinceId&untilId`, never `compare/diff`; only
  while the last reviewed commit is still one of the PR's commits (exact SHA)
  and no merge commit lies in the range, else full.
- An unreadable Jira ticket is not an error; dial timeout and bad JSON are.
- RE2: `\w \b \d \s` are ASCII. Use `[\pL\pN_]`, `(?:^|[^\pL\pN_])`,
  `\p{Nd}`, `[\s\p{Zs}]`, and check both directions. No lookaround: write
  boundary chars back.

## Webhook and HTTP

- Check order, never tidied: path team -> ping before the body -> raw body ->
  test-connection -> HMAC (hex strings, constant-time, uppercase fails) ->
  `pr:` prefix off the RAW json (never `Decode` first) -> validate -> repo
  from toRef then fromRef -> ownership -> exclude_repos (review-starting
  events and comments) -> exclude_source/target_branches (review-starting
  events only, never comments: a mention still answers; an excluded repo
  wins, user 2026-10-20) -> dispatch.
- The mention gate is the route's; `HandleMention` does not check. One
  exception, on the worker: a reply directly under the stored summary or
  under a stored bot reply is a mention, from anyone (user, 2026-10-19,
  2026-10-22); nested under either, not.
- `pr:modified` is never a review and never warns: only draft -> ready
  (`previousDraft` true, `draft` false) is queued, every other edit is
  ignored. Drafts are reviewed (user, 2026-10-22: a draft is the bot's
  turn, ready the humans'); never add a draft guard.
- 404/503 for a slug before 401. `/onboard` and the settings PUT decode the
  body AFTER auth.
- Response bodies are structs, never maps (`encoding/json` sorts map keys).
- One `teams.Runtime` snapshot per request. `UpdateSettings` mirrors both
  author lists onto the live Reviewer (it copies `config.Review` by value),
  not `exclude_repos` or the branch lists; `Reconcile` runs before any Reviewer is built; the
  onboarding orchestrators never mutate their team.
- Self-service handlers and `api/openapi.yaml` change together. Nullable =
  `type: [x, "null"]`, never `oneOf` with null.
- Dashboard: unauthenticated by design (the ingress is the boundary).
  `ui/src/routing.ts` and the `web/embed.go` allowlist are one list, plus one
  pattern (`/prs/{link token}` with an optional `/runs/{n}`, the same regex
  on both sides). The chart series ramp is CVD-validated: never re-picked by
  eye.
- PR page and its run pages: keyed only by `pull_requests.link_token` (16 B
  crypto/rand; rows from before 0015 a UUID's 122 bits; base64url, 22
  chars), linked only from the PR's summary comment; a run is
  `/prs/{token}/runs/{n}`, n its number in the PR's history (the row's life,
  as `prCostSQL`, any team), never a database id. The PR
  page is one shell: its rail links the PR and each run, and a run is a pane
  in it; nothing else on the dashboard links either. No token in any response or app log (`httpapi.logPath` masks
  it; the ingress log sees the path). Malformed, unknown, out-of-range n: one
  404. Old `/runs/{attempt token}` links 404 (user, 2026-10-20); the attempt
  column stays, unread. Posted and fixed findings: headline only, the text is
  on the PR (user, 2026-10-19); unposted findings and pre-existing bugs with
  their text, quoted code and file:line, on the run page only; never a secret
  (`inference.HasSecret`), discount or margin.
- PR page sums only what adds up (user, 2026-10-20): cost = `PRCost`, tokens
  and time over runs, waits one total (the step timeline stays per run, never
  summed). Lines and files = `Pipeline.PRScope` of the latest review, null
  ("—") when it did not read the whole diff; never an older run's (an
  earlier commit), never a sum of increments, and how it was derived never
  reaches the page. Files tab = every file any run listed.
- Dashboard text only through `ui/src/i18n` (i18next): `de` mirrors `en`
  (`satisfies Catalog`, same placeholders), Swiss German, no `ß`. Language =
  browser only, `de*` -> German, else English; no switch, no stored choice
  (decided). Review text (findings, replies) stays in the language it was
  produced in; the UI labels skip reasons by code, the server's label only as fallback.

## Review pipeline

- `Options.Owns` re-asked at the start of `prepare` and `post`.
- A keyword mention is a staged review; a Q&A mention stages its gateway
  call (a tool loop) and posts back on the worker. Neither writes a run.
- Mention tools: the PR, team and user come from the webhook, never a model
  argument; output is an explicit allowlist struct (author lists by name,
  decided; never env names, keys, tokens, hosts, riptide URL, disable
  reasons); claims and the exclude lists from the live snapshot, never the
  Reviewer's `cfg` copy; `Run` never calls Bitbucket.
  `set_my_language` accepts only when the commenter's own text names the
  language and asks (a switch word, or a comment of four words or fewer);
  the posting stage stores it, with the confirmation, only when an answer
  posts. At most `MaxToolRounds` rounds of `MaxToolCallsPerTurn` calls, then
  one round with `tool_choice: none`, tools still defined; every call id
  gets a result. Tool support
  is read off the profile at startup, never required: without it a Q&A is
  one plain call.
- Draft -> ready handoff: no model call, no run, no attempt. Open = unfixed
  in the DB and the Bitbucket thread unresolved; unreadable counts open,
  deleted closed; findings unreadable from the DB leave the summary as it
  is (never safeDB there: its "none" reads as "all closed"). Banner below cost and stale (their strippers look only at
  the top); nothing open writes nothing but the removal of an old banner.
- HTTP totals logged once at the end of `post`, never deferred in the entry.
  `post` runs on the review's ctx, never the job's.
- The plan across stages carries no file bodies and no raw diff.
- Team semaphore before the global one, released in reverse.
- `Queue.Stop` settles on "nothing runnable", not "no key held".
- `OutcomeError` posts no notice and writes no run row (its attempt and run
  detail are written).

## Store

- `review_runs` = successes only; failures and skips are `review_attempts`,
  written fail-open. A mention writes no attempt.
- `llm_calls` = what the AI Gateway reported per call, written after its
  attempt and apart from it (a refused call row never loses the attempt's
  billed cost). Its cost is detail, never summed beside runs/attempts. A lane
  the gateway did not send is NULL, never 0. No discount/margin column
  (decided).
- `ClaimRollup` stamps `riptide_emitted_at` in the statement that reads the
  snapshot. Never hang the decline-stamp clear on the reopen's review.

## Ops

- The PR diff is uncapped by default; the webhook and file bodies stay capped.
- Splunk-reserved keys renamed `splunk_<key>`, `timestamp` first.

## Diff engine

- The two header counting rules disagree on `\ No newline`: both required,
  never unify.
- Line splitting differs per call site on purpose (`\n` vs `splitlines()`).
