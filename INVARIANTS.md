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
- Own accounts (`PUT /teams/{team}/credentials`): Bitbucket and Jira only,
  the rest stays the admin's (user, 2026-10-26). Own Bitbucket token = own
  bot = own `@`; the shared `@` stops for that team. Every part proven before
  anything is stored; a refused part changes nothing. AES-GCM under
  `NOERGLER_SECRETS_KEY` (required), AAD `slug|kind`; never logged, never
  returned.
- Key missing, malformed or not the stored key check's = shared fault, boot
  aborts. A team's own row or token broken = that team only, never
  disabled: stays up (so it can push a fix), Bitbucket blocked =
  webhooks acknowledged not processed, Jira broken = no ticket check.
- Health leaves as a fixed status (`ok rejected unreachable undecryptable
  username_mismatch`), never a reason. 2xx = ok, transport error =
  unreachable; a 401 never sets rejected by itself (Bitbucket 401s a working
  token without rights on one repo): it rechecks the token, at most every
  30s, inline. A blocked rejected team is rechecked in the webhook route;
  that, the push proof and onboarding are the only Bitbucket calls off the
  worker. One log line per change.
- A push is all or none: both parts proven, one transaction, then swapped.

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
- The Almanach note call lives in `almanach_calls` only (decided): never
  `llm_calls`, never summed into a PR, a team or the rollup, never
  cap-checked.

## Prompt and findings

- Summary = the whole PR on every run, never one push (user, 2026-10-28:
  "it's called SUMMARY"): the comment is replaced each run. Prompt names it
  the PR review summary; issue list = every open finding (`EarlierOpen` +
  new); `SettleVerdict` sets the decision from that list, once, both ways;
  a match keeps the model's rationale unless a dropped finding could have carried it. List shows 30, the rest counted.
- Thread resolved or deleted = settled (user, 2026-10-28). `closeThreads`
  stamps `closed_at` on a resolved one only, never read by a review again
  (a review does not notice a reopen, accepted; the handoff reads stamped
  threads and unstamps a reopened one). A 404 is closed for that run only,
  never stamped (a passing fault would settle it for good). Settled: out of
  the list and verdict, never posted again (dedupe still holds it), in the
  prompt marked closed with no id, trimmed from it before any open one. No
  comment id (legacy): out of the list and verdict (no thread anyone could
  resolve), keeps its id in the prompt so a fix retires it.
  Unreadable = open. Read in prepare, once per run, at
  most 50 unsettled threads, issues first, the rest open unread (user).
  Unretireable after a full-review fallback (no ids): open until its
  thread is resolved (user kept it). The verdict counts new findings over
  the cap; the list names how many. Earlier findings unreadable from the
  DB: never safeDB ("none" would approve), the summary says it lists this
  run's only, the verdict is lowered only for drops.
  Heading "PR Review-Zusammenfassung" / "PR review summary", last push a
  line under it. Two schema slots, `overview` (the PR) and `latest_push`
  (that push, incremental only, rendered on the last-push line): scope is
  the schema's, never one prose field the model must keep PR-wide.
- The overview carries over (user, 2026-10-28): every incremental review
  gets the last completed run's `overview` (`pull_requests.last_summary`,
  written by `InsertRun`, never the posted comment), below the posted
  findings (changes per push, so last for the prefix cache), capped at 2000
  chars, as the model's own draft, never evidence: the diff wins; the PR as
  it stands, never its history (mimo narrated a revert without). Without it
  whatever of the PR does not fit the prompt drops out of the overview. An
  empty or sentinel overview never erases it; unreadable = review without;
  read on active rows only (`activeFilter`); `Reactivate` clears it. Its
  tokens are "rest" on the run page (small, no UI row). A full review never
  gets it, writes it fresh from the whole PR and stores it (mimo kept a
  reverted part on a full review with it). Only the overview today; the JSONB takes the other fields later.
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
- No em dash in anything posted (user, 2026-10-28): the Reviewer's client
  is wrapped (`outgoing`), so every comment write goes through
  `render.NoEmDash`, which leaves fences and inline code alone. Post only
  through `r.bitbucket`; a banner marker never holds an em dash, and a
  rendered body compared with a fetched one goes through `NoEmDash` first.
- Model prose is hard-wrapped (`WrapProse`, Bitbucket shows every newline):
  summary, inline findings and replies alike (user, 2026-10-28; replies in
  `outgoing`), lists at the prose width, never earlier. Fences (``` and
  ~~~, read the same by `NoEmDash`), indented code, headings, quotes and
  table rows pass through unbroken.
- `.properties` size gate is NOT in `IsReviewable` (the rollup must count the
  file).
- Test screenshots (user, 2026-10-28): summary only, never a finding (evals
  score the line, by keyword, never as one). Images ride the review call, in
  a second user message after the prompt (never inside it: prefix cache),
  only on a vision profile. List = the whole PR's, from its changes; before
  = the PR's base, never the target's tip; a line is stored per pair by blob
  id and reused, never re-sent or re-read; pixel diff and crop in code, never
  by the model. A screenshot left out is logged and its line says why; a
  removed one gets no line (user, 2026-10-28: noise), nor one the model
  marks `visible: false` (user, 2026-10-30: «kein klarer visueller
  Unterschied» lines are noise), stored as '' like same pixels; only a
  literal false hides, never on an added one. Visible drift no code
  explains keeps its line. At most 10 pairs a call (a changed pair takes 2
  of the 20 reads). Identical lines fold at render only (user, 2026-10-30):
  exact text after whitespace, never the model, never a status line; the
  store keeps one line per pair. Not a drop, so no log line: every screenshot
  is still named or counted on its line.

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
  wins, user 2026-10-20) -> own Bitbucket account blocked (200 ignored; new
  work only: review events, comments, `pr:modified`; lifecycle passes) ->
  dispatch.
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
- Pill words start with a capital in both languages, `OK` not `Ok`, the
  unknown-code fallback too (user, 2026-10-26).

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
- The plan across stages carries no file bodies and no raw diff. One
  exception, part of the call like the prompt: the cropped screenshot images
  (at most `MaxVisualImages` pairs, downscaled).
- Team semaphore before the global one, released in reverse.
- `Queue.Stop` settles on "nothing runnable", not "no key held".
- Pool work after a post (a call that posts nothing to Bitbucket) is
  `Spawn`, never a second `Stage`: Stage's tail ends the PR hold, so a Stage
  from inside post would drop the hold mid-call and later release a key a
  newer review holds. `Spawn` never touches `inflight`, so the PR's purge,
  decline, merge and next review run beside it. A spawned fn therefore:
  re-asks `Owns` and the PR's state right before every write, as `post`
  does; opens its own `httpstats` scope and logs it (the review's totals
  line is already out); and captures only the few fields it needs, built
  before Spawn, never the plan, the result or a diff. The one exception is
  the Almanach note's replayed conversation (the review prompt string and
  the model's answer): that is what the cache serves, and one prompt per
  spawned call is the documented cost of the piggyback.
- `OutcomeError` posts no notice and writes no run row (its attempt and run
  detail are written).

## Almanach

- The feed is per team, opt-in by the `almanach:` block; no block, no call,
  no row. The token is the team's identity at Almanach; the payload never
  names a team.
- A note never enters a prompt that reviews code. It is German always
  (`lang.SwissSS` in the validator), repo-scoped, and checked bluntly: shape,
  secret scan on prose only (names and paths are not prose), caps on the
  shape (name length, files, items per list) and on the whole note, never
  on prose: the note is documentation, its length follows the content, the
  total is the ceiling (user, 2026-10-28: "it's a live wiki", no sentence
  or paragraph quota). Over a cap the whole note drops, never truncated.
- The note call is a continuation of the review conversation (system,
  prompt, the answer verbatim, the note turn), one call, strict schema, no
  retry, after the review posted. Whether the prefix cache serves it is
  measured on `almanach_calls.cache_read_tokens`, not assumed (user,
  2026-10-28: welcome, not a dealbreaker).
- Every call leaves an `almanach_calls` row with a `note_status`, whatever
  became of the note. That ledger counts toward nothing else (Money).
- Merge stamps and sends; no model call. Emitted is marked after a 2xx, per
  run id. A merge during an in-flight round stamps and leaves the emit to the
  round, which sends its own note only: the drain of up to ten is a worker
  job, never a pool job. In flight is a count per PR, not a flag. Declined
  and deleted drop a note that is not final. A spawned round's own writes go
  through the store's PR-state guard: a declined, deleted or purged PR gets
  no note row back (`no_row`); a store fault is `store_failed`, never
  `no_row`.

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
- `almanach_notes`: one row per PR, replaced by run id (the round ran
  unheld), stamps set once by finalize and never touched by a round, a stub
  when finalize beats the first round (only for a PR with a row), emitted
  marked after a 2xx and never before, for the run id that was sent
  (Almanach dedupes; a lost note is the real cost). Delete never drops a
  final note. No FK to `pull_requests`: a final note survives
  `PurgeProject` until emitted; the purge drops the rest. The team slug
  follows the latest round. The review cost a note reports is `prCostSQL`.

## Ops

- The PR diff is uncapped by default; the webhook and file bodies stay capped.
- Splunk-reserved keys renamed `splunk_<key>`, `timestamp` first.

## Diff engine

- The two header counting rules disagree on `\ No newline`: both required,
  never unify.
- Line splitting differs per call site on purpose (`\n` vs `splitlines()`).
