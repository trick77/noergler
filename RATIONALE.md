# RATIONALE.md

Why the rules in `AGENTS.md` and `INVARIANTS.md` exist. Most were bought with
an incident or an eval run: read the reason before relaxing a rule.

## Money and llmwire

- USD as a string: a float rounds the money, `1E-9` breaks strict parsers.
- PR total only when priced: an unconditional read shows a total for an
  unpriced run and can trip the cost banner.
- Replies and Q&A count toward the cap, which a mention no longer overrides:
  as log lines only, a PR could talk its way past its limit, and the money
  was missing from the dashboard. One banner, not a reply per refusal: a
  thread of "limit reached" answers is noise the summary already says.
- Never streaming: a LiteLLM stream carries no cost header.
- Never `ReasoningMinimal()`: on some models minimal is thinking off.
- The gateway's window wins: it can cap below the vendor. LiteLLM often leaves
  `max_input_tokens` unset; prod once needed a hand-set override. Dropping
  `ListModels` warnings reports a garbage limit as a missing one.

## Findings

- Screenshots summary only: an image has no line to anchor on and nothing
  `ValidateFindings` can quote; checking a visual claim would take a judge
  model. Pixel diff and crop in code: vision models miss small differences
  between two whole screenshots, and a crop costs a fraction of the tokens.
  Exact compare, no tolerance: anti-aliasing and font drift are the change a
  reviewer most needs told about. The model, not a threshold, drops a pair
  nobody can see: a pixel tolerance is a guess a font fallback can slip
  under, and the model already says when it sees no difference. Stored per pair: the summary is the PR's on
  every run, and re-sending unchanged images every push pays for nothing.
  Keyed by blob id, not bytes: a byte hash needed both versions read on every
  push, on the single worker, to learn the answer was already stored. Before
  = the PR's base: the target's tip may hold another PR's baseline, which the
  model then reports as this PR's change, reversed.
- Numbered diff lines: when the model counted, findings landed past EOF.
- Secrets carved out of the failing-input rule: `password = "admin123"` was
  flagged in about half the runs; the remit, the "no failing input, drop"
  rule and the failing-path confidence scale all argued for dropping it.
  Evidence keeps the value: a masked quote matches nothing and is dropped.
  A placeholder exclusion stays narrow, or it swallows `admin123`;
  `changeme` in a production config is a default credential too. "Config"
  means a runtime lookup: a literal in a config file is the commonest
  committed secret. `HasSecret` by key name: its shapes miss a short default
  password, which verbatim evidence would otherwise put on the run page.
- Evidence from outside the diff: a real finding quoted the call outside the
  diff; matching only the own file dropped 6 of 7 correct cross-file findings.
  One own diff line is still required, or the line check accepts any line.
- Comment marker keeps its space: otherwise the tail of `*cfg = x` or
  `--count` matches an invented quote.
- Unwrapping `"…"`: a correct 95-confidence catch quoted two doc comment
  lines as JSON strings and was dropped, scoring the model's hit as a miss.
- Fixed claims: a wrong "fixed" resolves a live bug's thread. Requiring every
  quoted line changed refused real fixes. Reply before resolve: Bitbucket
  refuses replies on a resolved thread. Stamping despite a failed reply: no
  later diff holds the fix to retry on, and an unstamped row would dedup a real
  recurrence.
- Incremental diff: `compare/diff` serves JSON only and answered our
  `text/plain` with 406 on every push (Bitbucket 9.4), read as "rebase", so
  prod never ran an incremental review and never retired a finding. Rebase
  check by commit list, merge check by `merges=only`: a diff across a rewrite
  or a merge of the target passes upstream lines off as the push's, evidence
  for a wrong "fixed".
- Finding replies author-only: a reviewer's reply used up the one answer
  before the author could make their case.
- Language block above the untrusted input: below it the prompt calls
  everything data, so an instruction there is one the model may ignore; and
  the prefix up to it is the same for every PR in that language.
- Language by keyword, not by the model: the model reads the diff, so an
  injection could flip a user's language; code reads only the commenter's text.
- Language per user, instance-wide: it is the person's, whichever team
  reviews them. A review is the author's, so a reviewer's switch cannot
  change the author's PR.
- One-pass placeholders: sequential substitution would expand a `{files}`
  inside user text.
- `.properties` gate outside `IsReviewable`: hand edits yes, regenerated i18n
  bundles no, but the rollup must count the file over small pushes.

## Evals

- Not a gate: a pin would redden every work-in-progress prompt edit.
- Clean controls fail the run: before, a run inventing findings on every
  control exited 0.
- Two reds: exit 1 fired on two of three runs with the prompt unchanged.
- One pair per series: a weaker model or less thinking scores worse on the
  same prompt, so a mixed history cannot be compared: a model change and a
  prompt regression look the same.
- Every run committed: an unrecorded regression is invisible. Blunt scoring: a
  judge model makes a moved number two non-deterministic things to explain.

## Adapters and HTTP

- No `Client.Timeout`: Go's spans the body read and cut a 10 MiB diff.
- No redirects: a 3xx would replay the bearer token at the new host.
- Status before the cap: otherwise a big error page reports as an oversized
  diff.
- RE2 ASCII classes: `\bÖlservice\b` matches nothing; this hit the Jira key,
  the security keywords, question extraction and the markdown regexes.
- Decode after auth: validating first lets a 422 land before the 401, so an
  unauthenticated caller can probe the schema.
- One Runtime snapshot: otherwise a settings write lands between the
  ownership and the exclude check.
- `oneOf` with null crashes Spectral's nimma and takes the whole lint down.
- Mention gate in the route: without it every comment is an inference call.

## Pipeline

- Single worker: serialization is for Bitbucket, never for inference.
- `Owns` re-asked: a repo given up mid-review re-created its purged PR row.
- Keyword mention staged: inline it held the worker and ran outside the pool,
  so inference could reach cap+1.
- Q&A staged: a tool loop is up to four gateway calls of 300s each, which
  inline would hold the Bitbucket worker for every team's PRs.
- Mention tools read-only but the language: a write tool is a setting any
  commenter, or any injection the model reads, could change. The one write
  is the commenter's own language, and code checks their words first.
- Author lists by name in `team_config`: they are Bitbucket user names any
  commenter on the team's PRs sees in Bitbucket anyway, and "who is
  auto-reviewed" is what a settings question asks (user, 2026-10-18). Counts
  stay on the unauthenticated dashboard.
- Tool support never required: `Require(Needs{Tools})` would disable a team
  on a model without it, for a feature a Q&A can do without.
- Tools withheld get a log line: prod ran gpt-6.1-sol (profile: no tools)
  and every settings question came back "team_config steht mir nicht zur
  Verfügung", with nothing in the logs to say why.
- Assistant turn replayed as `resp.Assistant()`: Azure refuses tools with
  reasoning on /chat/completions for gpt-6.1-sol ("use /v1/responses"), so
  llmwire speaks /responses there, and a function_call sent back without
  its reasoning item is refused on round 2.
- First name on every question (user, 2026-10-19), Q&A included; it was
  held back as talking down.
- `handedOff`: the job path once released the key unconditionally, so a push
  or merge ran during a staged mention's inference.
- HTTP totals not deferred: a defer fires before the staged call (`inference=0`).
- `post` on the review's ctx: the job ctx has no `pr_tag` and no httpstats
  scope.
- A mention writes no attempt: it is a Q&A answer, not a review, and the
  dashboard feed means "what was reviewed".
- No content in the plan: a review waiting for a slot holds one prompt, not a
  PR's files.
- Team semaphore first: the other order holds a global slot while queuing for
  a team slot and deadlocks.
- `Queue.Stop` fixpoint: holds are released by inference goroutines already
  waited for, so a key still held is one nothing will release.

## Ops

- Exec probes: kit answers `/healthz` and `/readyz` on loopback only; 0.2.42
  never became ready and restart-looped on an httpGet probe.
- Disable reasons stay in the logs: one can name an env var, and `/readyz` and
  the dashboard are unauthenticated. Own-account health is the carve-out
  (user, 2026-10-26: "issues like that should be visible in the ui"): a
  fixed status names no env var and echoes no upstream body.
- Pushed tokens encrypted: before them the DB held no credential; a dump or
  replica would otherwise hand out every team's Bitbucket token. Key check
  row: without it a wrong key reads as every team's row broken, the shared
  fault disabling teams one by one.
- A broken own account never disables its team: a disabled team gets 503 on
  every self-service route, so it could not push the fix; an admin would
  have to delete its row and restart (user chose stay-up, 2026-10-26). No
  silent fallback to the shared bot either: reviews under the wrong account
  are worse than none.
- Summary edit 403 reposts: Bitbucket lets only a comment's author edit it,
  so after a bot switch every old summary would otherwise stop updating.
- Push proves no repo access: the new bot gets its rights from `grant-bot`,
  which grants the team's current bot, so requiring access first would
  deadlock. The answer lists `unreachable_claims` instead.
- A 401 rechecks, never rejects: Bitbucket answers 401 for missing rights
  too, so the first ungranted project after a bot switch blocked the whole
  team (code review, 2026-10-26). A blocked team sends no call that could
  clear it, hence the recheck in the webhook route.
- Run page by token, from the summary only: numeric ids were enumerable on the
  unauthenticated dashboard, which forced headlines only and prose without
  code. A token handed out in the PR reaches the PR's audience, so the page
  can show the finding text, the code and the classes and lines involved.
- One token per PR, not per run: a per-run link showed only the latest run and
  every summary edit stranded the previous one's page. Every run of a PR has
  the same audience, so one key for all of them gives nothing away.
- Diff uncapped: a diff is mostly files `IsReviewable` discards; a 19.4 MB PR
  with 5 KB of reviewable source was skipped. `GOMEMLIMIT` bounds the process.
- `timestamp` first: Splunk's auto timestamp guesses wrong otherwise.
