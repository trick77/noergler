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

- Numbered diff lines: when the model counted, findings landed past EOF.
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
  the dashboard are unauthenticated.
- Diff uncapped: a diff is mostly files `IsReviewable` discards; a 19.4 MB PR
  with 5 KB of reviewable source was skipped. `GOMEMLIMIT` bounds the process.
- `timestamp` first: Splunk's auto timestamp guesses wrong otherwise.
