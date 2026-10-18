# How noergler works

What happens between a Bitbucket webhook and a posted review.

## Pipeline overview

```
webhook  ->  route and authenticate  ->  queue  ->  review pipeline  ->  post
```

1. **Route and authenticate.** The path names the team, the team's HMAC secret
   verifies the body, the ownership check decides whether this team may review
   this repository.
2. **Queue.** A single worker, FIFO. It runs every Bitbucket call one at a
   time; only the model call leaves it, onto a bounded pool, so a slow review
   does not hold the next PR's turn. A new push for a PR already queued
   supersedes the older job, and a push for a PR still being reviewed waits
   for it.
3. **Review pipeline.** Guards, diff, file content, context expansion or
   compression, cross-file context, Jira, the model call, dedup, posting.
4. **Post.** Inline comments on the changed lines, one summary comment per PR,
   updated in place on later runs.

## 1. Webhook reception and routing

`POST /webhook/<slug>`. Team identity is three things together: **the path, that
team's HMAC secret, and the ownership check.** The payload's own `project.key` is
never trusted on its own.

In order: the slug must be known, the signature must verify (constant time), the
payload must be a `pr:` event, and the team must claim the repository. A signed
delivery for a repository the team does not claim is ignored, not reviewed.

Ownership comes from the `team_claims` table, not from `teams.yaml`. The yaml's
`projects:` only seeds it on first start.

A comment containing `@<BITBUCKET_USERNAME>` routes to the Q&A path instead of a
review. The trigger is the instance's bot name, not a per-team setting.

Events handled: `pr:opened`, `pr:modified`, `pr:from_ref_updated`, `pr:merged`,
`pr:declined`, `pr:deleted`, `pr:comment:added`, `pr:comment:deleted`.

## 2. Guard order

The pipeline refuses early and cheaply. In order:

1. Project and repository resolvable from the payload.
2. Author is in the auto-review list (an `@mention` skips this and the next two).
3. The actor who pushed is not in the ignore list.
4. Skip state: if a human deleted noergler's summary comment, the PR is left
   alone from then on.
5. The source branch does not contain the opt-out keyword.
6. `AGENTS.md` exists, when it is required.
7. `AGENTS.md` is under the hard token limit.
8. The PR's accumulated cost is under the cap. An `@mention` does not override
   it.

A skip that did not review the new commit writes the **prior** commit back as
the reviewed pointer, so raising a limit later re-reviews the accumulated range
instead of skipping past it.

Every skip that is a stable state of the PR (opt-out branch, missing or oversized
`AGENTS.md`, cost cap) posts a summary saying why. Guards 1 to 4 post nothing.

## 3. Incremental review detection

The last reviewed commit lives in the **database**, not in a marker inside the
summary comment.

On a push, noergler asks Bitbucket for the diff between the last reviewed commit
and the new head. Same SHA means nothing changed and the run stops. If Bitbucket
cannot produce an incremental diff (it answers 406 for a rewritten range), the
run falls back to the full PR diff.

The review then sees the incremental diff as the subject and the cumulative PR
diff as context.

## 4. Diff fetch and file splitting

The PR diff is fetched whole and split per file. Non-reviewable files are
dropped: binaries, lockfiles, vendored and generated paths, hidden path
components.

There is no byte cap by default. `BITBUCKET_MAX_DIFF_BYTES` can set one, but it
skips the entire PR when exceeded, and a diff's size is dominated by the files
this step is about to drop, so it refuses reviews over bytes that never become
resident. `GOMEMLIMIT` bounds the process instead.

## 5. Content enrichment

For each reviewable file, the new-version body is fetched (capped at
`BITBUCKET_MAX_FILE_BYTES`, 1 MiB), four fetches in flight. A file longer than
`REVIEW_MAX_FILE_LINES` is reviewed from its diff alone. A file whose body
cannot be fetched is reviewed from its diff alone.

Each file reaches the model as full content plus the minimal diff:

````
## File: util.py
### Full file content (new version):
```py
...
```
### Changes (diff: lines with `-` are REMOVED, lines with `+` are ADDED):
```diff
...
```
````

**File order is content-independent**: by group, then language, then path. Not
"larger files first".

## 6. Context expansion

Small PRs get their hunks widened with surrounding file content:
`REVIEW_DIFF_EXTRA_LINES_BEFORE` (3) before and
`REVIEW_DIFF_EXTRA_LINES_AFTER` (2) after. The asymmetry is deliberate: what a
changed line depends on is usually above it.

With `REVIEW_DIFF_ALLOW_DYNAMIC_CONTEXT`, the window also reaches up to
`REVIEW_DIFF_MAX_EXTRA_LINES_DYNAMIC_CONTEXT` lines further to pull in the
enclosing function or class header.

Hunks whose widened windows touch are merged into one. The overlap is taken off
the **first** hunk's added context, never off the second hunk's body: trimming
the body would drop real removal lines and leave the first hunk's context
claiming a line is unchanged that the second hunk deletes.

## 7. Cross-file context

Symbols defined in one changed file and used in another are collected and sent
as a short relationship section, capped at 30 lines. A reference that occurs on
a diff line is labelled as such, so the model can tell a call site it can see
from one it cannot.

## 8. Large PR compression

When the files plus room to expand them do not fit the budget, the PR is
compressed instead of expanded: files are taken in priority order until the
budget is spent, and **whole files** are dropped, never parts of them. The
dropped paths are still named to the model, under "other modified files",
"renamed files" and "deleted files", so it knows the change is wider than what
it can see.

There is **no 413 bisection.** A PR that cannot fit gets a summary saying so.

## 9. What is sent to the model

One call per review, never streaming: a LiteLLM stream carries no cost header.

The prompt is assembled in a fixed order so the gateway's prefix cache can work:
the template, then `{files}`, then `{cumulative_pr_diff}`, then
`{previously_posted_findings}`. Substitution is literal string replacement, not
templating: file content contains braces.

Budgets come off the resolved context window through a trust curve (large
advertised windows are the least trustworthy), minus an output reserve. The
cumulative diff and the previously-posted findings each get a small slice of
what is left.

A response is required to match a JSON schema with exactly these keys:
`overview`, `strengths`, `security_performance`, `test_coverage`, `verdict`,
`findings`, `fixed`, `compliance_requirements`.

## 10. Dedup, ordering and posting

Findings already posted by earlier runs are filtered out by reading the
**database**, not by fetching and parsing Bitbucket comments. What is left is
sorted by severity and capped at `REVIEW_MAX_COMMENTS`.

On an incremental review each earlier finding is listed to the model with an
id (`F12`). When the model names one under `fixed`, quoting a line of that
finding's file the push added or removed and no line it moved (across the
files the review showed), the thread gets a "Fixed in `<sha>`." reply and is
then resolved, and the finding leaves dedup. A claim quoting only unchanged
lines, or any moved one, is dropped and logged. A full review lists no ids:
its diff is the whole PR, so every line reads as added.

The summary lists only this run's findings. Under them, when there is any,
one line counts what this run checked: on an incremental review, the earlier
findings it listed that this push fixed and those still open, and how many
findings validation dropped. It is not a full ledger: repeats, earlier
findings trimmed from the prompt, the comment cap and inline posts Bitbucket
refused are in the log, not in the line.

Each finding becomes one inline comment on its line:

```
**Issue:** Callers of the old name will fail at import time.

**Suggested change:**
```suggestion
Keep an alias for one release.
```
```

The severity label is `Issue:` or `Suggestion:`. There is no emoji badge.
A `suggestion` block is only emitted when the model supplied a concrete
replacement; single-line replacements let Bitbucket render an "Apply suggestion"
button.

Then one summary comment per PR, edited in place on later runs. Its sections, in
order:

1. Overview
2. Strengths
3. Issues and suggestions
4. Security and performance
5. Test coverage
6. Ticket or Requirement compliance
7. Recommendation
8. A footer: a bullet for each thing the author can act on (no AGENTS.md or
   one over its warn size, no ticket while Jira is on, files not reviewed or
   reviewed without full context), then one line of model, files, lines,
   elapsed time and cost, ending in a link to the run's dashboard page when
   `NOERGLER_PUBLIC_URL` is set

There is no effort score and no "What changed" section. Tokens, prompt parts,
cross-file symbols, the gateway's cost split and the key spend are on the run
page, not in the PR.

The footer's model shows the llmwire profile id with the reasoning level
that was sent appended (`<profile>-<level>`), not the operator's gateway alias.
With no level configured, that is what the model's balanced level resolved to.

## 11. Cost

Only a cost the gateway **reports** counts. An unpriced run is stored with a
NULL cost and the review proceeds: cost fails open. Reviews, finding replies
and Q&A answers all count toward the PR's cost, and at the per-PR cap all of
them stop, mentions included. The PR gets one notice, the cost banner on its
summary; the comment that was not answered gets no reply of its own. Raising
the limit is the only way on.

Costs are stored as nano-USD integers and rendered as USD at the edges. The
gateway key's total spend is shown as a gauge where available, never summed.

## 12. Q&A by mention

A comment mentioning the bot is answered rather than reviewed. It runs without
the review gates and without compression, reads the PR diff and the ticket, and
replies in the comment's own thread. Keywords route the question; anything else
is answered as a question about the PR, or about noergler on it. On a model
that takes tools the answer may call `team_config` (the team's effective
settings: claimed projects, excluded repos, both author lists by name, the
built-in skipped dirs, files and extensions, and whether the asker is on the
lists; a settings answer is a Markdown table),
`pr_status` (runs, the latest outcomes and skip reasons, cost against the cap)
and `set_my_language` (the asker's language, only when their own comment names
it). Up to three tool rounds, four calls each, then a final round without
tools. The gateway calls are staged off the worker like a review; the reply is
posted back on it. Every call is recorded and counts toward the PR's cost; at
the cap a question is not answered.

The PR author's reply on one of noergler's finding threads needs no mention.
The route queues any non-empty reply not written by the bot (Bitbucket's
`commentParentId`); the worker answers only the author, only when the parent is
a stored, unfixed finding of that PR without an answer yet, and, for a plain
reply, only under the PR's cost cap. An `@bot` reply on such a thread is
answered as a reply too, unless it asks for a review or comes from someone
else, which the Q&A answers. The call (`finding_reply.txt`) is staged
like a review, and sees the finding, the file around it, AGENTS.md and the
reply. A dispute, explicit or not ("we do this in the UI"), is acknowledged;
when its reason is a project-wide rule AGENTS.md lacks, the answer suggests a
one-line AGENTS.md bullet. The dispute is stored and later prompts show it
beside the finding so the issue is not raised again. The thread stays open:
the author resolves it. A reply's call is recorded and counts toward the
PR's cost; at the cap, replies stop.

The answer may address the author by first name ("Ok Jan - ..."), not
always. `BITBUCKET_DISPLAY_NAME_PATTERN` extracts it from the Bitbucket
`displayName` (default `Last First (CODE)`: the last word before the bracket); no match, no name. The model
is told to use it only when acknowledging or accepting a dispute; on a question
or any other intent code cuts a leading address from the answer. The name is
never stored or logged. Q&A answers and reviews carry no name.

## 12a. Language

noergler writes German by default (Swiss spelling, generic masculine),
English on request. `REVIEW_LANGUAGE` / teams.yaml `language` set the team
default; a user's own choice wins. `@noergler english` (or `deutsch`, `de`,
`en`, `auf Englisch`, ...), or the same word alone as a reply under a
finding, sets the commenter's language: an exact keyword, matched by code,
no model call, confirmed in the new language. It is stored per Bitbucket user
name, instance-wide, and kept across PRs.

Whose language: a review (summary, inline findings, notices, banners, the
"Fixed in" reply) is the PR author's, whoever triggered it; a Q&A answer is
the asker's; a finding reply is the author's. Prompts, enums, JSON keys, logs
and alert names stay English. The prompt's language block sits above the
untrusted input, so the cache prefix forks once per language. A `ß` the model
writes is turned into `ss` outside code. Evals always run in English.

## 13. Terminal events and the riptide rollup

`pr:merged`, `pr:declined` and `pr:deleted` close the PR out. Except for a
deleted PR, the final diff is refreshed first, so the recorded line counts match
what was actually merged.

One rollup per PR is forwarded to riptide, claimed in the database in the same
statement that reads the snapshot, so a second terminal event emits nothing. It
is best effort: a riptide failure never fails a webhook and is never retried. An
unknown cost is omitted from the payload rather than sent as zero.

A declined PR that is reopened starts fresh.

## 14. Storage model

The schema records **runs**, not accumulators. A PR's totals are aggregates over
`review_runs`; its cost is NULL until at least one run is priced.

Every store call in the review path goes through a wrapper that warns and falls
back: a database fault must never fail a review.
