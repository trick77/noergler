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
8. The PR's accumulated cost is under the cap. Auto path only; an `@mention` is
   the explicit override.

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
`findings`, `compliance_requirements`.

## 10. Dedup, ordering and posting

Findings already posted by earlier runs are filtered out by reading the
**database**, not by fetching and parsing Bitbucket comments. What is left is
sorted by severity and capped at `REVIEW_MAX_COMMENTS`.

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
8. A footnote with scope, token and cost figures

There is no effort score and no "What changed" section.

The footnote's model line shows the llmwire profile id with the reasoning level
that was sent appended (`<profile>-<level>`), not the operator's gateway alias.
With no level configured, that is what the model's balanced level resolved to.

## 11. Cost

Only a cost the gateway **reports** counts. An unpriced run is stored with a
NULL cost and the review proceeds: cost fails open. The per-PR cap only stops
later automatic runs; an `@mention` always runs.

Costs are stored as nano-USD integers and rendered as USD at the edges. The
gateway key's total spend is shown as a gauge where available, never summed.

## 12. Q&A by mention

A comment mentioning the bot is answered rather than reviewed. It runs without
the review gates and without compression, reads the PR diff and the ticket, and
replies in the comment's own thread. Keywords route the question; anything else
is answered as a question about the PR.

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
