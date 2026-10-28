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
Then the team's globs: a repo in `exclude_repos` gets no review and no mention
answer; after that, a PR whose source branch matches `exclude_source_branches`
or whose target branch matches `exclude_target_branches` gets no automatic
review, silently (no summary, no DB row), while an `@mention` on it is still
answered. Merge, decline and delete pass both.

Ownership comes from the `team_claims` table, not from `teams.yaml`. The yaml's
`projects:` only seeds it on first start.

A comment containing `@<BITBUCKET_USERNAME>` routes to the Q&A path instead of a
review. The trigger is the shared bot's name, or, for a team that pushed its
own Bitbucket bot (`PUT /teams/{team}/credentials`), that bot's: such a team
posts as its bot and answers only to its `@`.

Last before dispatch: a team whose own Bitbucket token cannot be worked with
(rejected, unreadable, another account's) gets no new work: a review, a
comment or a draft handoff is acknowledged with `ignored` and nothing queued,
until it pushes a working token. Merge, decline and delete events still pass,
since Bitbucket never sends them again. No other team changes.

Events handled: `pr:opened`, `pr:from_ref_updated`, `pr:modified`, `pr:merged`,
`pr:declined`, `pr:deleted`, `pr:comment:added`, `pr:comment:deleted`.
`pr:modified` acts only when it takes a PR out of draft (§12b); every other
edit is ignored. A hook registered before `pr:modified` was required lacks
it until onboarding is run again.

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

On a push, noergler asks Bitbucket for the PR diff between the last reviewed
commit and the new head (`sinceId`/`untilId`). Same SHA means nothing changed
and the run stops. If the last reviewed commit is no longer one of the PR's
commits (a rebase or force push), or a merge commit lies between it and the
new head (the target branch merged in), the run falls back to the full PR diff.

The review then sees the incremental diff as the subject of its findings and
the cumulative PR diff as context. The summary is still the PR's (section 10).

## 4. Diff fetch and file splitting

The PR diff is fetched whole and split per file. Non-reviewable files are
dropped: binaries, lockfiles, vendored and generated paths, hidden path
components. Committed test screenshots are binaries too, never reviewed as
code; they go to the summary only (section 11c).

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
templating: file content contains braces. Changed test screenshots, when any
are sent, ride in a second user message after the prompt, so the prompt's
text stays byte-identical for the cache.

Budgets come off the resolved context window through a trust curve (large
advertised windows are the least trustworthy), minus an output reserve. The
cumulative diff and the previously-posted findings each get a small slice of
what is left.

A response is required to match a JSON schema with exactly these keys:
`overview`, `latest_push`, `strengths`, `security_performance`,
`test_coverage`, `verdict`, `findings`, `fixed`, `pre_existing`,
`compliance_requirements`, `visual_changes`.

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

The summary is the PR's, never one push's: the comment is replaced on every
run, so what it leaves out is gone. Its overview, strengths, security, test
coverage and verdict cover the whole PR, and its issue list is every open
finding, earlier ones (not fixed, thread neither resolved nor deleted; an
unreadable thread counts as open) and this run's: issues first, earlier before
new within a severity, the new ones marked when earlier ones stand beside
them. The list shows 30 and counts the rest under them.

The overview builds on the previous one: every incremental review gets the
overview the last completed review wrote, from the database, as its own
earlier draft to update. On a large PR whose diff does not fit the prompt,
what earlier pushes did stays in the overview; what the diff contradicts
(a reverted part) is dropped, because the code wins over the draft. A full
review writes the overview fresh from the whole PR.

A finding whose thread someone resolved or deleted is settled: it leaves
the list and the verdict, is never posted again, and the model still sees
it, marked closed, so it does not raise it again. A resolved thread is
stamped once and a review never reads it again; reopening it later is not
noticed, except by the draft-to-ready banner, which reads every thread. A
deleted thread is closed for the review that sees it, never stamped. A
review reads at most 50 unsettled threads, issues first; the rest count as
open without being read. An earlier finding a full review (after a rebase)
could not retire stays open until its thread is resolved. The verdict is settled
once against the whole list: any issue in it
requests changes, only suggestions approve with follow-ups, none approves.
So an earlier issue still open holds the verdict up even when the model saw
nothing new or the budget trimmed it from the prompt, and a request for
changes over an empty list (findings dropped, threads closed) is lowered. A
verdict that matches keeps the model's rationale; a changed one gets a
fixed line saying why. Under the list, when there
is any, one line counts the earlier findings this push fixed and the findings
validation dropped, or says the earlier findings could not be read from the
database and the list is this run's only. Repeats, the comment cap and
inline posts Bitbucket refused are in the log, not in the line.

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

Then one summary comment per PR, edited in place on later runs. It opens with
"PR review summary" ("PR Review-Zusammenfassung"), the last push's range
under it on an incremental run, with one sentence on what that push alone
changed. The model writes that sentence and the Overview into two separate
fields, so the Overview stays the whole PR's whatever the push was; a full
review's push sentence is not shown. Its sections, in order:

1. Overview
2. Strengths
3. Issues and suggestions
4. Security and performance
5. Test coverage
6. Visual changes, only when the PR changed test screenshots (section 11c)
7. Ticket or Requirement compliance
8. Recommendation
9. A footer: a bullet for each thing the author can act on (no AGENTS.md or
   one over its warn size, no ticket while Jira is on, files not reviewed or
   reviewed without full context), then one line of model, files, lines,
   elapsed time and cost, ending in a link to the PR's page when
   `NOERGLER_PUBLIC_URL` is set. The link carries the PR's random token and
   is the only way to the page: the dashboard never links to it. The PR page
   keeps its runs in a rail beside whichever view is open: every finding
   posted on the PR (headline, linked to its comment; the run that posted
   it and the one that fixed it), the files the reviews saw, the PR's cost
   against its cap, and its replies and answers. Each run has its own page
   under it, in the same shell, with the findings that were not posted,
   their text and quoted lines, the step timeline and the gateway call

There is no effort score and no "What changed" section. Tokens, prompt parts,
cross-file symbols, the gateway's cost split and the key spend are on the run
pages, not in the PR. On the PR page cost is the cap's total, tokens and time
add up over runs, the waits are one total, and the lines and files are the
whole PR's as the latest review read them (a dash when it could not read
the whole diff), never a sum of increments.

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

## 11b. Bugs that were already there

The model may also list up to five real bugs it saw in code the PR did not
change (`pre_existing`). They are not the author's, so they are never posted
and never touch the verdict. Validation keeps one whose evidence is in the
shown file and on no line the PR added or removed; the line moves to the
evidence when that is a shown line. The run page lists them with the model's
explanation, the quoted lines and a link to the line at the reviewed commit,
so someone can pick them up.

## 11c. Test screenshots

A UI repo commits its visual test baselines: an image under a `*-snapshots`
(Playwright), `__screenshots__` or `__image_snapshots__` directory. Any other
image is not one. The runner already compared the pixels; what a reviewer is
not told is what changed and whether the PR's code explains it. A baseline
that changed with no code to explain it is usually rendering drift (font,
anti-aliasing, browser version) committed as the new truth.

When the reviewed diff or the whole PR's names one, the PR's changes are
read (`/changes`, one paged call): the list of screenshots, each one's type
(added, deleted, moved) and blob ids, and the PR's base and head commits.
The list is the whole PR's on every run, and "before" is the PR's base, never
the target's tip, which may carry another PR's baseline: without a base a
changed screenshot is not compared, logged. "After" is the commit the run
reviews; when the PR moved on since, the blob ids belong to the newer head
and the bytes key the store instead. Without the changes at all the section
says it was not read this run.

Browser and platform variants of one screenshot (`-chromium`, `-firefox`,
`-webkit`, `-linux`, `-darwin`, `-win32`) are one line, chromium tried
first; a variant whose pixels did not change hands over to the next, so a
change only one browser shows is not lost. The line counts the rest.

A pair is known by its blob ids, so one described before is shown from the
store without a read. A new one is read at both commits (at most 20 reads a
run, under `BITBUCKET_MAX_FILE_BYTES`); a 404 says added or deleted where
the type does not, and bytes key the store where the ids are missing. A
model whose profile has no vision is sent none and nothing is read for it
(startup logs `vision=yes|no`). A baseline kept in Git LFS comes back as
its pointer and is named as such. The pixel diff runs in noergler, not the
model: an exact compare finds the changed region, both versions are cropped
to it (at least 256 px a side) and downscaled to 1024 px; an image over 16
megapixels is refused before decoding. Same pixels (a re-encode): not
listed, and stored so the pair is never read again.

At most six pairs go out per call; the rest wait for a later run, which
finds these in the store. The images and their labels are counted into the
fit check (w·h/750 tokens an image); one that does not fit is left out,
never a file. A diff that changes baselines and nothing else is reviewed
anyway on a vision model: that is the drift case, and a push that only
deletes or reverts one still refreshes the section.

Known limit: a stored line keeps the verdict it was written with. A
baseline described as unexplained drift stays so after a later push adds
the CSS that explains it; the pair (same blob ids) is never sent again.

The section lists each screenshot with the model's sentence, or why there is
none: removed, over the per-run limits, no room, too large, unreadable, in
Git LFS, no vision, or the model said nothing. Never a finding: an image has
no line to anchor on and no quotable evidence. Every screenshot left out is
in the log.

## 12. Q&A by mention

A comment mentioning the bot is answered rather than reviewed. It runs without
the review gates and without compression, reads the PR diff and the ticket, and
replies in the comment's own thread. Keywords route the question; anything else
is answered as a question about the PR, or about noergler on it. On a model
that takes tools the answer may call `team_config` (the team's effective
settings: claimed projects, excluded repos and branches, both author lists by name, the
built-in skipped dirs, files and extensions, and whether the asker is on the
lists; a settings answer is a Markdown table),
`pr_status` (runs, the latest outcomes and skip reasons, cost against the cap, the branches and the exclude list they match)
and `set_my_language` (the asker's language, only when their own comment names
it). Up to three tool rounds, four calls each, then a final round without
tools. Each answer logs whether the tools were offered, called or withheld and
why, and team startup logs `tools=yes|no`: a model without them answers a
settings question with "cannot see". The gateway calls are staged off the worker like a review; the reply is
posted back on it. Every call is recorded and counts toward the PR's cost; at
the cap a question is not answered. A mention writes the PR's row first (its
review pointer untouched), so a PR only ever asked about is capped too.

A mention posted as a reply also sees the comment it replies to: its author,
its file and line when it has them, and its text (capped at 4096
characters), so "@noergler can you answer Peter's question?" works.
Bitbucket sends the immediate parent only and anchors only a thread's root,
so a reply nested under a reply shows that reply, without a location. Under
the summary there is none to show; a parent that cannot be read is logged
and the question is answered without it. When the parent is someone else's
comment, the answer names no one; under the asker's own, it names the asker.

A reply directly under noergler's summary comment needs no mention either: the
worker matches its parent against the stored summary id and answers it as a
Q&A, whoever asks. The same holds under any reply noergler posted (a Q&A
answer, a finding-thread answer, a language confirmation): every reply the
bot posts is stored by its comment id (`bot_replies`), and a plain reply
directly under one is a Q&A too, whoever asks, with the bot's answer as the
comment it sees. The original question above that answer is not shown
(Bitbucket sends the immediate parent only). On a PR whose summary was
deleted such a reply is not answered: only an `@mention` brings a PR back.

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
is told to address a question by it, always, and may use it when acknowledging
or accepting a dispute; on any other intent code cuts a leading address from
the answer. A Q&A answer addresses the asker by the same name. The name is
never stored or logged. Reviews carry no name.

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

## 12b. Draft to ready

A draft is reviewed like any PR: teams open a draft so noergler reviews it
while humans stay away, and set it ready once the findings are dealt with.
That switch hands the PR to humans, so noergler checks its round is closed:
each unfixed finding's thread is read from Bitbucket, and when any is still
unresolved, a banner at the top of the summary counts them and links up to
five. All closed: nothing is written, and a banner from an earlier switch is
removed. A thread that cannot be read counts as open, a deleted one as
closed. No model call and no cost; a PR without a summary, closed, or
ignored after its summary was deleted is left alone. The banner sits below
a cost or staleness banner and goes with the next review, which rewrites
the summary.

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

## 15. Almanach feed

Almanach is a companion service like riptide: a living wiki of what the model
learns about each repository and project, built from one note per merged PR.
noergler prepares the notes; it never reads anything back, and nothing from
Almanach enters a review.

Opt-in per team with an `almanach:` block (`url`, `token_env`) in
`teams.yaml`, like `riptide:`. No block, no note call, no row. The token is
the team's identity at Almanach; the startup ping behaves like riptide's (a
401 disables the team, anything else warns).

**Per review round.** After a successful review has posted, one more call
continues the review conversation on the inference pool: the same system
message and prompt, the model's review answer verbatim as the assistant
turn, then the note instruction (`almanach_note.txt`). The replayed turns are
what the gateway's prefix cache can serve; whether it does under the note's
own response schema is read off `almanach_calls.cache_read_tokens` on the
first live calls, never assumed. The call runs without the PR's queue hold
(`Spawn`), so the review comment is never delayed by it; its writes re-ask
ownership and are ordered by run id, so a slow round never overwrites a later
one.

The note is German whatever the review's language, repo-scoped, one object:
a summary (or `null` when the PR teaches nothing), the change kind (bugfix,
feature, refactor, chore), for a bugfix its symptom, cause and cause class,
and up to sixteen components, rules, invariants and entities with the files
they rest on. It is written as documentation, not as a list of one-liners:
the prompt asks for what a thing is for, how it is structured, which rule it
enforces and why, what must always hold, and what a reader could not guess
from the code, kept essential. Prose has no length cap of its own; the
validator trims, spells Swiss, checks the shape, scans the prose for
secret-like tokens and applies the caps that exist: a name's length, the
files per item, the items per list, and the whole note against
`ALMANACH_NOTE_MAX_CHARS`, the one ceiling against bloat, counted on the
prose the model wrote, not on the JSON around it. Over any of them the
whole note drops and the previous round's stays; never truncated. An
answer the model's own output limit cut off is a length drop too, not a
shape fault. Identifiers in backticks are scanned for credential shapes
only: documentation names what it documents. Each round replaces the PR's
note. Every call, whatever became of it, leaves a row in `almanach_calls`
with the gateway's figures and a `note_status`; that ledger counts toward
nothing else, not the PR's cost, not the dashboard spend, not the rollup.

**At merge.** `pr:merged` stamps the note final with the merge facts and the
posted issue and suggestion counts (a stub when the first round has not
landed yet) and sends the team's pending notes on, oldest first, up to ten.
No model call. Almanach dedupes on the PR and its source commit, so a note is
marked emitted only after a 2xx and a refused one goes with the team's next
merge. A merge that lands while this PR's round is still in flight stamps the
row and leaves the emit to that round, which then sends its own note only;
the catch-up drain stays with the merges. A round whose PR was declined,
deleted or purged meanwhile writes nothing (`no_row`); a store fault on the
upsert is its own status (`store_failed`). `pr:declined` and `pr:deleted` forget
a note that is not final; a final one is kept until sent.

The payload carries the note, the PR's identity and the team's claims on the
project (the structure Almanach places the note by), the review totals and
every lane llmwire reported for the round's note call, money as exact
strings beside nano-USD integers.
