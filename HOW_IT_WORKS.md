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
`pr:modified` acts only when it takes a PR out of draft (§12b) or changes
its target (the SNAPSHOT task alone, §2); every other edit is ignored. A hook registered before `pr:modified` was required lacks
it until onboarding is run again.

## 2. Guard order

The pipeline refuses early and cheaply. In order:

1. Project and repository resolvable from the payload.
2. Author is in the auto-review list and not ignored: neither in the team's
   ignore list nor matching the instance-wide author globs
   (`REVIEW_IGNORE_AUTHOR_GLOBS`, default `*renovate*`, case-insensitive, no
   team can lift them). An `@mention` skips this and the next two.
3. The actor who pushed is not ignored by the same two lists.
4. Skip state: if a human deleted noergler's summary comment, the PR is left
   alone until an `@mention` brings it back (§12). If someone told noergler
   to stop reviewing it (§12c), it is left alone until a review is asked for.
5. The source branch does not contain the opt-out keyword.
5b. Not a guard: the SNAPSHOT task (below). It runs here, before every guard
   that can end the run without a review, since it calls no model.
6. `AGENTS.md` exists, when it is required.
7. `AGENTS.md` is under the hard token limit.
8. The PR's accumulated cost is under the cap. An `@mention` does not override
   it.

A skip that did not review the new commit writes the **prior** commit back as
the reviewed pointer, so raising a limit later re-reviews the accumulated range
instead of skipping past it.

Every skip that is a stable state of the PR (opt-out branch, missing or oversized
`AGENTS.md`, cost cap) posts a summary saying why. Guards 1 to 4 post nothing.

**SNAPSHOT task.** A PR into the repo's default branch (Bitbucket's
`default-branch`, read once an hour per repo) whose delta adds a dependency
on an unreleased Maven/Gradle build gets a Bitbucket task (a comment with
severity `BLOCKER`), in the author's language, naming the lines. Only `+`
lines count, only in `pom.xml`, `.mvn/maven.config`, any `*.gradle(.kts)`,
`gradle.properties` and `*.versions.toml`, and only `-SNAPSHOT` on a version
(in capitals after a word character, `}`, `>` or `=`,
`1.0.0-beta-SNAPSHOT`; in any case after a digit, `}` or `BUILD`; a
qualifier may follow, `1.0-SNAPSHOT-jdk17`; `endsWith("-SNAPSHOT")` is
Gradle logic, `libs-snapshot` a repository) or a resolved snapshot's
timestamp. The project's own version never counts, nor a reactor
module's: `version` in `gradle.properties`, or the key the build script
beside it (`build.gradle.kts`, `build.gradle`) sets the version from
(`version = projectVersion`); a Gradle script's `version = …` or
`project.version` (Groovy and Kotlin; a plugin's `id(…) version "…"`
and a dependency's named `version = "…"` argument are dependencies, so
is a `version =` line whose line before a hunk does not show);
`-Drevision`/`-Dchangelist`/`-Dsha1` in `maven.config`. A `pom.xml`
that adds a SNAPSHOT line is read whole at the PR's head: its
`revision`, `changelist` and `sha1` properties are own, and every
element whose text is the own version (its `<version>`, else the
inherited parent's), so sibling modules and their managed versions
bring no task. The parent is own only when its POM stands in the repo
where Maven looks first (`<relativePath>`, default `../pom.xml`) with
the same coordinates, read only when the module inherits its version or
the PR changes the parent's version line; a company parent from outside
counts. An outside dependency on the very same version string reads as
own. A POM that does not match its diff (a push in between, a charset)
or past 50 files read whole is told by the own version strings read so
far; one too large or not parsing keeps its lines (logged). A change
block that removes as many dependency SNAPSHOT lines as it adds brings
none in (`1.0-SNAPSHOT` → `1.1-SNAPSHOT`); the base POM is read only for
a block that may exempt a dependency, and when it cannot be told the
block's removed lines exempt only as many as they outnumber its added
own lines. Lines taken as the own version are logged. Only the changed
build files' diffs are read, one call each, or the whole PR diff once
past 50 of them. Build files with the blob ids of the last check are not
read again (in memory), with the head commit too when a file the PR does
not change was read (a parent, a build script). A PR touching none costs
one changes call; the open tasks are one call. The task blocks the merge only where the repo's
"incomplete tasks" merge check is on.
Each task names the lines it was posted for. A line no task named gets a
new task; the lines of a task the author resolved never get another while
they stay. An open task's text follows its lines (one released drops out),
and noergler resolves it once none is left, or once the PR targets another
branch. A SNAPSHOT removed and added again is new. Any read that fails
leaves everything as it is. A task a person reopened counts as open again;
noergler's own open task with the text it is about to post is taken over,
never posted twice. A `pr:modified` that changes the target of an open PR
runs the check alone, no review, unless the team's `exclude_repos` or
branch globs exclude the PR. `REVIEW_CREATE_TASKS` (instance, default
`true`) and the team's `create_tasks` (API, default `true`) must both be
on; off, no task call at all, an open task stays. A stopped PR, an
opt-out branch and an author or actor the review skips get no check.

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
dropped: binaries, lockfiles, data files (JSON, CSV, recorded HTTP traffic),
vendored and generated paths, hidden path components. Committed test screenshots are binaries too, never reviewed as
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

One call per review, streamed: the gateway cuts a connection that carries no
byte for 300 s, and only a stream sends anything before the answer is done.
The cost comes on the stream's usage (LiteLLM's
`include_cost_in_streaming_usage`), never from a header.

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

**Blocking rules.** A line of the target branch's AGENTS.md (never the
PR's own, so a PR cannot drop the rule it breaks) that starts with
`[blocking]` (any case, after optional indentation and one list marker,
`-`, `*`, `+` or `1.`/`1)`;
not inside a code fence; indented continuation lines belong to it) is a
blocking rule; at most 10, numbered `B1`, `B2`, ... for this review and
listed to the model under "Blocking rules". A finding the model
says violates one carries its id in `blocking_rule`. Code checks it: the id
must be one of the review's rules, the severity `issue`, the confidence at
least 90. A finding that passes is posted as a Bitbucket task on its line
(`blocker-comments` with the same anchor as a comment), naming the rule;
it sorts first within its severity, so the comment cap never cuts it. At
most 3 tasks a review; past them, with tasks off (same two switches as the
SNAPSHOT task), or when Bitbucket refuses the task (a 4xx), the finding is
a plain comment that names the rule and says why it is no task; any other
failure fails it like a comment, since the task may exist. A claim that fails
the check is a plain comment that does not name the rule. Nothing is
dropped for its claim. Each claim is one log line (rule, confidence,
outcome) and its rule text and outcome are kept on the finding's row.
The task blocks the merge where the repo's "incomplete tasks" merge check
is on, and the summary counts and marks open tasks. Counted, a task is a
task and never a finding: the run's `findings_posted` leaves it out, its
`tasks_posted` holds it, and every page shows the two apart (riptide's
total adds them back). Handled, a task is a finding: deduped like one, so one the author resolved is never posted again;
resolved, it is settled like a resolved thread; a `fixed` claim on it
replies and resolves the task, even when the reply failed; a fix
noergler committed itself resolves its task too, though it leaves the
thread open (the stamped row leaves every later review, so a task left
open would block unseen). A task whose rule the target's AGENTS.md no longer marks
(a rule past the cap of 10 is still marked), or marks reworded, is
resolved on the next review (never when that file could not be read);
the violation reported under the new wording is a task again. The cap
of 3 and the switches apply before the sort and the comment cap, so a
claim that ends a plain comment takes no slot ahead of its peers; a task
whose post failed is neither counted nor marked in the summary. A dispute
in the thread changes nothing.

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
7. Ticket or Requirement compliance. A requirement's criterion label from the
   ticket (`AK-1`) leads its line in a code span: bare, Bitbucket's Jira
   integration links it as an issue key, to an issue that does not exist.
   The model's prose elsewhere in the summary and in inline comments spans
   the same labels in place
8. Recommendation
9. A footer: a bullet for each thing the author can act on (no AGENTS.md or
   one over its warn size, no ticket while Jira is on, Jira unavailable so
   the ticket check was skipped or the parent went unchecked, files not reviewed or
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

The model answers once per change, listing every screenshot that shows it,
and each of those screenshots is stored with that one sentence. Screenshots
whose line is the same sentence, byte for byte after whitespace, are one
line at render time: three names, then a count of the rest, variants
included. A word apart stays apart, a status line never folds, and the store
keeps one line per pair.

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

At most ten pairs go out per call; the rest wait for a later run, which
finds these in the store. The images and their labels are counted into the
fit check (w·h/750 tokens an image); one that does not fit is left out,
never a file. A diff that changes baselines and nothing else is reviewed
anyway on a vision model: that is the drift case, and a push that only
deletes or reverts one still refreshes the section.

Known limit: a stored line keeps the verdict it was written with. A
baseline described as unexplained drift stays so after a later push adds
the CSS that explains it; the pair (same blob ids) is never sent again.
A call that sends screenshots also carries the sentences the section
already shows from the store (at most 20, counted into the fit check), and
a new screenshot of one of those changes copies its sentence when its
verdict still holds for the PR's code, so screenshots sent in different
runs (over ten pairs, or added by a later push) still fold. As many as fit
beside the images go; the rest are logged, and their new lines may stay
apart.

The section lists each screenshot by file name (the path where two share
one) with the model's sentence, or why there is none: over the per-run limits, no room, too large, unreadable, in Git LFS,
no vision, or the model said nothing. A removed baseline gets no line; a PR
that only removes baselines has no section. Neither does a pair the model
says shows no difference a person could see (a few edge pixels a shade
off; never an added baseline, which has no before): it
is stored like same pixels, and the next variant is tried on a later run. Never a finding: an image has
no line to anchor on and no quotable evidence. A bug in the code that a
screenshot also shows is still a finding, anchored and quoted from the code.
Every screenshot left out is in the log.

## 12. Q&A by mention

A comment mentioning the bot is answered rather than reviewed. It runs without
the review gates and without compression, reads the PR diff and the ticket, and
replies in the comment's own thread. Keywords route the question; anything else
is answered as a question about the PR, or about noergler on it. On a model
that takes tools the answer may call `team_config` (the team's effective
settings: claimed projects, excluded repos and branches, both author lists by name, the
instance-wide author globs, the built-in skipped dirs, files and extensions, and whether the asker is on the
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
comment it sees. One exception: the PR author's plain reply under
noergler's answer on a finding's thread is a reply on that finding (the
answer is stored with its finding), so "ok, fix it" there is heard. The
original question above that answer is not shown
(Bitbucket sends the immediate parent only). On a PR whose summary was
deleted such a reply is not answered: a plain reply never brings a PR
back. On a stopped PR (§12c) it is answered, and the PR stays stopped.

A reply that tags someone else (`@sjh`, `@"jan.saner"`) is addressed to that
person, not to noergler: the route ignores it on every thread, summary,
answers and findings alike, unless noergler is tagged as well. A tag inside
code (`` `@Transactional` ``) does not count; a bare annotation in prose does.

The PR author's reply on one of noergler's finding threads needs no mention.
The route queues any non-empty reply not written by the bot (Bitbucket's
`commentParentId`); the worker answers only the author, except that anyone
else who orders a change there ("fix it", "bitte beheben"; short, no
question, decided by code) is told kindly that only the author may, as a
mention asking a change is anywhere (author-only actions are never met with
silence); only when the parent is
a stored, unfixed finding of that PR without an answer yet, and, for a plain
reply, only under the PR's cost cap. On a thread answered already a plain
reply still gets a call, but only a fix request (below) is answered; an
`@bot` reply there is answered whatever it asks. An
`@bot` reply on such a thread is
answered as a reply too, unless it asks for a review or comes from someone
else, which the Q&A answers. The call (`finding_reply.txt`) is staged
like a review, and sees the finding, the file around it, AGENTS.md, the
repo's open dispute signals and the reply. A dispute, explicit or not ("we do
this in the UI"), is acknowledged. The dispute is stored and later prompts of
the same PR show it beside the finding so the issue is not raised again. The
thread stays open: the author resolves it. A reply's call is recorded and
counts toward the PR's cost; at the cap, replies stop.

### Auto-fix

A reply asking noergler to make the change itself ("tue dies selber", "flick
das", "setz das um", "fix it") is the intent `requests_fix`; "ich flicke das"
or "fixed" acknowledges. Always on, no setting. On that intent a second call
(`finding_fix.txt`, same profile, same pool slot) gets the file at the PR
head, 60 lines either side of the finding's line (its number when posted at
the head, else found again by its text in the file it was posted on, nearest
the old number, a blank line by its neighbour; that file or text gone = not
feasible; not UTF-8 = not feasible, no call) and AGENTS.md, and returns one
exact `old` text and its replacement, written in the team's style (AGENTS.md
conventions, else the surrounding code's), or `feasible: false` with a reason (another file,
a generator run, a decision only the author can make). `ValidateFix`, blunt
rules: `old` once in the file, inside the excerpt, at most 40 lines, no new
secret-like literal; the file is spliced byte for byte (line ends, BOM, last
newline kept). The commit goes to the source branch (a fork's repo for a fork
PR) through Bitbucket's edit-file API, based on the head the file was read
at: a branch that moved since is a 409, never an overwrite. Commit message:
the model's subject in the author's language, verb first, present tense,
lower case (`fix loop redirection`, `behebe Schleifenumleitung`), then a link
to the finding's thread.

Every request is answered in the thread, never silently: "Applied in
`<sha>`" with the subject and "please review the change carefully before you
merge", or why not: not feasible (the model's reason), rejected as unsafe,
no write permission (401/403: the token may be read-only; asked again, it is
tried again), the branch moved (409), refused by Bitbucket (a pre-receive
hook), or the call failed. Each request is a `finding_fixes` row: the
metrics page shows, per team beside the findings and tasks, fix requests
(every ask, its outcomes on hover) and fixes committed. Its Activity card
charts one kind of act per day, picked by a switch (`?activity=`):
findings, tasks (a review's and SNAPSHOT tasks), fixes or engagements,
each act under one kind only (`activity_daily`, `ActivitySince`). An
engagement is a comment that tags noergler or replies to one of its
comments, counted once whatever came of it (answered, only-the-author,
refused at the cap, failed); a fix or merge-conflict request counts as a
fix and a review asked by mention as a run instead. SNAPSHOT tasks and
engagements live in `activity_events`, one row per comment, counted from
0039 on. Findings, tasks and engagements stack by team; fixes stack
committed and not committed. A new figure is one more kind, never another
chart or column. The Runs page lists the same acts, each its own row: a
review (its findings and tasks apart), an engagement, a fix request, a
SNAPSHOT task, filtered by kind and outcome in the browser; an
engagement's time and cost are its own calls', a fix row shows none (its
call row names no request). A finding is committed once. The fix call is its own
`llm_calls` row (`fix`) and counts toward the PR's cost. The
next review retires a finding noergler committed without a "Fixed in" reply
and without resolving it: the author resolves the thread after looking.

The same works for a Q&A answer whose question sat on a line of code
(inline only). The answer keeps that line (`bot_replies.anchor_*`): the
question's own when it is the thread's root (one comment fetch per such
mention), else the inline root it replies to, else the line kept with the
noergler answer it replies to; never an old-side line or one gone from the
diff. A line on one commit's diff belongs to that commit and is found again
by its text. A question replying to a reply in an inline thread keeps no
line (known limit: Bitbucket sends the immediate parent only). A reply
under that answer asking noergler to apply it ("gute Idee, setz das um") is
the Q&A envelope's `apply`; the same fix call, checks, commit and answers
follow, the answer's text being what is applied. Every reply of that flow
(done, not possible, ask again, only the PR author) is kept about the
answer (`bot_replies.source_comment_id`), so "apply it" under it applies
the answer. Only the PR author may: anyone else is told so, with no call
paid and no row. No code line, the answer says so. An answer offers to
apply its suggestion only when "apply it" under it would apply it: its
question kept a line and it is no reply in a fix flow (there "apply it"
applies the source answer); the not-author notice names the reply only
on a line. `apply` anywhere else
gets a fixed "can't apply that here", never the model's sentence. An answer
is committed once.

### Merge conflicts

When Bitbucket shows a merge conflict, the PR author can write `@noergler
behebe den Merge-Konflikt` ("löse den Konflikt auf", "resolve the merge
conflict", "fix the conflicts"): a short command, a verb near "conflict",
decided by code, under a finding too. Anyone else is told only the author
can ask. A question ("how do I resolve the conflict?"), a negation or the
author speaking of doing it ("I'll resolve the conflict myself", "später")
goes to the Q&A.

A yes directly under noergler's open offer (below) is the same request: a
short one ("ja", "ok bitte", "👍") by code; any other wording ("lass uns das
versuchen") reaches the Q&A, whose model has a `resolve_merge_conflict`
tool there and only there. Code vetoes the tool on a no or a put-off in the
commenter's own words ("nicht jetzt", "mach ich selbst"); else the request
starts after the turn, answers with its own replies, and the model's
sentence is not posted. Without the tool, or after a veto, the answer adds
what starts it. Anyone but the author is told only the author can ask. On
a PR ignored since its summary was deleted a plain reply, a yes too, is
not answered (a plain reply never brings a PR back); the command with a
mention works there.

On the worker, bounded to 60 s: Bitbucket's merge status must say
`conflicted` (else "no merge conflict"); a fork PR is refused. noergler
fetches the source and target branches over git (smart HTTP, the account's
token, no disk: in memory, at most 300 MiB read and 600 MiB held, 50
commits deep, else 500) and finds their merge base. Every path both
branches changed since then is merged line by line, git's way
(`internal/merge`) four times: aligned by Myers and by patience, each as
found and with its changes slid as git slides them (`xdl_change_compact`);
a file the four merge differently is refused, since a server's histogram
merge might settle it otherwise. A path only one side changed takes that side.

Simple only, all or nothing: at most 10 paths both changed, 5 conflicts in 3
files, 15 lines a side; no lockfile, generated, built (`dist/`) or binary file, no file added
on both sides or deleted on one, no mode change, no file on one side where
the other has a directory, no conflict whose target or source names an
unreleased build (`-SNAPSHOT` in any case, a resolved snapshot's
timestamp), checked before the count and size bounds; an answer too long to quote every dropped line
in full. Past any bound the request
is refused with the reason, nothing asked, nothing pushed. A conflict where
both sides only added lines, not one block both added, share none (whitespace ignored) and set no key both (`"lodash":`
on both sides, `const X =` twice) is resolved by code: target's lines,
then source's. The rest goes to one call (`merge_conflict.txt`):
the model sees each conflict's base, target and source lines, numbered, and
answers per conflict the ids of the lines to keep (`T1`, `S2`), never text.
`ValidateResolution`: every id exists, none twice, each side's order kept,
no base id, and no base line one branch deleted or changed brought back
through the other branch's copy. The model may drop one side's version
only where the other already carries its change, else it says not
feasible and why. At the cost
cap a code-only resolution still runs; one that needs the model is refused
with an answer.

Two whole versions of the same lines, neither carrying the other's change
(each branch bumped one dependency to its own version), are the author's
pick (2026-11-18). The model names them in `ask`; code checks either side
alone passes `ValidateResolution` (`merge.Choosable`), else the answer is
refused. noergler resolves the rest as above but pushes nothing yet: it
replies with each asked spot, both versions quoted (`T` from the target,
`S` from the PR's branch), and asks for **T** or **S** (`1T 2S` for
several). The merge, its heads and every conflict's resolution are kept
on the PR row (`conflict_choice`). A reply directly under that question
naming the sides, read by code (letters only, never a version's text,
filler aside, a no or a put-off is none), settles it: same heads, same
conflicts, the kept resolutions and the picks, no second call, then the
commit and answer as below ("your pick"), the answer under the pick, the
fix row and the commit link naming the request. A branch moved (the
event's head says so before any fetch) or a conflict changed: nothing
pushed, ask again. A failure on the way keeps the question and writes no
row, so the same answer may be sent again; the request's one row comes
with the answer that ends it. The author's go-ahead naming no side ("Lets
go!") or letters of the wrong count get the letters asked again; anything
else there goes on as any reply; anyone else's pick is told only the
author may. Any other end of a request clears the question. A question
never answered leaves its call row and no fix row. Two sides sharing a
line are no pick (two additions T or S would halve); the question is
asked only if either pick assembles and the confirmation could quote
whichever side is dropped.

The merge commit (parents source then target, `Merge branch 'master' into
feature/x`, the resolved spots, a link to the request) is authored by the
bot account and pushed to the source branch without force, only if it still
points where it was read. The answer links the commit, lists each spot and
whether code, the model or the author's pick resolved it, quotes every dropped line, names the
files merged without a conflict, and asks the author to build and test
before merging. Failures answer like an auto-fix: no write permission,
branch moved, refused by a hook, failed. Each request is a `finding_fixes`
row (Activity, Fixes, on the metrics page), its call a `fix` row. The push
triggers a full re-review, as a merge by hand does.

A review offers it. After each completed review under the cost cap,
noergler asks Bitbucket for the merge status; on a conflict it fetches and
merges as a request would, without the model. A conflict code resolves
alone, or one within the bounds the model would answer, gets a comment of
its own that mentions the author and quotes the command (a plain reply
under it reaches noergler too); past the bounds nothing is said. One offer
is open at a time: a later review that finds no offer holding (conflict
gone, or past the bounds), a review at the cost cap, or noergler's own
push of the merge resolves its thread, and a conflict after that is
offered anew. A check that fails leaves the open offer as it is. A review
on the same two heads fetches nothing again. Bitbucket sends no event when the target branch
moves, so a conflict that arises after a review shows at the next push.

### Dispute signals

When a dispute's reason is a project-wide rule AGENTS.md lacks, the model
words it as a one-line AGENTS.md bullet. That dispute is an occurrence of a
dispute signal of the repo: the model is shown the repo's open signals,
numbered, and names the one that is the same rule (`matches_signal`), or none
for a new signal. A bare "no", a one-off, a quarrel about severity or a reply
the code contradicts gets no bullet and is no signal.

Each occurrence counts for `REVIEW_DISPUTE_SIGNAL_WINDOW_DAYS` (30), per
distinct PR. Below `REVIEW_DISPUTE_SIGNAL_THRESHOLD` (3) PRs the answer carries
no AGENTS.md text at all. The dispute that brings the signal to the threshold,
and every later one while it holds, adds a statement: the how-many-th PR this
is, links to the earlier disputed threads (10 at most), the number of
authors, and the bullet, merged from the earlier wording and this reply, so
that pasted into AGENTS.md the finding is not raised again. A signal nobody
disputes for the whole window is dormant: no longer listed, and the next
dispute on that rule starts over at one. Once AGENTS.md states the rule the
model writes no bullet, so nothing counts; a dispute that matches a signal
without a bullet is logged, since the review then raised a finding AGENTS.md
already rules out.

Signals are read only here, in the reply to a finding: never by a review, the
Q&A or Almanach, and they never drop a finding. The occurrence is recorded
once the answer is posted. A match whose merged bullet fails the format check
still counts, under the listed wording. Known limits: AGENTS.md is read at the
finding's commit, so a PR opened before the AGENTS.md edit may count once
more; two disputes of the same new rule answered at the same moment start a
signal each, and the count splits between them.

The answer may address the author by first name ("Ok Jan - ..."), not
always. `BITBUCKET_DISPLAY_NAME_PATTERN` extracts it from the Bitbucket
`displayName` (default `Last First (CODE)`: the last word before the bracket); no match, no name. The model
may use it, when it reads naturally, to answer a question, acknowledge or
accept a dispute; on any other intent code cuts a leading address from the
answer. A Q&A answer may address the asker by the same name. The name is
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

## 12c. Stopping reviews on a PR

Anyone who comments can tell noergler to stop reviewing a PR: `@noergler
stop monitoring`, `stop reviewing`, `Höre auf, diesen PR zu beobachten`,
or a plain reply saying so under the summary. New pushes are then not
reviewed (`pr_stopped` on the dashboard) and cost nothing. The rest carries
on: questions and replies, the author's on a finding included, are
answered, the summary stays, and at merge the riptide
rollup and the Almanach send run as for any PR. An Almanach note round runs
only after a review, so pushes after the stop add nothing to the note; the
last one is sent.

A short command (ten words at most, no question mark, a stop word and what
to stop, or "stop" with nothing but "please" beside it) is decided by code:
no model call, so it works at the cost cap too. "No longer" or "nicht mehr"
counts only with what to stop ("nicht mehr prüfen"), since a comment about
the code says it as often ("prüft den Token nicht mehr"). Other wording goes to the Q&A, whose model may call
`stop_reviewing`; code accepts that only when the comment itself says stop.
The reply confirms in the asker's language, and the summary gets a banner
naming who stopped it, in the author's language.

`@noergler review`, or a resume request ("resume", "weiter beobachten"),
brings the reviews back: the banner goes, and the review updates the same
summary and builds on its overview. A question does not; the PR stays
stopped, and on a PR that is not stopped "continue" is a question. The
last word wins: deleting the summary of a stopped PR turns it into the
deleted case (§12), where any mention brings it back with a fresh summary;
a stop after the summary was deleted makes it stopped, and the resume
posts a fresh summary.

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
merge. A 2xx answering `ignored` (the repo is not one of Almanach's source
repositories) is marked emitted too, as Almanach asks, but logs a warning,
never "taken": the note is kept by neither side. The row is marked ignored,
and the Teams page counts those per team over the last 7 days beside the
feed's word, which stays ok for them. A merge that lands while this PR's
round is still in flight stamps the row and leaves the emit to that round,
which then sends its own note only;
the catch-up drain stays with the merges. A round whose PR was declined,
deleted or purged meanwhile writes nothing (`no_row`); a store fault on the
upsert is its own status (`store_failed`). `pr:declined` and `pr:deleted` forget
a note that is not final; a final one is kept until sent.

The payload carries the note, the PR's identity and the team's claims on the
project (the structure Almanach places the note by), the review totals and
every lane llmwire reported for the round's note call, money as exact
strings beside nano-USD integers.
