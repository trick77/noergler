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
- Every call streams (user, 2026-11-06, LiteLLM timeouts stay where they
  are): the gateway cuts a connection that carries no byte for 300 s, and a
  call without streaming sends nothing until the whole answer is written, so
  every long review died there. Almanach hit the same and streams since its
  #91. The cost header is written before the body, on a stream as a literal
  0: a stream is priced from `usage.cost` only, which LiteLLM sends with
  `include_cost_in_streaming_usage`, else unpriced (llmwire once read the 0
  as a reported zero, a free call to the cap). The gateway's cost split is
  header-only, so a stream has none: the `llm_calls` lanes stay NULL.
- Header and idle bounds 4m30s, not llmwire's 60 s / 90 s: a reasoning model
  sends nothing for minutes before its first token (evals: `stream idle for
  1m30s`). Under the gateway's 300 s, not on it: at 300 s the gateway's cut
  won the race and read as a broken stream, an error with no notice, instead
  of a timeout (code review). The notice quotes the bound that fired. The
  call cap default 10m sits above them: a stream that keeps sending may run
  longer.
- Never `ReasoningMinimal()`: on some models minimal is thinking off.
- The gateway's window wins: it can cap below the vendor. LiteLLM often leaves
  `max_input_tokens` unset; prod once needed a hand-set override. Dropping
  `ListModels` warnings reports a garbage limit as a missing one.

## Findings

- Summary top-down by what the reader acts on (user, 2026-11-19): the
  summary often goes unread past its top, and a fixed skeleton put the
  recommendation last behind sections reading "None.". Sentinel-only
  sections are left out instead of shown. Review focus is the model's own
  hunk triage ("Where to spend attention"), which the old schema threw
  away; checked by path and line only, never by what the sentence says
  (no judge).
- Change line under the overview (user, 2026-11-25): under the heading,
  "Last push: a .. b" read as the review's scope, as if only that part had
  been reviewed, and the model's verb-first sentence ("Fasst … zusammen")
  as being about the summary itself. Rejected: kept on top with "covers the
  whole PR" (a line before the recommendation on every run); dropped (what
  is new no longer shows on the PR).
- Ticket check without ❌: a PR is a delta, so "not met" is never provable;
  the requirement can live in another file, an earlier PR, another repo or
  the pipeline. A requirement met by deployment scripts the PR did not
  touch read "not compliant" on the PR, the model quoting the changed
  config line it did see. Rejected: a third "not assessable" mark (still a
  verdict on what the review cannot see), the model sorting unmet from
  "outside the PR" (the same judgement that failed), a quote rule for ❌
  (an absence has nothing to quote; the changed line passes it). A count
  ("2 of 4") needs a total only the model gives. Loading files outside the
  diff would add ✅ lines, not prevent a wrong one; not built.
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
  model then reports as this PR's change, reversed. Grouped by the model, in
  the answer shape: asked for one sentence per screenshot, it reworded each
  for its screen ("in the first form", "in the second"), so the exact fold
  never matched; a fuzzy match would fold two different changes.
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
- Dispute signals, not a bullet per dispute (user, 2026-10-31): one dispute
  is weak evidence, the same reason across PRs is a rule AGENTS.md lacks.
  30 days so a reason nobody repeats fades; 3 PRs so a busy week of one PR
  does not count; no distinct-author rule, a solo repo would never reach it,
  the statement shows the count instead. The signal steers the answer only:
  author text is intent, so a review changes only when a human edits
  AGENTS.md. One call, not a second after the verdict: the list rides every
  finding reply and code ignores it unless the reply is a dispute.
- Auto-fix without a pre-check of write rights (user, 2026-11-13: "wir
  probieren es einfach"): Bitbucket does not tell a token's scope apart from
  a branch permission or a fork, and commenting needs read only, so no earlier
  call proves anything. Trying costs one call; the answer says why it failed.
- Auto-fix as one `old`/`new` replacement, not a whole file or a diff: a
  whole file from the model rewrites line ends, a BOM, unrelated lines and
  long files cut off; a diff's line numbers drift. An exact unique text
  inside the excerpt is checkable by code, and the splice leaves every other
  byte alone. The finding's line is found again by its text because no
  evidence quote is stored with a finding.
- Auto-fix call on the reply's pool slot: Stage's tail ends the PR's hold, so
  a second Stage from a post would release a key a newer job holds. The file
  is read on the worker before, for every author reply, accepted: rare.
- Merge conflict by a merge commit over git, never `EditFile`: a one-parent
  commit leaves the merge base where it was, so the same hunks conflict
  again. Bitbucket's REST makes neither a merge commit nor a rebase with
  content. go-git (no git binary: the image is distroless); its own merge
  only fast-forwards, so the three-way merge is `internal/merge`, checked
  against `git merge-file` (`TestAgreesWithGit`). Merged twice, Myers and
  patience, and refused where they differ: a server's merge (ort) diffs
  by histogram, so a file of repeated lines ("}") may settle there where
  Myers conflicts, or the other way round; pushing either guess would put
  code nobody was shown in "merged without a conflict".
- Merge conflict, code first (user, 2026-11-13: "auf den deterministischen
  teil möchte ich nicht verzichten"): insert/insert with nothing shared is
  exact without a model and free; the model only gets what code cannot
  settle. Nothing builds or tests the result (user: "noergler hat kein
  runtime/test environment"), hence simple only, all or nothing, and every
  dropped line quoted.
- Merge conflict as line ids, never text: the model cannot write a line no
  side has, line ends and a BOM stay byte for byte (the auto-fix reason),
  and dropped lines are computed by code, not claimed by the model. A side
  is dropped only where the kept version carries its change; else not
  feasible.
- Merge conflict, the author picks two whole versions (user, 2026-11-18):
  a dependency both branches bumped to their own version failed the whole
  request, the Q&A then suggested a side and offered an apply it could not
  do. IntelliJ leaves the same line to the developer, so noergler asks T
  or S instead of guessing newer or higher. The question keeps the merge,
  not just the picks: a second model call could answer the other
  conflicts differently, or ask again. Letters, not text: a reply quoting
  a version can negate it.
- Merge conflict prompt, read against B (eval 2026-11-14, EVALS.md): 10 of
  the 11 real wrong answers kept a base line one branch had deleted or
  changed while the other left it, e.g. a removed installer block came
  back. The rule alone went in (user, 2026-11-14); the abort clauses
  drafted with it showed no effect.
- Merge conflict base-line check in code (user, 2026-11-14): with the
  prompt rule the pattern drew 0 wrong, so the check is for a model or
  prompt that stops following it. Replayed offline on every stored answer:
  the 10 old-prompt wrong answers rejected, 0 of 313 correct ones; 1 of 39
  mined author resolutions refused (a version pin kept against a branch's
  bump: no push, the author resolves). Text exact (line end aside): a
  re-indent is a change in YAML or Python; whitespace-blind counting
  scored the same on the data but let a re-indent be undone. `}` and blank
  lines included: excluding them changed nothing measured. Counts carry no
  position, so a side that moves a `}` can get a correct resolution
  refused: safe, no push. Trimmed edges (`Head`, `Tail`) count with each
  side, or a block both added is refused. Misses a change lost where the
  kept line is the other branch's change (delete vs modify, git/git
  42afcb954b); "refuse a resolution that is one side verbatim" catches
  those but refused 22 of 139 correct answers.
- Merge3's slid witnesses (2026-11-15): a change that can slide (a blank
  inserted beside a blank) sits where git's `xdl_change_compact` puts it,
  at the bottom of its run unless it lines up with the other side's
  change. Myers and patience as found put it elsewhere: a blank after `}`
  touched the other branch's change of the `}` and conflicted where git
  merged cleanly (TestAgreesWithGit found it only once `merge.go`'s head,
  then its base, changed; the bases are fixed testdata now). Slid diffs
  are two more witnesses, never the merge alone: slid only, Myers and
  patience agreed far more often, and patience, histogram's stand-in, let
  through merges histogram conflicts on (code review). All four must
  agree; on a repeat-heavy stress (4 seeds, ~11 400 comparisons) merges
  clean where git conflicts or differs fell from 80 (master) to 4, a
  subset, and conflicts where git is clean to 0, refusals ~2.5%. User: "4
  in 11000 is an acceptable risk". No indent heuristic: `git
  merge-file` leaves it off.
- Merge conflict trivial rule (user, 2026-11-14): on 1500 git/git merges one
  trivial resolution broke a file (8f0d663e: both sides added one function,
  Merge3 settled the shared body, the headers one space apart were kept
  twice). Lines and keys compare without whitespace, and a conflict whose
  edges Merge3 trimmed (a block both added) is never code's: a header
  differing in more than whitespace would break the same way.
- Merge conflict build output (`dist/`) excluded like a lockfile (user,
  2026-11-14): asset hashes only a rebuild gets right, and the model
  declined them anyway. Web build files only (html, js, css, map, json):
  Go's `cmd/dist` and Python `dist` packages are source.
- Merge conflict with `SNAPSHOT` refused, never offered (user, 2026-11-15:
  "a snapshot must not go into master"): a resolution makes the PR
  mergeable, and the version on either side is the author's to settle.
  Code-resolvable ones too. Conflict regions only, not the whole PR; target
  and source only (base and trimmed edges are never a resolution's to
  bring). The loose rule (`snapshot.Mentions`): `-snapshot` in any case
  anywhere, every file (an image tag in YAML is a build too); a repository
  name or Gradle logic about the suffix refuses too: a stray match costs
  an offer, never a merge. The task's strict rule (`snapshot.Line`) was
  tried here and dropped: two reviews pulled it both ways at once.
- SNAPSHOT task (user, 2026-11-15): a Bitbucket task, not a finding: the
  author must resolve it on purpose, and the repo's merge check can block on
  it. Code decides, never the model: a task that blocks a merge must not
  hang on whether a model noticed a rule. Only into the default branch
  (user), from Bitbucket's API, no config. Only Maven/Gradle build files
  (user: nothing else uses SNAPSHOT; image tags in Helm/K8s YAML missed on
  purpose): a stray task blocks a merge, unlike the conflict refusal. So
  the strict rule (`snapshot.Line`): `-SNAPSHOT` in capitals after a word
  character, `}`, `>` or `=` (`1.0.0-beta-SNAPSHOT`,
  `<changelist>-SNAPSHOT`, `suffix=-SNAPSHOT`), any case only after a
  digit, `}` or BUILD; a quote, paren or blank before it is Gradle logic
  (`endsWith("-SNAPSHOT")`) or a comment, a lower-case name after a word
  a repository (`libs-snapshot`); a `-` after it a qualifier. Dependencies
  only (user, 2026-11-16: the own version "MUST" be a SNAPSHOT, "only
  dependencies from outside", "especially with Maven"): the first cut
  flagged a new component's `version=0.1.0-SNAPSHOT`. Own is told per
  file (`OwnLine`, a POM read whole), never by narrowing `Line`: a bare
  suffix may build an outside dependency's version too
  (`2.0${lib.qualifier}`). A POM's `<version>` reads the same for the
  project, its parent and a dependency, so a POM with a hit is read
  whole; the reactor is told by the version string, not the groupId (one
  groupId spans many repos, so a same-group dependency from another repo
  must still count): every element whose text is the own version is the
  reactor's. The parent by where Maven finds it: its POM in the repo at
  `<relativePath>` (default `../pom.xml`) with its coordinates, else from
  outside (a company parent on a SNAPSHOT is a dependency, and then
  nothing is inherited as own). Blunt, accepted: an outside dependency on
  the very same version string passes. Gradle siblings are
  `project(":x")`, no version; a named `version = "…"` argument is told
  from the project's by a `,`/`)` after it or a `,`/`(` ending the line
  before, blanks and comments skipped; a line before that a hunk does not
  show reads as an argument. gradle.properties' own key beyond `version`
  comes from the build script beside it (`version = projectVersion`),
  one read. The safe side throughout, without stalling: an error aborts
  the run, so a stable one (too large, no commit, a charset the two
  reads convert apart) must never be one, or no task is ever posted
  again. A head file that cannot be told keeps its lines; one not
  matching its diff, or past 50 reads (single worker), is told by the own
  version strings read (`>v<`). A base that cannot be told exempts only
  past the block's added own lines: the 1:1 dependency bump (user) still
  brings none, an own bump never exempts a dependency. The memo keys on
  the changed files' blob ids; a parent or script read outside them adds
  the head. PR
  delta only (user: "flag it only in the PR delta"): a SNAPSHOT already on
  the line before the change is never claimed; pairing per change block,
  by count, blunt: a SNAPSHOT moved to another block counts there, the safe
  way. A resolved task stays resolved for the lines it named (user), keyed
  by path + text without whitespace + the count of equal lines before it
  (two dependencies on one `<version>` line are two), no line number; keys
  leave a task once the line leaves the delta, so a SNAPSHOT removed and
  added again is new. A new line gets a task of its own, never an edit of
  an open one: an edit would name lines whose task the author already
  resolved. An open task's text follows its lines. Retarget (`pr:modified`)
  runs it alone: no push follows, and a task on a PR moved off the default
  branch would block it. Not on a stopped PR (user: decided; a stop before the first SNAPSHOT
  push gets no task, accepted) nor an opt-out branch, as a stop. Before the
  AGENTS.md gates and the cost cap: no model, nothing to save. Own diff,
  one changed build file at a time, not the review's: the cumulative one
  may be cut, a push's is not the delta, and the whole PR diff again would
  be a second large read per push on the single worker and fail at
  BITBUCKET_MAX_DIFF_BYTES. Past 50 build files (a reactor's
  `versions:set`) the whole diff once
  beats 50 calls. Hits remembered by the build files' blob ids, in memory:
  a push that leaves them alone reads no diff. Build files (user: only
  those): `pom.xml`, `.mvn/maven.config`, every Gradle script, properties,
  Version Catalog; Kotlin in buildSrc is code, left out. Off is off on both
  switches (user: "off means off!"): no call, an open task stays for a
  person. Team switch
  on for every team (user), instance switch on by default (user).
- Blocking rules (user, 2026-11-15): for rules only a model can check
  ("no DB access from controllers"); a rule code can check stays code (the
  SNAPSHOT task). The marker is read by code, the model never picks what
  blocks: a merge must not hang on a model's reading of which rule is
  strict. A task, not a finding flag: the merge check blocks on it and the
  author resolves it on purpose. Downgrade, never drop: a weak claim is
  still a finding. Confidence 90 (user), not the plan's 8: the scale is
  0-100 and the schema floor 80, so 80 filters nothing; 90 is the prompt's
  "exact line, failing input describable". Caps (10 rules, 3 tasks) bound a
  model that marks everything. Posted to `blocker-comments` with the
  anchor: the 7.6 reference's "Blocker file line comment"; `/comments` with
  anchor + severity is not documented. A rule removed resolves its task
  (user), which settles the finding: the team said it no longer blocks.
  Rules from the target branch (code review): read from the PR's head, a
  PR could delete the marker it violates and unblock itself; a rule a PR
  adds blocks from its merge on. One more GET per review. Nested files'
  rules (user, 2026-11-22) are read at the PR head instead, from the files
  the prompt already loaded: a module team's rule blocks without a second
  lookup at the target; that a PR can drop its own nested rule was judged
  not worth one ("das ist jetzt wirklich nicht wichtig"). A reworded rule
  is a removed one (text is the key), so its row holds back no new task:
  else the violation under the new wording would be a held-back repeat.
- Merge conflict author only: the push lands on the author's branch.
- A yes to the offer (user, 2026-11-15: "a reply with yes should simply be
  enough"; a word list misses "let's try that", so the model reads the
  rest). Not on a PR ignored since its summary was deleted: a plain reply
  never brings a PR back, so a yes there is not answered (known limit; the
  command with a mention works). Loosens 2026-11-13's "by code, never the model"
  only for whether a reply under the open offer agrees: a tool offered
  there alone, code's veto on a no or a put-off, the bounds and checks of
  the request untouched. A wrong yes is a merge commit on the author's
  branch, visible and revertable; the veto catches the plain no. Pronouns
  are no veto there ("ich bin einverstanden"), unlike the command's
  conflictSelf. The post starts the request through StageThen: the
  answer's post may stage again, the PR's hold kept.
- A reply that tags someone else is addressed to them (user, 2026-11-16):
  "@SJH, dann musst du …" under noergler's answer was answered as a Q&A.
  Tagged + no `@bot` = silence on every thread, the author-only answer
  included ("@jan bitte fixen" is for Jan). Decided at the route by text,
  no DB. A bare annotation in prose ("@Override fehlt") counts as a tag,
  accepted: checking real users costs a Bitbucket call per reply, and the
  payload's participants are not everyone tagged. Code spans never count.
- Author-only actions answer anyone else kindly, never silently (user,
  2026-11-15: general rule): a reviewer's "ok" under the offer is told only
  the author may ask; a reviewer's "fix it" under a finding, or a mention
  ordering a change, too (`changeRequest`, by code). Narrow on purpose: a
  short order only, no question ("how would you fix this?" is the Q&A's),
  nobody speaking of their own doing; a miss stays as before, silent under
  a finding. Under noergler's Q&A answer the answer's own notice stays (it
  is kept as a reply about the answer, so the author's "apply it" under it
  works).
- Merge conflict on the worker, not the pool: a fetch and a push are
  Bitbucket calls, and serialization is for Bitbucket. Bounded instead:
  60 s a stage, 300 MiB a fetch on the wire and 600 MiB inflated (one
  process serves every team: an OOM would take them all down), depth 50
  then 500. Paint-down merge base: a shallow fetch hides parents. An edge
  on one side's walk only is harmless when the other side's walk ended
  without one (a hidden commit would have been reached and fetched), so a
  target that merged an old branch keeps its base; both edges, or no base,
  is "fetch deeper", never a guessed base.
- Merge conflict request negated or self-spoken is none ("I'll resolve the
  conflict myself, just review"): a push the author did not ask for is
  worse than a command missed, which the Q&A then answers in words.
- Merge conflict's push re-reviews the whole PR (user, 2026-11-13: wanted):
  a merge commit in the range forces a full review, as after a merge by
  hand; skipping it only moves the full review to the next push.
- Merge conflict at the cap: code's resolution costs nothing, so it runs
  (like a stop); a model one is refused with an answer, because the author
  asked and a fix request never goes silent.
- Merge-conflict offer (user, 2026-11-14): authors did not know the command
  or that their conflict was simple. A comment of its own with a mention,
  not a summary line (the summary often goes unread) and not inline (the
  conflict sits in target's changes, which the PR diff does not show).
  Offered only, even when code alone resolves: a push nobody asked for on
  the author's branch. Past the bounds silent: Bitbucket shows the conflict,
  and the command names the reason if asked. At the cap nothing is checked
  and an open offer is withdrawn (user: stop there; a model resolution would
  be refused, so the offer would invite a refusal). At a review only, the first included (the common
  case: a branch cut from a stale master): `pr:to_ref_updated` is
  unavailable (BSERV-14136), and `repo:refs_changed` would deliver every
  push of every branch to check a few PRs.
- A finding noergler committed is retired silently and its thread left open:
  the confirmation already asks the author to review the change; resolving
  it would hide the change the author was told to look at.
- Q&A fix request as an envelope field, not a classifier call: every other
  reply under a Q&A answer keeps the full Q&A (diff, tools), and a second
  call to classify would cost every such reply. The Q&A prompt still never
  writes code; the fix call does, under its own system message.
- Q&A and finding-reply prompts never call noergler read-only: they decide
  `apply` and `requests_fix`, and told it was read-only the model refused
  every fix request with that reason ("kann ich als Read-only Reviewer nicht
  ändern"). The no-patches-in-the-answer rule stays.
- A Q&A answer's code line resolved when it is posted: Bitbucket anchors
  only a thread's root and sends a reply's immediate parent only, so at fix
  time the line would take a walk up the thread, a fetch per level. Posted,
  it is one fetch at most (a root question, paid also by a general one,
  accepted) and otherwise inherited. The fix's file is read before the
  call for every PR-author reply under an answer, accepted: a post cannot
  stage the fix call, so it must be ready when the model says `apply`.
- Kind `answer` apart from `question`: a notice or a language confirmation
  carries no suggestion, and "ok" under one must not start a fix that can
  only fail and count as not feasible.
- Q&A fix for the PR author only (user, 2026-11-13): a reviewer's question
  would otherwise push commits onto someone else's branch through the bot.
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

- Unchanged callers by Bitbucket code search, not by fetching the repo:
  go-git has no partial clone, so a depth-1 fetch pulls the whole working
  tree (3 MB for noergler itself, hundreds for a monorepo) on the single
  worker every push; one search is one request. The search reads the
  default branch only, which is exactly where a file the PR does not change
  stands as the PR sees it, when the PR goes there. One OR query for all
  names (the syntax takes 9 expressions, 250 characters), not one per name:
  the user asked for no burst of searches from one review (2026-11-20).
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
- Q&A staged: a tool loop is up to four gateway calls of the call timeout
  each (`REVIEW_INFERENCE_CALL_TIMEOUT`, 10m by default), which
  inline would hold the Bitbucket worker for every team's PRs.
- Mention tools read-only but the language and the stop: a write tool is a
  setting any commenter, or any injection the model reads, could change.
  The writes are the commenter's own language and a stop of this PR, and
  code checks their words first. A stop only ever saves money; a resume
  costs it, so there is no resume tool.
- Stop by command, not only by deleting the summary (user, 2026-11-04):
  deleting threw the summary away, and any later question brought the PR
  back. A stop keeps the summary (banner on top) and survives questions;
  only a review request resumes. A short command is code-decided so a PR
  at the cost cap can still be stopped.
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
- First name on a question: held back (talking down) -> always (user,
  2026-10-19) -> may, not must (user, 2026-10-31): always was too much.
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
