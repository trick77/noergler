# EVALS.md

The one place for what the evals FOUND: about the model, the prompt, the
corpus and the scorer. A new finding goes here, not into a results README,
a code comment or `AGENTS.md`.

- Rules for running evals: `AGENTS.md`.
- Run tables, one row per run: `backend/internal/evals/results/README.md`
  and `results/sampled/README.md`. Run names below (run4, rev5, ...) refer
  to those rows.

## Multi-review experiment (2026-10-02)

Question: misses vary run to run on an unchanged prompt. Do N reviews merged
by a fixed rule beat one?

Tooling, all in `cmd/evals`, none of it a series run:

- `-samples N`: N reviews per case, scored as single / vote / union / adaptive.
- `-confirm sampled.json`: union of three reviews + one consolidation call.
- `-mix`, `-rescore`, `-cases`, `-usage`.
- Results: `backend/internal/evals/results/sampled/`, never a series row.
- First sample alone, the rest after it: sent together they all miss the
  prefix cache.

Findings (gpt-5.4-mini high, 17 cases, 5 samples each):

- **Vote is wrong.** 2-of-3 scored like one review (10.3 vs 10.0 of 12) and
  dropped every bug found in under half the runs.
- **Union works.** 11.8 of 12, no invention on any control. Cost: the same
  bug posted twice, anchored on different lines.
- **File + line cannot merge duplicates.** Same bug, 3-5 lines apart.
- **A model that may drop findings loses real ones.** Keep-by-default judge
  dropped a real bug in 4 of 9 groups.
- **Merge-only call works.** Duplicates into one, drop nothing, add nothing:
  11.8 of 12, no duplicates. Fewer than two candidates = no call, the lone
  finding stands (handed one candidate, the model lost it once).
- **Self-rated severity is no filter.** An invented finding was rated `issue`
  while calling the change behaviour-preserving.
- **gpt-5.5 high misses too.** 4 of 5 on the hardest cases, nothing on
  `resource-leak`. One review each: thin.

Decision: production stays single-pass. Three mini reviews + merge cost about
what one gpt-5.5 review costs on small diffs (~$0.035 a case), and mini's 400K
window is under the 1M requirement. Worth revisiting if real PRs turn out
input-heavy (mini's input price is 6.67x lower) or a cheaper 1M model appears.

A second model merging findings is acceptable; one judging whether a finding
is true is not.

Why the cost came out equal: mini writes ~1,900 output tokens a review
(reasoning included), gpt-5.5 high ~600. Output is ~85% of a call's cost on
these diffs, so mini's 6.67x lower price shrinks to ~3x per call: ~$0.010
against $0.028 (15 calls over the corpus) to $0.037 (the five hardest cases).
Caching barely matters at this prompt size.

What the sample size can show: 5 samples a case sees a rate move by ~15
points, not by 5. The 10 triples per case share those 5 samples; precision is
the sample count's, never the group count's. The corpus is the bottleneck:
before rev6, 6 of 8 seeded cases were never missed.

Before the experiment, from the six anchored-scorer runs on rev5
(mimo-v2.5-pro high), per completed sample:

- seeded: 38 of 41 caught. All 3 misses on two cases:
  `context-not-propagated` 4 of 6, `struct-field-race` 4 of 5.
- controls: 2 invented findings in 24 samples, on two different controls.
- With independent runs a 2-of-3 vote is `3p^2 - 2p^3`: it helps above a 50%
  catch rate and hurts below. That threshold is why the vote fails on the
  hard bug.

Models allowed for eval runs: mimo and `gpt-5.4-mini`. No call on `gpt-5.5`,
`gpt-5.4` or anything above without the owner's okay for that run, with a
stated call count and budget. Use `-cases` and `-usage` to buy only the cases
that answer the question.

## Prompt experiment: completeness and newly broken lines (2026-10-02)

Question: do two edits taken from other review prompts raise recall? (1) "do
not stop at the first finding, a narrow trigger is still a bug"; (2) removed
lines and unmodified lines the diff breaks are in scope.

mimo-v2.6-pro high, 17 cases, 5 samples each, both prompts with severity and
confidence already untangled:

- without the edits: 54 of 60 seeded samples caught, controls clean in 25 of 25.
- with both: 52 of 60, controls clean in 25 of 25.

Decision: neither edit landed. No gain, and 2.6-pro leaves the corpus little
room to show one: 10 of 12 seeded cases were caught in every sample without the
edits. 5 samples see a 15-point move, so "slightly worse" is not shown either.

Side finding: on `cross-file-stale-caller` the model reported the bug in 7 of
10 samples and 6 were dropped as `evidence_not_found`. The finding on
`manager.go` quoted the changed `Touch` signature from `store.go`, and the
validator matched evidence against the finding's own file only. Fixed: a line
of another shown file is evidence, never an anchor, and one diff line of the
finding's own file is still required. Re-validated, 5 of the 6 are kept at
line 78 (57 and 55 of 60); the sixth quotes a line that is in no file. The
two JSONs are left as scored.

## Prompt experiment: committed secrets (2026-10-19)

Question: a real PR adding `public String password = "admin123";` was flagged
in about half its production runs. Does carving committed secrets out of the
failing-input rule raise that?

mimo-v2.6-pro high, `hardcoded-credential` + two controls (env read,
`changeme` test fixture), 5 samples each:

- before (`-prompt`, old template): caught 5 of 5, confidence 95-97, `issue`.
- after: caught 4 of 5; the miss has no finding at all.
- controls clean in 20 of 20 samples across both.
- full rev7 series run after: 11 of 13, controls clean 7 of 7.

The case does not reproduce the miss: the old prompt already catches it on
mimo. The real PR runs on the production model and carries other changes
("among other things"), neither of which the case has. With no room to
improve, the run cannot show a gain, and 4 vs 5 of 5 is noise.

After the code-review fixes (rev8: window on the password line, a
non-placeholder test fixture, the same password in `application.yaml`):

- Java case matched at line 21 in 5 of 5 samples, yaml case 5 of 5, both
  controls clean in 10 of 10.
- Series run: the Java finding was dropped `evidence_not_found` (two lines
  joined into one evidence entry, the known gap) and its comment named the
  value. The "never the value" rule holds most of the time, not always.

Decision: landed on the prompt's own contradictions, not on a measured gain.

## Split experiment: findings without the summary (2026-10-31)

Question: one call writes the findings and the PR summary (overview,
verdict, strengths, screenshot lines, ...). Does the summary work cost
findings recall, so that a findings-only call plus a summary call would
beat it?

mimo-v2.6-pro high, 26 cases rev10, 5 samples each. Arm B is
`briefs/review-findings-only.txt`: `review.txt` with every summary key
pinned to a fixed placeholder. Two confounds: pinning `visual_changes` to
`[]` also dropped "a screenshot is never a finding" from the brief, and the
model still wrote screenshot lines in 15 of 25 visual samples. Arm B did
less summary work, not none.

- full prompt: 68 of 75 seeded samples caught (single view), controls clean
  in 55 of 55.
- findings-only: 72 of 75, controls clean in 55 of 55.
- Misses with no finding at all: 6 against 3, on different cases each
  arm. Noise at 5 samples.
- `visual-hidden-submit` 3 of 5 against 5 of 5. Both full-prompt misses
  described the hidden button and its CSS cause correctly, in
  `visual_changes` only: the bug went to the summary, not into `findings`.
  `review.txt` says "a screenshot is never a finding", which the model can
  read as "a bug a screenshot shows is not a finding"; arm B had lost that
  sentence, so its 5 of 5 does not separate the two causes. The follow-up
  below does.
- Completion ~1,070 tokens a call against ~730; prompt 5,610 against 4,449.

+4 of 75 is about 5 points, under the ~15 a 5-sample run can show. The
split would add a second call that resends the whole diff, so it costs more
than the one call it replaces.

Decision: production stays one call. The `visual-hidden-submit` routing is
a prompt question for `review.txt`, not a reason to split.

Follow-up, the same day: the full prompt, summary work and screenshot rule
kept, plus one sentence after "a screenshot is never a finding": "A bug in
the code stays a finding when a screenshot also shows it: anchor and
evidence from the code, as if no screenshot were sent." Run with `-prompt`;
the variant is `review.txt` since this landed. The 5 visual cases, 5
samples each:

- `visual-hidden-submit` 5 of 5 (3 of 5 above), every finding on the CSS
  line, none naming the image.
- the four visual controls clean in 20 of 20: an explained or drifting
  screenshot did not turn into a finding.
- screenshot lines still written: 33 of 35 scored, the 2 misses correct
  lines worded outside the case's keywords.

The screenshot stays summary only (INVARIANTS: test screenshots); the
sentence keeps a code bug from leaving `findings` because an image shows it
too.

## Questions instead of silence (2026-11-20)

Question: `review.txt` withholds a suspected defect whose failing path
depends on unseen code ("when in doubt, withhold"). Would a question to the
author, for exactly that band (confidence under 80), turn misses into leads,
and at what cost on clean code?

mimo-v2.6-pro high, 40 cases rev16, 5 samples, two arms run side by side
(`sampled/2026-11-20-…-questions-base`, `…-questions`). The variant is
`-questions internal/evals/briefs/questions.txt`: the brief appended to the
assembled prompt and a `questions` key (0–2 items: file, line, evidence,
question, confidence) added to the schema, eval only. A question is
validated like a finding and scored by window: on a control, rescued (in a
seeded window with no finding there), redundant (beside one), elsewhere,
dropped.

- Questions: 2 in 200 reviews, both on controls, both plausible guesses
  about unseen code (`Store.Invoice` returning nil with nil error; a second
  `DataSource` bean taken by `@Primary`). 0 rescued, 0 redundant, 0 dropped,
  confidence 50 and 55. The run predates the per-item parse count: a
  question item that failed to decode would not show, though the strict
  schema types every field.
- Findings unchanged: 110 of 115 seeded samples caught in both arms, the
  misses on different cases each arm (noise). No question landed on a
  missed bug in the variant arm.
- Extras 2 against 3; the variant's third is a re-raise of an open finding
  on `incremental-moved-not-fixed`, not a question.
- Cost: +272 prompt tokens a call for the brief.

The corpus leaves the benefit almost no room (the series catches 30–31 of
31), so the run measures cost more than gain. On it the model all but
never asks, and when it does, it asks about clean code.

Recommendation, the decision open: no questions in production; the
withhold rule stands. Kept: the `-questions` flag and the brief, so the
experiment can be repeated on a corpus with withheld-band misses or on
another model.

## A fence inside a reviewed file (2026-11-20)

Question: does a ``` in a reviewed file, which closed the prompt's block
early before the block fence followed the file, cost recall?

mimo-v2.6-pro high, `fenced-docstring` (rev15) alone, five samples a side,
both on master, the before arm with only `blockFence` pinned to three
backticks (`sampled/2026-11-20-…-fence-before`, `…-fence-fixed`; prompts
5988 against 5989 tokens): caught 5 of 5 both ways.

- A file block that closes early costs nothing while the diff carries the
  bug: in the before arm the file's own closing fence opens a stray block
  that swallows the `### Changes` header and runs to the diff's end, so the
  changed line stays fenced, and every finding anchored on it (line 16).
  Code seen only as loose prompt text is not measured, and cannot be on a
  bug in the diff. The fix stays as hygiene.
- The 2026-11-19 pair shows nothing: rev13's fences were indented, which
  closes no block, and its before arm was an older prompt.

## Bait controls (2026-11-23)

Question: teams complained of nitpicking and irrelevant findings, a team on
`gpt-6.2-sol` among them. Does the corpus catch a model that does that, and
does mimo do it?

Four complained-about findings were looked at. All four built a failure on
something no shown code proves: a second tab and the server's 409, a 110 px
viewport, a business rule of the backend, a layout outside the diff; one
also against a restriction the diff states in its first line. The prompt
forbids each step ("do not speculate about unseen files", confidence 50 for
a path through unseen code, report from 80). So the rules were there and
were not followed. All four were posted, so their evidence had passed
`ValidateFindings`, which checks quoted lines and not the premise.

rev17 adds 11 baits (`bait-*`) and 4 seeded edge cases (`edge-*`). A bait is
a correct diff with one such temptation. Two edge cases are twins, the bait's
code with the condition made real: `edge-early-return-skips-check` of
`bait-state-excluded-by-domain`, `edge-guard-inverted` of
`bait-deliberate-scope-guard`. mimo-v2.6-pro high, 18 cases (three baits were
withdrawn after the runs, below), 5 samples; a second arm with
`briefs/review-no-restraint.txt`: "What NOT to flag" and the withhold
sentences cut, every failure asked for. It keeps the report-from-80 gate
and the rubric that puts a path through unseen code at 50 (the schema's
minimum is 80 too), so a bait silent there is not shown to be unable to
fire.

- **mimo does not reproduce the complaints.** The four rebuilt ones are
  silent in both arms. That is what `series.yaml` warns of: a false positive
  of the production model may not show on the stand-in. It does not show the
  model is the cause: which model wrote the four is not confirmed, no
  `gpt-6.x` run was made (it needs the owner's okay), and the rebuilds are a
  few dozen lines where the real PRs carry far more context to speculate
  about.
- **The 11 baits are silent on the production prompt**: 0 findings in 55
  samples and in the series run. In the second arm one fires,
  `bait-style-and-tests` (negative years) at 0.4 findings a sample; 0.04 a
  review over all 11.
- **Ten baits fire in no arm**, the four rebuilt complaints among them. They
  guard against a prompt edit that makes mimo start; whether they tempt
  another model is not measured.
- **Three baits were withdrawn.** `bait-invariant-assertion` (a non-null
  assertion on an optional field whose doc comment says it is always set:
  the type says otherwise) and `bait-paid-invoice-payment` (an unguarded
  `getPayments().get(0)` behind a PAID check: a senior reviewer would ask
  too) fired at 0.6 findings a sample on the production prompt, and most
  series runs would have been red on them alone. `bait-shortcut-field-guard`
  was wrong twice: its selector matched `contenteditable="false"`, then, fixed,
  every checkbox and radio against its own comment. Their samples are in the
  `-withdrawn` files.
- **The edge cases hold**, on the production prompt: `edge-failure-reported-as-success`
  and `edge-guard-inverted` 5 of 5, `edge-paid-without-payment` 4 of 5,
  `edge-early-return-skips-check` 3 of 5, the rest no finding at all
  (5, 5, 5 and 4 of 5 without restraint). Silence on edge cases does not
  win on these four.
- **Read every finding on a bait before scoring it.** Four of the first 18
  cases were built wrong, and the model was right: the panel handler (a bait
  and its twin) kept a pointer flag no release outside cleared, and after
  that fix a touch tap could reset it before `focusout` ran; a selector
  matched `contenteditable="false"`; a comment named a companion file the
  diff leaves alone. `…-bait-draft` and `…-bait-no-restraint-draft` hold the
  flag, touch and selector reports; the one on the companion file and the
  samples the reruns replaced were overwritten and cannot be checked.
- Five samples see a rate move by about 15 points, not 5.

## Blocking rules (2026-11-15)

Question: given `[blocking]` rules from AGENTS.md, does the model name the
broken rule on a real violation, and only there?

mimo-v2.6-pro high, the seven rev11 blocking cases, one review each:
four violations (SQL in a controller, a missing tenant check, float money,
PII in a log line), one unrelated bug in a repo with rules, two controls
that obey the rule.

- 4 of 4 violations claimed the right id and passed `CheckBlocking`
  (`issue`, confidence >= 90): no downgrade for severity or confidence.
- The unrelated bug was caught without a claim; both controls clean.
  0 false tasks.
- ~39k prompt and ~5k completion tokens for the run.

One sample per case: this shows the claim works, not how often. The
measure that matters for teams is false tasks, since a wrong one blocks a
merge; a larger run should weigh controls near a rule's topic.

## Merge-conflict resolution (2026-11-14)

Question: how often does a resolution go wrong unnoticed (a commit nobody
checked), on the code path and on the model path?

**Code path** (`merge.Trivial`, no model, runs past the cost cap).
`cmd/conflictmine` over 1500 git/git merges and 20 public trick77/ipverse
repos, every both-changed file checked as production checks it: 69
conflicts within production's bounds, 13 (19 %) resolved by code (12
distinct: git/git merged the replay.c conflict twice). Against the merge
author: 4 identical, 5 the same lines in another order, 3 differing in
blank lines (none of these 8 checked beyond that), 1 broken: git/git 8f0d663e
`t/lib-gpg.sh`, both branches added the same function with headers
differing in one space; target then source keeps both headers and the
script no longer parses (`sh -n`). Fixed since: lines and keys compare
without whitespace, and a conflict whose edges Merge3 trimmed (a block both
added) is never trivial. Mined again under that rule: code resolves 4 of
the 69 (6 %), each the author's lines in another order (two include lists,
two meson test lists), none broken; the other 9 go to the model. Order: the author put the branch
merged into first, code puts target first; in git/git that branch is
master, in production the PR branch, so the data cannot say which order
is right.

**Model path**, mimo-v2.6-pro high, 35 cases rev1, 10 answers each
(results/conflicts): 138 of 200 solvable answers correct, 44 declined;
controls 144 of 150 declined or rejected, 6 wrong. Wrong answers cluster
by case (0 or 5 of 10), so count cases, not answers: 2 of 15 controls and
1 of 20 solvable cases drew a real wrong. Read by hand, the 23 wrong:

- 10 keep base lines one branch deleted and the other left as they were
  (a removed installer block comes back; both fruits of the git docs'
  diff3 example kept). The one real pattern.
- 1 keeps a line the other branch deleted, where it changed it
  (delete vs modify).
- 12 are the corpus, not the model: the injected comment dropped, the
  rest correct; a version line merged three-way where the author kept
  the PR's. For rev2.

That pattern is mechanical: for a text the base region has, keep it at most
max(0, t + s - b) times. Rescored offline, a blunt check of that catches
all 10 and rejects none of the 138 correct answers; against the 45 author
resolutions of every mined draft it rejects one (the version case, where
the author kept a line the other branch changed: a refusal, never a bad
push). As a prompt rule (read each branch against B; a B line one branch
kept and the other changed or deleted follows the change): 0 of 10 wrong
on the two cases (was 10 of 20), the 7 solvable cases that were right
stayed right (60 answers, `-draft.json`). Drawn from the two cases it was
tested on, on 12 of 35 cases; the abort clauses added with it (delete vs
modify: 1 of 5 against 1 of 10, no effect seen; built files) are untested.

The rule alone went into `merge_conflict.txt` (user, 2026-11-14) and ran on
all 32 cases of rev2 (run2): 0 wrong of 190 solvable answers (rev1: 17 of
200, 5 of them real), 1 of 130 on controls (rev1: 6 of 150). No case with
braces or blank lines in its conflict lost a correct answer, the worry that
a text rule drops a `}` the other branch still has. Delete vs modify was
declined 10 of 10 without its abort clause. The one wrong keeps the PR's
version of a line both branches changed, where target's flag change is lost:
a line neither side carries in full, the case the prompt already declines.

Declines are the cost: version bumps both branches set (`agents-md-sync`,
`bd5c2827b2`) and a docstring both rewrote were declined in every answer.
The author picked; the prompt says to leave a value both set to them.

The check went into `merge.ValidateResolution` as well (`undoes_change`,
user, 2026-11-14), so a later run scores such an answer `rejected`, not
`wrong`. Replayed through it, every stored answer: rev1's 10 base-line
wrong answers rejected, none of 313 correct ones (rev1, draft, run2);
1 of 39 author resolutions from a fresh mine refused (the version case).
The run2 wrong (`42afcb954b`) passes it: no base line is kept.

## Known gaps

- `visual-hidden-submit` `button-gone` keywords miss correct lines worded
  "drops" or "loses" the button.
- Validator dropped 3 correct mini findings (`evidence_not_found`): two
  lines joined into one evidence string, or prose appended to a quoted line.
  The quote is not verbatim, so the drop stands.
- A doc comment line quoted without its `//` was `evidence_not_found`: the
  2.6-pro series row lost a correct `buried-behavioural-hunk` finding to it.
  Fixed: a quote that is a shown comment line minus its marker is that line.
  Re-validated, the finding is kept at line 64 and scores. The JSON is left
  as scored. Still open: a quote that is only PART of a comment line.
- Direct mimo host: any review thinking past ~98 s dies on llmwire's 90 s
  idle bound (`stream idle for 1m30s`). Not a gateway stall. A series run
  now redraws such a case like a sampled run does (twice at most, counted
  as `Redraws`); the redraw is the one that thought less.

## Single-run series (2026-09-20 to 2026-10-01, mimo-v2.5-pro high)

### The baseline

run4 and run5 are the pair. Same prompt, same model, same effort, same
corpus revision, **8 caught and 7 caught**. That spread is the baseline: a
single later run scoring 7 is inside the noise, and only a run scoring 6 or
less, or a second consecutive 7 after a prompt edit, is a signal.

Every run scored the same prompt text: the review prompt (then
`prompts/review.txt`, now `backend/internal/prompts/review.txt`) had one
commit at the time. Every difference between these rows is model
non-determinism, a corpus change or a scorer change, never a prompt change.

**Exit 1 has fired on two of the three rev4 runs with the prompt unchanged.**
It therefore does not mean "the prompt regressed"; it means the model had an
off run, and the cause is named in the row. Treat a single red as noise and a
second consecutive one as the signal, which is the same rule the caught count
already follows. Nothing in the tool enforces that: the exit code stays strict
so a CI job could gate on it later without a threshold to tune.

The three clean controls scored **0 findings in runs 1 through 5** (run1
predates two of them and had only `clean-refactor`). **run6 broke that**: one
finding on `clean-refactor`, the first in six runs, which is what the new gate
exists to catch. The controls are the honest false-positive floor and are what
makes Caught mean anything, so the run is red and stays red in the record.

### Extra is not the false-positive count

Extra has been inflated on a seeded case twice, neither time by invention:

- **`lock-not-released`, run2**: the model emitted the same finding twice,
  byte for byte, both at line 22. The duplicate pins nothing new and counts
  Extra. run5's single Extra is not this; it is the line-39 miss below.
- **`context-not-propagated`, on rev1**: the model reported the two
  `context.Background()` substitutions as two separate findings. rev1 had
  one expectation, which absorbs one of them, so the second honest finding
  scored Extra. rev2 split the expectation in two, and run2 then scored a
  *miss*, because that time the model emitted one finding naming both call
  sites. Both shapes are correct reviews, so rev3 settled on one
  expectation over a window spanning both. That rev1 run is not in the
  results directory: it was scored against a corpus state no committed row uses,
  and keeping it would have implied a comparison it cannot support.

Neither is a false positive. The clean controls are.

run6's third Extra is the control finding, and it is counted as invention on
purpose. Its text argues the refactor is *correct* ("No bug is introduced by
the extraction"), which is tempting to excuse. It is not excused: production
posts `severity: suggestion` findings, `severityOrder` only sorts them below
`issue` and nothing drops them, so that paragraph would have been a comment on
the PR. Filtering it out by reading what it says is the judge model the
bluntness rule forbids.

### Anchoring and line numbers

- `nil-deref` was found at line 19 in runs 1, 2 and 5, at 22 in run3 and at
  20 in run4:
  the dead `if r.Ticket == nil` check, the comment inside it, and the
  dereference. All are honest anchors for the same bug, which is why the
  window spans them. A narrower window scores a correct review as a miss.
- **A whole-function defect is anchored at the func declaration** about as
  often as at the first statement. run3 lost `resource-leak` that way. The
  four whole-function windows now start at the declaration; the
  narrow-statement cases do not, because widening `unchecked-error` to its
  func line would span the entire body and stop being falsifiable.
- **The model can cite a line past the end of the file.** Twice in six runs,
  on two different cases: run5 put the data race at line 39 of a 37-line
  file, run6 put the body leak at line 32 of a 29-line one. Both reviews were
  correct; no window can catch either, and widening one to reach a
  nonexistent line would only make the case unwinnable in the other
  direction. run6's finding also cites "line 24" for a call on line 17 in its
  own prose, so the number is unreliable rather than merely offset.
