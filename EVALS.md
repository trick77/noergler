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

## Known gaps

- `lock-not-released` keywords miss correct findings; inflates `extra`.
- Validator dropped 3 correct mini findings (`evidence_not_found`): two
  lines joined into one evidence string, or prose appended to a quoted line.
  The quote is not verbatim, so the drop stands.
- A doc comment line quoted without its `//` was `evidence_not_found`: the
  2.6-pro series row lost a correct `buried-behavioural-hunk` finding to it.
  Fixed: a quote that is a shown comment line minus its marker is that line.
  Re-validated, the finding is kept at line 64 and scores. The JSON is left
  as scored. Still open: a quote that is only PART of a comment line.
- Direct mimo host: any review thinking past ~98 s dies on llmwire's 90 s
  idle bound (`stream idle for 1m30s`). Not a gateway stall.

## Single-run series (2026-09-20 to 2026-10-01, mimo-v2.5-pro high)

### The baseline

run4 and run5 are the pair. Same prompt, same model, same effort, same
corpus revision, **8 caught and 7 caught**. That spread is the baseline: a
single later run scoring 7 is inside the noise, and only a run scoring 6 or
less, or a second consecutive 7 after a prompt edit, is a signal.

Every run scored the same prompt text: `prompts/review.txt` has one commit in
its entire history. Every difference between these rows is model
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
